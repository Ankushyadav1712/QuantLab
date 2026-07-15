"""Build the liquidity-ranked ``us_top1000`` universe.

WorldQuant Brain's TOP1000 is the top 1000 US names by *dollar-trading-volume*
(liquidity), not market cap. This script reproduces that selection method:

    1. Assemble a broad candidate pool (Russell 3000 = Russell 1000 ∪ 2000).
    2. Batch-download recent OHLCV from yfinance.
    3. Rank by median daily dollar volume (close × volume).
    4. Write the top 1000 symbols to backend/data/tickers/us_top1000.txt.

Run it OFFLINE (your machine or a Render build step) — it fetches a few
thousand tickers, which the free web dyno can't hold in memory. The running
server only ever reads the resulting 1000-name file (same footprint as
russell1000). Until this runs, the ``us_top1000`` universe transparently
falls back to Russell 1000.

    python backend/scripts/build_top1000_universe.py
    python backend/scripts/build_top1000_universe.py --top 1000 --lookback-days 90
    python backend/scripts/build_top1000_universe.py --pool-file my_symbols.txt

What it does NOT fix: this is still a current-snapshot, yfinance-sourced list —
it closes the *selection-method* gap with Brain, not the point-in-time
membership or data-vendor gaps. See docs/backtest-settings.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.universes import (  # noqa: E402
    _TICKERS_DIR,
    _get_russell1000_tickers,
    _get_russell2000_tickers,
    rank_by_dollar_volume,
)


def _candidate_pool(pool_file: str | None) -> list[str]:
    """Russell 3000 (1000 ∪ 2000) by default, or a user-supplied symbol file.

    Falls back to Russell 1000 alone if the 2000 fetch fails — the result then
    barely differs from the russell1000 universe, which the script warns about.
    """
    if pool_file:
        syms = [ln.strip().upper() for ln in Path(pool_file).read_text().splitlines()]
        return sorted({s for s in syms if s})

    r1000 = _get_russell1000_tickers()
    try:
        r2000 = _get_russell2000_tickers()
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: Russell 2000 fetch failed ({exc}); using Russell 1000 only")
        r2000 = []
    pool = sorted(set(r1000) | set(r2000))
    if len(pool) < 1500:
        print(
            f"  warning: candidate pool is only {len(pool)} names — the ranked "
            "top 1000 will barely differ from russell1000. Supply a broader "
            "--pool-file for a meaningful liquidity ranking."
        )
    return pool


def _dollar_volume_frame(tickers: list[str], lookback_days: int, batch: int):
    """Download OHLCV in batches and return a (dates × tickers) close×volume frame."""
    import pandas as pd
    import yfinance as yf

    period = f"{max(lookback_days + 15, 30)}d"  # pad for non-trading days
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        print(f"  fetching {i + 1}-{i + len(chunk)} / {len(tickers)}…", flush=True)
        try:
            data = yf.download(
                chunk,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    batch failed ({exc}); skipping")
            continue
        # yfinance's column shape depends on version + chunk size: multi-ticker
        # gives a (ticker, field) MultiIndex; a single ticker often gives flat
        # OHLCV columns.  Key off the actual columns, not len(chunk), so the
        # final 1-ticker chunk (when len(pool) % batch == 1) is handled too.
        is_multi = isinstance(data.columns, pd.MultiIndex)
        for t in chunk:
            try:
                sub = data[t] if is_multi else data
                dv = (sub["Close"] * sub["Volume"]).rename(t)
                if dv.notna().any():
                    frames.append(dv)
            except (KeyError, TypeError):
                continue
    if not frames:
        return None
    return pd.concat(frames, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the liquidity-ranked us_top1000 universe.")
    ap.add_argument("--top", type=int, default=1000, help="how many names to keep (default 1000)")
    ap.add_argument("--lookback-days", type=int, default=90, help="ranking window (default 90)")
    ap.add_argument("--batch", type=int, default=100, help="yfinance download batch size")
    ap.add_argument("--pool-file", default=None, help="candidate symbols file (one per line)")
    ap.add_argument("--dry-run", action="store_true", help="print the result, don't write the file")
    args = ap.parse_args()

    print("Assembling candidate pool…")
    pool = _candidate_pool(args.pool_file)
    print(f"  {len(pool)} candidates")
    if not pool:
        print("No candidates — aborting.")
        return 1

    print(f"Downloading ~{args.lookback_days}d of OHLCV…")
    dv = _dollar_volume_frame(pool, args.lookback_days, args.batch)
    if dv is None or dv.empty:
        print("No price data fetched — aborting (check network / yfinance).")
        return 1

    # Require roughly half the window of real trading days so recent IPOs /
    # delisting stubs with a day or two of data can't rank into the universe.
    min_days = max(10, int(args.lookback_days * 0.4))
    ranked = rank_by_dollar_volume(dv, args.top, min_days=min_days)
    print(f"Ranked {dv.shape[1]} names with data (≥{min_days}d); keeping top {len(ranked)}.")
    print("Top 10 by liquidity:", ", ".join(ranked[:10]))

    if args.dry_run:
        print("(dry run — not writing)")
        return 0

    _TICKERS_DIR.mkdir(parents=True, exist_ok=True)
    out = _TICKERS_DIR / "us_top1000.txt"
    out.write_text("\n".join(ranked) + "\n")
    print(f"Wrote {len(ranked)} tickers → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
