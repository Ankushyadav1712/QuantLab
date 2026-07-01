"""Pre-populate ticker lists and GICS cache for large universes.

Run once before using the sp500 or russell1000 universe presets so that
the first simulate request doesn't have to wait for the web fetch.

    python backend/scripts/download_universe_tickers.py

Optional flags:
    --universe sp500        Only refresh the S&P 500 list (default: all)
    --universe russell1000  Only refresh the Russell 1000 list
    --force                 Re-fetch even if cache is still fresh

After this script completes:
  backend/data/tickers/sp500.txt        ~503 tickers
  backend/data/tickers/russell1000.txt  ~1000 tickers
  backend/data/cache/gics_dynamic_cache.json  sector/industry data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure backend/ is on the path so we can import the data modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.universes import (  # noqa: E402
    _GICS_CATALOG,
    _GICS_DYNAMIC_CACHE,
    _TICKER_LIST_TTL,
    _TICKERS_DIR,
    _fetch_sp500_tickers_and_gics,
    _get_russell1000_tickers,
    _save_gics_dynamic_cache,
)


def _refresh_sp500(force: bool = False) -> None:
    path = _TICKERS_DIR / "sp500.txt"
    if not force and path.exists() and (time.time() - path.stat().st_mtime) < _TICKER_LIST_TTL:
        print(f"[sp500] cache is fresh ({path}), skipping (use --force to override)")
        return

    print("[sp500] fetching tickers + GICS from Wikipedia …")
    try:
        tickers, gics_map = _fetch_sp500_tickers_and_gics()
    except Exception as exc:
        print(f"[sp500] ERROR: {exc}")
        return

    _TICKERS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers))
    print(f"[sp500] saved {len(tickers)} tickers → {path}")

    need_cache = {t: v for t, v in gics_map.items() if t not in _GICS_CATALOG}
    if need_cache:
        _save_gics_dynamic_cache(need_cache)
        print(f"[sp500] cached GICS for {len(need_cache)} tickers → {_GICS_DYNAMIC_CACHE}")
    else:
        print("[sp500] all tickers already in static GICS catalog")


def _refresh_russell1000(force: bool = False) -> None:
    path = _TICKERS_DIR / "russell1000.txt"
    if not force and path.exists() and (time.time() - path.stat().st_mtime) < _TICKER_LIST_TTL:
        print(f"[russell1000] cache is fresh ({path}), skipping (use --force to override)")
        return

    print("[russell1000] fetching tickers from iShares IWB holdings …")
    try:
        tickers = _get_russell1000_tickers()
    except Exception as exc:
        print(f"[russell1000] ERROR: {exc}")
        return

    if not tickers:
        print("[russell1000] ERROR: no tickers returned")
        return

    print(f"[russell1000] saved {len(tickers)} tickers → {path}")

    # Fetch GICS for Russell 1000 stocks not already in sp500 GICS cache or catalog
    from data.universes import _load_gics_dynamic_cache

    existing_gics = set(_GICS_CATALOG.keys()) | set(_load_gics_dynamic_cache().keys())
    need_gics = [t for t in tickers if t not in existing_gics]
    if need_gics:
        print(f"[russell1000] fetching GICS for {len(need_gics)} new tickers via yfinance …")
        import yfinance as yf

        new_entries: dict[str, list[str]] = {}
        from data.universes import _YF_TO_GICS

        batch_size = 50
        for i in range(0, len(need_gics), batch_size):
            batch = need_gics[i : i + batch_size]
            for t in batch:
                try:
                    info = yf.Ticker(t).info
                    sector = _YF_TO_GICS.get(info.get("sector", "Unknown") or "Unknown", "Unknown")
                    industry = info.get("industry", "Unknown") or "Unknown"
                    new_entries[t] = [sector, sector, industry, industry]
                except Exception:
                    new_entries[t] = ["Unknown", "Unknown", "Unknown", "Unknown"]
            pct = min(100, round(100 * (i + batch_size) / len(need_gics)))
            print(f"  … {pct}% ({min(i + batch_size, len(need_gics))}/{len(need_gics)})")
            if i + batch_size < len(need_gics):
                time.sleep(0.5)

        _save_gics_dynamic_cache(new_entries)
        print(
            f"[russell1000] GICS cache updated ({len(new_entries)} entries) → {_GICS_DYNAMIC_CACHE}"
        )
    else:
        print("[russell1000] GICS already cached for all tickers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        choices=["sp500", "russell1000", "all"],
        default="all",
        help="Which universe to refresh (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if the cache is still fresh",
    )
    args = parser.parse_args()

    if args.universe in ("sp500", "all"):
        _refresh_sp500(force=args.force)

    if args.universe in ("russell1000", "all"):
        _refresh_russell1000(force=args.force)

    print("\nDone.  You can now select these universes in the QuantLab UI.")


if __name__ == "__main__":
    main()
