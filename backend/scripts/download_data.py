from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import DATA_END, DATA_START, UNIVERSE
from data.fetcher import DataFetcher


def main() -> int:
    print(f"QuantLab data download — {len(UNIVERSE)} tickers, {DATA_START} → {DATA_END}")
    fetcher = DataFetcher()

    successes: list[str] = []
    failures: list[str] = []
    started = time.time()

    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i:>2}/{len(UNIVERSE)}] {ticker} ...", end=" ", flush=True)
        # compute_derived=False inside the loop — we'd otherwise overwrite the
        # derived parquets with single-ticker data on every iteration.
        frames = fetcher.download_universe(
            tickers=[ticker],
            start=DATA_START,
            end=DATA_END,
            compute_derived=False,
        )
        if ticker in frames and not frames[ticker].empty:
            rows = len(frames[ticker])
            print(f"ok ({rows} rows)")
            successes.append(ticker)
        else:
            print("FAILED")
            failures.append(ticker)

    elapsed = time.time() - started
    print(
        f"\nDownload done in {elapsed:.1f}s — "
        f"{len(successes)} ok, {len(failures)} failed."
    )
    if failures:
        print("Failed tickers:", ", ".join(failures))

    if successes:
        print(f"Computing derived fields for {len(successes)} tickers...", flush=True)
        derived_started = time.time()
        fetcher.download_universe(
            tickers=successes, start=DATA_START, end=DATA_END, compute_derived=True
        )
        print(f"Derived fields ready in {time.time() - derived_started:.1f}s.")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
