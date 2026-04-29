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
        frames = fetcher.download_universe(
            tickers=[ticker], start=DATA_START, end=DATA_END
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
        f"\nDone in {elapsed:.1f}s — "
        f"{len(successes)} ok, {len(failures)} failed."
    )
    if failures:
        print("Failed tickers:", ", ".join(failures))
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
