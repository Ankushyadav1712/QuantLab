#!/usr/bin/env python3
"""Pre-seed the parquet cache at build time.

Run this during ``render.yaml`` buildCommand or ``Dockerfile`` build so that
the first cold start doesn't need to hit the network for financial data.

Usage:
    cd backend && python scripts/preseed_cache.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure the backend package root is on sys.path so flat imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("preseed")


def main() -> None:
    start = time.time()

    # Force production config so DATA_START matches what the server will use.
    os.environ.setdefault("ENVIRONMENT", "production")

    from config import CACHE_DIR
    from data.factors import download_ff5_daily
    from data.fetcher import DataFetcher
    from data.macro import download_macro
    from data.universes import all_tickers as universe_all_tickers

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Universe tickers (batch download) ----
    log.info("Downloading universe tickers...")
    t0 = time.time()
    fetcher = DataFetcher()
    pool = sorted(universe_all_tickers())
    fetcher.download_universe(tickers=pool)
    log.info(f"  Universe: {len(fetcher._frames)} tickers in {time.time() - t0:.1f}s")

    # ---- 2. SPY benchmark ----
    log.info("Downloading SPY benchmark...")
    t0 = time.time()
    spy_fetcher = DataFetcher()
    spy_fetcher.download_universe(tickers=["SPY"], compute_derived=False)
    log.info(f"  SPY: done in {time.time() - t0:.1f}s")

    # ---- 3. Fama-French 5 factors ----
    log.info("Downloading Fama-French 5 factors...")
    t0 = time.time()
    ff5 = download_ff5_daily(force=True)
    log.info(f"  FF5: {ff5.shape[0]} rows in {time.time() - t0:.1f}s")

    # ---- 4. Macro (FRED) ----
    log.info("Downloading macro data from FRED...")
    t0 = time.time()
    macro = download_macro()
    log.info(f"  Macro: {len(macro)} series in {time.time() - t0:.1f}s")

    # ---- 5. Fundamentals (optional, can be slow) ----
    close_mat = fetcher.get_data_matrix("close")
    if close_mat is not None and not close_mat.empty:
        log.info("Downloading fundamentals...")
        t0 = time.time()
        try:
            from data.fundamentals import download_fundamentals

            fund = download_fundamentals(
                tickers=list(close_mat.columns),
                daily_index=close_mat.index,
                close_matrix=close_mat,
            )
            log.info(f"  Fundamentals: {len(fund)} fields in {time.time() - t0:.1f}s")
        except Exception as exc:
            log.warning(f"  Fundamentals failed (non-fatal): {exc}")

    elapsed = time.time() - start
    n_files = len(list(CACHE_DIR.glob("*.parquet")))
    log.info(f"Pre-seed complete: {n_files} parquet files in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
