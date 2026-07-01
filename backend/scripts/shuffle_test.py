"""CLI for the shuffle leakage test.

    python backend/scripts/shuffle_test.py --expr "rank(close)" --n-shuffles 50

Loads the backend's data layer (so the test runs on the same parquet
cache + universe + GICS map as the API), then runs ``run_shuffle_test``
and prints the verdict.

Slow — 50 shuffles × ~2 sec each ≈ 1.5–3 minutes depending on the
expression complexity.  That's expected: shuffle tests are a once-per-
alpha confidence check, not a per-request operation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from analytics.shuffle_test import run_shuffle_test
from config import DATA_END, DATA_START, SECTOR_MAP, UNIVERSE
from data.fetcher import ALL_FIELDS, DataFetcher
from data.universes import default_universe_id, get_universe, gics_data_frames, gics_for
from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the shuffle leakage test on an alpha expression.")
    p.add_argument("--expr", required=True, help="DSL expression to test, e.g. 'rank(close)'")
    p.add_argument(
        "--n-shuffles",
        type=int,
        default=50,
        help="Number of random permutations (more → tighter p-value, slower).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility.")
    p.add_argument(
        "--universe",
        default=None,
        help="Universe id (defaults to the project default).  Use one of /api/universes.",
    )
    return p.parse_args()


def _load_data():
    """Use the same data pipeline as the API's lifespan handler — fetcher
    builds the (date × ticker) matrices, GICS map is loaded separately."""
    fetcher = DataFetcher()
    print(f"Loading data: {DATA_START} → {DATA_END} ({len(UNIVERSE)} tickers)...")
    t0 = time.time()
    fetcher.download_universe(UNIVERSE, DATA_START, DATA_END, compute_derived=True)
    data = {field: fetcher.get_data_matrix(field) for field in ALL_FIELDS}
    close_mat = data.get("close")
    gics_data = (
        gics_data_frames(close_mat.index, list(close_mat.columns))
        if close_mat is not None and not close_mat.empty
        else {}
    )
    data = {**data, **gics_data}
    print(f"  ({time.time() - t0:.1f}s)")
    return data, close_mat


def main():
    args = _parse_args()
    data, close_mat = _load_data()

    universe_id = args.universe or default_universe_id()
    try:
        u = get_universe(universe_id)
    except KeyError:
        print(f"Unknown universe: {universe_id!r}")
        sys.exit(2)
    tickers = u["tickers"] if isinstance(u, dict) else u.tickers
    # GICS map is built separately from the universe registry — same pattern
    # as ``main._resolve_universe`` does on the API path.
    gics_map = gics_for(tickers)

    cfg = SimulationConfig(
        universe=tickers,
        start_date=str(DATA_START),
        end_date=str(DATA_END),
    )

    def evaluator_factory(d):
        return AlphaEvaluator(d)

    def backtester_factory(d):
        return Backtester(d, sector_map=SECTOR_MAP, gics_map=gics_map)

    print(
        f"\nRunning shuffle test: expr={args.expr!r} · "
        f"n_shuffles={args.n_shuffles} · seed={args.seed}"
    )
    print("(this can take 1-3 minutes for a 50-shuffle run)\n")

    last_print = [0.0]

    def progress(i: int, total: int) -> None:
        # Throttle to one update per second so the terminal stays readable
        now = time.time()
        if now - last_print[0] >= 1.0 or i == total:
            print(f"  shuffle {i:>3d}/{total}", end="\r", flush=True)
            last_print[0] = now

    t0 = time.time()
    result = run_shuffle_test(
        args.expr,
        data=data,
        backtester_factory=backtester_factory,
        evaluator_factory=evaluator_factory,
        config=cfg,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        progress_callback=progress,
    )
    print()  # clear the progress line
    elapsed = time.time() - t0

    print(f"\n── RESULTS ({elapsed:.1f}s) ─────────────────────────────────")
    print(f"  Expression:        {args.expr}")
    print(f"  Real Sharpe:       {result.real_sharpe:+.3f}")
    if result.n_shuffles_completed >= 5:
        assert result.mean_shuffled is not None and result.median_shuffled is not None
        assert result.percentile is not None and result.p_value is not None
        print(
            f"  Shuffled (n={result.n_shuffles_completed}): "
            f"median {result.median_shuffled:+.3f}, mean {result.mean_shuffled:+.3f}"
        )
        print(f"  Percentile of real: {result.percentile:.1f}")
        print(f"  p-value:           {result.p_value:.4f}")
    if result.n_shuffles_failed:
        print(f"  Failed shuffles:   {result.n_shuffles_failed} (some shuffles errored)")
    print(f"  Verdict:           {result.verdict.upper()}")
    print(f"  {result.explanation}")
    print("──────────────────────────────────────────────────────────────")

    # Exit code: 0 if real-signal, 1 otherwise (so CI can fail-on-noise if desired)
    sys.exit(0 if result.verdict == "real-signal" else 1)


if __name__ == "__main__":
    main()
