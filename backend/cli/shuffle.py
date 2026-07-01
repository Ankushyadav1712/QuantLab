"""`alphatest shuffle <expr>` — shuffle leakage test.

Same engine the existing ``scripts/shuffle_test.py`` uses; this is the new
canonical entry point.  The standalone script is kept for backward
compatibility but new docs point here.
"""

from __future__ import annotations

import argparse
import sys
import time

from analytics.shuffle_test import run_shuffle_test

from cli._loader import load_context, make_backtester, make_config, make_evaluator


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("shuffle", help="Run the shuffle leakage test on an expression.")
    p.add_argument("expression", help="DSL expression to test.")
    p.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of random permutations (more → tighter p-value, slower).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--universe", default=None)
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    ctx = load_context(args.universe, verbose=True)
    cfg = make_config(ctx.tickers)

    # Throttle progress prints so a 50-shuffle run doesn't spam a thousand lines.
    last_print = [0.0]

    def progress(i: int, total: int) -> None:
        now = time.time()
        if now - last_print[0] >= 1.0 or i == total:
            print(f"  shuffle {i:>3d}/{total}", end="\r", flush=True)
            last_print[0] = now

    print(
        f"\nRunning shuffle test: expr={args.expression!r} · "
        f"n_shuffles={args.iters} · seed={args.seed}"
    )
    print("(this can take 1-3 minutes for a 50-shuffle run)\n")

    t0 = time.time()
    result = run_shuffle_test(
        args.expression,
        data=ctx.data,
        backtester_factory=lambda d: make_backtester(d, ctx.gics_map),
        evaluator_factory=lambda d: make_evaluator(d),
        config=cfg,
        n_shuffles=args.iters,
        seed=args.seed,
        progress_callback=progress,
    )
    print()  # clear the in-place progress line
    elapsed = time.time() - t0

    print(f"\n── RESULTS ({elapsed:.1f}s) " + "─" * (44 - len(f"{elapsed:.1f}")))
    print(f"  Expression:        {args.expression}")
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
    print("─" * 60)

    # Exit 0 on real-signal, 1 otherwise so `alphatest shuffle ... && deploy` works.
    return 0 if result.verdict == "real-signal" else 1


if __name__ == "__main__":  # pragma: no cover — manual invocation only
    sys.exit(handle(argparse.Namespace(expression=sys.argv[1], iters=50, seed=0, universe=None)))
