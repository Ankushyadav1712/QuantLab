"""`alphatest run <expr>` — evaluate an expression and print core metrics.

Same engine + analytics the FastAPI server uses, so numbers reproduce
byte-identically.  Default output is the in-sample window (no OOS split)
to keep the CLI's headline matching what users quote.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from analytics.performance import PerformanceAnalytics

from cli._loader import LoadedContext, load_context, make_backtester, make_config, make_evaluator


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("run", help="Evaluate an expression and print core metrics.")
    p.add_argument("expression", help="DSL expression, e.g. 'rank(close) - rank(open)'.")
    p.add_argument(
        "--universe",
        default=None,
        help="Universe id (defaults to project default).  See /api/universes.",
    )
    p.add_argument(
        "--neutralization",
        choices=("none", "market", "sector", "industry_group", "industry", "sub_industry"),
        default="market",
    )
    p.add_argument("--booksize", type=float, default=20_000_000)
    p.add_argument(
        "--oos",
        action="store_true",
        help="Run an in-sample/out-of-sample split and print both halves.",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    ctx = load_context(args.universe, verbose=True)
    return run_expression(
        args.expression,
        ctx,
        neutralization=args.neutralization,
        booksize=args.booksize,
        run_oos=args.oos,
    )


def run_expression(
    expression: str,
    ctx: LoadedContext,
    *,
    neutralization: str = "market",
    booksize: float = 20_000_000,
    run_oos: bool = False,
) -> int:
    """Backtest one expression against a pre-loaded context and print results.

    Returned exit code is 0 on success, 1 on engine/eval error (so CI pipelines
    can chain `alphatest run` together).
    """
    cfg = make_config(
        ctx.tickers, neutralization=neutralization, booksize=booksize, run_oos=run_oos
    )
    evaluator = make_evaluator(ctx.data)
    backtester = make_backtester(ctx.data, ctx.gics_map)
    perf = PerformanceAnalytics()

    print(f"\nExpression: {expression}")
    print(f"Universe:   {ctx.universe_id} ({len(ctx.tickers)} tickers)")
    print(f"Window:     {cfg.start_date} → {cfg.end_date}\n")

    t0 = time.time()
    try:
        alpha = evaluator.evaluate(expression)
        is_result, oos_result = backtester.run(alpha, cfg)
    except Exception as exc:  # noqa: BLE001 — CLI surfaces any engine error verbatim
        print(f"ERROR: {exc}")
        return 1

    is_metrics = perf.compute(is_result, gics_map=ctx.gics_map)
    _print_metrics_block("In-sample" if run_oos else "Full window", is_metrics)

    if oos_result is not None:
        oos_metrics = perf.compute(oos_result, gics_map=ctx.gics_map)
        _print_metrics_block("Out-of-sample", oos_metrics)

    print(f"\n({time.time() - t0:.1f}s)")
    return 0


def _print_metrics_block(label: str, m: dict[str, Any]) -> None:
    print(f"── {label} " + "─" * (58 - len(label)))
    print(f"  Sharpe:        {_fmt_num(m.get('sharpe'))}")
    print(f"  Annual return: {_fmt_pct(m.get('annual_return'))}")
    print(f"  Max drawdown:  {_fmt_pct(m.get('max_drawdown'))}")
    print(f"  Sortino:       {_fmt_num(m.get('sortino'))}")
    print(f"  Calmar:        {_fmt_num(m.get('calmar'))}")
    print(f"  Win rate:      {_fmt_pct(m.get('win_rate'))}")
    print(f"  Fitness:       {_fmt_num(m.get('fitness'))}")
    print(f"  Turnover:      {_fmt_num(m.get('avg_turnover'), width=10, prec=0)} $")


def _fmt_num(x: float | None, width: int = 8, prec: int = 3) -> str:
    if x is None:
        return "n/a".rjust(width)
    return f"{x:>{width}.{prec}f}"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "     n/a"
    return f"{x * 100:>7.2f}%"
