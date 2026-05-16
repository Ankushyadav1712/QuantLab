"""`alphatest stress <expr>` — stress-test an expression across known crises.

Runs a single backtest, then prints the per-regime breakdown that the web
app shows in its stress-test panel (GFC, COVID Crash, etc.).  Useful for
quickly checking that a candidate alpha survives historical regime shifts
before committing it to the saved-alpha library.
"""

from __future__ import annotations

import argparse
import time

from analytics.performance import PerformanceAnalytics

from cli._loader import load_context, make_backtester, make_config, make_evaluator


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "stress",
        help="Backtest an expression and print per-regime stress-test metrics.",
    )
    p.add_argument("expression", help="DSL expression to stress-test.")
    p.add_argument("--universe", default=None)
    p.add_argument(
        "--neutralization",
        choices=("none", "market", "sector", "industry_group", "industry", "sub_industry"),
        default="market",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    ctx = load_context(args.universe, verbose=True)
    cfg = make_config(ctx.tickers, neutralization=args.neutralization, run_oos=False)
    evaluator = make_evaluator(ctx.data)
    backtester = make_backtester(ctx.data, ctx.gics_map)
    perf = PerformanceAnalytics()

    print(f"\nStress-testing: {args.expression}\n")
    t0 = time.time()
    try:
        alpha = evaluator.evaluate(args.expression)
        is_result, _ = backtester.run(alpha, cfg)
        m = perf.compute(is_result, gics_map=ctx.gics_map)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1

    stress = m.get("stress_test") or []
    if not stress:
        print("(no stress-test windows applicable to this date range)")
        return 0

    _print_table(stress)
    elapsed = time.time() - t0
    print(f"\n({elapsed:.1f}s)")

    # CI signal: exit 1 if any regime had a meaningfully negative Sharpe
    # (below -0.5), so a fragile alpha can fail the gate without humans
    # eyeballing the table.
    worst = min((r.get("sharpe") or 0.0) for r in stress)
    return 0 if worst >= -0.5 else 1


def _print_table(stress: list[dict]) -> None:
    header = (
        f"{'Regime':<22}  {'Window':<24}  {'Sharpe':>7}  {'Ret%':>7}  {'MaxDD%':>7}  {'Days':>5}"
    )
    print(header)
    print("─" * len(header))
    for r in stress:
        # stress_test schema: name, label, start, end, sharpe, total_return,
        # annualised_return, max_drawdown, hit_rate, n_days
        name = (r.get("label") or r.get("name") or "Unnamed")[:22]
        start = r.get("start") or ""
        end = r.get("end") or ""
        window = f"{start[:10]}→{end[:10]}"[:24]
        sharpe = _fmt_num(r.get("sharpe"))
        ret = _fmt_pct(r.get("total_return"))
        dd = _fmt_pct(r.get("max_drawdown"))
        n = r.get("n_days") or 0
        print(f"{name:<22}  {window:<24}  {sharpe:>7}  {ret:>7}  {dd:>7}  {n:>5}")


def _fmt_num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"
