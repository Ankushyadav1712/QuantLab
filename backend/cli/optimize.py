"""`alphatest optimize <expression-with-{a..b:s}>` — parameter sweep.

Wraps the same expansion + grid-search engine the web app's Sweep mode
uses.  The expression must contain at least one ``{start..end:step}`` token
(e.g. ``rank(delta(close, {3..10:1}))`` sweeps the lookback over 3–10).

Output is a Sharpe-sorted table so the best cell is at the top.  Exit
code is 0 if at least one cell hit ``--min-sharpe`` (default 1.0); 1
otherwise.  Useful as a CI gate: "fail the build if no parameter setting
clears Sharpe 1.0".
"""

from __future__ import annotations

import argparse
import time

from analytics.performance import PerformanceAnalytics
from engine.sweep import expand_sweeps, has_sweep_syntax

from cli._loader import load_context, make_backtester, make_config, make_evaluator


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "optimize",
        help="Sweep parameters in an expression using {a..b:s} syntax; print Sharpe-ranked grid.",
    )
    p.add_argument(
        "expression",
        help="Expression with one or more {a..b:s} sweep tokens, e.g. 'rank(delta(close, {3..10:1}))'",
    )
    p.add_argument(
        "--max-combinations",
        type=int,
        default=50,
        help="Reject sweeps with > N total cells (default 50, matches web UI).",
    )
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=1.0,
        help="Exit 0 only if at least one cell hits Sharpe ≥ N (default 1.0).",
    )
    p.add_argument("--universe", default=None)
    p.add_argument(
        "--neutralization",
        choices=("none", "market", "sector", "industry_group", "industry", "sub_industry"),
        default="market",
    )
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    if not has_sweep_syntax(args.expression):
        print(
            "Expression has no sweep tokens. Add one like {3..10:1} — "
            "e.g. 'rank(delta(close, {3..10:1}))'"
        )
        return 2

    try:
        expansion = expand_sweeps(args.expression, max_combinations=args.max_combinations)
    except ValueError as exc:
        print(f"Sweep expansion failed: {exc}")
        return 2

    expressions: list[str] = expansion["expressions"]
    print(f"\nExpanding {args.expression!r} → {len(expressions)} cells")
    for d in expansion["dimensions"]:
        print(f"  {d['token']} = {d['values']}")
    print()

    ctx = load_context(args.universe, verbose=True)
    cfg = make_config(ctx.tickers, neutralization=args.neutralization, run_oos=False)
    evaluator = make_evaluator(ctx.data)
    backtester = make_backtester(ctx.data, ctx.gics_map)
    perf = PerformanceAnalytics()

    results: list[dict] = []
    t0 = time.time()
    for i, expr in enumerate(expressions, start=1):
        # Tight in-line progress — sweep runs can be 10s of seconds at 50 cells
        print(f"  [{i}/{len(expressions)}] {expr}", end="\r", flush=True)
        try:
            alpha = evaluator.evaluate(expr)
            is_result, _ = backtester.run(alpha, cfg)
            m = perf.compute(is_result, gics_map=ctx.gics_map)
            results.append(
                {
                    "expr": expr,
                    "sharpe": m.get("sharpe"),
                    "annual_return": m.get("annual_return"),
                    "max_drawdown": m.get("max_drawdown"),
                    "fitness": m.get("fitness"),
                    "ok": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"expr": expr, "ok": False, "error": str(exc)})
    print()  # clear the progress line

    # Sort by Sharpe descending so the winner is at the top.  Failed cells
    # and missing-sharpe cells sort to the bottom (coerced to -inf).
    def _sort_key(r: dict) -> float:
        if not r["ok"]:
            return float("-inf")
        s = r.get("sharpe")
        return float(s) if s is not None else float("-inf")

    results.sort(key=_sort_key, reverse=True)
    elapsed = time.time() - t0
    _print_table(results)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / max(1, len(expressions)):.1f}s/cell)")

    # CI gate
    best = (
        max((r.get("sharpe") or float("-inf")) for r in results if r["ok"])
        if any(r["ok"] for r in results)
        else float("-inf")
    )
    if best >= args.min_sharpe:
        return 0
    print(
        f"\n✗ No cell hit Sharpe ≥ {args.min_sharpe} (best: {best:.3f}). "
        f"Loosen --min-sharpe or widen the sweep."
    )
    return 1


def _print_table(rows: list[dict]) -> None:
    expr_w = 50
    header = f"{'Expression':<{expr_w}}  {'Sharpe':>7}  {'Ret%':>7}  {'DD%':>7}  {'Fitness':>8}"
    print(header)
    print("─" * len(header))
    for r in rows:
        expr = _ellip(r["expr"], expr_w)
        if not r["ok"]:
            print(f"{expr:<{expr_w}}  ERROR: {r['error']}")
            continue
        print(
            f"{expr:<{expr_w}}  "
            f"{_fmt_num(r['sharpe']):>7}  "
            f"{_fmt_pct(r['annual_return']):>7}  "
            f"{_fmt_pct(r['max_drawdown']):>7}  "
            f"{_fmt_num(r['fitness']):>8}"
        )


def _ellip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def _fmt_num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"
