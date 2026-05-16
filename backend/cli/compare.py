"""`alphatest compare expr1 expr2 [...]` — side-by-side metric table.

Runs each expression through the same engine + analytics that the web
app's compare endpoint uses.  Output is a fixed-width table with one row
per expression and columns for the headline metrics.

CI usage: ``alphatest compare`` exits 0 if every expression backtests
without error, 1 if any one fails (so the failure surfaces in pipeline
output without burying the working ones).
"""

from __future__ import annotations

import argparse
import time

from analytics.performance import PerformanceAnalytics

from cli._loader import load_context, make_backtester, make_config, make_evaluator

# Per-expression display width — long expressions get ellipsised.
MAX_EXPR_WIDTH = 50


def add_subparser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "compare",
        help="Backtest 2+ expressions and print a side-by-side metric table.",
    )
    p.add_argument(
        "expressions",
        nargs="+",
        help="Two or more DSL expressions, e.g. 'rank(close)' '-rank(open)'",
    )
    p.add_argument(
        "--universe",
        default=None,
        help="Universe id (defaults to project default).",
    )
    p.add_argument(
        "--neutralization",
        choices=("none", "market", "sector", "industry_group", "industry", "sub_industry"),
        default="market",
    )
    p.add_argument("--booksize", type=float, default=20_000_000)
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    if len(args.expressions) < 2:
        print("compare needs at least 2 expressions to be meaningful.")
        return 2

    ctx = load_context(args.universe, verbose=True)
    cfg = make_config(
        ctx.tickers, neutralization=args.neutralization, booksize=args.booksize, run_oos=False
    )
    evaluator = make_evaluator(ctx.data)
    backtester = make_backtester(ctx.data, ctx.gics_map)
    perf = PerformanceAnalytics()

    print(
        f"\nUniverse: {ctx.universe_id} ({len(ctx.tickers)} tickers) · "
        f"{cfg.start_date} → {cfg.end_date}\n"
    )

    rows: list[dict] = []
    n_ok = 0
    for expr in args.expressions:
        t0 = time.time()
        try:
            alpha = evaluator.evaluate(expr)
            is_result, _ = backtester.run(alpha, cfg)
            m = perf.compute(is_result, gics_map=ctx.gics_map)
            rows.append(
                {
                    "expr": expr,
                    "sharpe": m.get("sharpe"),
                    "annual_return": m.get("annual_return"),
                    "max_drawdown": m.get("max_drawdown"),
                    "fitness": m.get("fitness"),
                    "turnover": m.get("avg_turnover"),
                    "ok": True,
                    "elapsed": time.time() - t0,
                }
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 — surface any engine error verbatim
            rows.append(
                {
                    "expr": expr,
                    "error": str(exc),
                    "ok": False,
                    "elapsed": time.time() - t0,
                }
            )

    _print_table(rows)
    return 0 if n_ok == len(args.expressions) else 1


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'Expression':<{MAX_EXPR_WIDTH}}  "
        f"{'Sharpe':>7}  {'Ret%':>7}  {'DD%':>7}  {'Fitness':>8}  {'Turn $':>10}"
    )
    print(header)
    print("─" * len(header))
    for r in rows:
        expr = _ellip(r["expr"], MAX_EXPR_WIDTH)
        if not r["ok"]:
            print(f"{expr:<{MAX_EXPR_WIDTH}}  ERROR: {r['error']}")
            continue
        print(
            f"{expr:<{MAX_EXPR_WIDTH}}  "
            f"{_fmt_num(r['sharpe']):>7}  "
            f"{_fmt_pct(r['annual_return']):>7}  "
            f"{_fmt_pct(r['max_drawdown']):>7}  "
            f"{_fmt_num(r['fitness']):>8}  "
            f"{_fmt_dollar(r['turnover']):>10}"
        )


def _ellip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def _fmt_num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"


def _fmt_dollar(x: float | None) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"
