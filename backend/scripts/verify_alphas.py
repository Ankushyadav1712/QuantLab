"""Run a handful of canonical expressions and print their core metrics.

Bypasses the API — uses the engine + analytics modules directly so the
script is useful even without a running server.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from analytics.performance import PerformanceAnalytics
from config import (
    DATA_END,
    DATA_START,
    DEFAULT_BOOKSIZE,
    SECTOR_MAP,
    UNIVERSE,
)
from data.fetcher import DataFetcher
from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator

EXPRESSIONS = [
    "rank(delta(close, 5))",
    "rank(volume) * rank(returns)",
    "-ts_corr(rank(close), rank(volume), 10)",
]

DATA_FIELDS = ["open", "high", "low", "close", "volume", "returns", "vwap"]


def _fmt_pct(x: float | None) -> str:
    return "    n/a" if x is None else f"{x * 100:>7.2f}%"


def _fmt_num(x: float | None, width: int = 8, prec: int = 3) -> str:
    return "n/a".rjust(width) if x is None else f"{x:>{width}.{prec}f}"


def main() -> int:
    fetcher = DataFetcher()
    fetcher.download_universe()  # cache hit on each ticker
    data = {field: fetcher.get_data_matrix(field) for field in DATA_FIELDS}

    evaluator = AlphaEvaluator(data)
    backtester = Backtester(data, SECTOR_MAP)
    analytics = PerformanceAnalytics()

    header = f"{'Expression':<46} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} {'Fitness':>8}"
    print(header)
    print("-" * len(header))

    for expression in EXPRESSIONS:
        try:
            alpha = evaluator.evaluate(expression)
            cfg = SimulationConfig(
                universe=list(UNIVERSE),
                start_date=DATA_START,
                end_date=DATA_END,
                neutralization="market",
                booksize=DEFAULT_BOOKSIZE,
            )
            result = backtester.run(alpha, cfg)
            metrics = analytics.compute(result)
        except Exception as exc:  # noqa: BLE001
            print(f"{expression:<46}  ERROR: {exc}")
            continue

        line = (
            f"{expression:<46} "
            f"{_fmt_num(metrics['sharpe'])} "
            f"{_fmt_pct(metrics['annual_return'])} "
            f"{_fmt_pct(metrics['max_drawdown'])} "
            f"{_fmt_num(metrics['fitness'])}"
        )
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
