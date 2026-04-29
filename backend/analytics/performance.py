from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from engine.backtester import BacktestResult


TRADING_DAYS_PER_YEAR = 252
ROLLING_SHARPE_WINDOW = 63


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _safe_list(values) -> list[float | None]:
    return [_safe_float(v) for v in values]


class PerformanceAnalytics:
    def compute(
        self,
        result: BacktestResult,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        dates = pd.to_datetime(result.dates)
        daily_returns = pd.Series(result.daily_returns, index=dates).fillna(0.0)
        daily_pnl = pd.Series(result.daily_pnl, index=dates).fillna(0.0)
        turnover = pd.Series(result.turnover, index=dates).fillna(0.0)

        n = len(daily_returns)
        mean_dr = float(daily_returns.mean()) if n else 0.0
        std_dr = float(daily_returns.std(ddof=1)) if n > 1 else 0.0

        sharpe = (
            mean_dr / std_dr * math.sqrt(TRADING_DAYS_PER_YEAR) if std_dr > 0 else 0.0
        )

        # CAGR using booksize-relative additive returns: equity = 1 + cumsum(daily_returns)
        total_return = float(daily_returns.sum())
        years = n / TRADING_DAYS_PER_YEAR if n else 0.0
        if years > 0 and (1.0 + total_return) > 0:
            annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
        else:
            annual_return = 0.0

        annual_vol = std_dr * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Drawdown on the equity curve
        equity = 1.0 + daily_returns.cumsum()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min()) if n else 0.0

        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        neg = daily_returns[daily_returns < 0]
        downside_std = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
        sortino = (
            mean_dr / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)
            if downside_std > 0
            else 0.0
        )

        avg_turnover_dollars = float(turnover.mean()) if n else 0.0

        # Fitness uses turnover as a fraction of booksize so (1 - turnover) is bounded.
        positions = result.positions
        book_proxy = (
            float(positions.abs().sum(axis=1).max())
            if positions is not None and not positions.empty
            else 0.0
        )
        avg_turnover_frac = (
            avg_turnover_dollars / book_proxy if book_proxy > 0 else 0.0
        )
        fitness = (
            sharpe
            * math.sqrt(abs(annual_return))
            * max(0.0, 1.0 - avg_turnover_frac)
        )

        win_rate = float((daily_pnl > 0).mean()) if n else 0.0

        pos_sum = float(daily_pnl[daily_pnl > 0].sum())
        neg_sum = float(daily_pnl[daily_pnl < 0].sum())
        if neg_sum < 0:
            profit_factor = pos_sum / abs(neg_sum)
        elif pos_sum > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        beta: float | None = None
        information_ratio: float | None = None
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            bench = (
                pd.Series(benchmark_returns)
                .reindex(daily_returns.index)
                .fillna(0.0)
            )
            bench_var = float(bench.var(ddof=1))
            if bench_var > 0:
                cov = float(np.cov(daily_returns.values, bench.values, ddof=1)[0, 1])
                beta = cov / bench_var
            diff = daily_returns - bench
            diff_std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            if diff_std > 0:
                information_ratio = (
                    float(diff.mean()) / diff_std * math.sqrt(TRADING_DAYS_PER_YEAR)
                )

        # Time series for charts
        rolling_sharpe = daily_returns.rolling(ROLLING_SHARPE_WINDOW).apply(
            lambda x: (x.mean() / x.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
            if x.std(ddof=1) > 0
            else np.nan,
            raw=False,
        )

        monthly = daily_returns.groupby(
            [daily_returns.index.year, daily_returns.index.month]
        ).sum()
        monthly_returns = [
            [int(year), int(month), _safe_float(value)]
            for (year, month), value in monthly.items()
        ]

        return {
            "sharpe": _safe_float(sharpe),
            "annual_return": _safe_float(annual_return),
            "annual_vol": _safe_float(annual_vol),
            "max_drawdown": _safe_float(max_drawdown),
            "calmar_ratio": _safe_float(calmar),
            "sortino_ratio": _safe_float(sortino),
            "avg_turnover": _safe_float(avg_turnover_dollars),
            "fitness": _safe_float(fitness),
            "win_rate": _safe_float(win_rate),
            "profit_factor": _safe_float(profit_factor),
            "beta": _safe_float(beta) if beta is not None else None,
            "information_ratio": (
                _safe_float(information_ratio)
                if information_ratio is not None
                else None
            ),
            "rolling_sharpe": _safe_list(rolling_sharpe.tolist()),
            "monthly_returns": monthly_returns,
            "drawdown_series": _safe_list(drawdown.tolist()),
        }
