from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from engine import operators as ops


@dataclass
class SimulationConfig:
    universe: list[str]
    start_date: str
    end_date: str
    neutralization: Literal["none", "market", "sector"] = "market"
    truncation: float = 0.05
    booksize: float = 20_000_000
    transaction_cost_bps: float = 5.0
    decay: int = 0


@dataclass
class BacktestResult:
    dates: list[str]
    daily_pnl: list[float]
    cumulative_pnl: list[float]
    daily_returns: list[float]
    weights: pd.DataFrame
    turnover: list[float]
    positions: pd.DataFrame


class Backtester:
    def __init__(self, data: dict[str, pd.DataFrame], sector_map: dict[str, str]):
        self.data = data
        self.sector_map = sector_map

    def run(
        self, alpha_matrix: pd.DataFrame, config: SimulationConfig
    ) -> BacktestResult:
        # 1. Filter to universe + date range
        cols = [t for t in config.universe if t in alpha_matrix.columns]
        if not cols:
            raise ValueError(
                "No tickers in alpha_matrix match the configured universe"
            )

        alpha = alpha_matrix[cols].copy()
        alpha.index = pd.to_datetime(alpha.index)
        start = pd.to_datetime(config.start_date)
        end = pd.to_datetime(config.end_date)
        alpha = alpha.loc[(alpha.index >= start) & (alpha.index <= end)]

        if alpha.empty:
            raise ValueError(
                "Alpha matrix is empty after applying date/universe filter"
            )

        # 2. Drop dates where >50% of stocks have NaN alpha
        nan_frac = alpha.isna().sum(axis=1) / len(alpha.columns)
        alpha = alpha.loc[nan_frac <= 0.5]

        if alpha.empty:
            raise ValueError(
                "All dates dropped: every row had >50% NaN alpha"
            )

        # Optional decay before trading
        if config.decay and config.decay > 0:
            alpha = ops.decay_linear(alpha, config.decay)

        # 3. Neutralization
        alpha = self._neutralize(alpha, config.neutralization)

        # 5. Normalize to fractional weights (sum of abs ≈ 1 per row)
        abs_sum = alpha.abs().sum(axis=1)
        abs_sum = abs_sum.replace(0, np.nan)
        weights = alpha.div(abs_sum, axis=0).fillna(0.0)

        # 4. Truncation: cap each fractional weight at ±truncation
        weights = weights.clip(lower=-config.truncation, upper=config.truncation)

        # 6. Position sizing — scale to dollar positions
        positions = weights * config.booksize

        # 7. Daily PnL: yesterday's dollar position × today's return
        if "returns" not in self.data:
            raise ValueError("data['returns'] is required to compute PnL")
        returns = self.data["returns"].reindex(
            index=positions.index, columns=positions.columns
        )
        stock_pnl = positions.shift(1) * returns
        gross_pnl = stock_pnl.sum(axis=1, skipna=True)

        # 8. Transaction costs
        turnover = (positions - positions.shift(1)).abs().sum(axis=1, skipna=True)
        cost = turnover * config.transaction_cost_bps / 10_000.0
        net_pnl = gross_pnl - cost

        cumulative = net_pnl.cumsum()
        daily_returns = net_pnl / config.booksize

        date_strs = [d.strftime("%Y-%m-%d") for d in positions.index]

        return BacktestResult(
            dates=date_strs,
            daily_pnl=net_pnl.tolist(),
            cumulative_pnl=cumulative.tolist(),
            daily_returns=daily_returns.tolist(),
            weights=weights,
            turnover=turnover.tolist(),
            positions=positions,
        )

    def _neutralize(self, alpha: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "none":
            return alpha
        if mode == "market":
            return alpha.sub(alpha.mean(axis=1, skipna=True), axis=0)
        if mode == "sector":
            sectors = pd.Series(
                {t: self.sector_map.get(t, "Unknown") for t in alpha.columns}
            )
            out = alpha.copy()
            for _sector, group in sectors.groupby(sectors):
                cols = list(group.index)
                sub = alpha[cols]
                out[cols] = sub.sub(sub.mean(axis=1, skipna=True), axis=0)
            return out
        raise ValueError(f"Unknown neutralization mode: {mode!r}")
