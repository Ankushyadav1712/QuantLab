from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from data.sp100_history import build_membership_mask

from engine import operators as ops


def _quick_sharpe(daily_returns: list[float]) -> float:
    """Annualized Sharpe from a list of daily fractional returns.

    Used by walk_forward where we only need the headline number for each
    window — not the full PerformanceAnalytics output.
    """
    arr = np.asarray(
        [x for x in daily_returns if x is not None and not (isinstance(x, float) and np.isnan(x))],
        dtype=float,
    )
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


@dataclass
class SimulationConfig:
    universe: list[str]
    start_date: str
    end_date: str
    neutralization: Literal[
        "none", "market", "sector", "industry_group", "industry", "sub_industry"
    ] = "market"
    truncation: float = 0.05
    booksize: float = 20_000_000
    transaction_cost_bps: float = 5.0
    decay: int = 0
    # In/out-of-sample split: oos_split fraction of dates is held out as the
    # last block.  When run_oos is False, the whole window is treated as IS
    # (returned in the tuple's first slot, OOS slot is None).
    oos_split: float = 0.3
    run_oos: bool = True

    # Cost model.  `flat` keeps the original `bps × turnover` formula.
    # `sqrt_impact` adds an Almgren-Chriss-style permanent-impact term:
    #     impact_$_i = impact_coefficient · σ_i · |trade_$_i| · √(participation_i)
    # where `participation_i = |trade_$_i| / dollar_volume_i` and σ_i is each
    # stock's realized volatility.  This is the realistic upgrade — flat-bps
    # under-charges high-turnover, large-trade-relative-to-ADV strategies.
    cost_model: Literal["flat", "sqrt_impact"] = "flat"
    impact_coefficient: float = 0.1

    # Walk-forward analysis (rolling train→test windows over the full window).
    # Off by default since it costs ~20× the compute of a single backtest.
    run_walk_forward: bool = False
    walk_forward_train_days: int = 252
    walk_forward_test_days: int = 63
    walk_forward_step_days: int = 63

    # Execution lag.  With ``1`` (default) we use the original convention:
    # signal computed at close[t-1] earns close[t-1]→close[t] return on day t.
    # With ``2`` we shift PnL realization an extra day to model the realistic
    # T+1 case: a researcher can't see close[t] *and* trade at close[t] — the
    # earliest fill is the next session's open, so the close[t] signal earns
    # close[t+1]→close[t+2] (approximately).  The close-to-open gap captured by
    # the extra day is the ~10–30 bps of slippage real desks pay and a flat-bps
    # backtest silently ignores.
    execution_lag_days: int = 1

    # Point-in-time universe gating.  When True, alpha values for tickers that
    # weren't yet in the S&P 100 on a given date are NaN-ed out before sizing —
    # so e.g. TSLA can't be traded in 2019 just because it's in the index now.
    # Off by default so existing saved alphas keep their headline numbers
    # stable; opt in via the editor's settings panel.
    point_in_time_universe: bool = False

    # ADV liquidity filter (PDF Section 9.2).  When > 0, on each date any
    # ticker whose 20-day average dollar volume is below this threshold is
    # NaN-ed out of the alpha matrix before sizing — so the backtest can't
    # claim PnL from stocks it couldn't actually have traded that day.
    # Brain's US3000 approximation uses ``adv20 > $1M``; pass 1_000_000.0
    # for parity.  Default 0.0 = no filter (backwards compatible).
    min_adv_dollars: float = 0.0


@dataclass
class BacktestResult:
    dates: list[str]
    daily_pnl: list[float]
    cumulative_pnl: list[float]
    daily_returns: list[float]
    weights: pd.DataFrame
    turnover: list[float]
    positions: pd.DataFrame
    # Post-neutralization alpha signal (pre-truncation, pre-scaling).  This is
    # what IC / alpha-decay metrics correlate against forward returns — the
    # researcher's *prediction*, before portfolio construction shrinks it.
    # Optional so older saved results / tests that build a BacktestResult by
    # hand don't break.
    signal_matrix: pd.DataFrame | None = None
    # Per-stock daily returns aligned to the same (date × ticker) grid.  Used
    # by IC computation; stored here to avoid leaking the global data dict
    # into analytics modules.
    forward_returns: pd.DataFrame | None = None


class Backtester:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        sector_map: dict[str, str] | None = None,
        gics_map: dict[str, dict[str, str | None]] | None = None,
    ):
        """``gics_map`` maps ticker → {sector, industry_group, industry,
        sub_industry}.  When supplied it supersedes ``sector_map`` (the latter
        is kept for backwards compatibility with callers that haven't migrated
        to the GICS-aware path — the API and tests pass ``gics_map`` directly).
        """
        self.data = data
        self.sector_map = sector_map or {}
        self.gics_map = gics_map or {}
        # Synthesize sector_map from gics_map if only the latter was given —
        # keeps the legacy 'sector' code path working without branching.
        if gics_map and not sector_map:
            self.sector_map = {t: row.get("sector") or "Unknown" for t, row in gics_map.items()}

    def run(
        self, alpha_matrix: pd.DataFrame, config: SimulationConfig
    ) -> tuple[BacktestResult, BacktestResult | None]:
        """Run the backtest, optionally split into in-sample / out-of-sample.

        Returns ``(is_result, oos_result)``.  When ``config.run_oos`` is False,
        ``oos_result`` is None and ``is_result`` covers the full window — same
        behavior the engine had before this method was tuple-returning.
        """
        # 1. Universe + date filter (applies to both halves identically)
        cols = [t for t in config.universe if t in alpha_matrix.columns]
        if not cols:
            raise ValueError("No tickers in alpha_matrix match the configured universe")

        alpha = alpha_matrix[cols].copy()
        alpha.index = pd.to_datetime(alpha.index)
        start = pd.to_datetime(config.start_date)
        end = pd.to_datetime(config.end_date)
        alpha = alpha.loc[(alpha.index >= start) & (alpha.index <= end)]

        if alpha.empty:
            raise ValueError("Alpha matrix is empty after applying date/universe filter")

        # 2. Drop dates with >50% NaN — applies to both halves identically
        nan_frac = alpha.isna().sum(axis=1) / len(alpha.columns)
        alpha = alpha.loc[nan_frac <= 0.5]

        if alpha.empty:
            raise ValueError("All dates dropped: every row had >50% NaN alpha")

        if not config.run_oos:
            return self._run_pipeline(alpha, config), None

        # 3. Split by row-count (date count, not calendar year)
        n = len(alpha)
        split_idx = int(n * (1.0 - config.oos_split))
        # Guard against degenerate splits
        if split_idx <= 0 or split_idx >= n:
            return self._run_pipeline(alpha, config), None

        is_alpha = alpha.iloc[:split_idx]
        oos_alpha = alpha.iloc[split_idx:]

        is_result = self._run_pipeline(is_alpha, config)
        oos_result = self._run_pipeline(oos_alpha, config)
        return is_result, oos_result

    # ------------------------------------------------------------------
    # The signal-→-PnL pipeline that runs identically on each split.
    # Assumes its caller has already done the universe + date filter and
    # the >50%-NaN-row drop.
    # ------------------------------------------------------------------

    def _run_pipeline(self, alpha: pd.DataFrame, config: SimulationConfig) -> BacktestResult:
        if alpha.empty:
            raise ValueError("Empty alpha slice passed to pipeline")

        # Point-in-time gating: zero out alpha for any (date, ticker) where
        # ticker wasn't yet an S&P 100 member on date.  Done before decay/
        # neutralization so the masked entries don't leak through rolling ops.
        if config.point_in_time_universe:
            mask = build_membership_mask(pd.DatetimeIndex(alpha.index), list(alpha.columns))
            alpha = alpha.where(mask, other=0.0)

        # ADV liquidity gating: NaN out alpha cells where the 20-day average
        # dollar volume is below the threshold on that date.  Done after PIT
        # gating (since PIT zeros are valid trades for "in-index" stocks)
        # but before decay/neutralization so masked cells don't leak through
        # rolling operations downstream.
        if config.min_adv_dollars > 0:
            adv = self.data.get("adv20")
            if adv is None:
                # Compute it on the fly from dollar_volume so the filter still
                # works in test contexts that don't pre-load adv20.
                dv = self.data.get("dollar_volume")
                if dv is not None:
                    adv = dv.rolling(20, min_periods=1).mean()
            if adv is not None:
                adv_aligned = adv.reindex(index=alpha.index, columns=alpha.columns)
                liquid_mask = adv_aligned >= config.min_adv_dollars
                alpha = alpha.where(liquid_mask, other=np.nan)

        # Optional decay before trading
        if config.decay and config.decay > 0:
            alpha = ops.decay_linear(alpha, config.decay)

        # Neutralization
        alpha = self._neutralize(alpha, config.neutralization)
        # Snapshot the post-neutralization signal for IC analytics.  Done
        # before any downstream mutation (normalize/clip would distort the
        # ranks we want IC to measure).
        signal_snapshot = alpha.copy()

        # Normalize to fractional weights (sum of abs ≈ 1 per row)
        abs_sum = alpha.abs().sum(axis=1)
        abs_sum = abs_sum.replace(0, np.nan)
        weights = alpha.div(abs_sum, axis=0).fillna(0.0)

        # Truncation: cap each fractional weight at ±truncation
        weights = weights.clip(lower=-config.truncation, upper=config.truncation)

        # Position sizing — scale to dollar positions
        positions = weights * config.booksize

        # Daily PnL: yesterday's dollar position × today's return.  When
        # execution_lag_days > 1 we shift the position further into the past,
        # capturing the "we couldn't trade at the close we just saw" friction.
        if "returns" not in self.data:
            raise ValueError("data['returns'] is required to compute PnL")
        returns = self.data["returns"].reindex(index=positions.index, columns=positions.columns)
        exec_lag = max(1, int(config.execution_lag_days))
        stock_pnl = positions.shift(exec_lag) * returns
        gross_pnl = stock_pnl.sum(axis=1, skipna=True)

        # Transaction costs.  Per-stock |Δ$| matrix → flat spread/commission
        # always, optional Almgren-Chriss square-root impact on top.
        delta_pos = (positions - positions.shift(1)).abs()
        flat_cost_per_stock = delta_pos * config.transaction_cost_bps / 10_000.0

        if config.cost_model == "sqrt_impact":
            dv = self.data.get("dollar_volume")
            rv = self.data.get("realized_vol")
            if dv is not None and rv is not None and not dv.empty and not rv.empty:
                dv_aligned = dv.reindex(index=positions.index, columns=positions.columns).replace(
                    0, np.nan
                )
                rv_aligned = rv.reindex(index=positions.index, columns=positions.columns).fillna(
                    0.0
                )
                participation = (delta_pos / dv_aligned).fillna(0.0).clip(lower=0.0)
                # impact_$ = c · σ · |trade$| · √(participation)
                impact_per_stock = (
                    config.impact_coefficient * rv_aligned * delta_pos * participation.pow(0.5)
                )
                cost_per_stock = flat_cost_per_stock + impact_per_stock
            else:
                # Required fields missing — fall back silently to flat-bps
                cost_per_stock = flat_cost_per_stock
        else:
            cost_per_stock = flat_cost_per_stock

        turnover = delta_pos.sum(axis=1, skipna=True)
        cost = cost_per_stock.sum(axis=1, skipna=True)
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
            signal_matrix=signal_snapshot,
            forward_returns=returns,
        )

    # ------------------------------------------------------------------
    # Walk-forward analysis: rolling train→test windows over the alpha.
    # The senior-quant test for "does this generalize?" — a single 70/30
    # split can pass or fail by luck of which regime lands in OOS.  Sliding
    # the split forward in time removes that luck.
    # ------------------------------------------------------------------

    def walk_forward(self, alpha_matrix: pd.DataFrame, config: SimulationConfig) -> list[dict]:
        """Slide a (train_days → test_days) window across the full history.

        Returns one dict per window: {train/test period, train Sharpe, test
        Sharpe, test annual_return, test cumulative PnL}.  Empty list if the
        history is too short for even one window.
        """
        # Apply the same universe + date + nan-row filter we use in run()
        cols = [t for t in config.universe if t in alpha_matrix.columns]
        if not cols:
            return []
        alpha = alpha_matrix[cols].copy()
        alpha.index = pd.to_datetime(alpha.index)
        start = pd.to_datetime(config.start_date)
        end = pd.to_datetime(config.end_date)
        alpha = alpha.loc[(alpha.index >= start) & (alpha.index <= end)]
        if alpha.empty:
            return []
        nan_frac = alpha.isna().sum(axis=1) / len(alpha.columns)
        alpha = alpha.loc[nan_frac <= 0.5]
        if alpha.empty:
            return []

        train_n = max(1, int(config.walk_forward_train_days))
        test_n = max(1, int(config.walk_forward_test_days))
        step = max(1, int(config.walk_forward_step_days))

        results: list[dict] = []
        cursor = 0
        n = len(alpha)
        while cursor + train_n + test_n <= n:
            train_slice = alpha.iloc[cursor : cursor + train_n]
            test_slice = alpha.iloc[cursor + train_n : cursor + train_n + test_n]

            try:
                train_res = self._run_pipeline(train_slice, config)
                test_res = self._run_pipeline(test_slice, config)
            except ValueError:
                # Degenerate window (e.g. all-zero alpha after neutralization)
                cursor += step
                continue

            train_sharpe = _quick_sharpe(train_res.daily_returns)
            test_sharpe = _quick_sharpe(test_res.daily_returns)

            results.append(
                {
                    "train_start": train_slice.index[0].strftime("%Y-%m-%d"),
                    "train_end": train_slice.index[-1].strftime("%Y-%m-%d"),
                    "test_start": test_slice.index[0].strftime("%Y-%m-%d"),
                    "test_end": test_slice.index[-1].strftime("%Y-%m-%d"),
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "test_total_return": float(
                        sum(x for x in test_res.daily_returns if x is not None)
                    ),
                    "test_cumulative_pnl": float(
                        test_res.cumulative_pnl[-1] if test_res.cumulative_pnl else 0.0
                    ),
                }
            )
            cursor += step

        return results

    # Modes that key off the GICS map: per-group cross-sectional demean at the
    # level named by the mode string.  All four behave identically — they
    # differ only in which column of the gics_map they use to form groups.
    _GICS_MODES = ("sector", "industry_group", "industry", "sub_industry")

    def _neutralize(self, alpha: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "none":
            return alpha
        if mode == "market":
            return alpha.sub(alpha.mean(axis=1, skipna=True), axis=0)
        if mode in self._GICS_MODES:
            return self._neutralize_by_gics(alpha, mode)
        raise ValueError(f"Unknown neutralization mode: {mode!r}")

    def _neutralize_by_gics(self, alpha: pd.DataFrame, level: str) -> pd.DataFrame:
        """Cross-sectional demean per GICS group at ``level``.

        Tickers without a classification at this level (or universe members
        absent from the GICS map entirely) are bucketed together as
        ``"Unknown"`` — better than silently dropping them.
        """

        # The legacy `sector` path used `sector_map`; the GICS-aware paths use
        # `gics_map`.  When both are present, gics_map wins (it's the strictly
        # richer source).
        def lookup(t: str) -> str:
            row = self.gics_map.get(t)
            if row is not None:
                val = row.get(level)
                if val:
                    return val
            if level == "sector":
                return self.sector_map.get(t, "Unknown")
            return "Unknown"

        groups = pd.Series({t: lookup(t) for t in alpha.columns})
        out = alpha.copy()
        for _grp, members in groups.groupby(groups):
            cols = list(members.index)
            if not cols:
                continue
            sub = alpha[cols]
            out[cols] = sub.sub(sub.mean(axis=1, skipna=True), axis=0)
        return out
