from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from engine.backtester import BacktestResult

from analytics.attribution import compute_pnl_attribution
from analytics.deflated_sharpe import deflated_sharpe
from analytics.exposure import (
    compute_market_cap_distribution,
    compute_sector_exposure,
    compute_sector_exposure_timeseries,
    compute_size_exposure,
)
from analytics.ic_metrics import (
    DEFAULT_DECAY_HORIZONS,
    compute_alpha_decay,
    compute_ic_series,
    compute_ic_summary,
    compute_quintile_returns,
    compute_rank_stability,
)
from analytics.stress_test import compute_stress_metrics

TRADING_DAYS_PER_YEAR = 252
ROLLING_SHARPE_WINDOW = 63


def _tail_ratio(returns: pd.Series) -> float | None:
    """|95th-percentile return| / |5th-percentile return|.

    >1 means right-skewed payoffs (gains larger than losses at the tails),
    <1 means left-skewed (crash risk).  None if either tail is degenerate.
    """
    if len(returns) < 20:
        return None
    q95 = float(returns.quantile(0.95))
    q5 = float(returns.quantile(0.05))
    if q5 == 0:
        return None
    return abs(q95) / abs(q5)


def _positive_months_pct(daily_returns: pd.Series) -> float | None:
    """Share of calendar months with net-positive return."""
    if daily_returns.empty:
        return None
    monthly = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).sum()
    if len(monthly) == 0:
        return None
    return float((monthly > 0).mean())


def _drawdown_durations(equity: pd.Series) -> dict[str, float | int | None]:
    """Avg + max consecutive days underwater.

    "Underwater" = equity < running max, i.e. drawdown < 0.  We collapse the
    underwater periods into runs and return their mean / max lengths plus
    the current run (0 if equity is at a new high).
    """
    if equity.empty:
        return {"avg_dd_days": None, "max_dd_days": None, "current_dd_days": 0}
    running_max = equity.cummax()
    underwater = (equity < running_max).astype(int).values
    # Run-length encode: start a new run whenever we transition 0→1
    runs: list[int] = []
    cur = 0
    for u in underwater:
        if u == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    current = cur  # tail run, possibly still in progress
    if runs:
        avg = float(np.mean(runs))
        mx = int(max(runs + [current])) if current else int(max(runs))
    elif current > 0:
        avg = float(current)
        mx = int(current)
    else:
        return {"avg_dd_days": 0.0, "max_dd_days": 0, "current_dd_days": 0}
    return {
        "avg_dd_days": avg,
        "max_dd_days": mx,
        "current_dd_days": int(current),
    }


def _fitness_wq(
    sharpe: float | None, annual_return: float, avg_turnover_frac: float
) -> float | None:
    """WorldQuant Brain's Fitness composite.

    ``fitness = sign(returns) * sqrt(|annual_return| / max(turnover, 0.125)) * sharpe``

    The 0.125 floor stops fractional-turnover near zero from blowing up the
    score.  Sign carries through `annual_return` so loss-making alphas
    correctly score negative regardless of how good the Sharpe looks.
    """
    if sharpe is None:
        return None
    turnover_eff = max(abs(avg_turnover_frac), 0.125)
    sign = 1.0 if annual_return >= 0 else -1.0
    return sign * math.sqrt(abs(annual_return) / turnover_eff) * sharpe


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
        n_trials: int = 1,
        *,
        gics_map: dict[str, dict[str, str | None]] | None = None,
        size_field: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        dates = pd.to_datetime(result.dates)
        daily_returns = pd.Series(result.daily_returns, index=dates).fillna(0.0)
        daily_pnl = pd.Series(result.daily_pnl, index=dates).fillna(0.0)
        turnover = pd.Series(result.turnover, index=dates).fillna(0.0)

        n = len(daily_returns)
        mean_dr = float(daily_returns.mean()) if n else 0.0
        std_dr = float(daily_returns.std(ddof=1)) if n > 1 else 0.0

        sharpe = mean_dr / std_dr * math.sqrt(TRADING_DAYS_PER_YEAR) if std_dr > 0 else 0.0

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
            mean_dr / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0
        )

        avg_turnover_dollars = float(turnover.mean()) if n else 0.0

        # Fitness uses turnover as a fraction of booksize so (1 - turnover) is bounded.
        positions = result.positions
        book_proxy = (
            float(positions.abs().sum(axis=1).max())
            if positions is not None and not positions.empty
            else 0.0
        )
        avg_turnover_frac = avg_turnover_dollars / book_proxy if book_proxy > 0 else 0.0
        fitness = sharpe * math.sqrt(abs(annual_return)) * max(0.0, 1.0 - avg_turnover_frac)

        # ---- Cost breakdown (PDF "cost realism" gap) ---------------------
        # Sum each cost component over the backtest window so the researcher
        # can see which source ate into PnL.  Reported in dollars + as a %
        # of gross PnL.  All four are None when the backtester didn't emit
        # cost_components (older saved results / tests that hand-build a
        # BacktestResult by hand).
        cost_breakdown: dict[str, Any] | None = None
        if result.cost_components is not None:
            comps = result.cost_components
            flat_tot = float(sum(comps.get("flat", [])))
            spread_tot = float(sum(comps.get("spread", [])))
            impact_tot = float(sum(comps.get("impact", [])))
            borrow_tot = float(sum(comps.get("borrow", [])))
            total_cost = flat_tot + spread_tot + impact_tot + borrow_tot
            # Gross = net_pnl + total_cost (since net_pnl was already cost-deducted)
            gross_pnl = float(daily_pnl.sum()) + total_cost
            # Percentage of gross eaten by costs — bounded [0, 100] for the
            # common case but can exceed 100% if costs > gross (an alpha
            # whose gross PnL was wiped out by friction).
            cost_pct = (100.0 * total_cost / abs(gross_pnl)) if abs(gross_pnl) > 0 else 0.0
            cost_breakdown = {
                "gross_pnl": gross_pnl,
                "flat_bps_cost": flat_tot,
                "spread_cost": spread_tot,
                "impact_cost": impact_tot,
                "borrow_cost": borrow_tot,
                "total_cost": total_cost,
                "net_pnl": float(daily_pnl.sum()),
                "cost_pct_of_gross": cost_pct,
            }

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
            bench = pd.Series(benchmark_returns).reindex(daily_returns.index).fillna(0.0)
            bench_var = float(bench.var(ddof=1))
            if bench_var > 0:
                cov = float(np.cov(daily_returns.values, bench.values, ddof=1)[0, 1])
                beta = cov / bench_var
            diff = daily_returns - bench
            diff_std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            if diff_std > 0:
                information_ratio = float(diff.mean()) / diff_std * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Time series for charts
        rolling_sharpe = daily_returns.rolling(ROLLING_SHARPE_WINDOW).apply(
            lambda x: (
                (x.mean() / x.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
                if x.std(ddof=1) > 0
                else np.nan
            ),
            raw=False,
        )

        monthly = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).sum()
        monthly_returns = [
            [int(year), int(month), _safe_float(value)] for (year, month), value in monthly.items()
        ]

        # Per-year Sharpe + return — exposes regime fragility that the
        # single full-period Sharpe averages out.
        yearly_returns = []
        for year, group in daily_returns.groupby(daily_returns.index.year):
            mean_y = float(group.mean()) if len(group) else 0.0
            std_y = float(group.std(ddof=1)) if len(group) > 1 else 0.0
            year_sharpe = mean_y / std_y * math.sqrt(TRADING_DAYS_PER_YEAR) if std_y > 0 else 0.0
            yearly_returns.append(
                {
                    "year": int(year),
                    "sharpe": _safe_float(year_sharpe),
                    "annual_return": _safe_float(group.sum()),
                    "n_days": int(len(group)),
                }
            )

        # ---- IC / alpha-decay / signal-quality metrics -------------------
        # These need the post-neutralization signal matrix + per-stock
        # forward returns.  Older saved BacktestResult objects predate these
        # fields, so we gracefully degrade to None when they're missing.
        ic_summary: dict[str, Any] = {
            "ic": None,
            "icir": None,
            "ic_tstat": None,
            "ic_pct_positive": None,
            "n_days": 0,
        }
        alpha_decay: dict[str, Any] = {
            "ic_by_horizon": {},
            "half_life_days": None,
            "r_squared": None,
        }
        rank_stability: float | None = None
        ic_series_payload: dict[str, Any] | None = None
        quintile_returns: list[dict[str, Any]] = []
        if result.signal_matrix is not None and result.forward_returns is not None:
            ic_summary = compute_ic_summary(result.signal_matrix, result.forward_returns, horizon=1)
            alpha_decay = compute_alpha_decay(
                result.signal_matrix, result.forward_returns, horizons=DEFAULT_DECAY_HORIZONS
            )
            rank_stability = compute_rank_stability(result.signal_matrix)
            # Tier 4: raw daily IC series for the rolling-MA + z-score chart.
            # NaN-only / empty series collapse to None so the frontend hides
            # the panel cleanly.
            ic_raw = compute_ic_series(result.signal_matrix, result.forward_returns, horizon=1)
            ic_clean = ic_raw.dropna() if ic_raw is not None else None
            if ic_clean is not None and len(ic_clean) >= 2:
                ic_series_payload = {
                    "dates": [d.strftime("%Y-%m-%d") for d in ic_clean.index],
                    "ic": [_safe_float(v) for v in ic_clean.values],
                }
            # Tier 4: factor quintile returns — bucket each day's stocks by
            # signal value, time-average per-bucket forward returns.
            quintile_returns = compute_quintile_returns(
                result.signal_matrix, result.forward_returns, n_quantiles=5
            )

        # Tail ratio, positive-months %, DD durations
        tail_ratio = _tail_ratio(daily_returns)
        positive_months_pct = _positive_months_pct(daily_returns)
        dd_durations = _drawdown_durations(equity)
        fitness_wq = _fitness_wq(sharpe, annual_return, avg_turnover_frac)

        # ---- Long/Short exposure (PDF Section 5.1: "Long/Short Exposure") ---
        # Reported as a fraction of booksize so the numbers are scale-free and
        # comparable across saved alphas with different booksize settings.
        # All four values default to None (instead of 0) so an old saved
        # result with no positions matrix renders the row as "—" rather than
        # a misleading "0.00% exposure".
        #
        # Use ``positions`` (dollar-denominated holdings) — not ``weights``
        # (normalized signal in [-1, +1]) — to keep numerator and denominator
        # in the same units as ``book_proxy``.  Mixing the two yields 1e-8
        # ratios; ask me how I learned.
        long_exposure: float | None = None
        short_exposure: float | None = None
        gross_exposure: float | None = None
        net_exposure: float | None = None
        if positions is not None and not positions.empty and book_proxy > 0:
            longs_per_day = positions.where(positions > 0, 0.0).sum(axis=1)
            shorts_per_day = positions.where(positions < 0, 0.0).sum(axis=1)  # signed (neg)
            long_exposure = longs_per_day.mean() / book_proxy
            short_exposure = shorts_per_day.mean() / book_proxy
            # Gross = |L| + |S| (always ≥ 0); net = L + S (≈ 0 for dollar-neutral)
            gross_exposure = (longs_per_day - shorts_per_day).mean() / book_proxy
            net_exposure = (longs_per_day + shorts_per_day).mean() / book_proxy

        # ---- Tier 2: sector + size exposure, regime stress test ----------
        sector_exposure: dict[str, Any] | None = None
        sector_exposure_timeseries: dict[str, Any] | None = None
        size_exposure: dict[str, Any] | None = None
        market_cap_distribution: dict[str, Any] | None = None
        pnl_attribution: dict[str, Any] | None = None
        stress_test: list[dict[str, Any]] = []
        if result.weights is not None and not result.weights.empty:
            if gics_map is not None:
                sector_exposure = compute_sector_exposure(result.weights, gics_map)
                # PDF Section 5.3.2 — per-date per-sector heatmap data.
                # Downsampled to ~120 buckets so the payload stays bounded.
                sector_exposure_timeseries = compute_sector_exposure_timeseries(
                    result.weights, gics_map
                )
                # PDF Section 5.2 — Brinson-style allocation vs selection
                # split.  Only meaningful when both GICS map and per-stock
                # forward returns are available; older saved results that
                # lack forward_returns get None.
                if result.forward_returns is not None:
                    pnl_attribution = compute_pnl_attribution(
                        result.weights, result.forward_returns, gics_map
                    )
            if size_field is not None and not size_field.empty:
                # Heuristic: if the matrix's max value is < 10_000 it's
                # almost certainly a price series, not market_cap (which
                # ranges in the billions).  Flag as approximation so the UI
                # can mark it.
                try:
                    sample_max = float(size_field.abs().to_numpy().max())
                except (TypeError, ValueError):
                    sample_max = float("nan")
                is_approx = not math.isnan(sample_max) and sample_max < 10_000.0
                size_exposure = compute_size_exposure(
                    result.weights, size_field, is_approximation=is_approx
                )
                # PDF Section 5.3.4: per-decile $ exposure on long vs short
                # side.  Same approximation flag as size_exposure since both
                # share the size_field heuristic above.
                market_cap_distribution = compute_market_cap_distribution(
                    result.weights, size_field, is_approximation=is_approx
                )

        stress_test = compute_stress_metrics(daily_returns)

        # Deflated Sharpe (Bailey & López de Prado): adjusts the headline
        # Sharpe for the selection bias from running ``n_trials`` candidates.
        # Pandas' .skew()/.kurt() return the sample versions; .kurt() is
        # *excess* kurtosis, so add 3 for the formula's expected full kurt.
        skew_val = float(daily_returns.skew()) if n > 3 else 0.0
        kurt_val = float(daily_returns.kurt()) + 3.0 if n > 3 else 3.0
        deflated = deflated_sharpe(
            sharpe_annual=sharpe,
            n_trials=max(1, int(n_trials)),
            n_obs=n,
            skew=skew_val if not math.isnan(skew_val) else 0.0,
            kurt=kurt_val if not math.isnan(kurt_val) else 3.0,
        )

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
                _safe_float(information_ratio) if information_ratio is not None else None
            ),
            "rolling_sharpe": _safe_list(rolling_sharpe.tolist()),
            "monthly_returns": monthly_returns,
            "yearly_returns": yearly_returns,
            "drawdown_series": _safe_list(drawdown.tolist()),
            "deflated_sharpe": deflated,
            # Signal-quality block (Tier 1 research metrics)
            "ic": _safe_float(ic_summary.get("ic")),
            "icir": _safe_float(ic_summary.get("icir")),
            "ic_tstat": _safe_float(ic_summary.get("ic_tstat")),
            "ic_pct_positive": _safe_float(ic_summary.get("ic_pct_positive")),
            "ic_n_days": int(ic_summary.get("n_days") or 0),
            "alpha_decay": {
                "ic_by_horizon": {
                    int(h): _safe_float(v)
                    for h, v in (alpha_decay.get("ic_by_horizon") or {}).items()
                },
                "half_life_days": _safe_float(alpha_decay.get("half_life_days")),
                "r_squared": _safe_float(alpha_decay.get("r_squared")),
            },
            "rank_stability": _safe_float(rank_stability),
            "tail_ratio": _safe_float(tail_ratio),
            "positive_months_pct": _safe_float(positive_months_pct),
            "drawdown_durations": dd_durations,
            "fitness_wq": _safe_float(fitness_wq),
            # Tier 4 — daily IC series for time-series chart, quintile bars
            "ic_series": ic_series_payload,
            "quintile_returns": quintile_returns,
            # Long/Short dollar exposure as fractions of booksize.  For a true
            # dollar-neutral L/S strategy: net ≈ 0, gross ≈ 2·long, short < 0.
            "long_exposure": _safe_float(long_exposure),
            "short_exposure": _safe_float(short_exposure),
            "gross_exposure": _safe_float(gross_exposure),
            "net_exposure": _safe_float(net_exposure),
            # Tier 2 — exposure + stress test
            "sector_exposure": sector_exposure,
            # PDF Section 5.3.2 — sector × time matrix for the heatmap chart.
            # Downsampled (default ~120 buckets) so payload stays bounded
            # across long backtests.
            "sector_exposure_timeseries": sector_exposure_timeseries,
            "size_exposure": size_exposure,
            # PDF Section 5.3.4 — per-decile $ on long vs short side, reveals
            # size tilt that the single-scalar size_corr summarises.
            "market_cap_distribution": market_cap_distribution,
            # PDF Section 5.2 — Brinson allocation vs selection split.
            # None when forward_returns is unavailable (old saved results).
            "pnl_attribution": pnl_attribution,
            # Cost realism (flat / spread / impact / borrow split).  None if
            # the backtester didn't populate result.cost_components.
            "cost_breakdown": cost_breakdown,
            "stress_test": stress_test,
            # Date range carried forward so compare_is_oos can build period
            # labels without needing the original BacktestResult.
            "start_date": result.dates[0] if result.dates else None,
            "end_date": result.dates[-1] if result.dates else None,
        }

    def compare_is_oos(
        self, is_metrics: dict[str, Any], oos_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Diagnose whether OOS performance has decayed from IS.

        Returns sharpe/return decay (positive = OOS worse than IS) and a
        human-readable label / overfitting flag for UI use.
        """
        is_sharpe = is_metrics.get("sharpe") or 0.0
        oos_sharpe = oos_metrics.get("sharpe") or 0.0
        is_return = is_metrics.get("annual_return") or 0.0
        oos_return = oos_metrics.get("annual_return") or 0.0

        sharpe_decay = (
            (is_sharpe - oos_sharpe) / abs(is_sharpe) if is_sharpe not in (0, 0.0) else 0.0
        )
        return_decay = (
            (is_return - oos_return) / abs(is_return) if is_return not in (0, 0.0) else 0.0
        )

        overfitting_flag = bool(sharpe_decay > 0.5 or oos_sharpe < 0)

        # The decay-based label assumes the IS Sharpe is positive.  If OOS
        # actually loses money it doesn't matter whether decay is small —
        # the alpha is unusable and should be flagged accordingly.
        if oos_sharpe < 0:
            label = "Negative OOS — alpha lost money out-of-sample"
            severity = "severe"
        elif sharpe_decay < 0.2:
            label = "Robust — OOS performance close to IS"
            severity = "robust"
        elif sharpe_decay < 0.4:
            label = "Moderate decay — some overfitting likely"
            severity = "moderate"
        elif sharpe_decay < 0.6:
            label = "High decay — likely overfit, use with caution"
            severity = "high"
        else:
            label = "Severe overfit — alpha does not generalize"
            severity = "severe"

        return {
            "sharpe_decay": _safe_float(sharpe_decay),
            "return_decay": _safe_float(return_decay),
            "overfitting_flag": overfitting_flag,
            "overfitting_label": label,
            "severity": severity,
            "is_period": {
                "start": is_metrics.get("start_date"),
                "end": is_metrics.get("end_date"),
            },
            "oos_period": {
                "start": oos_metrics.get("start_date"),
                "end": oos_metrics.get("end_date"),
            },
        }
