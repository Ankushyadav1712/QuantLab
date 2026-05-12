"""Crisis-window stress test.

Headline Sharpe averages across regimes; an alpha can have Sharpe 1.2
overall while losing money during every crisis.  This module slices daily
returns to a curated list of well-known crisis windows and recomputes the
basic risk metrics inside each.

Regimes that don't overlap the backtest's date range (or that produce fewer
than ``MIN_REGIME_DAYS`` valid days) are silently skipped — so a 2019-onward
backtest naturally drops the GFC window without the caller having to think
about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# Annualization constant — keep in sync with performance.py
TRADING_DAYS_PER_YEAR = 252

# Minimum valid days in a window for it to produce a reportable result.
# Below this, Sharpe / std are too noisy to be meaningful.
MIN_REGIME_DAYS = 5


@dataclass(frozen=True)
class Regime:
    """A named crisis window.

    Attributes
    ----------
    name : str
        Stable identifier (also the dict key in the response).
    label : str
        Human-readable label for UI display.
    start : str
        ISO-8601 date (inclusive).
    end : str
        ISO-8601 date (inclusive).
    """

    name: str
    label: str
    start: str
    end: str


# ----------------------------------------------------------------------
# Default regime list.  These are the windows quants typically stress
# alphas against — peak-to-trough crashes plus a few slow-bleed bear
# regimes that decimate momentum strategies.
# ----------------------------------------------------------------------
DEFAULT_REGIMES: tuple[Regime, ...] = (
    Regime("gfc_crash", "GFC Crash", "2008-09-15", "2009-03-09"),
    Regime("euro_debt_2011", "Euro Debt Crisis", "2011-07-22", "2011-10-04"),
    Regime("china_2015", "China Devaluation", "2015-08-17", "2016-02-11"),
    Regime("vol_q4_2018", "Q4 2018 Selloff", "2018-10-03", "2018-12-24"),
    Regime("covid_crash", "COVID Crash", "2020-02-19", "2020-03-23"),
    Regime("covid_recovery", "COVID Recovery", "2020-03-24", "2020-08-31"),
    Regime("inflation_bear", "Inflation Bear 2022", "2022-01-03", "2022-10-12"),
    Regime("svb_crisis", "SVB Banking Crisis", "2023-03-08", "2023-03-31"),
)


def _compute_regime_metrics(window_returns: pd.Series, regime: Regime) -> dict | None:
    """Compute Sharpe / DD / hit-rate / annualised return inside one window."""
    if len(window_returns) < MIN_REGIME_DAYS:
        return None

    clean = window_returns.dropna()
    n = len(clean)
    if n < MIN_REGIME_DAYS:
        return None

    mean_dr = float(clean.mean())
    std_dr = float(clean.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean_dr / std_dr * math.sqrt(TRADING_DAYS_PER_YEAR) if std_dr > 0 else 0.0

    total_return = float(clean.sum())
    # Annualised return inside the window — useful for cross-regime comparison
    # even when windows have different lengths.
    years = n / TRADING_DAYS_PER_YEAR
    annualised = (
        ((1.0 + total_return) ** (1.0 / years) - 1.0)
        if years > 0 and (1.0 + total_return) > 0
        else 0.0
    )

    # Drawdown inside the window
    equity = 1.0 + clean.cumsum()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min()) if n else 0.0

    hit_rate = float((clean > 0).mean()) if n else 0.0

    return {
        "name": regime.name,
        "label": regime.label,
        "start": regime.start,
        "end": regime.end,
        "n_days": int(n),
        "sharpe": _safe_float(sharpe),
        "total_return": _safe_float(total_return),
        "annualised_return": _safe_float(annualised),
        "max_drawdown": _safe_float(max_dd),
        "hit_rate": _safe_float(hit_rate),
    }


def _safe_float(x: float) -> float | None:
    if x is None or math.isnan(x) or math.isinf(x):
        return None
    return float(x)


def compute_stress_metrics(
    daily_returns: pd.Series,
    regimes: tuple[Regime, ...] = DEFAULT_REGIMES,
) -> list[dict]:
    """Slice ``daily_returns`` into each regime window and compute per-regime metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        DatetimeIndex-indexed daily fractional returns (the same series used
        for headline Sharpe).
    regimes : tuple[Regime, ...]
        Override the default list — useful for tests.

    Returns
    -------
    list[dict] in the same order as ``regimes``.  Regimes that don't overlap
    the data range or produce <MIN_REGIME_DAYS valid days are omitted.
    """
    if daily_returns is None or len(daily_returns) == 0:
        return []

    # Ensure the index is DatetimeIndex — older saved results may carry
    # string dates.
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        try:
            daily_returns = pd.Series(
                daily_returns.values,
                index=pd.to_datetime(daily_returns.index),
            )
        except (TypeError, ValueError):
            return []

    results: list[dict] = []
    for r in regimes:
        start = pd.Timestamp(r.start)
        end = pd.Timestamp(r.end)
        window = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end)]
        metrics = _compute_regime_metrics(window, r)
        if metrics is not None:
            results.append(metrics)
    return results


def regime_severity(regime_result: dict) -> str:
    """Classify a regime's outcome for UI colouring.

    Returns one of ``'good'`` / ``'warn'`` / ``'bad'`` based on the regime
    Sharpe — the most regime-comparable headline number.
    """
    sharpe = regime_result.get("sharpe")
    if sharpe is None:
        return "warn"
    if sharpe >= 0.5:
        return "good"
    if sharpe >= -0.5:
        return "warn"
    return "bad"
