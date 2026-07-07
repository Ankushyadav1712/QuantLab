"""Backtester-vs-external correlation check (the "Brain validator").

Aligns a local backtest's daily-return series with an externally-produced one
(e.g. a WorldQuant Brain PnL export) by date, then reports how faithfully the
local backtester reproduces the external result: day-by-day return correlation
plus the annualised-Sharpe gap. The question it answers — "is my local sandbox a
trustworthy proxy for the competition server?" — is the whole point of the tool.

Pure module: pandas only, no FastAPI / main import, so it unit-tests fast without
spinning up the app or loading market data.

Both correlation and Sharpe are scale-invariant, so the external column may be
daily returns OR daily PnL (dollars) — only the per-day shape matters, not the
units. A cumulative series must be differenced to daily before upload.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

TRADING_DAYS = 252
# Below this many overlapping days, correlation/Sharpe are noise, not signal.
MIN_OVERLAP = 20
# |external Sharpe| below this makes a *relative* Sharpe-gap % meaningless.
_SHARPE_FLOOR = 0.05


def _to_series(dates: list[Any], values: list[Any]) -> pd.Series:
    """Date-indexed float series; drops unparseable dates/values and NaNs.

    Duplicate dates keep the last value (matches how a re-run would overwrite).
    """
    m = min(len(dates), len(values))
    idx = pd.to_datetime(pd.Series(list(dates[:m])), errors="coerce")
    vals = pd.to_numeric(pd.Series(list(values[:m])), errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(idx))
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()


def _annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    sd = float(returns.std(ddof=1))
    if sd <= 0 or math.isnan(sd):
        return float("nan")
    return float(returns.mean() / sd * math.sqrt(TRADING_DAYS))


def _safe(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def validate_correlation(
    local_dates: list[Any],
    local_returns: list[Any],
    external_dates: list[Any],
    external_returns: list[Any],
    *,
    sharpe_tolerance_pct: float = 3.0,
    min_overlap: int = MIN_OVERLAP,
) -> dict[str, Any]:
    """Compare a local return series with an external one on their common dates.

    Returns ``{"ok": False, "error": ...}`` when there isn't enough overlap to
    say anything; otherwise the correlation, both annualised Sharpes, the Sharpe
    gap (absolute + relative-to-external %), a tolerance check, and a verdict.
    """
    local = _to_series(local_dates, local_returns)
    external = _to_series(external_dates, external_returns)
    if local.empty or external.empty:
        return {"ok": False, "error": "A series is empty after parsing.", "n_overlap": 0}

    aligned = pd.concat({"local": local, "external": external}, axis=1).dropna()
    n = int(len(aligned))
    if n < min_overlap:
        return {
            "ok": False,
            "error": f"Only {n} overlapping day(s); need at least {min_overlap} to compare.",
            "n_overlap": n,
        }

    lcol = aligned["local"]
    ecol = aligned["external"]
    # Correlation is undefined if either side is constant over the window.
    both_vary = lcol.std(ddof=1) > 0 and ecol.std(ddof=1) > 0
    corr = float(lcol.corr(ecol)) if both_vary else float("nan")

    local_sharpe = _annualized_sharpe(lcol)
    external_sharpe = _annualized_sharpe(ecol)
    have_both = not (math.isnan(local_sharpe) or math.isnan(external_sharpe))
    sharpe_diff = local_sharpe - external_sharpe if have_both else float("nan")

    # A relative Sharpe gap only means something when the external Sharpe isn't ~0.
    if have_both and abs(external_sharpe) >= _SHARPE_FLOOR:
        sharpe_diff_pct = abs(sharpe_diff) / abs(external_sharpe) * 100.0
        within: bool | None = sharpe_diff_pct <= sharpe_tolerance_pct
    else:
        sharpe_diff_pct = float("nan")
        within = None

    if math.isnan(corr):
        corr_quality = "unknown"
    elif corr >= 0.95:
        corr_quality = "high"
    elif corr >= 0.8:
        corr_quality = "moderate"
    else:
        corr_quality = "low"

    # Verdict: the backtester is a faithful proxy when returns track closely AND
    # the Sharpe lands within tolerance. Weak correlation is the real red flag —
    # it means the two are modelling something structurally different.
    if not math.isnan(corr) and corr < 0.7:
        verdict = "fail"
        detail = "Low return correlation — the local and external runs diverge materially."
    elif corr >= 0.9 and within:
        verdict = "pass"
        detail = "Local backtester faithfully reproduces the external result."
    else:
        verdict = "review"
        detail = "Close but not within tolerance — check costs, execution lag, universe, and dates."

    return {
        "ok": True,
        "n_overlap": n,
        "correlation": _safe(corr),
        "correlation_quality": corr_quality,
        "local_sharpe": _safe(local_sharpe),
        "external_sharpe": _safe(external_sharpe),
        "sharpe_diff": _safe(sharpe_diff),
        "sharpe_diff_pct": _safe(sharpe_diff_pct),
        "sharpe_tolerance_pct": float(sharpe_tolerance_pct),
        "within_tolerance": within,
        "verdict": verdict,
        "verdict_detail": detail,
    }
