"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

When a researcher runs N alpha trials, the *highest* observed Sharpe is biased
upward — picking the best of N noisy estimates inflates the headline.  The DSR
asks: given N trials and the sample's higher-moment shape, what is the
probability that the *true* Sharpe is positive?

Inputs:
    * sharpe_annual — the observed annualized Sharpe of the chosen strategy
    * n_trials     — how many candidate strategies the researcher has tried
    * n_obs        — number of return observations in the chosen strategy
    * skew, kurt   — sample skew and *full* (not excess) kurtosis of returns

Returns the deflated annualized Sharpe (float) and a probability in [0,1]
that the true Sharpe exceeds zero.  ``None`` for degenerate inputs.

stdlib-only (`math`) so we don't need scipy.
"""

from __future__ import annotations

import math
from typing import Any

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF — Acklam's rational approximation.

    Accurate to ~1.15e-9 over the central region; relative error <1e-9 in the
    tails.  Returns ``nan`` outside (0, 1).
    """
    if not (0.0 < p < 1.0):
        return float("nan")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def deflated_sharpe(
    sharpe_annual: float | None,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any] | None:
    """Compute the deflated Sharpe ratio + p-value of true SR > 0.

    Args:
        sharpe_annual: Observed annualized Sharpe of the picked strategy.
        n_trials: How many candidate alphas the user has tested in this
            session.  ``1`` collapses to a no-deflation comparison.
        n_obs: Number of daily returns underlying ``sharpe_annual``.
        skew: Sample skewness of daily returns.
        kurt: *Full* (not excess) kurtosis — pandas' ``.kurt()`` returns
            *excess* by default, so the caller should add 3.
        periods_per_year: Annualization factor for SR.

    Returns:
        ``{deflated_sharpe_annualized, p_value, sharpe_threshold_annualized,
        n_trials, n_obs, is_significant}`` — or ``None`` for inputs the
        formula can't handle (too-short window, NaN inputs, …).
    """
    if (
        sharpe_annual is None
        or n_obs < 2
        or n_trials < 1
        or math.isnan(sharpe_annual)
        or math.isinf(sharpe_annual)
    ):
        return None

    sr = sharpe_annual / math.sqrt(periods_per_year)  # per-period SR

    # Expected max of N i.i.d. standard normals (Bailey & López de Prado eq. 9):
    # ((1-γ) * Φ⁻¹(1 - 1/N)) + (γ * Φ⁻¹(1 - 1/(N·e)))
    if n_trials == 1:
        e_max_z = 0.0
    else:
        e_max_z = (1.0 - EULER_MASCHERONI) * _norm_ppf(1.0 - 1.0 / n_trials) + (
            EULER_MASCHERONI * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    sr_0 = e_max_z / math.sqrt(n_obs)  # the per-period Sharpe one expects by chance

    excess_kurt = kurt - 3.0  # pandas kurt() is excess; callers pass full
    denom_sq = 1.0 - skew * sr + (excess_kurt / 4.0) * sr * sr
    if denom_sq <= 0.0:
        return None

    z = (sr - sr_0) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    p_value = _norm_cdf(z)

    deflated_annual = (sr - sr_0) * math.sqrt(periods_per_year)
    threshold_annual = sr_0 * math.sqrt(periods_per_year)

    return {
        "deflated_sharpe_annualized": deflated_annual,
        "p_value": p_value,
        "sharpe_threshold_annualized": threshold_annual,
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "is_significant": bool(p_value > 0.95),
    }
