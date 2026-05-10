"""Mean-variance + risk-parity weighting for multi-alpha blends.

All implementations are **pure numpy** — we deliberately don't pull in scipy
just for an unconstrained QP, since the closed-form mean-variance solution
is one matrix inversion away.  Long-only / box-constrained variants would
need scipy.optimize.minimize and are deferred until requested.

Four methods, all returning a 1-D numpy array of weights summing to 1:

    equal_weights(n)            : 1/n each.  Trivial baseline.
    inverse_variance(returns)   : w_i ∝ 1/σ_i².  Robust, ignores correlations.
    mv_optimal(returns, target_vol=None)
                                : Markowitz tangency: w = Σ⁻¹ μ, normalized.
                                  May produce shorts (no non-negativity constraint).
                                  If ``target_vol`` is set, scales to hit that
                                  annualized vol exactly.
    risk_parity(returns)        : equal vol contribution: σ_i × w_i = const.
                                  Iterative numerical solve, converges in
                                  <100 iterations on typical inputs.

All inputs:
    returns : pd.DataFrame
        Daily simple returns, columns = alphas, rows = dates.
        NaNs are dropped per-column-pair when computing covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def equal_weights(n: int) -> np.ndarray:
    """1/n each; the trivial baseline that the existing multi-blend uses."""
    if n <= 0:
        raise ValueError(f"equal_weights needs n > 0, got {n}")
    return np.full(n, 1.0 / n)


def inverse_variance(returns: pd.DataFrame) -> np.ndarray:
    """w_i ∝ 1/σ_i².  Practical default — robust to small samples and
    correlated alphas (no matrix inverse required)."""
    var = returns.var(ddof=1).values  # length-n
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        # Degenerate: fall back to equal weighting rather than blow up
        return equal_weights(len(var))
    inv_var = 1.0 / var
    return inv_var / inv_var.sum()


def mv_optimal(returns: pd.DataFrame, target_vol: float | None = None) -> np.ndarray:
    """Closed-form mean-variance tangency portfolio: w = Σ⁻¹ μ, normalized.

    No non-negativity constraint — the result may include shorts.  This is
    intentional for an alpha-blending context: alphas are signed signals,
    "shorting" one of them (i.e. negating its weight) is a valid choice.

    If ``target_vol`` (annualized) is provided, the weights are scaled so
    the portfolio vol matches.  Otherwise weights are normalized to sum to 1
    so they're directly comparable to the equal-weight baseline.
    """
    mu = returns.mean().values * TRADING_DAYS_PER_YEAR  # annualized
    cov = returns.cov().values * TRADING_DAYS_PER_YEAR  # annualized
    n = len(mu)

    # Regularize the covariance with a tiny ridge so it's invertible even
    # when alphas are near-perfectly correlated.
    ridge = 1e-6 * np.trace(cov) / max(1, n)
    cov_reg = cov + ridge * np.eye(n)

    try:
        raw = np.linalg.solve(cov_reg, mu)
    except np.linalg.LinAlgError:
        return equal_weights(n)

    if not np.all(np.isfinite(raw)):
        return equal_weights(n)

    if target_vol is not None:
        # Scale so √(wᵀ Σ w) == target_vol
        port_var = float(raw @ cov @ raw)
        if port_var <= 0:
            return equal_weights(n)
        scale = float(target_vol) / np.sqrt(port_var)
        return raw * scale

    # Default: normalize so weights sum to 1 (directly comparable to equal-weight)
    s = raw.sum()
    if abs(s) < 1e-12:
        return equal_weights(n)
    return raw / s


def risk_parity(returns: pd.DataFrame, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Equal-risk-contribution weights: each alpha contributes the same to
    portfolio variance.  Iterative; converges quickly for well-behaved cov.

    Algorithm (the classic Roncalli scheme): repeatedly update each weight by
    the ratio of its current risk contribution to the target (1/n of total),
    re-normalize, repeat until weights stabilize.
    """
    cov = returns.cov().values
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Start from inverse-vol weights — closer to the answer than equal weights
    sigma = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        return equal_weights(n)
    w = 1.0 / sigma
    w = w / w.sum()

    target = 1.0 / n
    for _ in range(max_iter):
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol <= 0:
            return equal_weights(n)
        # Marginal contribution of each name to portfolio vol
        marg = (cov @ w) / port_vol
        rc = w * marg / port_vol  # share of total variance from name i
        # Update weights toward equal contribution
        w_new = w * np.sqrt(target / np.maximum(rc, 1e-12))
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


WEIGHT_METHODS = {
    "equal": "Equal — 1/n each (no covariance estimation)",
    "inverse_variance": "Inverse-variance — robust, ignores correlations",
    "mv_optimal": "Mean-variance optimal — Σ⁻¹μ; may produce shorts",
    "risk_parity": "Risk parity — equal vol contribution per alpha",
}


def compute_weights(
    method: str,
    returns: pd.DataFrame,
    target_vol: float | None = None,
) -> np.ndarray:
    """Dispatch based on user-selected method.  Unknown methods fall back to
    equal weights with a clear error so the caller can surface it."""
    if method == "equal":
        return equal_weights(returns.shape[1])
    if method == "inverse_variance":
        return inverse_variance(returns)
    if method == "mv_optimal":
        return mv_optimal(returns, target_vol=target_vol)
    if method == "risk_parity":
        return risk_parity(returns)
    raise ValueError(f"Unknown weight method: {method!r}. Known: {sorted(WEIGHT_METHODS.keys())}")
