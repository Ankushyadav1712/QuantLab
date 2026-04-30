"""Fama-French 5-factor regression on a strategy's daily returns.

Strips the part of strategy return explained by market beta + size, value,
profitability, and investment factors.  The intercept of the regression
(annualized) is the *residual* alpha — the only definition of alpha most
quant firms accept.  R² tells you how much of your "alpha" was actually just
factor exposure.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analytics.performance import _safe_float

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
FACTOR_LABEL_MAP = {
    "Mkt-RF": "market",
    "SMB": "size",
    "HML": "value",
    "RMW": "profitability",
    "CMA": "investment",
}
TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 60  # below this, OLS is too noisy to report


class FactorDecomposition:
    """Run an OLS regression of strategy excess returns on FF5 factors."""

    def compute(
        self,
        daily_returns: list[float | None],
        dates: list[str],
        ff5: pd.DataFrame,
    ) -> dict[str, Any] | None:
        if ff5 is None or ff5.empty:
            return None

        s = pd.Series(daily_returns, index=pd.to_datetime(dates), dtype=float).dropna()
        aligned = ff5.reindex(s.index).dropna()
        if len(aligned) < MIN_OBSERVATIONS:
            return None
        s = s.reindex(aligned.index)

        # Excess strategy returns (subtract the risk-free rate from each day)
        rf = aligned["RF"] if "RF" in aligned.columns else 0.0
        y = (s - rf).to_numpy()
        X = aligned[FACTOR_COLS].to_numpy()

        # Augment with intercept column for alpha
        n, k_no_const = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        k = X_aug.shape[1]

        # OLS via lstsq (singular-aware)
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat = X_aug @ beta
        resid = y - y_hat

        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard errors → t-statistics
        try:
            sigma2 = ss_res / (n - k) if n > k else float("inf")
            xtx_inv = np.linalg.inv(X_aug.T @ X_aug)
            se = np.sqrt(np.diag(xtx_inv) * sigma2)
            t_stats = beta / np.where(se > 0, se, np.nan)
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
            t_stats = np.full(k, np.nan)

        alpha_daily = float(beta[0])
        alpha_annualized = alpha_daily * TRADING_DAYS_PER_YEAR
        alpha_t = float(t_stats[0]) if not math.isnan(t_stats[0]) else None

        loadings = {}
        for i, col in enumerate(FACTOR_COLS, start=1):
            label = FACTOR_LABEL_MAP[col]
            loadings[label] = {
                "beta": _safe_float(beta[i]),
                "t_stat": _safe_float(t_stats[i]),
            }

        # How much of the strategy's gross return is "explained" by factor
        # exposure?  Useful one-number summary that complements R².
        explained_by_factors = float((y_hat - alpha_daily).var())
        total_var = float(y.var())
        factor_share = (
            explained_by_factors / total_var if total_var > 0 else 0.0
        )

        return {
            "alpha_annualized": _safe_float(alpha_annualized),
            "alpha_t_stat": _safe_float(alpha_t),
            "alpha_significant": bool(alpha_t is not None and abs(alpha_t) > 2.0),
            "r_squared": _safe_float(r_squared),
            "factor_share": _safe_float(factor_share),
            "loadings": loadings,
            "sample_size": int(n),
            "period": {
                "start": s.index[0].strftime("%Y-%m-%d"),
                "end": s.index[-1].strftime("%Y-%m-%d"),
            },
        }
