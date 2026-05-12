"""Information Coefficient + alpha-decay analytics.

Three metrics that operate on the (date × ticker) signal matrix rather than
on the final daily PnL series:

- **IC (Rank IC)** — daily Spearman rank correlation between the signal at
  time t and the cross-sectional realized return at time t+1.  Tells you
  whether the *ranking* you produced predicts the next day's return ordering,
  before portfolio construction shrinks the differences.
- **ICIR** — mean(IC) / std(IC) * √252.  Stability of the IC across time.
- **Alpha decay** — IC at horizons 1, 2, 3, 5, 10, 21 days; fit an
  exponential decay to recover the half-life of the predictive power.

Plus a related signal-persistence metric:

- **Rank stability** — mean day-over-day Spearman correlation of the signal's
  cross-sectional rank.  Proxy for whether the alpha's ranking drifts
  smoothly or shuffles randomly day-to-day.

The computations are pure-numpy (no scipy) — Spearman is implemented as
Pearson-on-ranks since the two are mathematically equivalent and the rank
form vectorises cleanly across the date axis.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Horizons (in trading days) at which we sample IC for the decay fit.  Kept
# short — beyond ~21 days the IC of typical fast alphas is below the noise
# floor and adding more horizons just confuses the exponential fit.
DEFAULT_DECAY_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 21)

TRADING_DAYS_PER_YEAR = 252


def _rank_along_rows(arr: np.ndarray) -> np.ndarray:
    """Rank each row independently, with NaN preserved as NaN.

    Uses average-rank for ties (matches scipy.stats.rankdata's default).
    Vectorised across the row axis.
    """
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        valid = row[mask]
        # argsort-of-argsort gives ranks; convert ties to average rank below
        order = valid.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(valid), dtype=float)
        # Handle ties via average rank — group equal values and overwrite
        # their ranks with the group mean
        sorted_vals = valid[order]
        i_start = 0
        for j in range(1, len(sorted_vals) + 1):
            if j == len(sorted_vals) or sorted_vals[j] != sorted_vals[i_start]:
                if j - i_start > 1:
                    avg = (i_start + j - 1) / 2.0
                    for k in range(i_start, j):
                        ranks[order[k]] = avg
                i_start = j
        out[i, mask] = ranks
    return out


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation between two (T, N) matrices, NaN-safe.

    Returns a (T,) array.  Rows where either side has <2 valid observations
    or zero variance produce NaN.
    """
    out = np.full(a.shape[0], np.nan, dtype=float)
    for i in range(a.shape[0]):
        mask = ~np.isnan(a[i]) & ~np.isnan(b[i])
        if mask.sum() < 2:
            continue
        x = a[i, mask]
        y = b[i, mask]
        sx = x.std(ddof=1)
        sy = y.std(ddof=1)
        if sx <= 0 or sy <= 0:
            continue
        out[i] = float(
            ((x - x.mean()) * (y - y.mean())).mean() * (len(x) / (len(x) - 1)) / (sx * sy)
        )
    return out


def compute_ic_series(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """Daily rank IC: spearmanr(signal[t], forward_return[t+horizon]) per row.

    The signal and forward_returns frames are aligned to the intersection of
    their (date × ticker) grids.  Days where fewer than 2 stocks have valid
    pairs are NaN.

    Parameters
    ----------
    signal : DataFrame (date × ticker)
        Post-neutralization alpha values.
    forward_returns : DataFrame (date × ticker)
        Per-stock realized returns.  These are *contemporaneous* returns; the
        horizon shift is applied inside this function.
    horizon : int, default 1
        Trading-day forward horizon.  For h=1, IC[t] uses signal[t] and
        return[t+1] — which lines up with the backtester's T+1 execution
        convention (signal at close t earns the return into close t+1).
    """
    if signal is None or signal.empty or forward_returns is None or forward_returns.empty:
        return pd.Series(dtype=float)
    horizon = max(1, int(horizon))

    # Align columns to the intersection — order doesn't matter, ranks are
    # row-wise.  Reindexing prevents weird behavior when the two frames have
    # subtly different ticker sets.
    cols = signal.columns.intersection(forward_returns.columns)
    if len(cols) < 2:
        return pd.Series(dtype=float)

    # Shift returns *up* by `horizon` rows so that fwd_shifted.iloc[t] is the
    # return realized on day t+horizon.  The final `horizon` rows become NaN
    # and are dropped from the IC series.
    fwd_shifted = forward_returns[cols].shift(-horizon)
    # Align signal to forward-returns' index (it's the canonical one).
    sig = signal[cols].reindex(fwd_shifted.index)

    sig_ranks = _rank_along_rows(sig.values)
    ret_ranks = _rank_along_rows(fwd_shifted.values)
    ic_values = _row_corr(sig_ranks, ret_ranks)
    return pd.Series(ic_values, index=fwd_shifted.index, name=f"ic_h{horizon}")


def compute_ic_summary(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int = 1,
) -> dict[str, float | None]:
    """Headline IC + ICIR + t-stat from a single horizon.

    Returns
    -------
    dict with keys ``ic`` (mean IC), ``icir`` (mean / std × √252),
    ``ic_tstat`` (mean / SE), ``ic_pct_positive`` (share of days with IC>0),
    and ``n_days`` (number of usable days in the IC series).
    """
    ic = compute_ic_series(signal, forward_returns, horizon=horizon)
    ic_clean = ic.dropna()
    n = int(len(ic_clean))
    if n < 2:
        return {
            "ic": None,
            "icir": None,
            "ic_tstat": None,
            "ic_pct_positive": None,
            "n_days": n,
        }
    mean = float(ic_clean.mean())
    std = float(ic_clean.std(ddof=1))
    icir = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else None
    tstat = mean / (std / math.sqrt(n)) if std > 0 else None
    pct_pos = float((ic_clean > 0).mean())
    return {
        "ic": mean,
        "icir": icir,
        "ic_tstat": tstat,
        "ic_pct_positive": pct_pos,
        "n_days": n,
    }


def compute_alpha_decay(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_DECAY_HORIZONS,
) -> dict[str, float | dict | None]:
    """Sample IC at each horizon and fit an exponential decay to recover the
    half-life of the predictive power.

    Model: ``IC(h) = IC₀ · exp(-h / τ)`` with half-life = ``τ · ln(2)``.

    Fit done on ``log|IC(h)|`` via least-squares — works as long as at least
    two horizons produce a non-zero IC of the same sign as IC(1).  If the fit
    fails (e.g. ICs are noise around zero, or sign-flipping), the returned
    half-life is None and ``r_squared`` is None too.

    Returns
    -------
    dict with keys:
      ``ic_by_horizon`` — {1: 0.04, 2: 0.03, ...} (None if a horizon's IC is NaN)
      ``half_life_days`` — float or None
      ``r_squared`` — quality of the exponential fit, or None
    """
    ic_by_horizon: dict[int, float | None] = {}
    for h in horizons:
        s = compute_ic_summary(signal, forward_returns, horizon=h)
        ic_by_horizon[int(h)] = s.get("ic")

    valid = [(h, ic) for h, ic in ic_by_horizon.items() if ic is not None and ic != 0.0]
    if len(valid) < 2:
        return {"ic_by_horizon": ic_by_horizon, "half_life_days": None, "r_squared": None}

    # Anchor sign on the shortest-horizon IC — if it's negative, flip the
    # whole curve so the exponential model fits a positive amplitude.
    anchor = valid[0][1]
    sign = 1.0 if anchor > 0 else -1.0
    h_arr = np.array([h for h, _ in valid], dtype=float)
    ic_arr = np.array([ic * sign for _, ic in valid], dtype=float)

    # Filter out non-positive values that would blow up log()
    mask = ic_arr > 0
    if mask.sum() < 2:
        return {"ic_by_horizon": ic_by_horizon, "half_life_days": None, "r_squared": None}
    h_fit = h_arr[mask]
    log_ic = np.log(ic_arr[mask])

    # Linear regression of log(IC) on h: slope = -1/τ, intercept = log(IC₀)
    slope, intercept = np.polyfit(h_fit, log_ic, 1)
    if slope >= 0:
        # IC isn't decaying — could be flat or growing.  Don't claim a
        # half-life in that case; researcher should look at the curve.
        return {
            "ic_by_horizon": ic_by_horizon,
            "half_life_days": None,
            "r_squared": None,
        }
    tau = -1.0 / slope
    half_life = tau * math.log(2.0)

    # R² of the fit (standard 1 - SS_res / SS_tot)
    predicted = slope * h_fit + intercept
    ss_res = float(((log_ic - predicted) ** 2).sum())
    ss_tot = float(((log_ic - log_ic.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "ic_by_horizon": ic_by_horizon,
        "half_life_days": float(half_life),
        "r_squared": float(r_squared) if r_squared is not None else None,
    }


def compute_rank_stability(signal: pd.DataFrame) -> float | None:
    """Average day-over-day Spearman correlation of the signal's ranks.

    Returns a scalar in roughly [-1, 1] — close to 1 means the daily ranking
    barely changes (think slow value/quality alphas), close to 0 means the
    ranking reshuffles randomly day-to-day (likely noise).  Negative would
    mean systematic flipping, which is unusual.

    None if fewer than 2 valid day pairs exist.
    """
    if signal is None or signal.empty or len(signal) < 2:
        return None
    ranks = _rank_along_rows(signal.values)
    # Build lagged pairs: ranks[t] vs ranks[t-1]
    cur = ranks[1:]
    prev = ranks[:-1]
    corrs = _row_corr(cur, prev)
    clean = corrs[~np.isnan(corrs)]
    if len(clean) == 0:
        return None
    return float(clean.mean())
