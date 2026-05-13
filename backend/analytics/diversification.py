"""Portfolio diversification curve — ensemble Sharpe as f(portfolio size).

For each candidate size ``n`` in a list of sizes, draws ``n_samples`` random
subsets of size ``n`` from a pool of alphas, equal-weights their daily
returns, computes the resulting ensemble Sharpe, and aggregates the
distribution.  The output curve typically rises steeply for small n and
then plateaus — the plateau tells you when adding more alphas stops
contributing diversification benefit.

This is the empirical version of √n / σ shrinkage: with k independent
alphas of equal individual Sharpe ``s``, ensemble Sharpe = ``s · √k`` (the
classic diversification benefit).  With correlated alphas the curve
plateaus sooner, and how soon is the diagnostic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Default Fibonacci-ish sizes give good resolution at small n (where the
# curve moves fast) and sparse sampling at large n (where it plateaus).
DEFAULT_SIZES: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21)


def _ensemble_sharpe(returns_matrix: np.ndarray, indices: np.ndarray) -> float | None:
    """Equal-weight sum of selected alpha return columns → annualised Sharpe.

    ``returns_matrix`` shape: ``(T, n_alphas)``.  ``indices`` is a 1-D
    integer array selecting which columns to include.  Returns None when
    the resulting ensemble PnL has zero variance or too few observations.
    """
    if indices.size == 0:
        return None
    selected = returns_matrix[:, indices]
    # Equal-weight: simple mean across selected alphas per row.  Using mean
    # rather than sum keeps the per-row variance unit-comparable across
    # different ensemble sizes.
    ensemble = selected.mean(axis=1)
    # Drop NaN rows (alphas may have different valid date ranges)
    ensemble = ensemble[~np.isnan(ensemble)]
    if ensemble.size < 2:
        return None
    mean = float(ensemble.mean())
    std = float(ensemble.std(ddof=1))
    if std <= 0:
        return None
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def diversification_curve(
    alphas_pnl: dict[int, pd.Series],
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    n_samples: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample ensemble Sharpes across portfolio sizes.

    Parameters
    ----------
    alphas_pnl : dict[int, pd.Series]
        Mapping of alpha_id → daily returns series.  All series will be
        aligned on the union of their date indices (missing dates → NaN,
        dropped per-sample inside the ensemble computation).
    sizes : tuple[int, ...]
        Portfolio sizes to evaluate.  Sizes greater than ``len(alphas_pnl)``
        are silently clamped to the pool size.
    n_samples : int
        Number of random subsets to draw at each size.  More → tighter IQR
        bands but slower.  20 is enough for a usable median + IQR up to ~30
        alphas in the pool.
    seed : int
        Seed for the subset-sampling RNG so the curve is deterministic for
        a given pool.

    Returns
    -------
    list[dict] with one entry per size that produced at least one valid
    Sharpe sample::

        [
          {"n": 1,  "median_sharpe": 0.72, "q1": 0.21, "q3": 1.10, "n_samples": 8},
          {"n": 2,  "median_sharpe": 1.05, "q1": 0.55, "q3": 1.42, "n_samples": 20},
          ...
        ]

    Empty list if fewer than 2 alphas in the pool (curve isn't meaningful
    with a single alpha).
    """
    if alphas_pnl is None or len(alphas_pnl) < 2:
        return []

    # Align to the union of date indices.  pd.concat axis=1 gives us a
    # (T × n_alphas) frame with NaN-padded missing dates; that's what we want.
    aligned = pd.concat(alphas_pnl, axis=1, sort=True)
    # Columns are now the alpha_ids (dict keys); convert to numpy for speed.
    returns_matrix = aligned.to_numpy(dtype=float)
    n_total = returns_matrix.shape[1]

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for n in sizes:
        # Clamp to pool size — asking for size 21 when only 5 alphas exist
        # should still return the n=5 row, not skip.
        n_eff = min(int(n), n_total)
        if n_eff < 1:
            continue

        # If n == n_total, there's only one possible subset (all of them);
        # sampling is wasted work.  Otherwise draw up to n_samples random
        # combinations without replacement.
        if n_eff == n_total:
            subset_lists = [np.arange(n_total)]
        else:
            # Use sample-without-replacement at the index level.  We tolerate
            # some duplicate subsets at small n — n_samples is a budget,
            # not a uniqueness guarantee.
            subset_lists = [
                rng.choice(n_total, size=n_eff, replace=False) for _ in range(n_samples)
            ]

        sharpes: list[float] = []
        for indices in subset_lists:
            s = _ensemble_sharpe(returns_matrix, indices)
            if s is not None:
                sharpes.append(s)

        if not sharpes:
            continue

        arr = np.asarray(sharpes)
        results.append(
            {
                "n": int(n_eff),
                "median_sharpe": float(np.median(arr)),
                "q1": float(np.quantile(arr, 0.25)),
                "q3": float(np.quantile(arr, 0.75)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_samples": int(len(arr)),
            }
        )

    # If multiple input sizes clamped to the same n_eff (e.g. sizes=(13, 21)
    # but pool has 5 alphas), we'd duplicate the n=5 row.  Deduplicate.
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        if r["n"] in seen:
            continue
        seen.add(r["n"])
        deduped.append(r)
    return deduped


def extract_daily_returns_from_saved(
    saved_alphas: list[dict],
    *,
    result_field: str = "result_json",
) -> dict[int, pd.Series]:
    """Pull each saved alpha's daily PnL series out of the SQLite blob.

    Saved alphas store the full ``/api/simulate`` response in a JSON column.
    This walks ``is_timeseries.dates`` + ``is_timeseries.daily_returns``
    (built by ``main._compute_perf_pack``) and returns a ``{alpha_id: Series}``
    mapping suitable for ``diversification_curve``.

    Alphas whose result_json is missing / malformed / lacks a usable
    timeseries are silently skipped — they just don't contribute to the
    curve.  Logging the skip is the caller's responsibility.
    """
    import json

    out: dict[int, pd.Series] = {}
    for record in saved_alphas:
        alpha_id = record.get("id")
        if alpha_id is None:
            continue
        raw = record.get(result_field) or record.get("result")
        if raw is None:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if not isinstance(raw, dict):
            continue
        ts = raw.get("is_timeseries") or {}
        dates = ts.get("dates")
        rets = ts.get("daily_returns")
        if not dates or not rets or len(dates) != len(rets):
            continue
        try:
            series = pd.Series(
                [float(r) if r is not None else np.nan for r in rets],
                index=pd.to_datetime(dates),
                name=str(alpha_id),
            )
        except (ValueError, TypeError):
            continue
        out[int(alpha_id)] = series
    return out
