"""compute_size_exposure was vectorised out of a per-day Python loop; its
output must match the naive reference. compute_market_cap_distribution kept its
per-day bucketing (only the row access changed) so we assert it still produces a
well-formed, normalised distribution.
"""

import math

import numpy as np
import pandas as pd
import pytest
from analytics.exposure import compute_market_cap_distribution, compute_size_exposure


def _ref_size_exposure(weights, size_field, is_approximation=False):
    """Original per-day-loop implementation."""
    if weights is None or weights.empty or size_field is None or size_field.empty:
        return None
    cols = weights.columns.intersection(size_field.columns)
    if len(cols) < 3:
        return None
    size_aligned = size_field[cols].reindex(weights.index)
    size_log = size_aligned.where(size_aligned > 0, other=np.nan).apply(np.log)
    w = weights[cols]
    daily = []
    for date in w.index:
        x = w.loc[date].to_numpy(dtype=float)
        y = size_log.loc[date].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 3:
            continue
        xv, yv = x[mask], y[mask]
        sx, sy = xv.std(ddof=1), yv.std(ddof=1)
        if sx <= 0 or sy <= 0:
            continue
        c = float(((xv - xv.mean()) * (yv - yv.mean())).sum() / ((len(xv) - 1) * sx * sy))
        if not math.isnan(c):
            daily.append(c)
    if not daily:
        return None
    arr = np.asarray(daily)
    return {
        "size_corr": float(arr.mean()),
        "size_corr_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_days": int(len(arr)),
        "is_approximation": bool(is_approximation),
    }


def _fixtures(seed, t=120, n=25):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=t)
    cols = [f"T{i}" for i in range(n)]
    wa = rng.standard_normal((t, n))
    wa[rng.random((t, n)) < 0.05] = np.nan  # inject NaN on the array (writable)
    sa = np.abs(rng.standard_normal((t, n))) * 1e9 + 1e8
    sa[rng.random((t, n)) < 0.05] = np.nan
    w = pd.DataFrame(wa, index=idx, columns=cols)
    size = pd.DataFrame(sa, index=idx, columns=cols)
    return w, size


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_size_exposure_matches_reference(seed):
    w, size = _fixtures(seed)
    got = compute_size_exposure(w, size)
    ref = _ref_size_exposure(w, size)
    assert (got is None) == (ref is None)
    assert got is not None and ref is not None
    assert got["n_days"] == ref["n_days"]
    assert got["is_approximation"] == ref["is_approximation"]
    assert math.isclose(got["size_corr"], ref["size_corr"], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(got["size_corr_std"], ref["size_corr_std"], rel_tol=1e-9, abs_tol=1e-12)


def test_size_exposure_too_few_tickers_is_none():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=50)
    w = pd.DataFrame(rng.standard_normal((50, 2)), index=idx, columns=["A", "B"])
    size = pd.DataFrame(np.abs(rng.standard_normal((50, 2))) + 1, index=idx, columns=["A", "B"])
    assert compute_size_exposure(w, size) is None  # <3 tickers


def test_market_cap_distribution_is_well_formed():
    w, size = _fixtures(7)
    dist = compute_market_cap_distribution(w, size, n_buckets=10)
    assert dist is not None
    assert dist["n_buckets"] == 10
    assert len(dist["long_per_bucket"]) == 10
    assert len(dist["short_per_bucket"]) == 10
    # Each side is normalised to sum to 1 (averaged across days).
    assert math.isclose(sum(dist["long_per_bucket"]), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(dist["short_per_bucket"]), 1.0, abs_tol=1e-6)
