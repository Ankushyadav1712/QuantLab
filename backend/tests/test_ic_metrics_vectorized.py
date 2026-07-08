"""The vectorised `_rank_along_rows` / `_row_corr` must match the naive
reference (loop) implementations exactly — they were the backtest's hottest
Python loops, replaced with pandas/numpy vectorisation. This locks the behaviour.
"""

import numpy as np
import pytest
from analytics.ic_metrics import _rank_along_rows, _row_corr


def _ref_rank(arr):
    """Original triple-loop implementation (0-based average ranks, NaN kept)."""
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        valid = row[mask]
        order = valid.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(valid), dtype=float)
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


def _ref_corr(a, b):
    """Original per-row Pearson loop."""
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


def _fixture(seed, t=200, n=30):
    rng = np.random.default_rng(seed)
    a = np.round(rng.standard_normal((t, n)), 1)  # rounding forces ties
    a[rng.random((t, n)) < 0.1] = np.nan  # 10% missing
    return a


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_rank_matches_reference(seed):
    arr = _fixture(seed)
    np.testing.assert_allclose(_rank_along_rows(arr), _ref_rank(arr), equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_row_corr_matches_reference(seed):
    a, b = _fixture(seed), _fixture(seed + 100)
    ra, rb = _rank_along_rows(a), _rank_along_rows(b)
    np.testing.assert_allclose(
        _row_corr(ra, rb), _ref_corr(ra, rb), equal_nan=True, rtol=1e-9, atol=1e-12
    )


def test_edge_rows():
    # all-NaN, single-valid, zero-variance (constant), and strictly-increasing rows.
    arr = np.array(
        [
            [np.nan, np.nan, np.nan],
            [1.0, np.nan, np.nan],
            [5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0],
        ]
    )
    np.testing.assert_allclose(_rank_along_rows(arr), _ref_rank(arr), equal_nan=True)
    r = _rank_along_rows(arr)
    np.testing.assert_allclose(_row_corr(r, r), _ref_corr(r, r), equal_nan=True)


def test_empty_matrix():
    empty = np.empty((0, 5))
    assert _rank_along_rows(empty).shape == (0, 5)
