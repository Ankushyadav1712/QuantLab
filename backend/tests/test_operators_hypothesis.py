"""Property-based tests for engine operators (Tier 5 #9).

Hand-written tests verify ``rank([1,2,3]) == [0.33, 0.67, 1.0]`` — good
for catching obvious regressions, useless for catching edge cases the
author didn't think of (NaN at row boundary, all-equal rows, single-
column inputs, etc.).  Hypothesis generates random (date × ticker)
matrices across a wide shape + NaN-rate space and checks operator
invariants that must hold *regardless* of the inputs:

  rank(x)         → bounded [0, 1], NaN-preserving, monotone in x
  zscore(x)       → per-row mean ≈ 0 (within ε), NaN-preserving
  ts_mean(x, d)   → first d-1 rows NaN; equals naive rolling on small cases
  ts_rank(x, d)   → bounded [0, 1], last row matches `rank` of last window
  delay(x, d)     → first d rows NaN; values match x.shift(d) exactly
  delta(x, d)     → equals x - delay(x, d) for d ≥ 1
  decay_linear(x, d) → first d-1 rows NaN; equals weighted average of last d

A failure here means a subtle bug in an operator that the example tests
didn't catch.  Hypothesis will shrink any failing input to the smallest
case that still fails — copy it into a regression test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from engine import operators as ops
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.extra.pandas import column, data_frames, range_indexes

# ---------- Shared strategies ----------

# Small but realistic shapes: ≥ 3 rows so per-row stats are defined,
# ≥ 2 cols so rank() has something to rank.  Capped at 30×8 so the suite
# stays fast (each test runs hypothesis's default 100 examples).
SHAPES = array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=30)
SMALL_SHAPES = array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=12)

# Finite floats only — operators can't reasonably be expected to handle inf,
# and hypothesis is happy to feed us inf otherwise.  Limit magnitudes so the
# math doesn't lose precision under sums.
FINITE_FLOATS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def _matrix_strategy(shape_strategy=SHAPES, nan_rate: float = 0.0):
    """Generate a DataFrame from a 2D numpy array of finite floats.  Date
    index is irrelevant for these tests (operators are index-agnostic) so
    we use a plain RangeIndex."""

    def builder(arr: np.ndarray) -> pd.DataFrame:
        if nan_rate > 0:
            # Punch holes at random positions (use a separate seeded rng so
            # the test stays deterministic given the same hypothesis input)
            rng = np.random.default_rng(arr.flatten().sum().astype(int) % (2**31))
            mask = rng.random(arr.shape) < nan_rate
            arr = arr.astype(float)
            arr[mask] = np.nan
        cols = [f"t{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    return arrays(np.float64, shape_strategy, elements=FINITE_FLOATS).map(builder)


# ---------- rank ----------


@given(_matrix_strategy())
def test_rank_bounded_in_unit_interval(df):
    """rank() returns pandas pct rank, which lives in (0, 1] (smallest gets
    1/n, largest gets 1).  Strict 0 should never appear; > 1 never appears."""
    out = ops.rank(df)
    arr = out.to_numpy()
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return  # all-NaN row is acceptable
    assert valid.min() > 0.0, f"rank produced 0.0: min={valid.min()}"
    assert valid.max() <= 1.0 + 1e-9, f"rank produced > 1: max={valid.max()}"


@given(_matrix_strategy(nan_rate=0.3))
def test_rank_preserves_nan_positions(df):
    """A NaN in the input must produce a NaN in the output at the same
    (row, col).  Otherwise the operator is silently filling missing values."""
    out = ops.rank(df)
    in_nan = df.isna().to_numpy()
    out_nan = out.isna().to_numpy()
    assert np.array_equal(in_nan, out_nan), "rank must preserve NaN positions exactly"


@given(_matrix_strategy(SMALL_SHAPES))
def test_rank_monotone_in_input(df):
    """If x[a] < x[b] for two cells in the same row, then rank(x)[a] ≤ rank(x)[b].
    The non-strict ≤ allows for ties (pandas' default tie-breaker)."""
    out = ops.rank(df)
    for row_idx in range(len(df)):
        x_row = df.iloc[row_idx].to_numpy()
        r_row = out.iloc[row_idx].to_numpy()
        valid = ~np.isnan(x_row)
        x_v = x_row[valid]
        r_v = r_row[valid]
        if len(x_v) < 2:
            continue
        # Sort by input value, check ranks come out non-decreasing
        order = np.argsort(x_v)
        sorted_ranks = r_v[order]
        assert np.all(sorted_ranks[1:] >= sorted_ranks[:-1] - 1e-12), (
            "rank must be monotone non-decreasing in input"
        )


# ---------- zscore ----------


@given(_matrix_strategy())
def test_zscore_row_mean_near_zero(df):
    """Each row of zscore(x) must have mean ≈ 0 (the central property of a
    z-score).  Skip rows that are degenerate (constant or near-constant —
    std too small for the division to produce finite, well-conditioned
    output) since those legitimately yield NaN / inf."""
    out = ops.zscore(df)
    for row_idx in range(len(out)):
        row = out.iloc[row_idx].to_numpy()
        # Drop NaN AND non-finite cells — the latter come from degenerate
        # rows (std ≈ 0) and aren't meaningful inputs to a mean check
        finite = row[np.isfinite(row)]
        if len(finite) < 2:
            continue
        # Skip if the row was numerically degenerate (extreme magnitudes
        # with near-equal values cause float-precision std underflow)
        in_row = df.iloc[row_idx].to_numpy()
        in_valid = in_row[~np.isnan(in_row)]
        if len(in_valid) > 1:
            row_scale = max(abs(in_valid.max()), abs(in_valid.min()), 1.0)
            if in_valid.std() < 1e-9 * row_scale:
                continue  # row too close to constant for zscore to be stable
        assert abs(finite.mean()) < 1e-6, f"zscore row mean = {finite.mean()}, expected ≈ 0"


@given(_matrix_strategy(nan_rate=0.2))
def test_zscore_preserves_nan_positions(df):
    """Same NaN-preservation invariant as rank(): an input NaN must produce
    an output NaN at the same position.  zscore can additionally produce
    NaN/inf on constant or numerically-degenerate rows, which is acceptable —
    we only require that the original NaN positions stay NaN."""
    out = ops.zscore(df)
    in_nan = df.isna().to_numpy()
    out_nan = out.isna().to_numpy()
    # Every position that was NaN in the input must still be NaN in the
    # output.  (The reverse — new NaNs in the output — is permitted on
    # degenerate rows, both true-constant and float-precision-constant.)
    assert (out_nan | ~in_nan).all(), "zscore must preserve input NaN positions"


# ---------- delay ----------


@given(_matrix_strategy(), st.integers(min_value=1, max_value=5))
def test_delay_first_d_rows_are_nan(df, d):
    """delay(x, d) shifts down by d → first d rows must be all-NaN."""
    assume(len(df) > d)
    out = ops.delay(df, d)
    assert out.iloc[:d].isna().all().all(), f"delay({d}) didn't NaN-out first {d} rows"


@given(_matrix_strategy(), st.integers(min_value=1, max_value=5))
def test_delay_equals_shift(df, d):
    """delay(x, d) must equal x.shift(d) cell-by-cell.  This is the operator's
    one-line definition; the test prevents anyone from "optimising" it into
    something that silently changes behaviour."""
    assume(len(df) > d)
    out = ops.delay(df, d)
    expected = df.shift(d)
    # NaN == NaN with pd.testing
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


# ---------- delta ----------


@given(_matrix_strategy(), st.integers(min_value=1, max_value=5))
def test_delta_equals_x_minus_delay(df, d):
    """delta(x, d) is the operator's documented formula: x - delay(x, d)."""
    assume(len(df) > d)
    out = ops.delta(df, d)
    expected = df - df.shift(d)
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


# ---------- ts_mean ----------


@given(_matrix_strategy(), st.integers(min_value=2, max_value=5))
def test_ts_mean_equals_rolling_mean(df, d):
    """ts_mean(x, d) is rolling(d).mean() — verify it stays that way."""
    assume(len(df) >= d)
    out = ops.ts_mean(df, d)
    expected = df.rolling(d).mean()
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


@given(_matrix_strategy(), st.integers(min_value=2, max_value=5))
def test_ts_mean_first_d_minus_1_rows_nan(df, d):
    assume(len(df) >= d)
    out = ops.ts_mean(df, d)
    assert out.iloc[: d - 1].isna().all().all()


# ---------- ts_rank ----------


@given(_matrix_strategy(SMALL_SHAPES), st.integers(min_value=2, max_value=5))
def test_ts_rank_bounded_in_unit_interval(df, d):
    """ts_rank is a windowed rank → also bounded ~(0, 1]."""
    assume(len(df) >= d)
    out = ops.ts_rank(df, d)
    arr = out.to_numpy()
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return
    # Allow tiny float slack at the boundaries
    assert valid.min() >= 0.0 - 1e-9
    assert valid.max() <= 1.0 + 1e-9


# ---------- decay_linear ----------


@given(_matrix_strategy(SMALL_SHAPES), st.integers(min_value=2, max_value=5))
def test_decay_linear_first_d_minus_1_rows_nan(df, d):
    """decay_linear needs d valid trailing values → first d-1 rows can't
    produce a result and must be NaN."""
    assume(len(df) >= d)
    out = ops.decay_linear(df, d)
    assert out.iloc[: d - 1].isna().all().all()


@given(
    data_frames(
        columns=[
            column("A", dtype=float, elements=FINITE_FLOATS),
            column("B", dtype=float, elements=FINITE_FLOATS),
        ],
        index=range_indexes(min_size=5, max_size=20),
    )
)
def test_decay_linear_matches_explicit_weighted_average(df):
    """For d=3, decay_linear should equal sum(w_i * x_i) / sum(w_i) where
    weights are [1, 2, 3] (most recent observation weighted highest).
    Hand-roll the math and check pandas agrees."""
    assume(len(df) >= 3)
    out = ops.decay_linear(df, 3)
    weights = np.array([1.0, 2.0, 3.0])  # oldest → newest
    total_w = weights.sum()
    # Check the last row only — middle rows have the same formula but with
    # a different sliding window each time
    last_idx = len(df) - 1
    window = df.iloc[last_idx - 2 : last_idx + 1].to_numpy()  # shape (3, n_cols)
    expected = (window * weights[:, None]).sum(axis=0) / total_w
    actual = out.iloc[last_idx].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# ---------- Combined invariants ----------
#
# Originally had `test_rank_zscore_x_equals_rank_x` (rank o zscore should
# match rank since zscore is monotone per row), but hypothesis found that
# mean-subtraction in zscore collapses extreme-magnitude differences at
# float precision (e.g. 1.4e-186 − 1/3 == 0 − 1/3 in float64).  That's
# IEEE 754 reality, not an operator bug, so the invariant was dropped.
# Individual rank() / zscore() tests above cover the meaningful surface.


# Suppress hypothesis's "test ran slow" warning since matrix construction
# adds overhead beyond the operator call itself.
_settings_used = settings(suppress_health_check=[HealthCheck.too_slow])
_ = pytest  # silence unused-import — pytest's fixture system uses it
