"""Phase B derived fields — invariants on a synthetic OHLCV fixture.

Builds a `DataFetcher`, hands it a hand-crafted set of per-ticker OHLCV
frames, runs `_build_matrices()` to populate every derived field, then
asserts shape + a meaningful invariant for each new field.

We don't hit yfinance — `_frames` is set directly so the same fetcher's
matrix-building code path runs without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data.fetcher import (
    BASE_FIELDS,
    DERIVED_FIELDS,
    DataFetcher,
    _streak_count,
)


@pytest.fixture
def synth_fetcher(tmp_path):
    """A `DataFetcher` populated from a 300-day, 5-ticker synthetic universe.

    300 days is enough for the longest rolling window in Phase B (252-day
    momentum) to produce non-NaN values on the last few rows.
    """
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(42)

    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        # Geometric random walk for close, then derive plausible OHLV
        log_rets = rng.normal(0.0, 0.012, len(dates))
        close = 100.0 * np.exp(np.cumsum(log_rets))
        # Open is yesterday's close + a small overnight gap
        opens = np.concatenate(
            [[close[0]], close[:-1] * (1 + 0.002 * rng.standard_normal(len(dates) - 1))]
        )
        # High/low bracket the open-close range
        hi = np.maximum(opens, close) * (1 + np.abs(0.005 * rng.standard_normal(len(dates))))
        lo = np.minimum(opens, close) * (1 - np.abs(0.005 * rng.standard_normal(len(dates))))
        vol = rng.integers(500_000, 5_000_000, len(dates)).astype(float)

        df = pd.DataFrame(
            {"open": opens, "high": hi, "low": lo, "close": close, "volume": vol},
            index=dates,
        )
        df["returns"] = df["close"].pct_change()
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
        frames[t] = df

    fetcher = DataFetcher(cache_dir=tmp_path)
    fetcher._frames = frames
    fetcher._build_matrices(compute_derived=True)
    return fetcher


# ---------- Registration ----------


def test_all_phase_b_fields_present(synth_fetcher):
    """Every new field must exist in the matrix dict after build."""
    new_fields = {
        # momentum
        "momentum_3",
        "momentum_10",
        "momentum_60",
        "momentum_120",
        "momentum_252",
        "reversal_5",
        "reversal_20",
        "momentum_z_60",
        # volatility
        "realized_vol_5",
        "realized_vol_60",
        "realized_vol_120",
        "vol_of_vol_20",
        "parkinson_vol",
        "garman_klass_vol",
        # microstructure
        "roll_spread",
        "kyle_lambda",
        "vpin_proxy",
        "up_volume_ratio",
        "down_volume_ratio",
        "turnover_ratio",
        "dollar_amihud",
        "corwin_schultz",
        # range
        "atr_5",
        "atr_60",
        "range_z_20",
        "body_to_range",
        "consecutive_up",
        "consecutive_down",
    }
    missing = new_fields - set(synth_fetcher._matrix.keys())
    assert not missing, f"Phase B fields missing from matrix: {missing}"


def test_derived_fields_tuple_includes_new_fields():
    """The cache-loading code keys off DERIVED_FIELDS — every new field must
    be listed there or it won't survive a cache round-trip."""
    expected = {"momentum_60", "parkinson_vol", "kyle_lambda", "consecutive_up", "atr_5"}
    assert expected.issubset(set(DERIVED_FIELDS))


# ---------- Momentum invariants ----------


def test_momentum_definitions_match_close_ratios(synth_fetcher):
    c = synth_fetcher._matrix["close"]
    np.testing.assert_allclose(
        synth_fetcher._matrix["momentum_60"].values,
        (c / c.shift(60) - 1.0).values,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        synth_fetcher._matrix["momentum_252"].values,
        (c / c.shift(252) - 1.0).values,
        equal_nan=True,
    )


def test_reversal_is_negative_of_momentum(synth_fetcher):
    np.testing.assert_allclose(
        synth_fetcher._matrix["reversal_5"].values,
        -synth_fetcher._matrix["momentum_5"].values,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        synth_fetcher._matrix["reversal_20"].values,
        -synth_fetcher._matrix["momentum_20"].values,
        equal_nan=True,
    )


def test_momentum_z_60_is_momentum_over_vol(synth_fetcher):
    """momentum_z_60 = momentum_60 / 60-day return-vol."""
    r = synth_fetcher._matrix["returns"]
    expected = synth_fetcher._matrix["momentum_60"] / r.rolling(60).std().replace(0, np.nan)
    np.testing.assert_allclose(
        synth_fetcher._matrix["momentum_z_60"].values,
        expected.values,
        equal_nan=True,
    )


# ---------- Volatility invariants ----------


def test_realized_vol_windows_increase_in_smoothness(synth_fetcher):
    """Longer windows have lower stdev of the vol series itself (smoother).
    A flaky day jumps realized_vol_5 hard but barely moves realized_vol_120."""
    vol_5 = synth_fetcher._matrix["realized_vol_5"].iloc[120:].std().mean()
    vol_120 = synth_fetcher._matrix["realized_vol_120"].iloc[120:].std().mean()
    assert vol_5 > vol_120


def test_parkinson_vol_positive_and_finite(synth_fetcher):
    pv = synth_fetcher._matrix["parkinson_vol"].iloc[20:]
    finite = pv.values[~np.isnan(pv.values)]
    assert finite.size > 0
    assert (finite >= 0).all()
    assert np.isfinite(finite).all()


def test_garman_klass_vol_positive_and_finite(synth_fetcher):
    gk = synth_fetcher._matrix["garman_klass_vol"].iloc[20:]
    finite = gk.values[~np.isnan(gk.values)]
    assert finite.size > 0
    assert (finite >= 0).all()
    assert np.isfinite(finite).all()


def test_vol_of_vol_uses_realized_vol_20(synth_fetcher):
    expected = synth_fetcher._matrix["realized_vol"].rolling(20).std()
    np.testing.assert_allclose(
        synth_fetcher._matrix["vol_of_vol_20"].values,
        expected.values,
        equal_nan=True,
    )


# ---------- Microstructure invariants ----------


def test_kyle_lambda_finite_after_warmup(synth_fetcher):
    kl = synth_fetcher._matrix["kyle_lambda"].iloc[40:]
    finite = kl.values[~np.isnan(kl.values)]
    assert finite.size > 0
    assert (finite >= 0).all()


def test_up_down_volume_ratios_sum_to_at_most_one(synth_fetcher):
    """Up + down volume can't exceed total volume — flat days (return=0)
    contribute to neither, so the sum is ≤ 1, not = 1."""
    up = synth_fetcher._matrix["up_volume_ratio"].iloc[40:]
    dn = synth_fetcher._matrix["down_volume_ratio"].iloc[40:]
    total = (up + dn).values
    finite = total[~np.isnan(total)]
    assert (finite <= 1.0 + 1e-9).all()
    # Most cells should sum to nearly 1 (flat days are rare in a random walk)
    assert finite.mean() > 0.95


def test_vpin_proxy_in_unit_interval(synth_fetcher):
    vp = synth_fetcher._matrix["vpin_proxy"].iloc[40:]
    finite = vp.values[~np.isnan(vp.values)]
    assert (finite >= 0).all()
    assert (finite <= 1.0 + 1e-9).all()


def test_turnover_ratio_is_volume_over_adv60(synth_fetcher):
    v = synth_fetcher._matrix["volume"]
    expected = v / v.rolling(60).mean().replace(0, np.nan)
    np.testing.assert_allclose(
        synth_fetcher._matrix["turnover_ratio"].values,
        expected.values,
        equal_nan=True,
    )


def test_corwin_schultz_when_defined_is_positive(synth_fetcher):
    """The CS estimator returns NaN where the formula breaks down (negative
    α → negative spread); kept cells must be positive by construction."""
    cs = synth_fetcher._matrix["corwin_schultz"].iloc[40:]
    finite = cs.values[~np.isnan(cs.values)]
    if finite.size > 0:
        assert (finite > 0).all()


def test_roll_spread_nonnegative_when_defined(synth_fetcher):
    """Roll's spread is 2·sqrt(-cov); when negative cov is missing the cell
    is NaN, otherwise it's >= 0."""
    rs = synth_fetcher._matrix["roll_spread"].iloc[40:]
    finite = rs.values[~np.isnan(rs.values)]
    if finite.size > 0:
        assert (finite >= 0).all()


def test_dollar_amihud_smooths_raw_amihud(synth_fetcher):
    """dollar_amihud is the rolling mean of the raw |returns|/dollar_volume,
    so its 20-day-and-later std should be lower than the unsmoothed input."""
    r = synth_fetcher._matrix["returns"]
    dv = synth_fetcher._matrix["dollar_volume"]
    raw = (r.abs() / dv.replace(0, np.nan)).iloc[40:]
    smooth = synth_fetcher._matrix["dollar_amihud"].iloc[40:]
    assert smooth.std().mean() < raw.std().mean()


# ---------- Range / candle invariants ----------


def test_atr_windows_match_true_range_rolling_mean(synth_fetcher):
    tr = synth_fetcher._matrix["true_range"]
    np.testing.assert_allclose(
        synth_fetcher._matrix["atr_5"].values,
        tr.rolling(5).mean().values,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        synth_fetcher._matrix["atr_60"].values,
        tr.rolling(60).mean().values,
        equal_nan=True,
    )


def test_range_z_20_centered_near_zero(synth_fetcher):
    """A z-score of a stationary-ish series should hover near 0 with std ~1
    (looser tolerance — synthetic data, finite sample)."""
    rz = synth_fetcher._matrix["range_z_20"].iloc[40:]
    finite_mean = rz.values[~np.isnan(rz.values)].mean()
    assert abs(finite_mean) < 0.5  # generous bound; just ensures not biased


def test_body_to_range_in_unit_interval(synth_fetcher):
    btr = synth_fetcher._matrix["body_to_range"]
    finite = btr.values[~np.isnan(btr.values)]
    assert (finite >= 0).all()
    assert (finite <= 1.0 + 1e-9).all()


def test_consecutive_up_resets_on_down_day(synth_fetcher):
    """For each ticker, the streak counter must drop to 0 immediately after
    any down day (return < 0).  Equivalently: where return < 0, the streak
    on that day is 0."""
    r = synth_fetcher._matrix["returns"]
    streak = synth_fetcher._matrix["consecutive_up"]
    down_mask = (r < 0).values
    assert (streak.values[down_mask] == 0.0).all()


def test_consecutive_up_increments_on_up_day(synth_fetcher):
    """If today is up, the streak should be exactly (yesterday's streak + 1)."""
    r = synth_fetcher._matrix["returns"]
    streak = synth_fetcher._matrix["consecutive_up"]
    # Pick row 50 (post-warmup) and check at least one ticker has a 2+ streak
    up_today = r.iloc[50] > 0
    up_yesterday = r.iloc[49] > 0
    both_up = up_today & up_yesterday
    if both_up.any():
        for ticker in both_up[both_up].index:
            assert streak.iloc[50][ticker] == streak.iloc[49][ticker] + 1


# ---------- Helper unit test ----------


def test_streak_count_helper_directly():
    """The vectorized streak-count trick is the trickiest piece of new code;
    test it directly on a known-answer mask."""
    mask = pd.DataFrame(
        {
            "X": [1, 1, 0, 1, 1, 1, 0, 1],
            "Y": [0, 1, 1, 1, 0, 0, 1, 1],
        }
    )
    expected = pd.DataFrame(
        {
            "X": [1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
            "Y": [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0],
        }
    )
    out = _streak_count(mask)
    pd.testing.assert_frame_equal(out, expected)
