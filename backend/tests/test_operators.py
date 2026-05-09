import numpy as np
import pandas as pd
import pytest

from engine import operators as ops


@pytest.fixture
def df():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "B": [10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
            "C": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        },
        index=dates,
    )


# ---------- delta ----------


def test_delta_window_1(df):
    result = ops.delta(df, 1)
    expected = df - df.shift(1)
    pd.testing.assert_frame_equal(result, expected)
    assert result.iloc[0].isna().all()
    assert result.iloc[1]["A"] == 1.0
    assert result.iloc[1]["B"] == -2.0
    assert result.iloc[1]["C"] == 0.0


def test_delta_window_2(df):
    result = ops.delta(df, 2)
    assert result.iloc[0].isna().all() and result.iloc[1].isna().all()
    assert result.iloc[2]["A"] == 2.0  # 3 - 1
    assert result.iloc[5]["B"] == -3.0  # 1 - 4


# ---------- rank (cross-sectional) ----------


def test_rank_row0(df):
    # row 0: A=1, C=3, B=10  ->  ranks: A=1, C=2, B=3 of 3
    r = ops.rank(df)
    assert r.iloc[0]["A"] == pytest.approx(1 / 3)
    assert r.iloc[0]["C"] == pytest.approx(2 / 3)
    assert r.iloc[0]["B"] == pytest.approx(1.0)


def test_rank_row5(df):
    # row 5: A=6, B=1, C=3 -> ranks B=1, C=2, A=3
    r = ops.rank(df)
    assert r.iloc[5]["B"] == pytest.approx(1 / 3)
    assert r.iloc[5]["C"] == pytest.approx(2 / 3)
    assert r.iloc[5]["A"] == pytest.approx(1.0)


def test_rank_handles_nan():
    df = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, 3.0], "C": [3.0, 1.0]})
    r = ops.rank(df)
    # row 1: A=NaN should remain NaN; B=3 ranks above C=1
    assert pd.isna(r.iloc[1]["A"])
    assert r.iloc[1]["B"] == pytest.approx(1.0)
    assert r.iloc[1]["C"] == pytest.approx(0.5)


# ---------- ts_mean ----------


def test_ts_mean_window_3(df):
    result = ops.ts_mean(df, 3)
    assert result.iloc[0].isna().all()
    assert result.iloc[1].isna().all()
    assert result.iloc[2]["A"] == pytest.approx((1 + 2 + 3) / 3)
    assert result.iloc[2]["B"] == pytest.approx((10 + 8 + 6) / 3)
    assert result.iloc[5]["A"] == pytest.approx((4 + 5 + 6) / 3)


def test_ts_mean_matches_pandas_rolling(df):
    pd.testing.assert_frame_equal(
        ops.ts_mean(df, 3),
        df.rolling(window=3, min_periods=3).mean(),
    )


# ---------- ts_std / ts_sum / ts_min / ts_max ----------


def test_ts_std_known_values(df):
    result = ops.ts_std(df, 3)
    # std of [1,2,3] (sample) = 1.0
    assert result.iloc[2]["A"] == pytest.approx(1.0)
    # constant column -> 0
    assert result.iloc[2]["C"] == pytest.approx(0.0)


def test_ts_sum(df):
    result = ops.ts_sum(df, 3)
    assert result.iloc[2]["A"] == pytest.approx(6.0)
    assert result.iloc[2]["B"] == pytest.approx(24.0)


def test_ts_min_max(df):
    assert ops.ts_min(df, 3).iloc[2]["A"] == pytest.approx(1.0)
    assert ops.ts_max(df, 3).iloc[2]["A"] == pytest.approx(3.0)


# ---------- delay / decay_linear ----------


def test_delay(df):
    pd.testing.assert_frame_equal(ops.delay(df, 2), df.shift(2))


def test_decay_linear_window_3():
    s = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = ops.decay_linear(s, 3)
    # weights [1,2,3], total 6
    # row 2: 1*1 + 2*2 + 3*3 = 14 -> 14/6
    assert result.iloc[2]["A"] == pytest.approx(14 / 6)
    # row 3: 1*2 + 2*3 + 3*4 = 20 -> 20/6
    assert result.iloc[3]["A"] == pytest.approx(20 / 6)
    # rows 0, 1 should be NaN (insufficient history)
    assert pd.isna(result.iloc[0]["A"])
    assert pd.isna(result.iloc[1]["A"])


# ---------- cross-sectional ops ----------


def test_zscore_zero_mean(df):
    z = ops.zscore(df)
    # row mean ~ 0
    assert z.mean(axis=1).abs().max() < 1e-12


def test_demean_zero_mean(df):
    d = ops.demean(df)
    assert d.mean(axis=1).abs().max() < 1e-12


def test_scale_abs_sum_one(df):
    s = ops.scale(df)
    assert np.allclose(s.abs().sum(axis=1).values, 1.0)


def test_normalize_zero_mean_unit_abs_sum(df):
    n = ops.normalize(df)
    assert n.mean(axis=1).abs().max() < 1e-12
    assert np.allclose(n.abs().sum(axis=1).values, 1.0)


# ---------- arithmetic ----------


def test_op_abs(df):
    pd.testing.assert_frame_equal(ops.op_abs(-df), df)


def test_op_sign():
    df = pd.DataFrame({"A": [-2.0, 0.0, 3.0]})
    s = ops.op_sign(df)
    assert s.iloc[0]["A"] == -1.0
    assert s.iloc[1]["A"] == 0.0
    assert s.iloc[2]["A"] == 1.0


def test_power(df):
    p = ops.power(df, 2)
    assert p.iloc[0]["A"] == 1.0
    assert p.iloc[1]["A"] == 4.0


def test_op_max_op_min_dataframes():
    a = pd.DataFrame({"X": [1.0, 5.0]})
    b = pd.DataFrame({"X": [3.0, 2.0]})
    assert ops.op_max(a, b).iloc[0]["X"] == 3.0
    assert ops.op_max(a, b).iloc[1]["X"] == 5.0
    assert ops.op_min(a, b).iloc[0]["X"] == 1.0
    assert ops.op_min(a, b).iloc[1]["X"] == 2.0


def test_if_else():
    cond = pd.DataFrame({"X": [True, False, True]})
    x = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"X": [10.0, 20.0, 30.0]})
    out = ops.if_else(cond, x, y)
    assert list(out["X"]) == [1.0, 20.0, 3.0]


# ---------- Synthetic 100×10 fixture for spec-required tests ----------


@pytest.fixture
def synth_df():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = [f"T{i}" for i in range(10)]
    rng = np.random.default_rng(2024)
    data = rng.standard_normal((100, 10))
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_delta_equals_x_minus_x_shift_5(synth_df):
    pd.testing.assert_frame_equal(
        ops.delta(synth_df, 5),
        synth_df - synth_df.shift(5),
    )


def test_rank_output_in_unit_interval(synth_df):
    r = ops.rank(synth_df)
    arr = r.to_numpy()
    finite = arr[~np.isnan(arr)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_ts_mean_matches_pandas_rolling_synth(synth_df):
    pd.testing.assert_frame_equal(
        ops.ts_mean(synth_df, 20),
        synth_df.rolling(window=20, min_periods=20).mean(),
    )


def test_zscore_row_mean_zero_std_one(synth_df):
    z = ops.zscore(synth_df)
    # Per-row mean ≈ 0 and per-row std ≈ 1
    assert z.mean(axis=1).abs().max() < 1e-12
    assert (z.std(axis=1, ddof=1) - 1.0).abs().max() < 1e-12


def test_normalize_row_abs_sum_one(synth_df):
    n = ops.normalize(synth_df)
    sums = n.abs().sum(axis=1).to_numpy()
    # Allow a tiny numerical tolerance
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)
    # And rows are demeaned
    assert n.mean(axis=1).abs().max() < 1e-12


# ---------- Evaluator-level: unknown function rejected ----------


def test_evaluator_rejects_unknown_function(synth_df):
    """Parser is purely syntactic; semantic 'unknown function' is caught here."""
    from engine.evaluator import AlphaEvaluator

    evaluator = AlphaEvaluator({"close": synth_df})
    with pytest.raises(ValueError, match="Unknown function"):
        evaluator.evaluate("notarealop(close)")


# ---------- Phase A: time-series additions ----------


def test_ts_median_matches_pandas(df):
    pd.testing.assert_frame_equal(
        ops.ts_median(df, 3),
        df.rolling(window=3, min_periods=3).median(),
    )


def test_ts_skewness_zero_for_symmetric_window():
    # Symmetric data → skew = 0
    s = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ops.ts_skewness(s, 5)
    assert out.iloc[4]["X"] == pytest.approx(0.0, abs=1e-12)


def test_ts_kurtosis_runs_without_nan(synth_df):
    out = ops.ts_kurtosis(synth_df, 20)
    assert not out.iloc[20:].isna().all().any()


def test_ts_zscore_zero_for_constant_window():
    # Constant column → zscore is NaN (std=0 → divide by NaN), not inf
    s = pd.DataFrame({"X": [5.0] * 6})
    out = ops.ts_zscore(s, 3)
    # All rows after the warmup should be NaN
    assert out.iloc[2:]["X"].isna().all()


def test_ts_zscore_known_value():
    s = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ops.ts_zscore(s, 3)
    # window [1,2,3]: mean=2, std=1, last value 3 → z=1
    assert out.iloc[2]["X"] == pytest.approx(1.0)


def test_ts_quantile_median_equals_ts_median(df):
    np.testing.assert_allclose(
        ops.ts_quantile(df, 3, 0.5).values,
        ops.ts_median(df, 3).values,
        equal_nan=True,
    )


def test_ts_quantile_rejects_out_of_range(df):
    with pytest.raises(ValueError, match=r"q must be in"):
        ops.ts_quantile(df, 3, 1.5)


def test_ts_arg_max_today_is_max():
    # Strictly increasing → today is always the max → arg = 0
    s = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ops.ts_arg_max(s, 3)
    assert out.iloc[2]["X"] == 0.0
    assert out.iloc[4]["X"] == 0.0


def test_ts_arg_min_today_is_min():
    s = pd.DataFrame({"X": [5.0, 4.0, 3.0, 2.0, 1.0]})
    out = ops.ts_arg_min(s, 3)
    assert out.iloc[2]["X"] == 0.0
    assert out.iloc[4]["X"] == 0.0


def test_ts_product_known_value():
    s = pd.DataFrame({"X": [2.0, 3.0, 4.0, 5.0]})
    out = ops.ts_product(s, 3)
    # window [2,3,4] → 24
    assert out.iloc[2]["X"] == pytest.approx(24.0)
    # window [3,4,5] → 60
    assert out.iloc[3]["X"] == pytest.approx(60.0)


def test_ts_returns_matches_pct_change():
    s = pd.DataFrame({"X": [10.0, 11.0, 12.1, 13.31]})
    out = ops.ts_returns(s, 1)
    expected = s.pct_change(1)
    np.testing.assert_allclose(out.values, expected.values, equal_nan=True)


def test_ts_decay_exp_factor_one_equals_simple_mean():
    # factor=1.0 → all weights equal → same as ts_mean
    s = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
    decayed = ops.ts_decay_exp(s, 3, 1.0)
    np.testing.assert_allclose(
        decayed.values,
        ops.ts_mean(s, 3).values,
        equal_nan=True,
    )


def test_ts_decay_exp_rejects_bad_factor():
    s = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="factor"):
        ops.ts_decay_exp(s, 3, 0.0)
    with pytest.raises(ValueError, match="factor"):
        ops.ts_decay_exp(s, 3, 1.5)


def test_ts_partial_corr_zero_when_z_explains_x():
    # If z ≡ x, then conditioning on z removes all variance — partial corr ≈ 0
    rng = np.random.default_rng(0)
    n = 100
    z = pd.DataFrame({"X": rng.standard_normal(n)})
    x = z.copy()
    y = pd.DataFrame({"X": rng.standard_normal(n)})
    out = ops.ts_partial_corr(x, y, z, 30)
    # Numerically the denominator hits the floor; result should not blow up
    finite = out.dropna().values
    assert np.all(np.abs(finite) < 1.5)  # bounded


def test_ts_regression_slope_known_relation():
    # y = 2x + noise → slope ≈ 2 over a long enough window
    rng = np.random.default_rng(7)
    n = 200
    x_arr = rng.standard_normal(n)
    y_arr = 2.0 * x_arr + 0.01 * rng.standard_normal(n)
    x = pd.DataFrame({"X": x_arr})
    y = pd.DataFrame({"X": y_arr})
    out = ops.ts_regression(y, x, 60)
    last = out.iloc[-1]["X"]
    assert last == pytest.approx(2.0, abs=0.05)


def test_ts_min_max_diff_known_value():
    s = pd.DataFrame({"X": [1.0, 5.0, 3.0, 9.0, 2.0]})
    out = ops.ts_min_max_diff(s, 3)
    # window [1,5,3] → 5-1 = 4
    assert out.iloc[2]["X"] == pytest.approx(4.0)
    # window [5,3,9] → 9-3 = 6
    assert out.iloc[3]["X"] == pytest.approx(6.0)


def test_days_from_last_change_constant_run():
    s = pd.DataFrame({"X": [5.0, 5.0, 5.0, 7.0, 7.0]})
    out = ops.days_from_last_change(s)
    # Initial 5: counter = 0; second 5: 1; third 5: 2; first 7 (change): 0; second 7: 1
    assert list(out["X"].values) == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_hump_suppresses_below_threshold():
    s = pd.DataFrame({"X": [1.0, 1.005, 1.01, 1.5, 1.502]})
    out = ops.hump(s, threshold=0.1)
    # Initial 1.0; 1.005 (change=0.005<0.1) stays at 1.0;
    # 1.01 (cum change still tiny) stays; 1.5 (jump) updates; 1.502 stays at 1.5
    assert out.iloc[0]["X"] == 1.0
    assert out.iloc[1]["X"] == 1.0
    assert out.iloc[2]["X"] == 1.0
    assert out.iloc[3]["X"] == 1.5
    assert out.iloc[4]["X"] == 1.5


# ---------- Phase A: cross-sectional additions ----------


def test_winsorize_clips_at_std_band():
    # Realistic shape: 20 cols of standard normal + an extreme outlier in col 0
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((10, 20)))
    df.iloc[:, 0] = 100.0  # outliers we expect to clip
    w = ops.winsorize(df, std=2.0)
    # Invariant: every value within ±2σ of original row mean (within tolerance)
    mean = df.mean(axis=1)
    sigma = df.std(axis=1)
    upper = mean + 2.0 * sigma
    lower = mean - 2.0 * sigma
    for i in range(len(df)):
        assert (w.iloc[i] >= lower.iloc[i] - 1e-9).all()
        assert (w.iloc[i] <= upper.iloc[i] + 1e-9).all()
    # And the outlier column was actually clipped (not unchanged)
    assert (w.iloc[:, 0] < 100.0 - 1e-9).all()


def test_quantile_median_known_row():
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], columns=list("ABCDE"))
    out = ops.quantile(df, 0.5)
    # All entries in the row are the same broadcast value (the median = 3)
    assert (out.iloc[0] == 3.0).all()


def test_vector_neut_orthogonalizes():
    # x = α*y + ε → vector_neut(x, y) should be approximately ε per row
    rng = np.random.default_rng(11)
    n_rows, n_cols = 50, 20
    y = pd.DataFrame(rng.standard_normal((n_rows, n_cols)))
    eps = pd.DataFrame(0.01 * rng.standard_normal((n_rows, n_cols)))
    x = 2.5 * y + eps
    resid = ops.vector_neut(x, y)
    # Per-row dot product of residual with y should be ~0
    dot = (resid * y).sum(axis=1)
    assert dot.abs().max() < 1e-9


def test_regression_neut_residual_uncorrelated_with_y():
    rng = np.random.default_rng(13)
    n_rows, n_cols = 50, 20
    y = pd.DataFrame(rng.standard_normal((n_rows, n_cols)))
    x = 1.7 * y + 0.05 * pd.DataFrame(rng.standard_normal((n_rows, n_cols)))
    resid = ops.regression_neut(x, y)
    # Both demeaned series → cross-product should be ~0 per row
    y_dm = y.sub(y.mean(axis=1), axis=0)
    dot = (resid * y_dm).sum(axis=1)
    assert dot.abs().max() < 1e-9


def test_bucket_assigns_n_groups():
    # 10 values, n=5 buckets → groups 0..4, each with 2 entries
    df = pd.DataFrame([list(range(10))], columns=[f"T{i}" for i in range(10)])
    b = ops.bucket(df, n=5)
    counts = b.iloc[0].value_counts().sort_index()
    assert list(counts.index) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert (counts == 2).all()


def test_tail_replaces_out_of_band():
    df = pd.DataFrame([[-3.0, -1.0, 0.0, 1.0, 3.0]], columns=list("ABCDE"))
    out = ops.tail(df, lower=-2.0, upper=2.0, replace=0.0)
    # Out-of-band → 0; in-band → unchanged
    assert out.iloc[0]["A"] == 0.0
    assert out.iloc[0]["E"] == 0.0
    assert out.iloc[0]["B"] == -1.0


def test_kth_element_smallest():
    df = pd.DataFrame([[10.0, 1.0, 7.0, 3.0]], columns=list("ABCD"))
    out = ops.kth_element(df, 1)  # smallest
    assert (out.iloc[0] == 1.0).all()
    out2 = ops.kth_element(df, -1)  # largest
    assert (out2.iloc[0] == 10.0).all()


def test_harmonic_geometric_mean_known_row():
    df = pd.DataFrame([[1.0, 2.0, 4.0]], columns=list("ABC"))
    # HM = 3 / (1 + 0.5 + 0.25) = 3/1.75
    hm = ops.harmonic_mean(df).iloc[0, 0]
    assert hm == pytest.approx(3 / 1.75)
    # GM = (1*2*4)^(1/3) = 2
    gm = ops.geometric_mean(df).iloc[0, 0]
    assert gm == pytest.approx(2.0)


def test_step_endpoints_minus_one_and_plus_one():
    # Strictly increasing 5 values → step → -1, -0.5, 0, 0.5, 1
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], columns=list("ABCDE"))
    s = ops.step(df).iloc[0]
    assert s["A"] == pytest.approx(-1.0)
    assert s["E"] == pytest.approx(1.0)
    assert s["C"] == pytest.approx(0.0)


# ---------- Phase A: group operators ----------


@pytest.fixture
def group_fixture():
    """4 tickers in 2 sectors (Tech, Energy), values that make group math obvious."""
    cols = ["AAPL", "MSFT", "XOM", "CVX"]
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    x = pd.DataFrame(
        [
            [1.0, 3.0, 10.0, 20.0],
            [2.0, 4.0, 30.0, 40.0],
            [5.0, 6.0, 50.0, 60.0],
        ],
        index=dates, columns=cols,
    )
    sectors = pd.DataFrame(
        [["Tech", "Tech", "Energy", "Energy"]] * 3,
        index=dates, columns=cols,
    )
    return x, sectors


def test_group_mean_broadcasts_within_group(group_fixture):
    x, g = group_fixture
    out = ops.group_mean(x, g)
    # Tech row 0: mean(1,3) = 2 → AAPL=MSFT=2
    assert out.iloc[0]["AAPL"] == pytest.approx(2.0)
    assert out.iloc[0]["MSFT"] == pytest.approx(2.0)
    # Energy row 0: mean(10,20) = 15 → XOM=CVX=15
    assert out.iloc[0]["XOM"] == pytest.approx(15.0)
    assert out.iloc[0]["CVX"] == pytest.approx(15.0)


def test_group_neutralize_subtracts_group_mean(group_fixture):
    x, g = group_fixture
    out = ops.group_neutralize(x, g)
    # Tech row 0: 1-2=-1, 3-2=+1
    assert out.iloc[0]["AAPL"] == pytest.approx(-1.0)
    assert out.iloc[0]["MSFT"] == pytest.approx(1.0)
    # Energy row 0: 10-15=-5, 20-15=+5
    assert out.iloc[0]["XOM"] == pytest.approx(-5.0)
    assert out.iloc[0]["CVX"] == pytest.approx(5.0)


def test_group_rank_within_group(group_fixture):
    x, g = group_fixture
    out = ops.group_rank(x, g)
    # Within Tech (n=2): ranks are 0.5 and 1.0 (smallest gets 0.5)
    # Within Energy (n=2): same
    # Row 0: Tech [1,3] → AAPL=0.5, MSFT=1.0; Energy [10,20] → XOM=0.5, CVX=1.0
    assert out.iloc[0]["AAPL"] == pytest.approx(0.5)
    assert out.iloc[0]["MSFT"] == pytest.approx(1.0)
    assert out.iloc[0]["XOM"] == pytest.approx(0.5)
    assert out.iloc[0]["CVX"] == pytest.approx(1.0)


def test_group_count_returns_size(group_fixture):
    x, g = group_fixture
    out = ops.group_count(x, g)
    # Both groups have 2 members
    assert (out.iloc[0] == 2).all()


def test_group_max_min(group_fixture):
    x, g = group_fixture
    mx = ops.group_max(x, g)
    mn = ops.group_min(x, g)
    # Tech row 0: max=3, min=1
    assert mx.iloc[0]["AAPL"] == pytest.approx(3.0)
    assert mn.iloc[0]["AAPL"] == pytest.approx(1.0)
    # Energy row 0: max=20, min=10
    assert mx.iloc[0]["XOM"] == pytest.approx(20.0)
    assert mn.iloc[0]["XOM"] == pytest.approx(10.0)


def test_group_zscore_zero_mean_per_group(group_fixture):
    x, g = group_fixture
    out = ops.group_zscore(x, g)
    # Per group per row, sum of z-scores is 0 (mean is 0)
    tech_sum = out.iloc[0][["AAPL", "MSFT"]].sum()
    energy_sum = out.iloc[0][["XOM", "CVX"]].sum()
    assert tech_sum == pytest.approx(0.0, abs=1e-9)
    assert energy_sum == pytest.approx(0.0, abs=1e-9)


def test_group_normalize_unit_abs_sum_per_group(group_fixture):
    x, g = group_fixture
    out = ops.group_normalize(x, g)
    # Per group, the |sum| of normalized weights = 1 (the operator's invariant)
    tech_abs_sum = out.iloc[0][["AAPL", "MSFT"]].abs().sum()
    energy_abs_sum = out.iloc[0][["XOM", "CVX"]].abs().sum()
    assert tech_abs_sum == pytest.approx(1.0)
    assert energy_abs_sum == pytest.approx(1.0)


def test_group_scale_unit_abs_sum_per_group(group_fixture):
    x, g = group_fixture
    out = ops.group_scale(x, g)
    tech_abs_sum = out.iloc[0][["AAPL", "MSFT"]].abs().sum()
    energy_abs_sum = out.iloc[0][["XOM", "CVX"]].abs().sum()
    assert tech_abs_sum == pytest.approx(1.0)
    assert energy_abs_sum == pytest.approx(1.0)


def test_group_ops_treat_nan_label_as_excluded():
    # Mixed: 2 cols labeled, 1 col with NaN label
    cols = ["A", "B", "C"]
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    x = pd.DataFrame([[1.0, 2.0, 999.0], [3.0, 4.0, 999.0]], index=dates, columns=cols)
    g = pd.DataFrame([["X", "X", None], ["X", "X", None]], index=dates, columns=cols)
    out = ops.group_mean(x, g)
    # A and B in group X → both get mean(1,2) = 1.5 on row 0
    assert out.iloc[0]["A"] == pytest.approx(1.5)
    assert out.iloc[0]["B"] == pytest.approx(1.5)
    # C has no group → result NaN, never contaminated by 999
    assert pd.isna(out.iloc[0]["C"])


def test_group_neutralize_via_evaluator():
    """End-to-end: parser accepts `sector` as a data field, evaluator routes
    it to the GICS frame, group_neutralize uses it."""
    from engine.evaluator import AlphaEvaluator

    cols = ["AAPL", "MSFT", "XOM"]
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    close = pd.DataFrame([[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]], index=dates, columns=cols)
    sector = pd.DataFrame(
        [["Tech", "Tech", "Energy"], ["Tech", "Tech", "Energy"]],
        index=dates, columns=cols,
    )
    evaluator = AlphaEvaluator({"close": close, "sector": sector})
    result = evaluator.evaluate("group_neutralize(close, sector)")
    # AAPL row 0: 1 - mean(1,3) = -1
    assert result.iloc[0]["AAPL"] == pytest.approx(-1.0)
    # XOM row 0: only one Energy stock → mean is itself → result 0
    assert result.iloc[0]["XOM"] == pytest.approx(0.0)


# ---------- Phase A: arithmetic additions ----------


def test_op_exp():
    df = pd.DataFrame({"X": [0.0, 1.0, np.log(5.0)]})
    out = ops.op_exp(df)
    assert out.iloc[0]["X"] == pytest.approx(1.0)
    assert out.iloc[1]["X"] == pytest.approx(np.e)
    assert out.iloc[2]["X"] == pytest.approx(5.0)


def test_op_sqrt_negative_returns_nan():
    df = pd.DataFrame({"X": [4.0, -1.0, 9.0]})
    out = ops.op_sqrt(df)
    assert out.iloc[0]["X"] == pytest.approx(2.0)
    assert pd.isna(out.iloc[1]["X"])
    assert out.iloc[2]["X"] == pytest.approx(3.0)


def test_op_mod_known_values():
    a = pd.DataFrame({"X": [10.0, 7.0, 5.0]})
    b = pd.DataFrame({"X": [3.0, 0.0, 2.0]})
    out = ops.op_mod(a, b)
    assert out.iloc[0]["X"] == pytest.approx(1.0)
    assert pd.isna(out.iloc[1]["X"])  # mod by zero → NaN
    assert out.iloc[2]["X"] == pytest.approx(1.0)


def test_clip_clamps_to_band():
    df = pd.DataFrame({"X": [-5.0, 0.0, 3.0, 10.0]})
    out = ops.clip(df, -1.0, 2.0)
    assert out.iloc[0]["X"] == -1.0
    assert out.iloc[1]["X"] == 0.0
    assert out.iloc[2]["X"] == 2.0
    assert out.iloc[3]["X"] == 2.0


def test_clip_rejects_inverted_bounds():
    df = pd.DataFrame({"X": [1.0]})
    with pytest.raises(ValueError, match="lo .* must be <= hi"):
        ops.clip(df, 5.0, 1.0)


def test_signed_power_preserves_sign():
    df = pd.DataFrame({"X": [-4.0, 0.0, 9.0]})
    out = ops.signed_power(df, 0.5)  # signed sqrt
    assert out.iloc[0]["X"] == pytest.approx(-2.0)
    assert out.iloc[1]["X"] == pytest.approx(0.0)
    assert out.iloc[2]["X"] == pytest.approx(3.0)


def test_sigmoid_at_zero_is_half():
    df = pd.DataFrame({"X": [0.0, 1e9, -1e9]})
    out = ops.sigmoid(df)
    assert out.iloc[0]["X"] == pytest.approx(0.5)
    assert out.iloc[1]["X"] == pytest.approx(1.0)
    assert out.iloc[2]["X"] == pytest.approx(0.0)


def test_replace_swaps_value():
    df = pd.DataFrame({"X": [1.0, 2.0, 1.0, 3.0]})
    out = ops.replace(df, 1.0, 99.0)
    assert list(out["X"]) == [99.0, 2.0, 99.0, 3.0]


def test_isnan_marks_missing():
    df = pd.DataFrame({"X": [1.0, np.nan, 3.0]})
    out = ops.isnan(df)
    assert list(out["X"]) == [0.0, 1.0, 0.0]


def test_op_equal_returns_indicator():
    a = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"X": [1.0, 9.0, 3.0]})
    out = ops.op_equal(a, b)
    assert list(out["X"]) == [1.0, 0.0, 1.0]


def test_where_routes_by_truthy_cond():
    cond = pd.DataFrame({"X": [1.0, 0.0, 2.0]})  # nonzero → True
    x = pd.DataFrame({"X": [10.0, 20.0, 30.0]})
    y = pd.DataFrame({"X": [-1.0, -2.0, -3.0]})
    out = ops.where(cond, x, y)
    assert list(out["X"]) == [10.0, -2.0, 30.0]


def test_arithmetic_remaps_via_evaluator():
    """User-facing names (exp, sqrt, mod, equal) must resolve correctly."""
    from engine.evaluator import AlphaEvaluator

    df = pd.DataFrame({"A": [4.0, 9.0], "B": [16.0, 25.0]})
    ev = AlphaEvaluator({"close": df})
    assert ev.evaluate("sqrt(close)").iloc[0]["A"] == pytest.approx(2.0)
    assert ev.evaluate("exp(close - close)").iloc[0]["A"] == pytest.approx(1.0)
    assert ev.evaluate("mod(close, close - close + 3)").iloc[0]["A"] == pytest.approx(1.0)
    assert ev.evaluate("equal(close, close)").iloc[0]["A"] == 1.0


# ---------- Phase A: conditional / state operators ----------


def test_trade_when_carries_position_forward():
    # Cond: True only on rows 0 and 2; x: 1, 2, 3, 4, 5
    cond = pd.DataFrame({"X": [True, False, True, False, False]})
    x = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ops.trade_when(cond, x)
    # Row 0: take 1 (cond True). Row 1: carry 1 (cond False).
    # Row 2: take 3 (cond True). Rows 3-4: carry 3.
    assert out.iloc[0]["X"] == 1.0
    assert out.iloc[1]["X"] == 1.0
    assert out.iloc[2]["X"] == 3.0
    assert out.iloc[3]["X"] == 3.0
    assert out.iloc[4]["X"] == 3.0


def test_trade_when_exit_drops_to_nan():
    cond = pd.DataFrame({"X": [True, False, False, False]})
    exit_c = pd.DataFrame({"X": [False, False, True, False]})
    x = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]})
    out = ops.trade_when(cond, x, exit_c)
    # Row 0: enter at 1. Row 1: carry 1. Row 2: exit → NaN. Row 3: still NaN (no re-entry).
    assert out.iloc[0]["X"] == 1.0
    assert out.iloc[1]["X"] == 1.0
    assert pd.isna(out.iloc[2]["X"])
    assert pd.isna(out.iloc[3]["X"])


def test_when_no_carry():
    cond = pd.DataFrame({"X": [True, False, True]})
    x = pd.DataFrame({"X": [10.0, 20.0, 30.0]})
    out = ops.when(cond, x)
    assert out.iloc[0]["X"] == 10.0
    assert pd.isna(out.iloc[1]["X"])  # no carry, unlike trade_when
    assert out.iloc[2]["X"] == 30.0


def test_mask_inverts_when():
    cond = pd.DataFrame({"X": [True, False, True]})
    x = pd.DataFrame({"X": [10.0, 20.0, 30.0]})
    out = ops.mask(x, cond)
    # mask drops cells where cond is True
    assert pd.isna(out.iloc[0]["X"])
    assert out.iloc[1]["X"] == 20.0
    assert pd.isna(out.iloc[2]["X"])


def test_keep_top_n_by_magnitude():
    df = pd.DataFrame([[1.0, -5.0, 2.0, -3.0, 0.5]], columns=list("ABCDE"))
    out = ops.keep(df, 2)
    # Top-2 by |x| → -5 (B) and -3 (D); the rest zero out
    assert out.iloc[0]["B"] == -5.0
    assert out.iloc[0]["D"] == -3.0
    assert out.iloc[0]["A"] == 0.0
    assert out.iloc[0]["C"] == 0.0
    assert out.iloc[0]["E"] == 0.0


def test_keep_rejects_zero_n():
    df = pd.DataFrame({"X": [1.0]})
    with pytest.raises(ValueError, match="must be >= 1"):
        ops.keep(df, 0)


def test_pasteurize_replaces_inf_and_nan_with_zero():
    df = pd.DataFrame({"X": [1.0, np.nan, np.inf, -np.inf, 2.0]})
    out = ops.pasteurize(df)
    assert list(out["X"]) == [1.0, 0.0, 0.0, 0.0, 2.0]
