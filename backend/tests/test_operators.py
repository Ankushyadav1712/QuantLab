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
