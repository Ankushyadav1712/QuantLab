"""IC / ICIR / alpha-decay / rank-stability — sanity tests.

All tests use small synthetic frames with known structure so we can assert
on the recovered values directly (no fixtures needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.ic_metrics import (
    compute_alpha_decay,
    compute_ic_series,
    compute_ic_summary,
    compute_rank_stability,
)

# ---------- Helpers ----------


def _frames(T: int = 100, N: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two (T × N) DataFrames sharing the same date index and tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="B")
    tickers = [f"T{i:02d}" for i in range(N)]
    rets = pd.DataFrame(rng.standard_normal((T, N)) * 0.01, index=dates, columns=tickers)
    return rets, pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=tickers)


# ---------- IC series ----------


def test_ic_perfect_signal_is_one():
    rets, _ = _frames()
    # Signal at t = next-day return → IC at horizon 1 should be ≈ 1.0
    signal = rets.shift(-1)
    ic = compute_ic_series(signal, rets, horizon=1).dropna()
    assert ic.mean() == pytest.approx(1.0, abs=1e-9)
    # Each daily IC is exactly 1 (perfect ranking), so std should be 0
    assert ic.std(ddof=1) == pytest.approx(0.0, abs=1e-9)


def test_ic_inverted_signal_is_minus_one():
    rets, _ = _frames()
    signal = -rets.shift(-1)
    ic = compute_ic_series(signal, rets, horizon=1).dropna()
    assert ic.mean() == pytest.approx(-1.0, abs=1e-9)


def test_ic_random_signal_is_near_zero():
    rets, sig = _frames(T=400, N=30, seed=42)
    ic = compute_ic_series(sig, rets, horizon=1).dropna()
    # With 400 days, |mean IC| should be near 0 — well under 0.1
    assert abs(ic.mean()) < 0.1


def test_ic_empty_inputs_return_empty_series():
    empty = pd.DataFrame()
    out = compute_ic_series(empty, empty, horizon=1)
    assert isinstance(out, pd.Series)
    assert out.empty


def test_ic_handles_single_column():
    # Cannot rank a single column cross-sectionally → empty result
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    sig = pd.DataFrame({"A": np.random.randn(10)}, index=dates)
    rets = pd.DataFrame({"A": np.random.randn(10) * 0.01}, index=dates)
    out = compute_ic_series(sig, rets, horizon=1)
    assert out.empty


# ---------- IC summary ----------


def test_ic_summary_returns_n_days():
    rets, _ = _frames(T=50, N=10)
    sig = rets.shift(-1)
    s = compute_ic_summary(sig, rets, horizon=1)
    # 50 rows minus the 1 lost to shift = 49 valid IC days
    assert s["n_days"] == 49
    assert s["ic"] == pytest.approx(1.0, abs=1e-9)


def test_ic_summary_handles_too_few_days():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    tickers = ["A", "B", "C"]
    sig = pd.DataFrame(np.random.randn(2, 3), index=dates, columns=tickers)
    rets = pd.DataFrame(np.random.randn(2, 3) * 0.01, index=dates, columns=tickers)
    s = compute_ic_summary(sig, rets, horizon=1)
    # Only 1 IC value after the shift → can't compute std → all None
    assert s["icir"] is None
    assert s["ic_tstat"] is None


def test_ic_summary_pct_positive():
    # Start from a perfect predictor (signal[t] = return[t+1]) and flip the
    # sign on 30% of days.  The IC should then be +1 on 70% of days and -1
    # on 30%, so ic_pct_positive ≈ 0.7.
    rng = np.random.default_rng(123)
    T, N = 200, 30
    dates = pd.date_range("2022-01-01", periods=T, freq="B")
    tickers = [f"T{i}" for i in range(N)]
    rets = pd.DataFrame(rng.standard_normal((T, N)) * 0.01, index=dates, columns=tickers)
    sig_arr = rets.shift(-1).to_numpy().copy()
    flip_days = rng.random(T) < 0.3
    sig_arr[flip_days] *= -1
    sig = pd.DataFrame(sig_arr, index=rets.index, columns=rets.columns)
    s = compute_ic_summary(sig, rets, horizon=1)
    # Expect roughly 70% positive — allow wide slack for the 200-day sample
    assert 0.55 < s["ic_pct_positive"] < 0.85


# ---------- Alpha decay ----------


def test_alpha_decay_recovers_known_half_life():
    """Synthetic process with embedded ~10-day decay → half-life ~ 7 days."""
    rng = np.random.default_rng(7)
    T, N = 500, 50
    dates = pd.date_range("2022-01-01", periods=T, freq="B")
    tickers = [f"T{i}" for i in range(N)]
    sig_arr = rng.standard_normal((T, N))
    ret_arr = np.zeros((T, N))
    # return[t] = sum_k 0.05 * exp(-k/10) * signal[t-k]  + small noise
    for t in range(1, T):
        for k in range(1, min(t + 1, 25)):
            ret_arr[t] += 0.05 * np.exp(-k / 10.0) * sig_arr[t - k]
        ret_arr[t] += 0.005 * rng.standard_normal(N)
    sig = pd.DataFrame(sig_arr, index=dates, columns=tickers)
    rets = pd.DataFrame(ret_arr, index=dates, columns=tickers)
    dec = compute_alpha_decay(sig, rets, horizons=(1, 2, 3, 5, 10, 21))

    # Recovered half-life should be 4–12 (synthetic effective half-life ~7)
    hl = dec["half_life_days"]
    assert hl is not None
    assert 4.0 < hl < 12.0
    # Decay fit should be tight (linear in log space)
    assert dec["r_squared"] > 0.8
    # ICs should be monotonically decreasing in horizon
    ics = [dec["ic_by_horizon"][h] for h in (1, 2, 3, 5, 10, 21)]
    for prev, curr in zip(ics, ics[1:]):
        assert prev >= curr or abs(prev - curr) < 0.05  # allow tiny non-monotonicity


def test_alpha_decay_handles_pure_noise():
    """No signal → no half-life claim; must not crash."""
    rets, sig = _frames(T=300, N=20, seed=99)
    dec = compute_alpha_decay(sig, rets, horizons=(1, 2, 5, 10))
    # ICs near zero → fit is unstable; half-life should be None or huge
    if dec["half_life_days"] is not None:
        # If a fit happened, it should be either a fluke (small) or huge
        assert dec["half_life_days"] > 0


def test_alpha_decay_records_each_horizon():
    rets, _ = _frames(T=200, N=15, seed=3)
    sig = rets.shift(-1)  # perfect h=1 predictor
    dec = compute_alpha_decay(sig, rets, horizons=(1, 2, 5))
    # ic_by_horizon should have all 3 keys
    assert set(dec["ic_by_horizon"].keys()) == {1, 2, 5}
    assert dec["ic_by_horizon"][1] == pytest.approx(1.0, abs=1e-9)


# ---------- Rank stability ----------


def test_rank_stability_perfect_persistence_is_one():
    # Same ranking every day
    T, N = 50, 10
    dates = pd.date_range("2024-01-01", periods=T, freq="B")
    tickers = [f"T{i}" for i in range(N)]
    arr = np.tile(np.arange(N), (T, 1)).astype(float)
    sig = pd.DataFrame(arr, index=dates, columns=tickers)
    rs = compute_rank_stability(sig)
    assert rs == pytest.approx(1.0, abs=1e-9)


def test_rank_stability_random_is_near_zero():
    rng = np.random.default_rng(11)
    T, N = 300, 30
    dates = pd.date_range("2023-01-01", periods=T, freq="B")
    tickers = [f"T{i}" for i in range(N)]
    sig = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=tickers)
    rs = compute_rank_stability(sig)
    assert abs(rs) < 0.1


def test_rank_stability_handles_empty_or_single_row():
    assert compute_rank_stability(pd.DataFrame()) is None
    assert (
        compute_rank_stability(
            pd.DataFrame({"A": [1.0], "B": [2.0]}, index=[pd.Timestamp("2024-01-01")])
        )
        is None
    )
