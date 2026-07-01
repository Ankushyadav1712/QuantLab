"""Portfolio diversification curve — synthetic-correlation invariants."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from analytics.diversification import (
    DEFAULT_SIZES,
    diversification_curve,
    extract_daily_returns_from_saved,
)

# ---------- Helpers ----------


def _make_pnl(n_alphas: int, n_days: int = 504, seed: int = 0, sharpe_per: float = 1.0):
    """Build a pool of `n_alphas` independent daily-PnL series, each with
    approximate annualised Sharpe ``sharpe_per``."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    # daily mean s.t. mean/std * sqrt(252) ≈ sharpe_per
    # using std=0.01, mean = sharpe_per * 0.01 / sqrt(252)
    daily_vol = 0.01
    daily_mean = sharpe_per * daily_vol / np.sqrt(252)
    out: dict[int, pd.Series] = {}
    for i in range(n_alphas):
        # Each series gets its own RNG draw → independent
        noise = rng.standard_normal(n_days) * daily_vol
        rets = noise + daily_mean
        out[i + 1] = pd.Series(rets, index=dates)
    return out


# ---------- diversification_curve ----------


def test_independent_alphas_show_sqrt_n_diversification():
    """k independent Sharpe-1 alphas → ensemble Sharpe ≈ √k.

    With 5 independent alphas, n=1 median should be ~1.0 and n=5 should
    be ~√5 ≈ 2.24.  Tolerance is wide because of sample-size noise on
    only ~500 trading days.
    """
    pnl = _make_pnl(n_alphas=5, n_days=2520, seed=1, sharpe_per=1.0)
    curve = diversification_curve(pnl, sizes=(1, 2, 5), n_samples=20, seed=0)
    by_n = {r["n"]: r for r in curve}
    # n=1 should be near the per-alpha sharpe of 1.0
    assert 0.7 < by_n[1]["median_sharpe"] < 1.4
    # n=5 with independent alphas should approach sqrt(5) ≈ 2.24
    assert by_n[5]["median_sharpe"] > 1.6
    # Curve should be monotonically increasing in the independent case
    assert by_n[2]["median_sharpe"] > by_n[1]["median_sharpe"] - 0.2
    assert by_n[5]["median_sharpe"] > by_n[2]["median_sharpe"] - 0.2


def test_identical_alphas_curve_is_flat():
    """k copies of the same PnL → no diversification benefit.  Sharpe
    should be ~constant across n (the equal-weight mean of identical
    series is just the series itself)."""
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-03", periods=1260, freq="B")
    single = pd.Series(rng.standard_normal(1260) * 0.01 + 0.001, index=dates)
    pnl = {i + 1: single.copy() for i in range(5)}
    curve = diversification_curve(pnl, sizes=(1, 2, 5), n_samples=10, seed=0)
    medians = [r["median_sharpe"] for r in curve]
    # All medians should be near-identical (within ~5%)
    assert max(medians) - min(medians) < 0.05


def test_empty_or_single_alpha_returns_empty():
    """Need at least 2 alphas for a meaningful curve."""
    assert diversification_curve({}, n_samples=5) == []
    pnl = _make_pnl(n_alphas=1)
    assert diversification_curve(pnl, n_samples=5) == []


def test_sizes_clamped_to_pool_size():
    """Asking for n=21 when only 4 alphas exist should produce n=4 row
    (not skipped, not error)."""
    pnl = _make_pnl(n_alphas=4)
    curve = diversification_curve(pnl, sizes=(1, 2, 8, 13, 21), n_samples=8, seed=0)
    # Sizes 8, 13, 21 all clamp to 4; should appear only once
    ns = [r["n"] for r in curve]
    assert 4 in ns
    # Dedup: size 21 → 4 already → don't repeat
    assert ns.count(4) == 1
    # 1 and 2 still come through
    assert 1 in ns
    assert 2 in ns


def test_result_includes_iqr_bands():
    pnl = _make_pnl(n_alphas=5, seed=3)
    curve = diversification_curve(pnl, sizes=(2,), n_samples=20)
    r = curve[0]
    assert r["q1"] <= r["median_sharpe"] <= r["q3"]
    assert r["min"] <= r["q1"]
    assert r["q3"] <= r["max"]


def test_deterministic_with_seed():
    pnl = _make_pnl(n_alphas=4, seed=4)
    a = diversification_curve(pnl, sizes=(2, 3), n_samples=10, seed=42)
    b = diversification_curve(pnl, sizes=(2, 3), n_samples=10, seed=42)
    for ra, rb in zip(a, b):
        assert ra["median_sharpe"] == pytest.approx(rb["median_sharpe"], abs=1e-9)


def test_default_sizes_is_fibonacci_ish():
    assert DEFAULT_SIZES == (1, 2, 3, 5, 8, 13, 21)


# ---------- extract_daily_returns_from_saved ----------


def test_extract_from_saved_parses_json_string():
    """SQLite stores result_json as a TEXT column → arrives as str."""
    payload = {
        "is_timeseries": {
            "dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "daily_returns": [0.001, -0.002, 0.0005],
        }
    }
    rows = [{"id": 7, "result_json": json.dumps(payload)}]
    out = extract_daily_returns_from_saved(rows)
    assert 7 in out
    assert len(out[7]) == 3
    assert out[7].iloc[0] == pytest.approx(0.001)


def test_extract_skips_malformed():
    """Rows without a usable timeseries should be silently dropped."""
    rows = [
        {"id": 1, "result_json": "not json"},
        {"id": 2, "result_json": json.dumps({"no_timeseries": True})},
        {"id": 3, "result_json": json.dumps({"is_timeseries": {"dates": [], "daily_returns": []}})},
        {"id": 4, "result_json": None},
        {"id": 5},  # missing field
    ]
    out = extract_daily_returns_from_saved(rows)
    assert out == {}


def test_extract_handles_dict_result_field():
    """When the result_json blob has already been parsed (e.g. /api/alphas/{id}
    returns a dict), extraction should still work."""
    payload = {
        "is_timeseries": {
            "dates": ["2024-01-02", "2024-01-03"],
            "daily_returns": [0.01, -0.01],
        }
    }
    rows = [{"id": 9, "result": payload}]
    out = extract_daily_returns_from_saved(rows, result_field="result_json")
    # Falls back to "result" key when "result_json" missing
    assert 9 in out


def test_extract_handles_none_in_returns():
    """JSON serialises NaN as null → string round-trip turns them to None;
    must coerce back to np.nan."""
    payload = {
        "is_timeseries": {
            "dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "daily_returns": [0.001, None, 0.002],
        }
    }
    rows = [{"id": 1, "result_json": json.dumps(payload)}]
    out = extract_daily_returns_from_saved(rows)
    assert 1 in out
    assert np.isnan(out[1].iloc[1])
    assert out[1].iloc[0] == 0.001
