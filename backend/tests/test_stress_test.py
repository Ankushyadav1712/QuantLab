"""Stress-test (regime window) analytics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from analytics.stress_test import (
    DEFAULT_REGIMES,
    MIN_REGIME_DAYS,
    Regime,
    compute_stress_metrics,
    regime_severity,
)


def _series(start: str, n_days: int, ret: float = 0.0) -> pd.Series:
    """Daily returns series starting at ``start`` for ``n_days`` business days."""
    idx = pd.date_range(start, periods=n_days, freq="B")
    return pd.Series([ret] * n_days, index=idx)


def test_stress_window_inside_data_is_picked_up():
    """Returns spanning 2020 → COVID windows should produce results."""
    rng = np.random.default_rng(0)
    n = 800
    rets = pd.Series(
        rng.standard_normal(n) * 0.005, index=pd.date_range("2019-01-01", periods=n, freq="B")
    )
    out = compute_stress_metrics(rets)
    names = [r["name"] for r in out]
    assert "covid_crash" in names
    assert "covid_recovery" in names


def test_regime_skipped_when_no_overlap():
    """A 2010–2012 series should miss every default regime except euro_debt_2011."""
    rng = np.random.default_rng(1)
    rets = pd.Series(
        rng.standard_normal(500) * 0.005, index=pd.date_range("2010-01-01", periods=500, freq="B")
    )
    out = compute_stress_metrics(rets)
    names = {r["name"] for r in out}
    # Should hit euro_debt_2011 but no later regimes
    assert "euro_debt_2011" in names
    assert "covid_crash" not in names
    assert "svb_crisis" not in names


def test_too_few_days_skipped():
    """A regime that only overlaps for 2 days is below MIN_REGIME_DAYS."""
    # Construct a series that only has the last 2 days of COVID_CRASH inside
    idx = pd.date_range("2020-03-22", periods=10, freq="B")
    rets = pd.Series([0.001] * 10, index=idx)
    out = compute_stress_metrics(rets)
    names = {r["name"] for r in out}
    # COVID_CRASH ends 2020-03-23 — only 1-2 days of overlap, below threshold
    assert "covid_crash" not in names


def test_regime_metrics_recover_known_sharpe():
    """Hand-craft a constant-return series and verify the metrics."""
    # 30 business days of constant +0.001 daily return inside SVB window
    idx = pd.date_range("2023-03-08", periods=18, freq="B")
    rets = pd.Series([0.001] * 18, index=idx)
    out = compute_stress_metrics(rets, regimes=(DEFAULT_REGIMES[7],))  # svb_crisis only
    assert len(out) == 1
    r = out[0]
    # Constant +0.001 → std=0 → Sharpe falls back to 0 (no risk-adjusted excess)
    assert r["sharpe"] == 0.0
    # Total return = 18 × 0.001 = 0.018
    assert r["total_return"] == pytest.approx(0.018, abs=1e-9)
    # No drawdowns on monotone-up curve
    assert r["max_drawdown"] == 0.0
    # Every day positive → hit rate = 1.0
    assert r["hit_rate"] == 1.0


def test_regime_metrics_hit_rate_and_drawdown():
    """Mixed positive/negative series — verify hit rate + DD."""
    idx = pd.date_range("2022-01-03", periods=20, freq="B")
    # Alternating +0.01 / -0.005 → 50% hit rate
    pattern = np.array([0.01, -0.005] * 10)
    rets = pd.Series(pattern, index=idx)
    out = compute_stress_metrics(rets, regimes=(DEFAULT_REGIMES[6],))  # inflation_bear
    assert len(out) == 1
    r = out[0]
    assert r["hit_rate"] == pytest.approx(0.5)
    # Some drawdown must exist (negative days follow positive ones)
    assert r["max_drawdown"] is not None and r["max_drawdown"] <= 0
    # Total = 10 * 0.01 + 10 * -0.005 = 0.05
    assert r["total_return"] == pytest.approx(0.05, abs=1e-9)


def test_empty_input_returns_empty_list():
    assert compute_stress_metrics(pd.Series(dtype=float)) == []


def test_non_datetime_index_is_coerced():
    """Saved alphas may carry string-date indices."""
    idx_str = pd.date_range("2020-02-19", periods=20, freq="B").strftime("%Y-%m-%d")
    rets = pd.Series([0.001] * 20, index=idx_str)
    out = compute_stress_metrics(rets)
    # Must produce at least one regime hit (COVID windows are in this range)
    assert len(out) >= 1


def test_results_ordered_by_regime_definition():
    rng = np.random.default_rng(2)
    rets = pd.Series(
        rng.standard_normal(1500) * 0.005, index=pd.date_range("2018-01-01", periods=1500, freq="B")
    )
    out = compute_stress_metrics(rets)
    # Default regime order: GFC, euro, china, vol_2018, covid_crash, covid_recovery, inflation, svb
    # The 2018-onward data should hit vol_q4_2018 → covid → inflation → svb in that order
    names = [r["name"] for r in out]
    expected = ["vol_q4_2018", "covid_crash", "covid_recovery", "inflation_bear", "svb_crisis"]
    # Filter our output to those expected names — should be in same order
    filtered = [n for n in names if n in expected]
    assert filtered == expected


def test_regime_severity_classification():
    assert regime_severity({"sharpe": 1.5}) == "good"
    assert regime_severity({"sharpe": 0.0}) == "warn"
    assert regime_severity({"sharpe": -1.0}) == "bad"
    assert regime_severity({"sharpe": None}) == "warn"


def test_regime_dataclass_immutability():
    r = Regime("test", "Test", "2020-01-01", "2020-12-31")
    with pytest.raises(Exception):
        r.name = "changed"  # type: ignore[misc]


def test_custom_regime_list_used():
    """Caller can override the default regime list."""
    custom = (Regime("only_one", "Only", "2024-01-01", "2024-06-30"),)
    rets = pd.Series([0.001] * 200, index=pd.date_range("2024-01-01", periods=200, freq="B"))
    out = compute_stress_metrics(rets, regimes=custom)
    names = {r["name"] for r in out}
    assert names == {"only_one"}


def test_min_regime_days_constant_is_sensible():
    """Sanity check — the threshold is small but non-trivial."""
    assert 3 <= MIN_REGIME_DAYS <= 20
