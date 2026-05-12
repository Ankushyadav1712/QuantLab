"""Tier-1 signal-quality additions to performance.py.

The four helpers tested here (`_tail_ratio`, `_positive_months_pct`,
`_drawdown_durations`, `_fitness_wq`) are private but worth pinning down
with focused unit tests since they feed the user-visible Signal Quality
panel in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.performance import (
    _drawdown_durations,
    _fitness_wq,
    _positive_months_pct,
    _tail_ratio,
)

# ---------- Tail ratio ----------


def test_tail_ratio_symmetric_distribution_is_near_one():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.standard_normal(1000))
    tr = _tail_ratio(rets)
    assert 0.8 < tr < 1.25


def test_tail_ratio_right_skewed_is_above_one():
    # Big positive tail, small negative tail
    arr = np.concatenate([np.full(95, -0.001), np.full(5, 0.10)])
    rets = pd.Series(arr)
    tr = _tail_ratio(rets)
    assert tr > 3.0


def test_tail_ratio_left_skewed_is_below_one():
    # Big negative tail, small positive tail (crash-like distribution)
    arr = np.concatenate([np.full(5, -0.10), np.full(95, 0.001)])
    rets = pd.Series(arr)
    tr = _tail_ratio(rets)
    assert tr < 0.4


def test_tail_ratio_too_few_returns():
    assert _tail_ratio(pd.Series([0.01, -0.02, 0.03])) is None


# ---------- Positive months % ----------


def test_positive_months_all_positive():
    # 6 months of positive daily returns
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rets = pd.Series(np.full(120, 0.001), index=dates)
    assert _positive_months_pct(rets) == 1.0


def test_positive_months_all_negative():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rets = pd.Series(np.full(120, -0.001), index=dates)
    assert _positive_months_pct(rets) == 0.0


def test_positive_months_handles_empty():
    assert _positive_months_pct(pd.Series(dtype=float)) is None


def test_positive_months_mixed():
    # Hand-build: month 1 positive, month 2 negative, month 3 positive
    idx = pd.DatetimeIndex(
        ["2024-01-15", "2024-01-20", "2024-02-10", "2024-02-25", "2024-03-05", "2024-03-22"]
    )
    rets = pd.Series([0.01, 0.02, -0.01, -0.02, 0.03, 0.01], index=idx)
    pct = _positive_months_pct(rets)
    assert pct == pytest.approx(2.0 / 3.0)


# ---------- Drawdown duration ----------


def test_dd_duration_no_drawdown():
    # Strictly monotonically increasing equity → never underwater
    eq = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04])
    out = _drawdown_durations(eq)
    assert out["max_dd_days"] == 0
    assert out["avg_dd_days"] == 0.0
    assert out["current_dd_days"] == 0


def test_dd_duration_single_known_run():
    # Equity goes up, dips for exactly 5 days, recovers
    eq = pd.Series([1.0, 1.10, 1.05, 1.04, 1.03, 1.02, 1.01, 1.11])
    out = _drawdown_durations(eq)
    # Days where equity < running max: indices 2..6 = 5 days
    assert out["max_dd_days"] == 5
    assert out["current_dd_days"] == 0


def test_dd_duration_still_underwater_at_end():
    # Equity peaks then never recovers
    eq = pd.Series([1.0, 1.10, 1.05, 1.04, 1.03, 1.02])
    out = _drawdown_durations(eq)
    # 4 consecutive days underwater, still in DD at the last point
    assert out["max_dd_days"] == 4
    assert out["current_dd_days"] == 4


def test_dd_duration_multiple_runs():
    eq = pd.Series([1.0, 1.10, 1.05, 1.04, 1.11, 1.09, 1.08, 1.07, 1.12])
    out = _drawdown_durations(eq)
    # Run 1: 2 days underwater (idx 2..3).  Run 2: 3 days (idx 5..7).
    assert out["max_dd_days"] == 3
    assert out["avg_dd_days"] == pytest.approx(2.5)
    assert out["current_dd_days"] == 0


# ---------- Fitness (WQ-style) ----------


def test_fitness_wq_positive_alpha():
    # sharpe=1.5, annual_return=0.10, turnover_frac=0.5
    # = +1 * sqrt(0.10 / 0.5) * 1.5 = sqrt(0.2) * 1.5 ≈ 0.671
    f = _fitness_wq(1.5, 0.10, 0.5)
    assert f == pytest.approx(0.671, abs=1e-3)


def test_fitness_wq_negative_returns_score_negative():
    f = _fitness_wq(1.0, -0.05, 0.3)
    assert f is not None and f < 0


def test_fitness_wq_low_turnover_floor():
    # turnover below 0.125 floor should be clamped
    f_low = _fitness_wq(1.0, 0.10, 0.01)
    f_floor = _fitness_wq(1.0, 0.10, 0.125)
    assert f_low == pytest.approx(f_floor)


def test_fitness_wq_handles_none_sharpe():
    assert _fitness_wq(None, 0.1, 0.3) is None  # type: ignore[arg-type]
