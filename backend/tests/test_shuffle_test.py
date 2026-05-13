"""Shuffle leakage test — verdict logic + permutation correctness.

We don't run the full backtester here (too slow for a unit test).  Instead
we inject lightweight factories that simulate the three relevant cases:

  1. Clean alpha: high real Sharpe, shuffle distribution near zero
  2. Pure-noise alpha: real Sharpe inside the shuffle distribution
  3. Leaky alpha: high real Sharpe AND high shuffle mean (structural exploit)

This isolates the verdict logic from the (slow, real-data) backtest path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from analytics.shuffle_test import _sharpe_from_returns, _shuffle_data, run_shuffle_test

# ---------- Test infrastructure: a fake backtester + evaluator ----------


@dataclass
class _FakeResult:
    daily_returns: list[float]


class _FakeBacktester:
    """Returns a precomputed daily-returns series, seeded externally."""

    def __init__(self, returns: list[float]):
        self._returns = returns

    def run(self, alpha, cfg):
        return _FakeResult(self._returns), None


class _FakeEvaluator:
    def __init__(self, data):
        self.data = data

    def evaluate(self, expression):
        return None  # the fake backtester ignores it


def _make_factories_v2(
    real_sharpe: float,
    shuffle_mean: float,
    shuffle_std: float,
    *,
    n_days: int = 2000,  # long enough that empirical Sharpe ≈ target
    raise_on_shuffle: bool = False,
):
    """Counter-based factories: the *first* backtester invocation returns
    the "real" Sharpe; every subsequent invocation returns a "shuffled"
    Sharpe drawn from N(shuffle_mean, shuffle_std).

    Long n_days keeps empirical Sharpe noise low (≈ 1/√n_days) so the
    target values manifest reliably.
    """
    counter = {"n": 0}

    def evaluator_factory(d):
        return _FakeEvaluator(d)

    def backtester_factory(d):
        idx = counter["n"]
        counter["n"] += 1
        if idx == 0:
            target = real_sharpe
            rng = np.random.default_rng(42)
        else:
            if raise_on_shuffle:
                raise ValueError("simulated backtest failure")
            rng = np.random.default_rng(1000 + idx)
            target = shuffle_mean + shuffle_std * rng.standard_normal()

        # Build a returns array whose *empirical* Sharpe is exactly ``target``.
        # Naïve "scale + shift" of standard normals leaves residual noise from
        # the finite sample (e.g. seed 42's first 2000 draws have mean ≈ -0.04
        # which would knock a target=2.0 down to ~1.1).  We normalise away
        # that sample noise so the test's verdict checks are deterministic.
        raw = rng.standard_normal(n_days)
        raw = (raw - raw.mean()) / raw.std(ddof=1)  # mean=0, std=1
        daily_mean = target / math.sqrt(252)
        returns = (raw + daily_mean).tolist()
        return _FakeBacktester(returns)

    return evaluator_factory, backtester_factory


def _make_data(n_days: int = 200):
    """Minimal data dict — only `close` is required for permutation length."""
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    arr = np.random.default_rng(0).standard_normal((n_days, 5))
    df = pd.DataFrame(arr, index=dates, columns=list("ABCDE"))
    return {"close": df}


# ---------- _sharpe_from_returns ----------


def test_sharpe_basic_correctness():
    rng = np.random.default_rng(0)
    # mean 0.001, std 0.01 → annualised ≈ 1.0 * sqrt(252) / 1.0 ≈ 1.58
    returns = (rng.standard_normal(1000) * 0.01 + 0.001).tolist()
    s = _sharpe_from_returns(returns)
    assert s is not None
    # Loose: depends on RNG, but should be in (0, 3) for this many samples
    assert 0.5 < s < 3.0


def test_sharpe_zero_variance_returns_none():
    s = _sharpe_from_returns([0.001] * 100)
    assert s is None


def test_sharpe_handles_nans():
    s = _sharpe_from_returns([0.01, math.nan, 0.02, math.nan, -0.005, 0.015])
    assert s is not None


# ---------- _shuffle_data ----------


def test_shuffle_preserves_cross_section():
    """A row permutation must keep each row's cross-section intact —
    only the date axis is scrambled."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    close = pd.DataFrame(rng.standard_normal((10, 5)), index=dates, columns=list("ABCDE"))
    volume = pd.DataFrame(rng.standard_normal((10, 5)), index=dates, columns=list("ABCDE"))
    data = {"close": close, "volume": volume}

    perm = np.array([3, 1, 4, 0, 2, 7, 9, 5, 8, 6])
    shuffled = _shuffle_data(data, perm)

    # Each row of shuffled["close"] should match SOME row of original close
    # (specifically: shuffled[i] = original[perm[i]] for all fields).
    for i in range(10):
        np.testing.assert_array_equal(shuffled["close"].iloc[i].values, close.iloc[perm[i]].values)
        np.testing.assert_array_equal(
            shuffled["volume"].iloc[i].values, volume.iloc[perm[i]].values
        )
    # And the index itself is unchanged (still 2024-01-02, etc.)
    assert (shuffled["close"].index == close.index).all()


def test_shuffle_handles_mismatched_field_lengths():
    """Macro fields can be shorter than OHLCV; shuffler shouldn't crash."""
    dates_full = pd.date_range("2024-01-02", periods=10, freq="B")
    dates_short = pd.date_range("2024-01-02", periods=5, freq="B")
    close = pd.DataFrame(np.zeros((10, 3)), index=dates_full, columns=list("ABC"))
    macro = pd.DataFrame(np.zeros((5, 3)), index=dates_short, columns=list("ABC"))
    out = _shuffle_data({"close": close, "macro": macro}, np.arange(10))
    assert out["close"].shape == (10, 3)
    assert out["macro"].shape == (5, 3)


# ---------- run_shuffle_test verdict logic ----------


def _cfg():
    return type("Cfg", (), {})()  # backtester factory ignores it


def test_clean_alpha_verdict_real_signal():
    """High real Sharpe vs near-zero shuffle distribution → real-signal."""
    eval_fac, bt_fac = _make_factories_v2(real_sharpe=2.5, shuffle_mean=0.0, shuffle_std=0.3)
    out = run_shuffle_test(
        "fake",
        data=_make_data(),
        backtester_factory=bt_fac,
        evaluator_factory=eval_fac,
        config=_cfg(),
        n_shuffles=30,
        seed=0,
    )
    assert out.verdict == "real-signal"
    assert out.p_value is not None and out.p_value < 0.05
    assert out.mean_shuffled is not None and abs(out.mean_shuffled) < 0.5


def test_noise_alpha_verdict_indistinguishable():
    """Real Sharpe inside the shuffle distribution → indistinguishable-from-noise."""
    eval_fac, bt_fac = _make_factories_v2(real_sharpe=0.1, shuffle_mean=0.0, shuffle_std=0.5)
    out = run_shuffle_test(
        "fake",
        data=_make_data(),
        backtester_factory=bt_fac,
        evaluator_factory=eval_fac,
        config=_cfg(),
        n_shuffles=30,
        seed=0,
    )
    assert out.verdict in ("indistinguishable-from-noise", "borderline")


def test_leaky_alpha_verdict_leakage_suspected():
    """Real Sharpe high AND shuffle mean high → structural artifact."""
    eval_fac, bt_fac = _make_factories_v2(real_sharpe=2.0, shuffle_mean=1.2, shuffle_std=0.3)
    out = run_shuffle_test(
        "fake",
        data=_make_data(),
        backtester_factory=bt_fac,
        evaluator_factory=eval_fac,
        config=_cfg(),
        n_shuffles=30,
        seed=0,
    )
    # When both real AND shuffled-mean are high, we flag leakage even
    # though the p-value is low (real beats most shuffles).
    assert out.verdict == "leakage-suspected"


def test_p_value_bounded():
    eval_fac, bt_fac = _make_factories_v2(real_sharpe=0.5, shuffle_mean=0.3, shuffle_std=0.4)
    out = run_shuffle_test(
        "fake",
        data=_make_data(),
        backtester_factory=bt_fac,
        evaluator_factory=eval_fac,
        config=_cfg(),
        n_shuffles=30,
        seed=0,
    )
    # p-value strictly in (0, 1] due to the +1 correction
    assert out.p_value is not None
    assert 0 < out.p_value <= 1.0


def test_too_few_shuffles_verdict():
    """When most shuffles fail (the backtester errors), we shouldn't claim
    a verdict — just report 'too-few-shuffles-completed'."""
    eval_fac, bt_fac = _make_factories_v2(
        real_sharpe=2.0, shuffle_mean=0.0, shuffle_std=0.3, raise_on_shuffle=True
    )
    out = run_shuffle_test(
        "fake",
        data=_make_data(),
        backtester_factory=bt_fac,
        evaluator_factory=eval_fac,
        config=_cfg(),
        n_shuffles=10,
        seed=0,
    )
    assert out.verdict == "too-few-shuffles-completed"
    assert out.n_shuffles_failed == 10


def test_to_dict_safe_for_json():
    eval_fac, bt_fac = _make_factories_v2(real_sharpe=1.0, shuffle_mean=0.0, shuffle_std=0.3)
    out = run_shuffle_test(
        "fake",
        data=_make_data(),
        backtester_factory=bt_fac,
        evaluator_factory=eval_fac,
        config=_cfg(),
        n_shuffles=10,
        seed=0,
    )
    d = out.to_dict()
    import json

    s = json.dumps(d)  # must be serialisable
    assert "verdict" in s
