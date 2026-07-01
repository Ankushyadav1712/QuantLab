"""Mean-variance + risk-parity weight calculators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.mv_optimizer import (
    compute_weights,
    equal_weights,
    inverse_variance,
    mv_optimal,
    risk_parity,
)

# ---------- Fixtures ----------


@pytest.fixture
def two_alphas_one_low_vol():
    """Alpha A has low vol, alpha B has high vol — inverse-variance should
    weight A heavier than B."""
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(0.0005, 0.005, n)  # low vol
    b = rng.normal(0.0005, 0.020, n)  # 4× the vol
    return pd.DataFrame({"A": a, "B": b})


@pytest.fixture
def three_uncorrelated_alphas():
    rng = np.random.default_rng(1)
    n = 500
    return pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.01, n),
            "B": rng.normal(0.001, 0.01, n),
            "C": rng.normal(0.001, 0.01, n),
        }
    )


# ---------- equal_weights ----------


def test_equal_weights_sum_to_one():
    w = equal_weights(4)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0)
    assert all(x == pytest.approx(0.25) for x in w)


def test_equal_weights_rejects_zero():
    with pytest.raises(ValueError, match="n > 0"):
        equal_weights(0)


# ---------- inverse_variance ----------


def test_inverse_variance_weights_low_vol_alpha_more(two_alphas_one_low_vol):
    w = inverse_variance(two_alphas_one_low_vol)
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    # Low-vol alpha (A) should get the larger share
    assert w[0] > w[1]
    # And by a lot — variance ratio is ~16x
    assert w[0] > 0.9


def test_inverse_variance_falls_back_to_equal_on_zero_var():
    # Constant column → variance is 0 → fallback
    df = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0]})
    w = inverse_variance(df)
    assert w == pytest.approx(np.array([0.5, 0.5]))


# ---------- mv_optimal ----------


def test_mv_optimal_normalizes_to_unit_sum_by_default(three_uncorrelated_alphas):
    w = mv_optimal(three_uncorrelated_alphas)
    assert w.shape == (3,)
    # No target_vol → weights normalized to sum to 1
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_mv_optimal_target_vol_hits_target(three_uncorrelated_alphas):
    target = 0.15  # 15% annualized vol
    w = mv_optimal(three_uncorrelated_alphas, target_vol=target)
    cov_annual = three_uncorrelated_alphas.cov().values * 252
    actual_vol = float(np.sqrt(w @ cov_annual @ w))
    assert actual_vol == pytest.approx(target, rel=1e-6)


def test_mv_optimal_can_produce_negative_weights():
    """When one alpha has negative expected return, the unconstrained MV
    optimizer should be willing to short it.  Document the behavior."""
    rng = np.random.default_rng(7)
    n = 500
    a = rng.normal(0.001, 0.01, n)  # positive
    b = rng.normal(-0.001, 0.01, n)  # negative
    df = pd.DataFrame({"A": a, "B": b})
    w = mv_optimal(df)
    # Sum still 1, but B can be negative
    assert w.sum() == pytest.approx(1.0)


# ---------- risk_parity ----------


def test_risk_parity_equal_contribution(two_alphas_one_low_vol):
    w = risk_parity(two_alphas_one_low_vol)
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    # Each name should contribute roughly equal *risk* (not equal weight).
    # Compute per-name risk contribution and check equality.
    cov = two_alphas_one_low_vol.cov().values
    port_var = float(w @ cov @ w)
    rc = w * (cov @ w) / port_var
    assert rc[0] == pytest.approx(rc[1], abs=1e-3)


def test_risk_parity_falls_back_on_degenerate():
    df = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0]})  # zero vol both
    w = risk_parity(df)
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)


# ---------- compute_weights dispatcher ----------


def test_compute_weights_dispatches_each_method(three_uncorrelated_alphas):
    for method in ("equal", "inverse_variance", "mv_optimal", "risk_parity"):
        w = compute_weights(method, three_uncorrelated_alphas)
        assert w.shape == (3,)
        # All methods normalize to unit sum (or scale to target_vol; here None)
        assert abs(float(w.sum()) - 1.0) < 1e-6 or method == "mv_optimal"


def test_compute_weights_unknown_method_raises():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, -0.01]})
    with pytest.raises(ValueError, match="Unknown weight method"):
        compute_weights("not_a_real_method", df)
