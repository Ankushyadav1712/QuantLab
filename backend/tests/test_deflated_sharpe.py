"""Deflated Sharpe Ratio — sanity checks against the published formula."""

import math

import pytest
from analytics.deflated_sharpe import _norm_cdf, _norm_ppf, deflated_sharpe

# ---------- Helpers ----------


def test_norm_cdf_known_values():
    # CDF(0) = 0.5, CDF(1.96) ≈ 0.975, CDF(-1.96) ≈ 0.025
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert _norm_cdf(1.96) == pytest.approx(0.9750021, abs=1e-5)
    assert _norm_cdf(-1.96) == pytest.approx(0.0249979, abs=1e-5)


def test_norm_ppf_inverse_cdf():
    for x in (-2.0, -0.5, 0.0, 0.7, 1.5, 2.5):
        p = _norm_cdf(x)
        # Acklam's algorithm is accurate to ~1e-9
        assert _norm_ppf(p) == pytest.approx(x, abs=1e-6)


def test_norm_ppf_out_of_range_returns_nan():
    assert math.isnan(_norm_ppf(0.0))
    assert math.isnan(_norm_ppf(1.0))
    assert math.isnan(_norm_ppf(-0.1))


# ---------- Deflated Sharpe ----------


def test_deflated_sharpe_n_trials_one_collapses_threshold_to_zero():
    """With one trial there's no selection bias — threshold is exactly 0."""
    out = deflated_sharpe(sharpe_annual=1.0, n_trials=1, n_obs=500, skew=0.0, kurt=3.0)
    assert out is not None
    assert out["sharpe_threshold_annualized"] == pytest.approx(0.0, abs=1e-9)
    # Deflated == headline when threshold is zero
    assert out["deflated_sharpe_annualized"] == pytest.approx(1.0, rel=1e-6)


def test_deflated_sharpe_more_trials_higher_threshold():
    """Trying more alphas raises the bar — threshold grows monotonically."""
    sr_kwargs = dict(sharpe_annual=1.0, n_obs=500, skew=0.0, kurt=3.0)
    t1 = deflated_sharpe(n_trials=1, **sr_kwargs)["sharpe_threshold_annualized"]
    t10 = deflated_sharpe(n_trials=10, **sr_kwargs)["sharpe_threshold_annualized"]
    t100 = deflated_sharpe(n_trials=100, **sr_kwargs)["sharpe_threshold_annualized"]
    assert t1 < t10 < t100


def test_deflated_sharpe_more_trials_lower_pvalue():
    """More trials → lower probability the headline beat luck."""
    sr_kwargs = dict(sharpe_annual=1.0, n_obs=500, skew=0.0, kurt=3.0)
    p1 = deflated_sharpe(n_trials=1, **sr_kwargs)["p_value"]
    p100 = deflated_sharpe(n_trials=100, **sr_kwargs)["p_value"]
    assert p1 > p100


def test_deflated_sharpe_high_sharpe_significant():
    """A genuinely strong Sharpe (3.0) over 500 days survives 100 trials."""
    out = deflated_sharpe(sharpe_annual=3.0, n_trials=100, n_obs=500, skew=0.0, kurt=3.0)
    assert out["is_significant"] is True
    assert out["p_value"] > 0.95


def test_deflated_sharpe_returns_none_on_degenerate_input():
    assert deflated_sharpe(None, n_trials=10, n_obs=500) is None
    assert deflated_sharpe(float("nan"), n_trials=10, n_obs=500) is None
    assert deflated_sharpe(1.0, n_trials=10, n_obs=1) is None  # too few obs


def test_deflated_sharpe_negative_skew_lowers_threshold_significance():
    """Crash-prone return distributions (negative skew) hurt the deflated SR
    via the higher-moment correction term."""
    base = deflated_sharpe(sharpe_annual=1.5, n_trials=20, n_obs=500, skew=0.0, kurt=3.0)
    skewed = deflated_sharpe(sharpe_annual=1.5, n_trials=20, n_obs=500, skew=-0.5, kurt=5.0)
    # Negative skew + fat tails → narrower confidence in the same Sharpe,
    # i.e. lower p-value than the symmetric case.
    assert skewed["p_value"] < base["p_value"]
