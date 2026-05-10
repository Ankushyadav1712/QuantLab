"""Tests for the FF5 factor decomposition.

We don't hit the network in unit tests — instead we synthesize a small
ff5 DataFrame and a strategy whose returns are a known linear combo of those
factors plus pure alpha, then verify the regression recovers the loadings.
"""

import numpy as np
import pandas as pd
from analytics.factor_decomp import FactorDecomposition


def _synth_ff5(n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0005, 0.01, n),
            "SMB": rng.normal(0.0, 0.005, n),
            "HML": rng.normal(0.0, 0.005, n),
            "RMW": rng.normal(0.0, 0.004, n),
            "CMA": rng.normal(0.0, 0.004, n),
            "RF": np.full(n, 0.00005),
        },
        index=dates,
    )


def test_recovers_known_loadings():
    """If strategy returns = 0.8·Mkt + 0.3·SMB + alpha + noise, the
    regression should recover those betas (to within ~0.05) and a positive
    alpha."""
    ff5 = _synth_ff5(n=400, seed=7)
    rng = np.random.default_rng(11)

    daily_alpha = 0.0004  # ~10% annualized pure alpha
    strategy = (
        0.8 * ff5["Mkt-RF"]
        + 0.3 * ff5["SMB"]
        + daily_alpha
        + rng.normal(0.0, 0.001, len(ff5))  # tiny noise
        + ff5["RF"]  # gross-of-RF
    )

    out = FactorDecomposition().compute(
        daily_returns=strategy.tolist(),
        dates=[d.strftime("%Y-%m-%d") for d in strategy.index],
        ff5=ff5,
    )
    assert out is not None

    assert out["loadings"]["market"]["beta"] == pytest_approx(0.8, abs=0.05)
    assert out["loadings"]["size"]["beta"] == pytest_approx(0.3, abs=0.05)
    # Other factor loadings should be near zero
    for k in ("value", "profitability", "investment"):
        assert abs(out["loadings"][k]["beta"]) < 0.1

    assert out["alpha_annualized"] == pytest_approx(daily_alpha * 252, rel=0.20)
    assert out["r_squared"] > 0.8  # most variance explained by Mkt + SMB


def test_returns_none_when_too_few_observations():
    ff5 = _synth_ff5(n=300)
    out = FactorDecomposition().compute(
        daily_returns=[0.001] * 30,
        dates=[d.strftime("%Y-%m-%d") for d in ff5.index[:30]],
        ff5=ff5,
    )
    assert out is None


def test_returns_none_when_ff5_missing():
    out = FactorDecomposition().compute(
        daily_returns=[0.001] * 200,
        dates=[d.strftime("%Y-%m-%d") for d in pd.date_range("2023-01-02", periods=200, freq="B")],
        ff5=pd.DataFrame(),
    )
    assert out is None


def test_pure_market_beta_alpha_near_zero():
    """A strategy that's literally 1.0× the market should have pure alpha ≈ 0
    and `factor_share` close to 1.0 (almost all variance explained by Mkt)."""
    ff5 = _synth_ff5(n=400, seed=42)
    strategy = ff5["Mkt-RF"] + ff5["RF"]  # pure beta of 1.0

    out = FactorDecomposition().compute(
        daily_returns=strategy.tolist(),
        dates=[d.strftime("%Y-%m-%d") for d in strategy.index],
        ff5=ff5,
    )
    assert out is not None
    assert abs(out["alpha_annualized"]) < 0.005
    assert out["loadings"]["market"]["beta"] == pytest_approx(1.0, abs=0.05)
    assert out["factor_share"] > 0.98


# A tiny shim so we don't have to import pytest at module top to use approx
def pytest_approx(value, rel=None, abs=None):
    import pytest

    return pytest.approx(value, rel=rel, abs=abs)
