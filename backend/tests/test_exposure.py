"""Sector + size exposure analytics — synthetic-data sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.exposure import compute_sector_exposure, compute_size_exposure

# ---------- Helpers ----------


def _frame(n_days: int = 30, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["A", "B", "C", "D"]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(0.0, index=dates, columns=tickers)


def _gics(mapping: dict[str, str]) -> dict[str, dict[str, str | None]]:
    return {t: {"sector": s} for t, s in mapping.items()}


# ---------- Sector exposure ----------


def test_sector_exposure_all_long_one_sector():
    weights = _frame(tickers=["A", "B", "C", "D"])
    # All 0.25 long in A,B (Tech), zero elsewhere
    weights[["A", "B"]] = 0.5
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_sector_exposure(weights, gics)
    assert out["by_sector"]["Tech"]["avg_net"] == pytest.approx(1.0)
    assert out["by_sector"]["Energy"]["avg_net"] == pytest.approx(0.0)
    assert out["headline"]["max_long_sector"] == "Tech"
    assert out["headline"]["max_long_exposure"] == pytest.approx(1.0)


def test_sector_exposure_neutral_within_sector():
    weights = _frame(tickers=["A", "B", "C", "D"])
    # Tech: +0.5 in A, -0.5 in B → net 0, gross 1
    weights["A"] = 0.5
    weights["B"] = -0.5
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_sector_exposure(weights, gics)
    assert out["by_sector"]["Tech"]["avg_net"] == pytest.approx(0.0)
    assert out["by_sector"]["Tech"]["avg_gross"] == pytest.approx(1.0)


def test_sector_exposure_long_short_sectors():
    weights = _frame(tickers=["A", "B", "C", "D"])
    weights[["A", "B"]] = 0.4  # long Tech
    weights[["C", "D"]] = -0.3  # short Energy
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_sector_exposure(weights, gics)
    assert out["headline"]["max_long_sector"] == "Tech"
    assert out["headline"]["max_short_sector"] == "Energy"
    assert out["headline"]["max_long_exposure"] == pytest.approx(0.8)
    assert out["headline"]["max_short_exposure"] == pytest.approx(-0.6)


def test_sector_exposure_missing_gics_bucketed_as_unknown():
    weights = _frame(tickers=["A", "B", "C"])
    weights["A"] = 0.5
    weights["B"] = 0.5
    weights["C"] = 0.5
    gics = _gics({"A": "Tech", "B": "Tech"})  # C absent
    out = compute_sector_exposure(weights, gics)
    assert "Unknown" in out["by_sector"]
    assert out["by_sector"]["Unknown"]["n_tickers"] == 1


def test_sector_exposure_returns_none_on_empty_weights():
    assert compute_sector_exposure(pd.DataFrame(), {"A": {"sector": "Tech"}}) is None


def test_sector_exposure_returns_none_without_gics_map():
    weights = _frame()
    weights["A"] = 0.5
    assert compute_sector_exposure(weights, None) is None


def test_sector_exposure_records_ticker_counts():
    weights = _frame(tickers=["A", "B", "C", "D", "E"])
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Energy"})
    out = compute_sector_exposure(weights, gics)
    assert out["by_sector"]["Tech"]["n_tickers"] == 3
    assert out["by_sector"]["Energy"]["n_tickers"] == 2


# ---------- Size exposure ----------


def test_size_exposure_long_largest_caps():
    """Weights ranked the same as market_cap → corr ≈ 1.0."""
    rng = np.random.default_rng(0)
    n_days, n_t = 100, 10
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_t)]
    mcap_row = rng.uniform(1e8, 1e12, n_t)
    mcap = pd.DataFrame(np.tile(mcap_row, (n_days, 1)), index=dates, columns=tickers)
    # weights perfectly aligned with log(mcap)
    weights_row = np.log(mcap_row)
    weights = pd.DataFrame(np.tile(weights_row, (n_days, 1)), index=dates, columns=tickers)
    out = compute_size_exposure(weights, mcap)
    assert out["size_corr"] == pytest.approx(1.0, abs=1e-6)
    assert out["n_days"] == n_days


def test_size_exposure_short_largest_caps_is_negative():
    rng = np.random.default_rng(1)
    n_days, n_t = 80, 8
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_t)]
    mcap_row = rng.uniform(1e8, 1e12, n_t)
    mcap = pd.DataFrame(np.tile(mcap_row, (n_days, 1)), index=dates, columns=tickers)
    weights = pd.DataFrame(np.tile(-np.log(mcap_row), (n_days, 1)), index=dates, columns=tickers)
    out = compute_size_exposure(weights, mcap)
    assert out["size_corr"] == pytest.approx(-1.0, abs=1e-6)


def test_size_exposure_random_weights_near_zero():
    rng = np.random.default_rng(7)
    n_days, n_t = 300, 30
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_t)]
    weights = pd.DataFrame(rng.standard_normal((n_days, n_t)), index=dates, columns=tickers)
    mcap = pd.DataFrame(rng.uniform(1e8, 1e12, (n_days, n_t)), index=dates, columns=tickers)
    out = compute_size_exposure(weights, mcap)
    assert abs(out["size_corr"]) < 0.1


def test_size_exposure_handles_missing_inputs():
    assert compute_size_exposure(pd.DataFrame(), None) is None
    weights = _frame()
    weights["A"] = 0.5
    assert compute_size_exposure(weights, None) is None
    assert compute_size_exposure(weights, pd.DataFrame()) is None


def test_size_exposure_too_few_tickers():
    # Need at least 3 tickers for a daily correlation
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    weights = pd.DataFrame({"A": [0.5] * 10, "B": [0.3] * 10}, index=dates)
    mcap = pd.DataFrame({"A": [1e9] * 10, "B": [5e8] * 10}, index=dates)
    assert compute_size_exposure(weights, mcap) is None


def test_size_exposure_approximation_flag_passed_through():
    rng = np.random.default_rng(0)
    n_days, n_t = 50, 8
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_t)]
    weights = pd.DataFrame(rng.standard_normal((n_days, n_t)), index=dates, columns=tickers)
    close = pd.DataFrame(rng.uniform(10, 500, (n_days, n_t)), index=dates, columns=tickers)
    out = compute_size_exposure(weights, close, is_approximation=True)
    assert out["is_approximation"] is True
