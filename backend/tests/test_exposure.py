"""Sector + size exposure analytics — synthetic-data sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.exposure import (
    compute_market_cap_distribution,
    compute_sector_exposure,
    compute_sector_exposure_timeseries,
    compute_size_exposure,
)

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


# ---------- Market-cap distribution (PDF Section 5.3.4) ----------


def _build_book(
    n_days: int = 30,
    n_tickers: int = 10,
    *,
    long_indices: list[int] | None = None,
    short_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct (weights, market_cap) matrices where ticker index → market-cap
    rank is fixed (T0 = smallest, T9 = largest), and the caller picks which
    tickers are long vs short.  Used by the bias-detection tests below."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    # Market cap monotonically increasing in ticker index — T0 = $100M ... T9 = $1T
    mcap_values = np.geomspace(1e8, 1e12, n_tickers)
    mcap = pd.DataFrame(np.tile(mcap_values, (n_days, 1)), index=dates, columns=tickers)
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    if long_indices:
        for i in long_indices:
            weights.iloc[:, i] = 1e6 / len(long_indices)
    if short_indices:
        for i in short_indices:
            weights.iloc[:, i] = -1e6 / len(short_indices)
    return weights, mcap


def test_mcap_distribution_long_large_short_small_has_positive_tilt():
    """Classic large-cap-long / small-cap-short carry: longs in top deciles,
    shorts in bottom deciles → decile_tilt should be strongly positive."""
    # Long the top 3 tickers (largest caps), short the bottom 3 (smallest)
    weights, mcap = _build_book(long_indices=[7, 8, 9], short_indices=[0, 1, 2])
    out = compute_market_cap_distribution(weights, mcap)
    assert out is not None
    assert out["long_avg_decile"] > 7.0, f"Expected long_avg ≈ 8, got {out['long_avg_decile']}"
    assert out["short_avg_decile"] < 2.0, f"Expected short_avg ≈ 1, got {out['short_avg_decile']}"
    assert out["decile_tilt"] > 4.0  # Clearly large-cap-long carry


def test_mcap_distribution_long_small_short_large_has_negative_tilt():
    """Opposite direction: small-cap-long / large-cap-short — decile_tilt
    should be strongly negative."""
    weights, mcap = _build_book(long_indices=[0, 1, 2], short_indices=[7, 8, 9])
    out = compute_market_cap_distribution(weights, mcap)
    assert out is not None
    assert out["decile_tilt"] < -4.0


def test_mcap_distribution_uniform_book_has_near_zero_tilt():
    """An evenly-spread book across the size spectrum should land both
    long_avg and short_avg near the median decile (~4.5)."""
    weights, mcap = _build_book(long_indices=[1, 3, 5, 7], short_indices=[0, 2, 4, 6])
    out = compute_market_cap_distribution(weights, mcap)
    assert out is not None
    assert abs(out["decile_tilt"]) < 2.0


def test_mcap_distribution_bucket_counts_match_n_buckets():
    weights, mcap = _build_book(long_indices=[5], short_indices=[4])
    out = compute_market_cap_distribution(weights, mcap)
    assert len(out["long_per_bucket"]) == out["n_buckets"]
    assert len(out["short_per_bucket"]) == out["n_buckets"]
    # Each side sums to ~1 (it's a per-bucket fraction of that side's book)
    assert sum(out["long_per_bucket"]) == pytest.approx(1.0, abs=1e-9)
    assert sum(out["short_per_bucket"]) == pytest.approx(1.0, abs=1e-9)


def test_mcap_distribution_returns_none_on_empty_inputs():
    assert compute_market_cap_distribution(pd.DataFrame(), pd.DataFrame()) is None
    assert compute_market_cap_distribution(pd.DataFrame(), None) is None


def test_mcap_distribution_too_few_tickers_returns_none():
    """Default n_buckets=10 needs at least 10 columns to produce one entry
    per bucket; fewer than that should bail cleanly rather than crash."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    weights = pd.DataFrame({"A": [1.0] * 5, "B": [-1.0] * 5}, index=dates)
    mcap = pd.DataFrame({"A": [1e9] * 5, "B": [5e8] * 5}, index=dates)
    assert compute_market_cap_distribution(weights, mcap) is None


def test_mcap_distribution_long_only_book_has_zero_short_buckets():
    """A long-only book → short_per_bucket should be all zeros, short_avg=0."""
    weights, mcap = _build_book(long_indices=[5, 6, 7], short_indices=None)
    out = compute_market_cap_distribution(weights, mcap)
    assert out is not None
    assert all(v == 0.0 for v in out["short_per_bucket"])
    assert out["short_avg_decile"] == 0.0


def test_mcap_distribution_approximation_flag_passes_through():
    weights, mcap = _build_book(long_indices=[8], short_indices=[1])
    out = compute_market_cap_distribution(weights, mcap, is_approximation=True)
    assert out["is_approximation"] is True


def test_mcap_distribution_custom_bucket_count():
    weights, mcap = _build_book(n_tickers=12, long_indices=[10, 11], short_indices=[0, 1])
    out = compute_market_cap_distribution(weights, mcap, n_buckets=4)
    assert out["n_buckets"] == 4
    assert len(out["long_per_bucket"]) == 4
    # With 12 tickers across 4 buckets and longs in indices 10,11 → top bucket
    assert out["long_per_bucket"][3] == pytest.approx(1.0, abs=1e-9)


# ---------- Sector × time heatmap (PDF Section 5.3.2) ----------


def test_sector_timeseries_basic_shape():
    """Output should be n_sectors × n_buckets, with stable alphabetical row order."""
    weights = _frame(n_days=20, tickers=["A", "B", "C", "D"])
    weights[["A", "B"]] = 0.5
    weights[["C", "D"]] = -0.5
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_sector_exposure_timeseries(weights, gics)
    assert out is not None
    # Sectors sorted alphabetically — Energy before Tech
    assert out["sectors"] == ["Energy", "Tech"]
    # 20 days, default max_buckets=120, so bucket_size=1 → 20 buckets
    assert len(out["dates"]) == 20
    assert len(out["matrix"]) == 2
    assert len(out["matrix"][0]) == 20


def test_sector_timeseries_values_match_aggregation():
    """A long-Tech / short-Energy book should have positive Tech values and
    negative Energy values in every column."""
    weights = _frame(n_days=10, tickers=["A", "B", "C", "D"])
    weights[["A", "B"]] = 0.3  # long Tech
    weights[["C", "D"]] = -0.2  # short Energy
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_sector_exposure_timeseries(weights, gics)
    energy_row = out["matrix"][out["sectors"].index("Energy")]
    tech_row = out["matrix"][out["sectors"].index("Tech")]
    # Tech net = +0.6/day, Energy net = -0.4/day
    assert all(v == pytest.approx(0.6, abs=1e-9) for v in tech_row)
    assert all(v == pytest.approx(-0.4, abs=1e-9) for v in energy_row)


def test_sector_timeseries_downsamples_long_backtests():
    """500-day backtest with max_buckets=100 should bucket into ~100 entries
    (501/100 → bucket_size=6 → ceil(501/6)=84 buckets)."""
    weights = _frame(n_days=500, tickers=["A", "B"])
    weights["A"] = 0.5
    weights["B"] = -0.5
    gics = _gics({"A": "Tech", "B": "Energy"})
    out = compute_sector_exposure_timeseries(weights, gics, max_buckets=100)
    # Bucket size should be 5 (500/100), giving 100 buckets
    assert out["n_periods_per_bucket"] == 5
    assert len(out["dates"]) == 100
    assert len(out["matrix"][0]) == 100


def test_sector_timeseries_keeps_daily_for_short_backtests():
    """If n_days <= max_buckets, no downsampling — bucket_size stays at 1."""
    weights = _frame(n_days=30, tickers=["A"])
    weights["A"] = 1.0
    gics = _gics({"A": "Tech"})
    out = compute_sector_exposure_timeseries(weights, gics, max_buckets=120)
    assert out["n_periods_per_bucket"] == 1
    assert len(out["dates"]) == 30


def test_sector_timeseries_returns_none_on_empty():
    assert compute_sector_exposure_timeseries(pd.DataFrame(), None) is None
    assert compute_sector_exposure_timeseries(_frame(), None) is None


def test_sector_timeseries_unknown_bucket_for_missing_gics():
    """Tickers with no GICS row land in 'Unknown' (same as the static
    sector_exposure behaviour)."""
    weights = _frame(n_days=5, tickers=["A", "ZZZ"])
    weights["A"] = 0.5
    weights["ZZZ"] = -0.3
    # Only A has a GICS entry
    gics = _gics({"A": "Tech"})
    out = compute_sector_exposure_timeseries(weights, gics)
    assert "Unknown" in out["sectors"]
    unknown_row = out["matrix"][out["sectors"].index("Unknown")]
    assert all(v == pytest.approx(-0.3, abs=1e-9) for v in unknown_row)


def test_sector_timeseries_captures_regime_change():
    """The heatmap's whole point — sector-timing patterns.  Long Tech for
    first half, short Tech for second half → row should flip sign."""
    n = 20
    weights = _frame(n_days=n, tickers=["A"])
    weights.iloc[: n // 2, weights.columns.get_loc("A")] = 1.0  # long Tech first half
    weights.iloc[n // 2 :, weights.columns.get_loc("A")] = -1.0  # short Tech second half
    gics = _gics({"A": "Tech"})
    out = compute_sector_exposure_timeseries(weights, gics)
    tech_row = out["matrix"][out["sectors"].index("Tech")]
    # First half should be ≈ +1, second half ≈ -1
    assert tech_row[0] == pytest.approx(1.0)
    assert tech_row[-1] == pytest.approx(-1.0)
    # Find the flip point
    sign_changes = sum(
        1
        for i in range(1, len(tech_row))
        if tech_row[i] is not None
        and tech_row[i - 1] is not None
        and (tech_row[i] >= 0) != (tech_row[i - 1] >= 0)
    )
    assert sign_changes >= 1, "Expected at least one sign change in the row"
