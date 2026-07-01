"""FRED macro loader — parser + downloader + broadcast helper.

Network is stubbed via the injectable ``fetch_fn`` parameter on
``download_macro`` so tests stay hermetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data.macro import (
    DERIVED_MACRO_BUILDERS,
    FRED_SERIES,
    _parse_fred_csv,
    broadcast_to_matrix,
    download_macro,
)

# ---------- CSV parser ----------


def test_parse_fred_csv_basic():
    csv = "observation_date,VIXCLS\n2024-01-02,13.20\n2024-01-03,12.99\n"
    s = _parse_fred_csv(csv)
    assert isinstance(s, pd.Series)
    assert len(s) == 2
    assert s.iloc[0] == pytest.approx(13.20)
    assert s.iloc[1] == pytest.approx(12.99)
    assert s.index[0] == pd.Timestamp("2024-01-02")


def test_parse_fred_csv_handles_dot_as_nan():
    """FRED encodes missing values as a literal ``.`` — must coerce to NaN."""
    csv = "observation_date,VIXCLS\n2024-01-02,13.20\n2024-01-03,.\n2024-01-04,15.00\n"
    s = _parse_fred_csv(csv)
    assert s.iloc[0] == pytest.approx(13.20)
    assert pd.isna(s.iloc[1])
    assert s.iloc[2] == pytest.approx(15.00)


def test_parse_fred_csv_legacy_date_column():
    """Older FRED CSVs use ``DATE`` instead of ``observation_date``."""
    csv = "DATE,DGS10\n2024-01-02,4.05\n"
    s = _parse_fred_csv(csv)
    assert len(s) == 1
    assert s.iloc[0] == pytest.approx(4.05)


def test_parse_fred_csv_rejects_too_few_columns():
    with pytest.raises(ValueError, match="expected 2"):
        _parse_fred_csv("DATE\n2024-01-02\n")


# ---------- download_macro with stubbed fetcher ----------


def _stub_fetcher(canned: dict[str, pd.Series]):
    """Build a fetch_fn that returns the canned series for known field names
    and None for everything else (simulating network failure)."""

    def fn(field, series_id):
        return canned.get(field)

    return fn


def test_download_macro_returns_known_fields(tmp_path, monkeypatch):
    # Point the cache at a temp dir so we don't pollute the real cache
    monkeypatch.setattr("data.macro.CACHE_DIR", tmp_path)
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    canned = {
        "vix": pd.Series([13.0, 14.0, 12.5, 13.2, 14.1], index=dates),
        "treasury_10y_yield": pd.Series([4.0, 4.05, 4.1, 4.15, 4.2], index=dates),
        "treasury_2y_yield": pd.Series([4.5, 4.55, 4.6, 4.65, 4.7], index=dates),
        "treasury_3m_yield": pd.Series([5.2, 5.21, 5.22, 5.23, 5.24], index=dates),
    }
    out = download_macro(fetch_fn=_stub_fetcher(canned))

    assert "vix" in out
    assert out["vix"].iloc[0] == pytest.approx(13.0)
    # Derived spread is computed from base series
    assert "term_spread_10y_2y" in out
    assert out["term_spread_10y_2y"].iloc[0] == pytest.approx(4.0 - 4.5)
    assert "term_spread_10y_3m" in out
    assert out["term_spread_10y_3m"].iloc[0] == pytest.approx(4.0 - 5.2)


def test_download_macro_skips_failed_fetches(tmp_path, monkeypatch):
    """If a series fetch returns None, the field simply isn't in the output —
    not an error.  Operators referencing it later will get a clear evaluator
    error instead of a silent zero."""
    monkeypatch.setattr("data.macro.CACHE_DIR", tmp_path)
    out = download_macro(fetch_fn=lambda field, sid: None)
    # Nothing fetched → nothing returned; no exception
    assert out == {}


def test_download_macro_writes_cache(tmp_path, monkeypatch):
    """Successful fetch must persist to parquet so a second call hits the cache."""
    monkeypatch.setattr("data.macro.CACHE_DIR", tmp_path)
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    canned = {"vix": pd.Series([10.0, 11.0, 12.0], index=dates)}

    download_macro(fetch_fn=_stub_fetcher(canned))
    # Cache file written
    cache_file = tmp_path / "macro__vix.parquet"
    assert cache_file.exists()

    # Second call with a fetcher that always fails should still serve `vix`
    # from the cache because it's fresh
    out = download_macro(fetch_fn=lambda field, sid: None)
    assert "vix" in out
    assert out["vix"].iloc[0] == pytest.approx(10.0)


def test_download_macro_falls_back_to_stale_cache(tmp_path, monkeypatch):
    """If the fresh fetch fails AND the cache is stale, we still serve the
    stale data rather than dropping the field — degraded > broken."""
    monkeypatch.setattr("data.macro.CACHE_DIR", tmp_path)
    monkeypatch.setattr("data.macro.CACHE_TTL_SECONDS", 0)  # everything is stale
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    canned = {"vix": pd.Series([20.0, 21.0], index=dates)}

    # First call writes a cache file
    download_macro(fetch_fn=_stub_fetcher(canned))
    # Second call: fetch fails, but the (now-stale) cache is still consulted
    out = download_macro(fetch_fn=lambda field, sid: None)
    assert "vix" in out
    assert out["vix"].iloc[0] == pytest.approx(20.0)


# ---------- broadcast_to_matrix ----------


def test_broadcast_to_matrix_repeats_per_ticker():
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    tickers = ["A", "B", "C"]
    out = broadcast_to_matrix(series, dates, tickers)
    assert out.shape == (3, 3)
    # Every column matches the source series
    for t in tickers:
        np.testing.assert_allclose(out[t].values, series.values)


def test_broadcast_to_matrix_ffills_missing_dates():
    """FRED holidays don't always match equity holidays — gaps get
    forward-filled so the resulting matrix has no holes inside the window."""
    series = pd.Series(
        [10.0, 12.0],  # observations at days 0 and 2 only
        index=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")],
    )
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    out = broadcast_to_matrix(series, dates, ["X"])
    assert out.iloc[0]["X"] == 10.0
    # Day 2 (no observation) ffilled from day 1
    assert out.iloc[1]["X"] == 10.0
    assert out.iloc[2]["X"] == 12.0
    assert out.iloc[3]["X"] == 12.0


def test_broadcast_to_matrix_pre_history_stays_nan():
    """Dates before the series' first observation should remain NaN, not
    back-fill from the future (that would be a silent look-ahead leak)."""
    series = pd.Series(
        [5.0, 6.0],
        index=[pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
    )
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    out = broadcast_to_matrix(series, dates, ["X"])
    # Day 0 and 1 are before any observation → NaN
    assert pd.isna(out.iloc[0]["X"])
    assert pd.isna(out.iloc[1]["X"])
    assert out.iloc[2]["X"] == 5.0
    assert out.iloc[3]["X"] == 6.0


# ---------- Registration sanity ----------


def test_fred_series_dict_well_formed():
    """Each entry must be a (series_id, description) tuple."""
    for field, value in FRED_SERIES.items():
        assert isinstance(field, str)
        assert len(value) == 2
        assert isinstance(value[0], str) and len(value[0]) > 0
        assert isinstance(value[1], str) and len(value[1]) > 0


def test_derived_macro_names_unique_and_distinct_from_base():
    base_names = set(FRED_SERIES.keys())
    derived_names = {name for name, _ in DERIVED_MACRO_BUILDERS}
    assert len(derived_names) == len(DERIVED_MACRO_BUILDERS)  # no dupes within derived
    assert base_names.isdisjoint(derived_names)  # no name reuse across base+derived
