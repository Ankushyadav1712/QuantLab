"""Liquidity-ranked us_top1000 universe.

Covers the pure ranking function (the selection IP), the universe registration,
and the file-load / russell1000-fallback behaviour of the lazy loader.  The
offline build script's network fetch is out of scope (it runs on the user's
machine, not in CI).
"""

import numpy as np
import pandas as pd
import pytest
from data import universes
from data.universes import _UNIVERSES, list_universes, rank_by_dollar_volume

# ---------- rank_by_dollar_volume (the selection criterion) ----------


def _dv_frame(cols_to_level: dict[str, float], n_days: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    return pd.DataFrame({t: [lvl] * n_days for t, lvl in cols_to_level.items()}, index=idx)


def test_rank_orders_by_median_dollar_volume_desc():
    dv = _dv_frame({"LOW": 1e6, "HIGH": 9e6, "MID": 5e6})
    assert rank_by_dollar_volume(dv, 3) == ["HIGH", "MID", "LOW"]


def test_rank_truncates_to_n():
    dv = _dv_frame({"A": 4e6, "B": 3e6, "C": 2e6, "D": 1e6})
    assert rank_by_dollar_volume(dv, 2) == ["A", "B"]


def test_rank_uses_median_not_mean_so_spikes_dont_dominate():
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    # STEADY has higher *median* liquidity; SPIKY has one huge day but is
    # otherwise illiquid — median (Brain's robust proxy) must prefer STEADY.
    dv = pd.DataFrame(
        {"STEADY": [5e6, 5e6, 5e6, 5e6, 5e6], "SPIKY": [1e5, 1e5, 9e8, 1e5, 1e5]},
        index=idx,
    )
    assert rank_by_dollar_volume(dv, 1) == ["STEADY"]


def test_rank_drops_all_nan_columns():
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    dv = pd.DataFrame(
        {"GOOD": [2e6, 2e6, 2e6, 2e6], "NODATA": [np.nan] * 4},
        index=idx,
    )
    assert rank_by_dollar_volume(dv, 10) == ["GOOD"]


def test_rank_min_days_excludes_thin_history():
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    # IPO traded once, huge that day; STEADY trades every day at lower volume.
    # With min_days, the one-day name is dropped before ranking.
    ipo = [np.nan] * 9 + [9e9]
    steady = [4e6] * 10
    dv = pd.DataFrame({"IPO": ipo, "STEADY": steady}, index=idx)
    assert rank_by_dollar_volume(dv, 2, min_days=5) == ["STEADY"]
    # Default (min_days=1) keeps the spike name and it outranks on that day.
    assert rank_by_dollar_volume(dv, 1) == ["IPO"]


def test_rank_ignores_partial_nans_in_median():
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    dv = pd.DataFrame(
        {"A": [np.nan, 8e6, 8e6, np.nan], "B": [3e6, 3e6, 3e6, 3e6]},
        index=idx,
    )
    # A's median over its non-NaN days (8e6) beats B (3e6).
    assert rank_by_dollar_volume(dv, 2) == ["A", "B"]


# ---------- registration ----------


def test_us_top1000_is_registered():
    assert "us_top1000" in _UNIVERSES
    entry = _UNIVERSES["us_top1000"]
    assert entry["tickers"] is None  # lazy
    assert entry["preload"] is False  # never loaded at startup
    ids = {u["id"] for u in list_universes()}
    assert "us_top1000" in ids


# ---------- lazy loader: file-load + fallback ----------


def test_loader_reads_prebuilt_file(tmp_path, monkeypatch):
    tickers_dir = tmp_path / "tickers"
    tickers_dir.mkdir()
    (tickers_dir / "us_top1000.txt").write_text("AAA\nBBB\nCCC\n")
    monkeypatch.setattr(universes, "_TICKERS_DIR", tickers_dir)
    assert universes._get_us_top1000_tickers() == ["AAA", "BBB", "CCC"]


def test_loader_falls_back_to_russell1000_when_unbuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(universes, "_TICKERS_DIR", tmp_path / "tickers")  # no file
    monkeypatch.setattr(universes, "_get_russell1000_tickers", lambda: ["FALLBACK1", "FALLBACK2"])
    with pytest.warns(UserWarning, match="not built"):
        assert universes._get_us_top1000_tickers() == ["FALLBACK1", "FALLBACK2"]


def test_loader_ignores_blank_lines(tmp_path, monkeypatch):
    tickers_dir = tmp_path / "tickers"
    tickers_dir.mkdir()
    (tickers_dir / "us_top1000.txt").write_text("AAA\n\n  \nBBB\n")
    monkeypatch.setattr(universes, "_TICKERS_DIR", tickers_dir)
    assert universes._get_us_top1000_tickers() == ["AAA", "BBB"]
