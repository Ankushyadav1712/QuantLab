"""Tests for analytics.attribution — Brinson-style PnL decomposition.

Verifies the central identity (sum of allocation + selection equals total
PnL) plus pure-case scenarios that map cleanly to expected outcomes:

  * pure sector bet (all weight in one sector, uniform within) → selection ≈ 0
  * pure stock picking (zero net per sector, picks the winner) → allocation ≈ 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.attribution import compute_pnl_attribution


def _gics(mapping: dict[str, str]) -> dict[str, dict[str, str | None]]:
    return {t: {"sector": s} for t, s in mapping.items()}


def _frame(values: dict[str, list[float]], n_days: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(values, index=dates)


# ---------- Identity: alloc + selection = total PnL ----------


def test_identity_allocation_plus_selection_equals_total():
    """Core invariant: per sector and aggregated, allocation + selection must
    sum to the actual PnL (sum_i w[i] * r[i]).  Any drift means a bug in the
    decomposition algebra."""
    rng = np.random.default_rng(42)
    n_days, n_tickers = 30, 12
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    weights = pd.DataFrame(rng.standard_normal((n_days, n_tickers)), index=dates, columns=tickers)
    returns = pd.DataFrame(
        rng.standard_normal((n_days, n_tickers)) * 0.01, index=dates, columns=tickers
    )
    # 4 sectors of 3 tickers each
    gics = _gics({t: f"S{i % 4}" for i, t in enumerate(tickers)})
    out = compute_pnl_attribution(weights, returns, gics)

    # Per-sector: allocation + selection == total (the dict's own "total" key)
    for sector, vals in out["by_sector"].items():
        assert vals["allocation"] + vals["selection"] == pytest.approx(vals["total"], abs=1e-9), (
            f"sector {sector}: alloc+sel != total"
        )

    # Aggregated: matches the truly-computed sum_i w[i] * r[i]
    actual = float((weights * returns).sum().sum())
    assert out["totals"]["allocation"] + out["totals"]["selection"] == pytest.approx(
        actual, abs=1e-6
    )
    assert out["totals"]["total_pnl"] == pytest.approx(actual, abs=1e-6)


# ---------- Pure sector bet → selection ≈ 0 ----------


def test_pure_sector_bet_has_zero_selection():
    """Equal weight within a sector → no stock picking → selection should be
    exactly zero, all PnL attributed to allocation."""
    weights = _frame(
        {"A": [0.5] * 5, "B": [0.5] * 5, "C": [0.0] * 5, "D": [0.0] * 5},
    )
    # Both A and B (same sector) get the same return on each day → within-sector
    # spread is zero, so selection must be zero.
    returns = _frame(
        {"A": [0.01] * 5, "B": [0.01] * 5, "C": [0.02] * 5, "D": [-0.01] * 5},
    )
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_pnl_attribution(weights, returns, gics)

    # Within each sector with equal weights, selection should be 0 (within
    # numerical precision)
    assert out["by_sector"]["Tech"]["selection"] == pytest.approx(0.0, abs=1e-9)
    assert out["by_sector"]["Energy"]["selection"] == pytest.approx(0.0, abs=1e-9)
    # Tech allocation = w_net (1.0) * r_avg (0.01) per day * 5 days = 0.05
    assert out["by_sector"]["Tech"]["allocation"] == pytest.approx(0.05, abs=1e-9)


# ---------- Pure stock picking → allocation ≈ 0 ----------


def test_dollar_neutral_within_sector_has_zero_allocation():
    """Long winner / short loser in same sector → net sector weight = 0 →
    allocation = 0, all PnL attributed to selection."""
    # A and B both in Tech: A long, B short, equal magnitude → w_net=0 every day
    weights = _frame(
        {"A": [0.5] * 5, "B": [-0.5] * 5, "C": [0.0] * 5, "D": [0.0] * 5},
    )
    returns = _frame(
        {"A": [0.02] * 5, "B": [-0.01] * 5, "C": [0.0] * 5, "D": [0.0] * 5},
    )
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"})
    out = compute_pnl_attribution(weights, returns, gics)
    # Net Tech weight is 0 → allocation must be 0
    assert out["by_sector"]["Tech"]["allocation"] == pytest.approx(0.0, abs=1e-9)
    # All Tech PnL goes to selection: 0.5*0.02 + (-0.5)*(-0.01) = 0.015/day × 5
    assert out["by_sector"]["Tech"]["selection"] == pytest.approx(0.075, abs=1e-9)


# ---------- Percentages + headline shape ----------


def test_pct_split_sums_to_about_100():
    rng = np.random.default_rng(0)
    n_days, n_tickers = 30, 8
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    weights = pd.DataFrame(rng.standard_normal((n_days, n_tickers)), index=dates, columns=tickers)
    returns = pd.DataFrame(
        rng.standard_normal((n_days, n_tickers)) * 0.01, index=dates, columns=tickers
    )
    gics = _gics({t: f"S{i % 3}" for i, t in enumerate(tickers)})
    out = compute_pnl_attribution(weights, returns, gics)
    total_pct = out["totals"]["allocation_pct"] + out["totals"]["selection_pct"]
    assert total_pct == pytest.approx(100.0, abs=1e-6)


def test_zero_pnl_book_returns_zero_pcts_without_dividing_by_zero():
    """A book with all-zero weights → 0 PnL → pct should be 0, not NaN."""
    weights = _frame({"A": [0.0] * 5, "B": [0.0] * 5})
    returns = _frame({"A": [0.01] * 5, "B": [-0.01] * 5})
    gics = _gics({"A": "Tech", "B": "Tech"})
    out = compute_pnl_attribution(weights, returns, gics)
    assert out["totals"]["allocation_pct"] == 0.0
    assert out["totals"]["selection_pct"] == 0.0


# ---------- Edge cases ----------


def test_returns_none_when_inputs_are_empty():
    assert compute_pnl_attribution(pd.DataFrame(), None, {}) is None
    assert compute_pnl_attribution(pd.DataFrame(), pd.DataFrame(), {}) is None
    weights = _frame({"A": [0.5] * 5})
    assert compute_pnl_attribution(weights, pd.DataFrame(), {"A": {"sector": "Tech"}}) is None
    assert compute_pnl_attribution(weights, weights, None) is None


def test_unknown_bucket_for_tickers_missing_from_gics():
    """A ticker not in the gics_map should land in 'Unknown' rather than
    silently being dropped."""
    weights = _frame({"A": [0.5] * 5, "ZZ": [0.5] * 5})
    returns = _frame({"A": [0.01] * 5, "ZZ": [-0.02] * 5})
    out = compute_pnl_attribution(weights, returns, {"A": {"sector": "Tech"}})
    assert "Unknown" in out["by_sector"]
    assert out["by_sector"]["Unknown"]["n_tickers"] == 1


def test_n_tickers_per_sector_reported():
    weights = _frame({"A": [0.5] * 5, "B": [0.3] * 5, "C": [0.2] * 5})
    returns = _frame({"A": [0.01] * 5, "B": [0.01] * 5, "C": [0.01] * 5})
    gics = _gics({"A": "Tech", "B": "Tech", "C": "Energy"})
    out = compute_pnl_attribution(weights, returns, gics)
    assert out["by_sector"]["Tech"]["n_tickers"] == 2
    assert out["by_sector"]["Energy"]["n_tickers"] == 1


def test_attribution_at_industry_level():
    """The same decomposition should work at any GICS level."""
    weights = _frame({"A": [1.0] * 5, "B": [1.0] * 5})
    returns = _frame({"A": [0.01] * 5, "B": [0.02] * 5})
    gics = {
        "A": {"sector": "Tech", "industry": "Software"},
        "B": {"sector": "Tech", "industry": "Semis"},
    }
    out = compute_pnl_attribution(weights, returns, gics, level="industry")
    assert set(out["by_sector"].keys()) == {"Software", "Semis"}
    assert out["level"] == "industry"
