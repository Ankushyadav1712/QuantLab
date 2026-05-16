"""Tests for the new metrics/fields added in the PDF audit pass:

- Fundamentals: ``net_margin``, ``earnings_yield``, ``dividend_yield``,
  ``shares_outstanding`` (S-5 / S-7)
- Long/Short dollar exposure: ``long_exposure``, ``short_exposure``,
  ``gross_exposure``, ``net_exposure`` (S-6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analytics.performance import PerformanceAnalytics
from data.fundamentals import RATIO_FIELDS, _compute_ratios
from engine.backtester import BacktestResult

# ---------- fundamentals: ratio coverage ----------


def test_ratio_fields_include_pdf_additions():
    """The four PDF-required ratios + shares_outstanding must be listed."""
    expected_new = {"net_margin", "earnings_yield", "dividend_yield", "shares_outstanding"}
    assert expected_new.issubset(set(RATIO_FIELDS)), (
        f"Missing from RATIO_FIELDS: {expected_new - set(RATIO_FIELDS)}"
    )


def test_compute_ratios_emits_new_fields():
    """When the raw inputs needed by each new ratio are present, the ratio
    matrix should contain a key for each."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    raw = {
        # Net income / EPS / revenue / equity needed for the basic profitability
        "net_income": pd.DataFrame({"AAA": [10.0] * 4, "BBB": [20.0] * 4}, index=idx),
        "eps": pd.DataFrame({"AAA": [1.0] * 4, "BBB": [2.0] * 4}, index=idx),
        "revenue": pd.DataFrame({"AAA": [100.0] * 4, "BBB": [400.0] * 4}, index=idx),
        "total_equity": pd.DataFrame({"AAA": [50.0] * 4, "BBB": [100.0] * 4}, index=idx),
        "dividends_paid": pd.DataFrame({"AAA": [-3.0] * 4, "BBB": [-6.0] * 4}, index=idx),
    }
    close = pd.DataFrame({"AAA": [10.0] * 4, "BBB": [20.0] * 4}, index=idx)

    out = _compute_ratios(raw, close)
    # Each PDF-required field present
    assert "net_margin" in out
    assert "earnings_yield" in out
    assert "dividend_yield" in out
    assert "shares_outstanding" in out
    # Sanity-check the math on one cell of each
    # net_margin = 10 / 100 = 0.10
    assert out["net_margin"].iloc[0, 0] == pytest.approx(0.10)
    # earnings_yield = ni / mcap = 10 / (price 10 * shares ni/eps=10) = 10/100 = 0.10
    assert out["earnings_yield"].iloc[0, 0] == pytest.approx(0.10)
    # dividend_yield: |-3| / 100 = 0.03 (abs() of the negative cashflow)
    assert out["dividend_yield"].iloc[0, 0] == pytest.approx(0.03)
    # shares_outstanding = ni / eps = 10
    assert out["shares_outstanding"].iloc[0, 0] == pytest.approx(10.0)


def test_compute_ratios_handles_missing_dividends_gracefully():
    """A ticker that pays no dividends → no dividends_paid in raw → field
    should be silently omitted, not raise."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    raw = {
        "net_income": pd.DataFrame({"AAA": [10.0, 10.0]}, index=idx),
        "eps": pd.DataFrame({"AAA": [1.0, 1.0]}, index=idx),
        "revenue": pd.DataFrame({"AAA": [100.0, 100.0]}, index=idx),
    }
    close = pd.DataFrame({"AAA": [10.0, 10.0]}, index=idx)
    out = _compute_ratios(raw, close)
    assert "dividend_yield" not in out  # No dividend data → no field
    assert "net_margin" in out  # but other ratios still computed


# ---------- L/S exposure metric ----------


def _fake_backtest_result(weights: pd.DataFrame) -> BacktestResult:
    """Minimal BacktestResult plumbed with just enough fields for the
    performance.compute() L/S exposure block.  Daily returns are flat so all
    the other metrics stay numerically benign."""
    dates = [d.strftime("%Y-%m-%d") for d in weights.index]
    n = len(dates)
    zeros = [0.0] * n
    return BacktestResult(
        dates=dates,
        daily_pnl=zeros,
        cumulative_pnl=zeros,
        daily_returns=zeros,
        weights=weights,
        turnover=zeros,
        positions=weights.copy(),
    )


def test_exposure_dollar_neutral_book_has_zero_net():
    """Equal $1M long and $1M short → net_exposure ≈ 0, gross ≈ 2.0
    (in units of booksize, since book_proxy = max gross = 2.0)."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    # Each row: A=+0.5M, B=+0.5M (long $1M); C=-0.5M, D=-0.5M (short $1M)
    weights = pd.DataFrame(
        {"A": [5e5] * 3, "B": [5e5] * 3, "C": [-5e5] * 3, "D": [-5e5] * 3},
        index=idx,
    )
    result = _fake_backtest_result(weights)
    m = PerformanceAnalytics().compute(result)
    # book_proxy = max gross |w| sum per day = 2.0M
    # long_exposure / book_proxy = 1.0M / 2.0M = 0.5
    assert m["long_exposure"] == pytest.approx(0.5, abs=1e-9)
    assert m["short_exposure"] == pytest.approx(-0.5, abs=1e-9)
    # gross = (long - short) / book = (1.0M - (-1.0M)) / 2.0M = 1.0
    assert m["gross_exposure"] == pytest.approx(1.0, abs=1e-9)
    # net = (long + short) / book = (1.0M + (-1.0M)) / 2.0M = 0
    assert m["net_exposure"] == pytest.approx(0.0, abs=1e-9)


def test_exposure_long_only_book_has_positive_net():
    """All-long book: net_exposure should equal gross_exposure (both > 0),
    short_exposure should be 0."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.DataFrame({"A": [1e6] * 3, "B": [1e6] * 3}, index=idx)
    result = _fake_backtest_result(weights)
    m = PerformanceAnalytics().compute(result)
    assert m["long_exposure"] == pytest.approx(1.0, abs=1e-9)
    assert m["short_exposure"] == pytest.approx(0.0, abs=1e-9)
    assert m["gross_exposure"] == pytest.approx(1.0, abs=1e-9)
    assert m["net_exposure"] == pytest.approx(1.0, abs=1e-9)


def test_exposure_none_when_weights_empty():
    """Old saved results with no weights matrix → all four fields None
    (UI renders as "—" instead of misleading 0%)."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    # Construct a result with an empty weights frame
    result = _fake_backtest_result(pd.DataFrame(index=idx))
    m = PerformanceAnalytics().compute(result)
    assert m["long_exposure"] is None
    assert m["short_exposure"] is None
    assert m["gross_exposure"] is None
    assert m["net_exposure"] is None


def test_exposure_metric_keys_in_perf_pack():
    """The exposure keys must be in main's _METRIC_KEYS allowlist or they
    won't make it into the saved alpha payload."""
    # Import here to avoid loading main.py at module-import time
    from main import _METRIC_KEYS

    for k in ("long_exposure", "short_exposure", "gross_exposure", "net_exposure"):
        assert k in _METRIC_KEYS, f"{k!r} missing from _METRIC_KEYS"


# ---------- /api/operators metadata coverage ----------


def test_new_operators_documented_in_api():
    """Every new operator added in the PDF pass must have an entry in
    main.OPERATORS so /api/operators surfaces it for the editor's autocomplete."""
    from main import OPERATORS

    names = {op["name"] for op in OPERATORS}
    expected = {"neutralize", "correlation", "cap_weight", "adv"}
    missing = expected - names
    assert not missing, f"Missing /api/operators entries: {missing}"


def test_new_fundamentals_documented_in_api():
    """Same check for the new fundamentals fields → /api/operators FIELDS."""
    from main import DATA_FIELDS

    expected = {"net_margin", "earnings_yield", "dividend_yield", "shares_outstanding"}
    missing = expected - set(DATA_FIELDS)
    assert not missing, f"Missing FIELDS entries: {missing}"


# ---------- silence unused-import warning ----------
_ = np  # numpy was used implicitly via pandas DataFrame construction
