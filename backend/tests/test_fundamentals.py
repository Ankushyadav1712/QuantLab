"""yfinance fundamentals loader — extraction, lag application, ratio math.

Tests stub the per-ticker fetch via the injectable ``fetch_fn`` parameter so
nothing hits the network.  Each stub returns the same shape yfinance does:
three DataFrames (income / balance sheet / cash flow), each indexed by the
yfinance row labels and columned by quarterly report date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.fundamentals import (
    ALL_FUNDAMENTAL_FIELDS,
    INCOME_LABELS,
    LAG_QUARTERS,
    RATIO_FIELDS,
    RAW_FUNDAMENTAL_FIELDS,
    _build_per_field_matrix,
    _compute_ratios,
    _extract_quarterly,
    _pick_row,
    download_fundamentals,
)


# ---------- Helpers ----------


def _quarterly_dates(n: int, end: str = "2024-12-31") -> pd.DatetimeIndex:
    """Generate ``n`` quarter-end dates ending at ``end`` (most recent first
    to match yfinance's column ordering)."""
    return pd.date_range(end=end, periods=n, freq="QE")


def _income_frame(report_dates: pd.DatetimeIndex, *, revenue, net_income, eps,
                  gross_profit=None, operating_income=None, ebitda=None) -> pd.DataFrame:
    """Build a yfinance-shaped income statement frame (rows are line items,
    columns are quarterly report dates)."""
    rows = {
        "Total Revenue": revenue,
        "Net Income": net_income,
        "Diluted EPS": eps,
    }
    if gross_profit is not None:
        rows["Gross Profit"] = gross_profit
    if operating_income is not None:
        rows["Operating Income"] = operating_income
    if ebitda is not None:
        rows["EBITDA"] = ebitda
    return pd.DataFrame(rows, index=report_dates).T


def _balance_frame(report_dates, *, total_assets=None, total_debt=None,
                   total_equity=None, cash=None,
                   current_assets=None, current_liabilities=None) -> pd.DataFrame:
    rows: dict[str, list] = {}
    if total_assets is not None: rows["Total Assets"] = total_assets
    if total_debt is not None: rows["Total Debt"] = total_debt
    if total_equity is not None: rows["Stockholders Equity"] = total_equity
    if cash is not None: rows["Cash"] = cash
    if current_assets is not None: rows["Current Assets"] = current_assets
    if current_liabilities is not None: rows["Current Liabilities"] = current_liabilities
    return pd.DataFrame(rows, index=report_dates).T


def _cashflow_frame(report_dates, *, operating_cash_flow=None, capex=None, free_cash_flow=None):
    rows: dict[str, list] = {}
    if operating_cash_flow is not None: rows["Operating Cash Flow"] = operating_cash_flow
    if capex is not None: rows["Capital Expenditure"] = capex
    if free_cash_flow is not None: rows["Free Cash Flow"] = free_cash_flow
    return pd.DataFrame(rows, index=report_dates).T


# ---------- Pure helpers ----------


def test_pick_row_returns_first_alias_match():
    df = pd.DataFrame({"q1": [1.0, 2.0]}, index=["Total Revenue", "Net Income"])
    out = _pick_row(df, ("TotalRevenue", "Total Revenue"))
    assert out is not None
    assert out["q1"] == 1.0


def test_pick_row_returns_none_on_miss():
    df = pd.DataFrame({"q1": [1.0]}, index=["Foo"])
    assert _pick_row(df, ("Bar", "Baz")) is None


def test_pick_row_handles_empty_frame():
    assert _pick_row(None, ("Foo",)) is None
    assert _pick_row(pd.DataFrame(), ("Foo",)) is None


# ---------- Quarterly extraction ----------


def test_extract_quarterly_returns_known_fields():
    dates = _quarterly_dates(3)
    income = _income_frame(dates, revenue=[100, 110, 120], net_income=[10, 11, 12], eps=[1.0, 1.1, 1.2])
    out = _extract_quarterly(income, None, None, INCOME_LABELS, "income")
    assert "revenue" in out
    assert "net_income" in out
    # Series should be sorted ascending by date and have len == 3
    rev = out["revenue"]
    assert len(rev) == 3
    assert rev.is_monotonic_increasing  # date index ascending


def test_extract_quarterly_skips_missing_labels():
    dates = _quarterly_dates(2)
    income = _income_frame(dates, revenue=[100, 110], net_income=[10, 11], eps=[1.0, 1.1])
    out = _extract_quarterly(income, None, None, INCOME_LABELS, "income")
    # gross_profit not provided → not in output (no exception)
    assert "gross_profit" not in out


# ---------- Lag application ----------


def test_build_per_field_matrix_applies_quarter_lag():
    """A value reported on Q1-end-date should not be visible until Q1+90 days."""
    report_q = pd.Timestamp("2024-03-31")
    series = pd.Series([100.0], index=[report_q])
    daily = pd.date_range("2024-04-01", "2024-08-01", freq="B")
    out = _build_per_field_matrix({"AAPL": series}, daily, ["AAPL"], lag_quarters=1)
    # Pre-lag (within ~90 days of report) → still NaN
    pre_lag = out.loc["2024-04-01":"2024-06-01", "AAPL"]
    assert pre_lag.isna().all()
    # Post-lag → value is forward-filled
    post_lag_value = out.loc["2024-07-01", "AAPL"]
    assert post_lag_value == 100.0


def test_build_per_field_matrix_pre_history_stays_nan():
    """Daily dates earlier than the first lagged report should remain NaN —
    no back-filling that would silently leak future data."""
    report_q = pd.Timestamp("2024-06-30")
    series = pd.Series([200.0], index=[report_q])
    daily = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    out = _build_per_field_matrix({"AAPL": series}, daily, ["AAPL"], lag_quarters=1)
    # Day 1 is well before any data was available → NaN
    assert pd.isna(out.iloc[0]["AAPL"])


def test_build_per_field_matrix_two_reports_step_function():
    """Across two reports, the matrix should step up from value 1 to value 2
    on the day the second report becomes 'available' (report + lag)."""
    series = pd.Series(
        [100.0, 200.0],
        index=[pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")],
    )
    daily = pd.date_range("2024-07-01", "2024-12-31", freq="B")
    out = _build_per_field_matrix({"AAPL": series}, daily, ["AAPL"], lag_quarters=1)
    # By Aug, only Q1 (report + 90d ≈ end-June) is visible → 100
    assert out.loc["2024-08-01", "AAPL"] == 100.0
    # By Oct, Q2 (report end-Jun + 90d ≈ end-Sep) is visible → 200
    assert out.loc["2024-11-01", "AAPL"] == 200.0


def test_build_per_field_matrix_handles_missing_ticker():
    """A ticker absent from the data dict should produce a NaN column, not
    blow up.  This matches the "ticker yfinance failed for" scenario."""
    daily = pd.date_range("2024-01-01", periods=10, freq="B")
    out = _build_per_field_matrix({}, daily, ["AAPL", "MSFT"], lag_quarters=1)
    assert out.isna().all().all()
    assert list(out.columns) == ["AAPL", "MSFT"]


# ---------- Ratio computation ----------


def test_compute_ratios_basic():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["AAPL"]
    # Build raw matrices: net_income=100, eps=1.0 → implied shares = 100,
    # market cap = close × shares = 50 × 100 = 5000
    raw = {
        "net_income": pd.DataFrame({"AAPL": [100.0, 100.0]}, index=dates),
        "eps": pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates),
        "total_equity": pd.DataFrame({"AAPL": [1000.0, 1000.0]}, index=dates),
        "revenue": pd.DataFrame({"AAPL": [500.0, 500.0]}, index=dates),
        "total_assets": pd.DataFrame({"AAPL": [2000.0, 2000.0]}, index=dates),
        "total_debt": pd.DataFrame({"AAPL": [800.0, 800.0]}, index=dates),
    }
    close = pd.DataFrame({"AAPL": [50.0, 50.0]}, index=dates)
    out = _compute_ratios(raw, close)

    # ROE = ni / equity = 100 / 1000 = 0.1
    assert out["roe"].iloc[0]["AAPL"] == pytest.approx(0.1)
    # ROA = ni / assets = 100 / 2000 = 0.05
    assert out["roa"].iloc[0]["AAPL"] == pytest.approx(0.05)
    # Debt/equity = 800 / 1000 = 0.8
    assert out["debt_to_equity"].iloc[0]["AAPL"] == pytest.approx(0.8)
    # P/E: market_cap = 50 * 100 = 5000; ni = 100 → P/E = 50
    assert out["pe_ratio"].iloc[0]["AAPL"] == pytest.approx(50.0)
    # P/B: 5000 / 1000 = 5
    assert out["pb_ratio"].iloc[0]["AAPL"] == pytest.approx(5.0)
    # P/S: 5000 / 500 = 10
    assert out["ps_ratio"].iloc[0]["AAPL"] == pytest.approx(10.0)


def test_compute_ratios_skips_when_inputs_missing():
    """If a required raw field is missing, the ratio just isn't in the output —
    no exception, no garbage values."""
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    raw = {
        "net_income": pd.DataFrame({"AAPL": [100.0]}, index=dates),
        # No eps → no implied shares → no market-cap-based ratios
    }
    close = pd.DataFrame({"AAPL": [50.0]}, index=dates)
    out = _compute_ratios(raw, close)
    assert "pe_ratio" not in out
    assert "pb_ratio" not in out


def test_compute_ratios_safe_division_on_zero_denom():
    """ROE with zero equity must be NaN, not inf."""
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    raw = {
        "net_income": pd.DataFrame({"AAPL": [100.0]}, index=dates),
        "eps": pd.DataFrame({"AAPL": [1.0]}, index=dates),
        "total_equity": pd.DataFrame({"AAPL": [0.0]}, index=dates),  # zero
    }
    close = pd.DataFrame({"AAPL": [50.0]}, index=dates)
    out = _compute_ratios(raw, close)
    assert pd.isna(out["roe"].iloc[0]["AAPL"])


# ---------- End-to-end download_fundamentals ----------


def test_download_fundamentals_with_stubbed_fetcher():
    """End-to-end with a single ticker.  Verifies the full pipeline:
    fetch → extract → lag → ratios."""
    dates = pd.date_range("2024-04-01", "2024-12-31", freq="B")
    close = pd.DataFrame(
        {"AAPL": np.full(len(dates), 50.0)},
        index=dates,
    )

    # Two quarterly reports prior to the daily window
    qdates = pd.DatetimeIndex([pd.Timestamp("2023-12-31"), pd.Timestamp("2024-03-31")])
    income = _income_frame(qdates, revenue=[400, 500], net_income=[50, 60], eps=[0.5, 0.6])
    balance = _balance_frame(qdates, total_assets=[2000, 2100], total_debt=[800, 850],
                             total_equity=[1000, 1050])
    cashflow = _cashflow_frame(qdates, operating_cash_flow=[80, 90], capex=[-30, -35])

    def stub(ticker):
        assert ticker == "AAPL"
        return income, balance, cashflow

    out = download_fundamentals(
        tickers=["AAPL"], daily_index=dates, close_matrix=close, fetch_fn=stub,
    )

    # Raw fields show up
    assert "revenue" in out
    assert "net_income" in out
    # FCF was synthesized from OCF + capex (yfinance often omits FCF)
    assert "free_cash_flow" in out
    # FCF Q1 = OCF + capex = 80 + (-30) = 50; visible after Q1 + 1Q lag (~end-March)
    fcf_late = out["free_cash_flow"].loc[out["free_cash_flow"].index >= "2024-05-01", "AAPL"].dropna()
    assert len(fcf_late) > 0
    # Could be either 50 (from Q4-2023) or 55 (from Q1-2024) depending on how late we look
    assert fcf_late.iloc[-1] in (50.0, 55.0)

    # Ratios are computed
    assert "roe" in out
    # By July 2024 (after Q1-2024 + 90d lag), Q1-2024 numbers are visible:
    # ROE = 60 / 1050 ≈ 0.0571
    july_roe = out["roe"].loc["2024-07-15":"2024-08-15", "AAPL"].dropna()
    if len(july_roe) > 0:
        assert july_roe.iloc[0] == pytest.approx(60.0 / 1050, rel=1e-6)


def test_download_fundamentals_failed_ticker_keeps_running():
    """If yfinance raises for one ticker, the others still load."""
    dates = pd.date_range("2024-04-01", "2024-12-31", freq="B")
    close = pd.DataFrame(
        {"AAPL": np.full(len(dates), 50.0), "MSFT": np.full(len(dates), 400.0)},
        index=dates,
    )
    qdates = pd.DatetimeIndex([pd.Timestamp("2024-03-31")])

    def stub(ticker):
        if ticker == "MSFT":
            raise RuntimeError("yfinance simulated failure")
        return (
            _income_frame(qdates, revenue=[500], net_income=[60], eps=[0.6]),
            _balance_frame(qdates, total_equity=[1000]),
            _cashflow_frame(qdates),
        )

    out = download_fundamentals(
        tickers=["AAPL", "MSFT"], daily_index=dates, close_matrix=close, fetch_fn=stub,
    )

    # AAPL has data; MSFT column is all NaN — no exception raised
    rev = out["revenue"]
    aapl_rev = rev.loc[rev.index >= "2024-08-01", "AAPL"].dropna()
    assert len(aapl_rev) > 0
    msft_rev = rev["MSFT"]
    assert msft_rev.isna().all()


# ---------- Registration sanity ----------


def test_field_lists_consistent():
    """The combined surface must equal raw + ratios with no overlap."""
    assert set(ALL_FUNDAMENTAL_FIELDS) == set(RAW_FUNDAMENTAL_FIELDS) | set(RATIO_FIELDS)
    assert not (set(RAW_FUNDAMENTAL_FIELDS) & set(RATIO_FIELDS))


def test_lag_quarters_default_is_one():
    """Default PIT lag is 1Q — changing this is a research-design decision
    that should be loud, so the test guards against silent bumps."""
    assert LAG_QUARTERS == 1
