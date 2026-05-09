"""yfinance fundamentals loader.

Pulls quarterly income statement, balance sheet, and cash flow data per
ticker, applies a **one-quarter lag** to approximate point-in-time
availability (yfinance returns latest filings with restatements; without a
lag, backtests would silently use future revisions), then forward-fills
to a daily index and broadcasts into (dates × tickers) matrices.

Computed ratios (P/E, P/B, ROE, etc.) are derived in this module — they
combine fundamentals with the close-price matrix that the OHLCV pipeline
already loaded.

Limitations honestly disclosed in the data-quality banner:
  * One-quarter lag is a coarse PIT proxy; real PIT needs the actual report
    release date, which yfinance often doesn't expose reliably.
  * yfinance is rate-limited; ~5% of tickers may return missing data on any
    given day.  Failures degrade gracefully (the ticker just gets NaN).
  * Restatements are baked in — a 2023 backtest using 2020 revenue uses the
    *current* (possibly restated) 2020 number, not the originally reported one.

Network call budget: ~3 yfinance calls per ticker (.quarterly_financials,
.quarterly_balance_sheet, .quarterly_cashflow), ~500ms each → ~1.5s per ticker.
For ~95 tickers that's ~2 minutes on cold cache.  Cached weekly thereafter.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import CACHE_DIR

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 1 week — fundamentals change quarterly
LAG_QUARTERS = 1  # PIT proxy; values "available" 1 quarter after report

# yfinance's row labels (case-sensitive) we care about, mapped to our
# user-facing field name.  yfinance schema occasionally rotates label names
# (e.g. "TotalRevenue" vs "Total Revenue") — we accept several aliases per field.
INCOME_LABELS: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "TotalRevenue", "Revenue"),
    "gross_profit": ("Gross Profit", "GrossProfit"),
    "operating_income": ("Operating Income", "OperatingIncome"),
    "net_income": ("Net Income", "NetIncome"),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "eps": ("Diluted EPS", "Basic EPS", "EPS"),
}
BALANCE_LABELS: dict[str, tuple[str, ...]] = {
    "total_assets": ("Total Assets", "TotalAssets"),
    "total_debt": ("Total Debt", "TotalDebt"),
    "total_equity": ("Total Equity Gross Minority Interest", "Stockholders Equity",
                     "Total Stockholder Equity"),
    "cash": ("Cash And Cash Equivalents", "Cash"),
    "current_assets": ("Current Assets", "Total Current Assets"),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
}
CASHFLOW_LABELS: dict[str, tuple[str, ...]] = {
    "operating_cash_flow": ("Operating Cash Flow", "Total Cash From Operating Activities"),
    "capex": ("Capital Expenditure", "Capital Expenditures"),
    "free_cash_flow": ("Free Cash Flow",),  # often computed below if missing
}

# Names ratios depend on (and the close-price matrix, supplied at compute time).
RATIO_FIELDS: tuple[str, ...] = (
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa", "debt_to_equity", "current_ratio",
    "gross_margin", "operating_margin", "fcf_yield",
)

# Combined surface used by parser/editor/FIELDS metadata
RAW_FUNDAMENTAL_FIELDS: tuple[str, ...] = (
    *INCOME_LABELS.keys(),
    *BALANCE_LABELS.keys(),
    *CASHFLOW_LABELS.keys(),
)
ALL_FUNDAMENTAL_FIELDS: tuple[str, ...] = (*RAW_FUNDAMENTAL_FIELDS, *RATIO_FIELDS)

log = logging.getLogger("quantlab.fundamentals")


def _cache_path(field: str) -> Path:
    return Path(CACHE_DIR) / f"fundamental__{field}.parquet"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _pick_row(frame: pd.DataFrame | None, aliases: tuple[str, ...]) -> pd.Series | None:
    """Find the first matching row label in ``frame`` (yfinance schema is loose)."""
    if frame is None or frame.empty:
        return None
    for alias in aliases:
        if alias in frame.index:
            return frame.loc[alias]
    return None


def _extract_quarterly(
    income: pd.DataFrame | None,
    balance: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
    labels: dict[str, tuple[str, ...]],
    source: str,
) -> dict[str, pd.Series]:
    """Pull labeled rows from one of the three yfinance frames.

    Returns ``{field_name: Series indexed by report_date}``.  Fields not
    present in the source frame are silently omitted — caller treats missing
    fields as NaN downstream.
    """
    src = {"income": income, "balance": balance, "cashflow": cashflow}[source]
    out: dict[str, pd.Series] = {}
    for field, aliases in labels.items():
        row = _pick_row(src, aliases)
        if row is None:
            continue
        # yfinance frames have report dates as columns — transpose to a Series
        # indexed by date with float values.
        s = pd.Series(pd.to_numeric(row.values, errors="coerce"),
                      index=pd.to_datetime(row.index))
        s = s.sort_index()
        out[field] = s
    return out


def _per_ticker_frames(
    ticker: str, fetch_fn
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch the 3 yfinance frames for one ticker with graceful failure."""
    try:
        return fetch_fn(ticker)
    except Exception as exc:
        warnings.warn(f"[fundamentals:{ticker}] fetch failed: {exc}")
        return None, None, None


def _yfinance_fetch(ticker: str):
    """Default fetcher — hits the network.  Tests pass a stub instead."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    return t.quarterly_financials, t.quarterly_balance_sheet, t.quarterly_cashflow


def _build_per_field_matrix(
    series_by_ticker: dict[str, pd.Series],
    daily_index: pd.DatetimeIndex,
    tickers: list[str],
    *,
    lag_quarters: int = LAG_QUARTERS,
) -> pd.DataFrame:
    """Assemble per-ticker quarterly Series into a (daily_index × tickers) matrix.

    Steps:
      1. Lag each Series by ``lag_quarters`` (90 calendar days per quarter)
         to approximate when the data was actually publicly available.
      2. Reindex to the daily backtest index and forward-fill (a quarterly
         observation persists until the next report).
      3. Pre-history values stay NaN — we don't back-fill into time the
         data didn't exist.
    """
    out = pd.DataFrame(np.nan, index=daily_index, columns=tickers)
    lag_days = pd.Timedelta(days=90 * lag_quarters)
    for ticker, series in series_by_ticker.items():
        if ticker not in tickers or series is None or series.empty:
            continue
        lagged = series.copy()
        lagged.index = lagged.index + lag_days
        # Reindex to daily, ffill — but only forward, never backward
        daily = lagged.reindex(daily_index, method="ffill")
        out[ticker] = daily.values
    return out


def _compute_ratios(
    raw: dict[str, pd.DataFrame],
    close_matrix: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Derive market-relative ratios from raw fundamentals + close prices.

    All ratios are aligned to the close matrix's (dates × tickers) shape.
    Division-by-zero cells return NaN, not inf.

    For ratios involving a market-cap proxy we use ``close × shares_outstanding``.
    Since we don't have shares_outstanding from yfinance reliably, we instead
    use ``close × net_income / eps`` (implied shares) where eps is non-NaN —
    a cheap workaround that's stable for most large-caps.
    """
    out: dict[str, pd.DataFrame] = {}

    def safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
        return num / den.replace(0, np.nan)

    rev = raw.get("revenue")
    ni = raw.get("net_income")
    eps = raw.get("eps")
    book = raw.get("total_equity")
    debt = raw.get("total_debt")
    assets = raw.get("total_assets")
    cur_a = raw.get("current_assets")
    cur_l = raw.get("current_liabilities")
    gross = raw.get("gross_profit")
    op = raw.get("operating_income")
    fcf = raw.get("free_cash_flow")
    ebitda = raw.get("ebitda")
    cash = raw.get("cash")

    # Implied shares = net_income / eps (per ticker, per day) — only valid
    # where both are finite and non-zero.
    if ni is not None and eps is not None:
        shares = (ni / eps.replace(0, np.nan)).abs()
        market_cap = close_matrix * shares
    else:
        shares = None
        market_cap = None

    # ---- Pure profitability ratios (no market cap needed) ----
    if ni is not None and book is not None:
        out["roe"] = safe_div(ni, book)
    if ni is not None and assets is not None:
        out["roa"] = safe_div(ni, assets)
    if debt is not None and book is not None:
        out["debt_to_equity"] = safe_div(debt, book)
    if cur_a is not None and cur_l is not None:
        out["current_ratio"] = safe_div(cur_a, cur_l)
    if gross is not None and rev is not None:
        out["gross_margin"] = safe_div(gross, rev)
    if op is not None and rev is not None:
        out["operating_margin"] = safe_div(op, rev)

    # ---- Market-relative ratios (need market cap) ----
    if market_cap is not None:
        if ni is not None:
            out["pe_ratio"] = safe_div(market_cap, ni)
        if book is not None:
            out["pb_ratio"] = safe_div(market_cap, book)
        if rev is not None:
            out["ps_ratio"] = safe_div(market_cap, rev)
        if fcf is not None:
            out["fcf_yield"] = safe_div(fcf, market_cap)
        # Enterprise value / EBITDA — EV ≈ market_cap + total_debt - cash
        if ebitda is not None and debt is not None:
            ev = market_cap + debt
            if cash is not None:
                ev = ev - cash
            out["ev_ebitda"] = safe_div(ev, ebitda)

    return out


def download_fundamentals(
    tickers: list[str],
    daily_index: pd.DatetimeIndex,
    close_matrix: pd.DataFrame,
    *,
    fetch_fn=None,
    lag_quarters: int = LAG_QUARTERS,
) -> dict[str, pd.DataFrame]:
    """Top-level loader.  Returns ``{field_name: (dates × tickers) DataFrame}``.

    ``fetch_fn`` is injectable for testing — defaults to the real yfinance
    fetcher.  Tickers that fail fetch contribute NaN columns to every field
    (no exception raised).
    """
    fetch_fn = fetch_fn or _yfinance_fetch

    # Per-field accumulator: {field: {ticker: Series}}
    raw_collected: dict[str, dict[str, pd.Series]] = {
        field: {} for field in RAW_FUNDAMENTAL_FIELDS
    }

    # Load (cache or fetch) each ticker's three frames, then extract every
    # labeled row we know about.
    for ticker in tickers:
        income, balance, cashflow = _per_ticker_frames(ticker, fetch_fn)
        for field, series in _extract_quarterly(income, balance, cashflow,
                                                INCOME_LABELS, "income").items():
            raw_collected[field][ticker] = series
        for field, series in _extract_quarterly(income, balance, cashflow,
                                                BALANCE_LABELS, "balance").items():
            raw_collected[field][ticker] = series
        for field, series in _extract_quarterly(income, balance, cashflow,
                                                CASHFLOW_LABELS, "cashflow").items():
            raw_collected[field][ticker] = series

        # Synthesize free_cash_flow if missing but OCF + capex are present
        ocf = raw_collected["operating_cash_flow"].get(ticker)
        capex = raw_collected["capex"].get(ticker)
        existing_fcf = raw_collected["free_cash_flow"].get(ticker)
        if existing_fcf is None and ocf is not None and capex is not None:
            ocf_a, capex_a = ocf.align(capex, join="inner")
            # capex is reported negative in yfinance; FCF = OCF + capex
            raw_collected["free_cash_flow"][ticker] = ocf_a + capex_a

    # Materialize each raw field as a (daily × tickers) DataFrame
    raw_matrices: dict[str, pd.DataFrame] = {}
    for field in RAW_FUNDAMENTAL_FIELDS:
        raw_matrices[field] = _build_per_field_matrix(
            raw_collected[field], daily_index, tickers, lag_quarters=lag_quarters,
        )

    # Compute ratios (need close matrix; aligned to the same daily index)
    aligned_close = close_matrix.reindex(index=daily_index, columns=tickers)
    ratio_matrices = _compute_ratios(raw_matrices, aligned_close)

    return {**raw_matrices, **ratio_matrices}
