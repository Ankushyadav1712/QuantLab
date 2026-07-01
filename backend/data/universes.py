"""Preset trading universes + their GICS classification metadata.

Each universe is a named ticker list with a `default` flag and a
description.  The GICS metadata (sector / industry_group / industry /
sub_industry) lives in a *shared* catalog because the same ticker has the
same classification regardless of which universe it appears in.

Adding a new built-in universe:
    1. Append its tickers to the relevant ``_UNIVERSES`` entry.
    2. Make sure every ticker has an entry in ``_GICS_CATALOG``.
    3. The lifespan handler picks it up automatically.

Adding a new ticker to the catalog:
    Use the post-2023 GICS taxonomy (the major reclassification moved
    Visa/Mastercard from Information Technology → Financials, e-commerce
    retailers into Consumer Discretionary Distribution & Retail, etc.).

Custom universes (Phase 2) supply a ticker list at request time; tickers
not present in the catalog get a ``None`` GICS row, which the backtester
handles by limiting the available neutralization modes.
"""

from __future__ import annotations

import json
import time
import warnings as _warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths for lazy ticker lists and dynamic GICS cache
# ---------------------------------------------------------------------------

_TICKERS_DIR = Path(__file__).parent / "tickers"
_GICS_DYNAMIC_CACHE = Path(__file__).parent / "cache" / "gics_dynamic_cache.json"
_TICKER_LIST_TTL = 30 * 24 * 3600   # 30 days before re-fetching from source

# yfinance sector label → GICS sector name
_YF_TO_GICS: dict[str, str] = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
    # Pass-through when yfinance already returns GICS names
    "Information Technology": "Information Technology",
    "Health Care": "Health Care",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Financials": "Financials",
    "Materials": "Materials",
}

# ---------------------------------------------------------------------------
# Dynamic GICS cache  (JSON on disk, 90-day TTL per-entry)
# ---------------------------------------------------------------------------


def _load_gics_dynamic_cache() -> dict[str, list[str]]:
    """Return {ticker: [sector, industry_group, industry, sub_industry]} from cache."""
    try:
        if _GICS_DYNAMIC_CACHE.exists():
            return json.loads(_GICS_DYNAMIC_CACHE.read_text()).get("entries", {})
    except Exception:
        pass
    return {}


def _save_gics_dynamic_cache(new_entries: dict[str, list[str]]) -> None:
    """Merge new_entries into the persistent GICS cache file."""
    existing: dict[str, list[str]] = {}
    try:
        if _GICS_DYNAMIC_CACHE.exists():
            existing = json.loads(_GICS_DYNAMIC_CACHE.read_text()).get("entries", {})
    except Exception:
        pass
    existing.update(new_entries)
    try:
        _GICS_DYNAMIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _GICS_DYNAMIC_CACHE.write_text(json.dumps({"entries": existing}))
    except Exception as exc:
        _warnings.warn(f"[gics_cache] write failed: {exc}")


# ---------------------------------------------------------------------------
# Ticker list helpers (file cache + web fetch)
# ---------------------------------------------------------------------------


def _load_or_fetch_ticker_list(
    name: str,
    fetch_fn,  # () -> list[str]
) -> list[str]:
    """Return ticker list from cache file, or call fetch_fn() and save the result.

    The cache file lives at backend/data/tickers/{name}.txt and expires after
    _TICKER_LIST_TTL seconds.  Falls back to a stale cache if the fetch fails.
    """
    _TICKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = _TICKERS_DIR / f"{name}.txt"

    if path.exists() and (time.time() - path.stat().st_mtime) < _TICKER_LIST_TTL:
        tickers = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if tickers:
            return tickers

    try:
        tickers = fetch_fn()
        if tickers:
            path.write_text("\n".join(tickers))
            return tickers
    except Exception as exc:
        _warnings.warn(f"[universe:{name}] ticker fetch failed: {exc}")
        # Fall back to stale cache if it exists
        if path.exists():
            stale = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            if stale:
                _warnings.warn(f"[universe:{name}] using stale cache ({len(stale)} tickers)")
                return stale

    return []


def _fetch_sp500_tickers_and_gics() -> tuple[list[str], dict[str, list[str]]]:
    """Scrape Wikipedia for S&P 500 symbols + GICS data (one HTTP request)."""
    import pandas as _pd  # noqa: PLC0415

    tables = _pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]

    sym_col = next(
        (c for c in df.columns if str(c).strip().lower().startswith("symbol")),
        df.columns[0],
    )
    sector_col = next(
        (c for c in df.columns if "gics sector" in str(c).strip().lower()), None
    )
    sub_col = next(
        (c for c in df.columns if "sub-industry" in str(c).strip().lower()), None
    )

    tickers: list[str] = []
    gics_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        raw = str(row[sym_col]).strip()
        if not raw or raw.lower() == "nan":
            continue
        t = raw.replace(".", "-")
        tickers.append(t)
        sector = (
            str(row[sector_col]).strip()
            if sector_col and str(row[sector_col]).strip().lower() != "nan"
            else "Unknown"
        )
        sub = (
            str(row[sub_col]).strip()
            if sub_col and str(row[sub_col]).strip().lower() != "nan"
            else "Unknown"
        )
        gics_map[t] = [sector, sector, sub, sub]

    return tickers, gics_map


def _get_sp500_tickers() -> list[str]:
    """Return S&P 500 ticker list, populating the GICS dynamic cache as a side-effect."""

    def _fetch() -> list[str]:
        tickers, gics_map = _fetch_sp500_tickers_and_gics()
        need_cache = {t: v for t, v in gics_map.items() if t not in _GICS_CATALOG}
        if need_cache:
            _save_gics_dynamic_cache(need_cache)
        return tickers

    return _load_or_fetch_ticker_list("sp500", _fetch)


def _get_russell1000_tickers() -> list[str]:
    """Return Russell 1000 tickers from the iShares IWB ETF holdings CSV."""

    def _fetch() -> list[str]:
        import io
        import urllib.request

        url = (
            "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf"
            "/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        import pandas as _pd  # noqa: PLC0415

        lines = raw.splitlines()
        # iShares CSV has metadata rows before the actual header
        start = next(
            (i for i, ln in enumerate(lines) if ln.strip().startswith("Ticker,")), None
        )
        if start is None:
            return []
        import io as _io  # noqa: PLC0415

        df = _pd.read_csv(_io.StringIO("\n".join(lines[start:])))
        raw_tickers: list[str] = list(map(str, df["Ticker"].dropna()))
        result: list[str] = []
        for t in raw_tickers:
            clean = str(t).strip().replace(".", "-")
            if clean and clean not in ("-", "nan") and len(clean) <= 5:
                result.append(clean)
        return result

    return _load_or_fetch_ticker_list("russell1000", _fetch)

# GICS classification.  Values are the post-2023 taxonomy.
# Format: ticker -> (sector, industry_group, industry, sub_industry)
_GICS_CATALOG: dict[str, tuple[str, str, str, str]] = {
    # ----- Information Technology -----
    "AAPL": (
        "Information Technology",
        "Technology Hardware & Equipment",
        "Technology Hardware, Storage & Peripherals",
        "Technology Hardware, Storage & Peripherals",
    ),
    "MSFT": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Systems Software",
    ),
    "NVDA": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "AVGO": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "ADBE": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Application Software",
    ),
    "CSCO": (
        "Information Technology",
        "Technology Hardware & Equipment",
        "Communications Equipment",
        "Communications Equipment",
    ),
    "CRM": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Application Software",
    ),
    "ACN": (
        "Information Technology",
        "Software & Services",
        "IT Services",
        "IT Consulting & Other Services",
    ),
    "TXN": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "INTC": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "ORCL": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Systems Software",
    ),
    "AMD": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "QCOM": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "IBM": (
        "Information Technology",
        "Software & Services",
        "IT Services",
        "IT Consulting & Other Services",
    ),
    "INTU": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Application Software",
    ),
    "AMAT": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductor Materials & Equipment",
    ),
    "MU": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "LRCX": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductor Materials & Equipment",
    ),
    "KLAC": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductor Materials & Equipment",
    ),
    "ADI": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "MRVL": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors",
    ),
    "PANW": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Systems Software",
    ),
    "FTNT": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Systems Software",
    ),
    "CDNS": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Application Software",
    ),
    "SNPS": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Application Software",
    ),
    "NOW": (
        "Information Technology",
        "Software & Services",
        "Software",
        "Systems Software",
    ),
    "ASML": (
        "Information Technology",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductors & Semiconductor Equipment",
        "Semiconductor Materials & Equipment",
    ),
    # ----- Communication Services -----
    "GOOG": (
        "Communication Services",
        "Media & Entertainment",
        "Interactive Media & Services",
        "Interactive Media & Services",
    ),
    "META": (
        "Communication Services",
        "Media & Entertainment",
        "Interactive Media & Services",
        "Interactive Media & Services",
    ),
    "NFLX": (
        "Communication Services",
        "Media & Entertainment",
        "Entertainment",
        "Movies & Entertainment",
    ),
    "VZ": (
        "Communication Services",
        "Telecommunication Services",
        "Diversified Telecommunication Services",
        "Integrated Telecommunication Services",
    ),
    "T": (
        "Communication Services",
        "Telecommunication Services",
        "Diversified Telecommunication Services",
        "Integrated Telecommunication Services",
    ),
    "CMCSA": (
        "Communication Services",
        "Media & Entertainment",
        "Media",
        "Cable & Satellite",
    ),
    "DIS": (
        "Communication Services",
        "Media & Entertainment",
        "Entertainment",
        "Movies & Entertainment",
    ),
    "TMUS": (
        "Communication Services",
        "Telecommunication Services",
        "Wireless Telecommunication Services",
        "Wireless Telecommunication Services",
    ),
    # ----- Consumer Discretionary -----
    "AMZN": (
        "Consumer Discretionary",
        "Consumer Discretionary Distribution & Retail",
        "Broadline Retail",
        "Broadline Retail",
    ),
    "TSLA": (
        "Consumer Discretionary",
        "Automobiles & Components",
        "Automobiles",
        "Automobile Manufacturers",
    ),
    "HD": (
        "Consumer Discretionary",
        "Consumer Discretionary Distribution & Retail",
        "Specialty Retail",
        "Home Improvement Retail",
    ),
    "MCD": (
        "Consumer Discretionary",
        "Consumer Services",
        "Hotels, Restaurants & Leisure",
        "Restaurants",
    ),
    "NKE": (
        "Consumer Discretionary",
        "Consumer Durables & Apparel",
        "Textiles, Apparel & Luxury Goods",
        "Footwear",
    ),
    "BKNG": (
        "Consumer Discretionary",
        "Consumer Services",
        "Hotels, Restaurants & Leisure",
        "Hotels, Resorts & Cruise Lines",
    ),
    "ABNB": (
        "Consumer Discretionary",
        "Consumer Services",
        "Hotels, Restaurants & Leisure",
        "Hotels, Resorts & Cruise Lines",
    ),
    "SBUX": (
        "Consumer Discretionary",
        "Consumer Services",
        "Hotels, Restaurants & Leisure",
        "Restaurants",
    ),
    "LOW": (
        "Consumer Discretionary",
        "Consumer Discretionary Distribution & Retail",
        "Specialty Retail",
        "Home Improvement Retail",
    ),
    "F": (
        "Consumer Discretionary",
        "Automobiles & Components",
        "Automobiles",
        "Automobile Manufacturers",
    ),
    "GM": (
        "Consumer Discretionary",
        "Automobiles & Components",
        "Automobiles",
        "Automobile Manufacturers",
    ),
    # ----- Consumer Staples -----
    "WMT": (
        "Consumer Staples",
        "Consumer Staples Distribution & Retail",
        "Consumer Staples Distribution & Retail",
        "Consumer Staples Merchandise Retail",
    ),
    "PG": (
        "Consumer Staples",
        "Household & Personal Products",
        "Household Products",
        "Household Products",
    ),
    "COST": (
        "Consumer Staples",
        "Consumer Staples Distribution & Retail",
        "Consumer Staples Distribution & Retail",
        "Consumer Staples Merchandise Retail",
    ),
    "PEP": (
        "Consumer Staples",
        "Food, Beverage & Tobacco",
        "Beverages",
        "Soft Drinks & Non-alcoholic Beverages",
    ),
    "KO": (
        "Consumer Staples",
        "Food, Beverage & Tobacco",
        "Beverages",
        "Soft Drinks & Non-alcoholic Beverages",
    ),
    "PM": (
        "Consumer Staples",
        "Food, Beverage & Tobacco",
        "Tobacco",
        "Tobacco",
    ),
    "MO": (
        "Consumer Staples",
        "Food, Beverage & Tobacco",
        "Tobacco",
        "Tobacco",
    ),
    "MDLZ": (
        "Consumer Staples",
        "Food, Beverage & Tobacco",
        "Food Products",
        "Packaged Foods & Meats",
    ),
    "MNST": (
        "Consumer Staples",
        "Food, Beverage & Tobacco",
        "Beverages",
        "Soft Drinks & Non-alcoholic Beverages",
    ),
    # ----- Health Care -----
    "UNH": (
        "Health Care",
        "Health Care Equipment & Services",
        "Health Care Providers & Services",
        "Managed Health Care",
    ),
    "LLY": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Pharmaceuticals",
        "Pharmaceuticals",
    ),
    "JNJ": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Pharmaceuticals",
        "Pharmaceuticals",
    ),
    "MRK": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Pharmaceuticals",
        "Pharmaceuticals",
    ),
    "ABBV": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Biotechnology",
        "Biotechnology",
    ),
    "TMO": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Life Sciences Tools & Services",
        "Life Sciences Tools & Services",
    ),
    "ABT": (
        "Health Care",
        "Health Care Equipment & Services",
        "Health Care Equipment & Supplies",
        "Health Care Equipment",
    ),
    "DHR": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Life Sciences Tools & Services",
        "Life Sciences Tools & Services",
    ),
    "AMGN": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Biotechnology",
        "Biotechnology",
    ),
    "GILD": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Biotechnology",
        "Biotechnology",
    ),
    "PFE": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Pharmaceuticals",
        "Pharmaceuticals",
    ),
    "ISRG": (
        "Health Care",
        "Health Care Equipment & Services",
        "Health Care Equipment & Supplies",
        "Health Care Equipment",
    ),
    "REGN": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Biotechnology",
        "Biotechnology",
    ),
    "VRTX": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Biotechnology",
        "Biotechnology",
    ),
    "MDT": (
        "Health Care",
        "Health Care Equipment & Services",
        "Health Care Equipment & Supplies",
        "Health Care Equipment",
    ),
    "BMY": (
        "Health Care",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Pharmaceuticals",
        "Pharmaceuticals",
    ),
    # ----- Financials -----
    "BRK-B": (
        "Financials",
        "Insurance",
        "Insurance",
        "Multi-line Insurance",
    ),
    "JPM": (
        "Financials",
        "Banks",
        "Banks",
        "Diversified Banks",
    ),
    "V": (
        "Financials",
        "Financial Services",
        "Financial Services",
        "Transaction & Payment Processing Services",
    ),
    "MA": (
        "Financials",
        "Financial Services",
        "Financial Services",
        "Transaction & Payment Processing Services",
    ),
    "MS": (
        "Financials",
        "Financial Services",
        "Capital Markets",
        "Investment Banking & Brokerage",
    ),
    "GS": (
        "Financials",
        "Financial Services",
        "Capital Markets",
        "Investment Banking & Brokerage",
    ),
    "SPGI": (
        "Financials",
        "Financial Services",
        "Capital Markets",
        "Financial Exchanges & Data",
    ),
    "BLK": (
        "Financials",
        "Financial Services",
        "Capital Markets",
        "Asset Management & Custody Banks",
    ),
    "BAC": (
        "Financials",
        "Banks",
        "Banks",
        "Diversified Banks",
    ),
    "WFC": (
        "Financials",
        "Banks",
        "Banks",
        "Diversified Banks",
    ),
    "C": (
        "Financials",
        "Banks",
        "Banks",
        "Diversified Banks",
    ),
    "AXP": (
        "Financials",
        "Financial Services",
        "Consumer Finance",
        "Consumer Finance",
    ),
    "SCHW": (
        "Financials",
        "Financial Services",
        "Capital Markets",
        "Investment Banking & Brokerage",
    ),
    "PYPL": (
        "Financials",
        "Financial Services",
        "Financial Services",
        "Transaction & Payment Processing Services",
    ),
    # ----- Energy -----
    "XOM": (
        "Energy",
        "Energy",
        "Oil, Gas & Consumable Fuels",
        "Integrated Oil & Gas",
    ),
    "CVX": (
        "Energy",
        "Energy",
        "Oil, Gas & Consumable Fuels",
        "Integrated Oil & Gas",
    ),
    "COP": (
        "Energy",
        "Energy",
        "Oil, Gas & Consumable Fuels",
        "Oil & Gas Exploration & Production",
    ),
    "SLB": (
        "Energy",
        "Energy",
        "Energy Equipment & Services",
        "Oil & Gas Equipment & Services",
    ),
    "OXY": (
        "Energy",
        "Energy",
        "Oil, Gas & Consumable Fuels",
        "Oil & Gas Exploration & Production",
    ),
    # ----- Industrials -----
    "RTX": (
        "Industrials",
        "Capital Goods",
        "Aerospace & Defense",
        "Aerospace & Defense",
    ),
    "ADP": (
        "Industrials",
        "Commercial & Professional Services",
        "Professional Services",
        "Human Resource & Employment Services",
    ),
    "HON": (
        "Industrials",
        "Capital Goods",
        "Industrial Conglomerates",
        "Industrial Conglomerates",
    ),
    "BA": (
        "Industrials",
        "Capital Goods",
        "Aerospace & Defense",
        "Aerospace & Defense",
    ),
    "CAT": (
        "Industrials",
        "Capital Goods",
        "Machinery",
        "Construction Machinery & Heavy Transportation Equipment",
    ),
    "GE": (
        "Industrials",
        "Capital Goods",
        "Aerospace & Defense",
        "Aerospace & Defense",
    ),
    "LMT": (
        "Industrials",
        "Capital Goods",
        "Aerospace & Defense",
        "Aerospace & Defense",
    ),
    "UPS": (
        "Industrials",
        "Transportation",
        "Air Freight & Logistics",
        "Air Freight & Logistics",
    ),
    "FDX": (
        "Industrials",
        "Transportation",
        "Air Freight & Logistics",
        "Air Freight & Logistics",
    ),
    "UNP": (
        "Industrials",
        "Transportation",
        "Ground Transportation",
        "Rail Transportation",
    ),
    "MMM": (
        "Industrials",
        "Capital Goods",
        "Industrial Conglomerates",
        "Industrial Conglomerates",
    ),
    "PCAR": (
        "Industrials",
        "Capital Goods",
        "Machinery",
        "Construction Machinery & Heavy Transportation Equipment",
    ),
    # ----- Materials -----
    "LIN": (
        "Materials",
        "Materials",
        "Chemicals",
        "Industrial Gases",
    ),
}


# ---------- Universe definitions ----------

# The 50-name S&P 100 subset that has been the platform's default since v1.
_SP100_50 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "AVGO", "COST", "PEP", "KO", "ADBE", "CSCO", "CRM", "ACN", "MCD",
    "TMO", "ABT", "NFLX", "LIN", "DHR", "TXN", "NKE", "VZ", "AMGN", "PM",
    "MS", "GS", "RTX", "INTC", "SPGI", "BLK", "MDLZ", "ADP", "GILD", "T",
]

# Extended S&P 100 — adds 25 of the larger non-50 members.  Not an exhaustive
# enumeration of all 100 names; that would require curating ~50 more GICS
# rows for marginal value.  Add more by extending the list + catalog.
_SP100_EXTENDED = sorted(set(_SP100_50 + [
    "AMD", "ORCL", "QCOM", "IBM", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC",
    "CMCSA", "DIS", "TMUS", "BKNG", "SBUX", "LOW", "F", "GM",
    "PFE", "ISRG", "MDT", "BMY", "BAC", "WFC", "C", "AXP", "SCHW",
    "COP", "OXY", "HON", "BA", "CAT", "GE", "LMT", "UPS", "FDX", "UNP", "MMM",
    "MO",
]))

# NASDAQ-100 leaning subset — heavily tech/communications, no banks, no oil.
_NASDAQ100 = sorted({
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
    "NFLX", "PEP", "ADBE", "CSCO", "INTC", "AMGN", "GILD", "AMD", "QCOM",
    "INTU", "AMAT", "MU", "LRCX", "KLAC", "ADI", "MRVL", "PANW", "FTNT",
    "CDNS", "SNPS", "ASML", "BKNG", "ABNB", "SBUX", "MNST", "ISRG",
    "REGN", "VRTX", "MDLZ", "PYPL", "TMUS", "CMCSA", "TXN", "ORCL",
    "PCAR", "NOW",
})

# Tech / Comm Services focus — useful for sector-specific cross-sectional alphas.
_TECH_FOCUS = sorted({
    t for t, gics in _GICS_CATALOG.items()
    if gics[0] in ("Information Technology", "Communication Services")
})


_UNIVERSES: dict[str, dict[str, Any]] = {
    "sp100_50": {
        "name": "S&P 100 — top 50",
        "description": "The original 50-name megacap subset; default for legacy alphas.",
        "tickers": _SP100_50,
        "is_default": True,
        "preload": True,
    },
    "sp100_extended": {
        "name": "S&P 100 — extended (75)",
        "description": "Top 50 plus 25 of the largest other S&P 100 members.",
        "tickers": _SP100_EXTENDED,
        "is_default": False,
        "preload": True,
    },
    "nasdaq100": {
        "name": "NASDAQ-100 subset (45)",
        "description": "Tech/Comm Services–heavy; no banks, no oil, no industrials.",
        "tickers": _NASDAQ100,
        "is_default": False,
        "preload": True,
    },
    "tech_focus": {
        "name": "Tech & Comm Services focus",
        "description": "All catalog tickers in IT or Communication Services.",
        "tickers": _TECH_FOCUS,
        "is_default": False,
        "preload": True,
    },
    # ---- Large universes: loaded on-demand, NOT preloaded at startup ----
    "sp500": {
        "name": "S&P 500 (~503 stocks)",
        "description": (
            "All S&P 500 components. Ticker list and GICS data fetched from "
            "Wikipedia on first use and cached for 30 days. Expands the active "
            "data pool on first simulate request (~2–3 min if uncached)."
        ),
        "tickers": None,          # lazy-loaded via _get_sp500_tickers()
        "ticker_count_estimate": 503,
        "is_default": False,
        "preload": False,
    },
    "russell1000": {
        "name": "Russell 1000 (~1000 stocks)",
        "description": (
            "Largest 1000 US stocks by market cap (iShares IWB ETF). "
            "Ticker list fetched on first use and cached for 30 days. "
            "First simulate request may take 5–10 min if data is uncached."
        ),
        "tickers": None,          # lazy-loaded via _get_russell1000_tickers()
        "ticker_count_estimate": 1000,
        "is_default": False,
        "preload": False,
    },
}


# ---------- Public API ----------


GICS_LEVELS = ("sector", "industry_group", "industry", "sub_industry")


def list_universes() -> list[dict[str, Any]]:
    """All built-in universes with metadata for the /api/universes endpoint."""
    out = []
    for uid, u in _UNIVERSES.items():
        # For lazy universes use the cached file count, or the estimate
        tickers = u["tickers"]
        if tickers is None:
            cache_path = _TICKERS_DIR / f"{uid}.txt"
            if cache_path.exists():
                ticker_count = sum(
                    1 for ln in cache_path.read_text().splitlines() if ln.strip()
                )
            else:
                ticker_count = u.get("ticker_count_estimate", 0)
        else:
            ticker_count = len(tickers)
        out.append(
            {
                "id": uid,
                "name": u["name"],
                "description": u["description"],
                "ticker_count": ticker_count,
                "is_default": u["is_default"],
                "preload": u.get("preload", True),
            }
        )
    return out


def get_universe(universe_id: str) -> dict[str, Any]:
    """Look up a built-in universe.

    Raises ``KeyError`` if the id is unknown.  Callers (the API layer) should
    translate to a 400 with a useful message.

    For large lazy universes (sp500, russell1000) the ticker list is fetched
    from the web on first call and cached on disk for 30 days.
    """
    if universe_id not in _UNIVERSES:
        raise KeyError(
            f"Unknown universe {universe_id!r}. "
            f"Known: {sorted(_UNIVERSES.keys())}"
        )
    u = _UNIVERSES[universe_id]
    tickers = u["tickers"]

    if tickers is None:
        # Lazy-load large universe ticker lists
        _fetchers: dict[str, Any] = {
            "sp500": _get_sp500_tickers,
            "russell1000": _get_russell1000_tickers,
        }
        fetch_fn = _fetchers.get(universe_id)
        if fetch_fn:
            tickers = fetch_fn()
        if not tickers:
            _warnings.warn(
                f"[universe:{universe_id}] ticker list unavailable; "
                "falling back to sp100_extended"
            )
            return get_universe("sp100_extended")

    return {
        "id": universe_id,
        "name": u["name"],
        "tickers": list(tickers),
        "gics": gics_for(list(tickers)),
    }


def gics_for(tickers: list[str]) -> dict[str, dict[str, str | None]]:
    """Per-ticker GICS row.

    Priority: hardcoded _GICS_CATALOG → dynamic on-disk cache (populated by
    Wikipedia / iShares fetch) → None (backtester falls back to market
    neutralization for unknown tickers).
    """
    dynamic = _load_gics_dynamic_cache()
    out: dict[str, dict[str, str | None]] = {}
    for t in tickers:
        catalog = _GICS_CATALOG.get(t)
        if catalog is not None:
            out[t] = dict(zip(GICS_LEVELS, catalog))
        elif t in dynamic:
            entry = dynamic[t]
            if isinstance(entry, list) and len(entry) >= 4:
                out[t] = dict(zip(GICS_LEVELS, entry[:4]))
            else:
                out[t] = {level: None for level in GICS_LEVELS}
        else:
            out[t] = {level: None for level in GICS_LEVELS}
    return out


def all_tickers() -> set[str]:
    """Union of every *preloaded* built-in universe's tickers.

    Large lazy universes (sp500, russell1000) are excluded so startup time
    stays fast.  They are loaded on-demand when a simulate request selects them.
    """
    out: set[str] = set()
    for u in _UNIVERSES.values():
        if u.get("preload", True) and u.get("tickers"):
            out.update(u["tickers"])
    return out


def default_universe_id() -> str:
    for uid, u in _UNIVERSES.items():
        if u["is_default"]:
            return uid
    # Fallback if no preset is marked default
    return next(iter(_UNIVERSES))


def available_neutralizations(
    gics_map: dict[str, dict[str, str | None]],
    *,
    min_groups: int = 2,
) -> list[str]:
    """Which neutralization modes are usable for this universe + GICS map.

    A mode is usable if ``min_groups`` distinct non-None values exist for that
    GICS level.  Single-group neutralization would zero out every weight.
    """
    modes = ["none", "market"]
    for level in GICS_LEVELS:
        groups = {row[level] for row in gics_map.values() if row.get(level)}
        if len(groups) >= min_groups:
            modes.append(level)
    return modes
