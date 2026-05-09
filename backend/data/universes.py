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

from typing import Any

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
    },
    "sp100_extended": {
        "name": "S&P 100 — extended (75)",
        "description": "Top 50 plus 25 of the largest other S&P 100 members.",
        "tickers": _SP100_EXTENDED,
        "is_default": False,
    },
    "nasdaq100": {
        "name": "NASDAQ-100 subset (45)",
        "description": "Tech/Comm Services–heavy; no banks, no oil, no industrials.",
        "tickers": _NASDAQ100,
        "is_default": False,
    },
    "tech_focus": {
        "name": "Tech & Comm Services focus",
        "description": "All catalog tickers in IT or Communication Services.",
        "tickers": _TECH_FOCUS,
        "is_default": False,
    },
}


# ---------- Public API ----------


GICS_LEVELS = ("sector", "industry_group", "industry", "sub_industry")


def list_universes() -> list[dict[str, Any]]:
    """All built-in universes with metadata for the /api/universes endpoint."""
    out = []
    for uid, u in _UNIVERSES.items():
        out.append({
            "id": uid,
            "name": u["name"],
            "description": u["description"],
            "ticker_count": len(u["tickers"]),
            "is_default": u["is_default"],
        })
    return out


def get_universe(universe_id: str) -> dict[str, Any]:
    """Look up a built-in universe.

    Raises ``KeyError`` if the id is unknown.  Callers (the API layer) should
    translate to a 400 with a useful message.
    """
    if universe_id not in _UNIVERSES:
        raise KeyError(
            f"Unknown universe {universe_id!r}. "
            f"Known: {sorted(_UNIVERSES.keys())}"
        )
    u = _UNIVERSES[universe_id]
    return {
        "id": universe_id,
        "name": u["name"],
        "tickers": list(u["tickers"]),
        "gics": gics_for(u["tickers"]),
    }


def gics_for(tickers: list[str]) -> dict[str, dict[str, str | None]]:
    """Per-ticker GICS row.

    Tickers absent from the catalog get a row of ``None`` values — the
    backtester's neutralization will skip these gracefully (or fall back to
    ``market`` neutralization).
    """
    out: dict[str, dict[str, str | None]] = {}
    for t in tickers:
        gics = _GICS_CATALOG.get(t)
        if gics is None:
            out[t] = {level: None for level in GICS_LEVELS}
        else:
            out[t] = dict(zip(GICS_LEVELS, gics))
    return out


def all_tickers() -> set[str]:
    """Union of every built-in universe's tickers — preloaded at startup."""
    out: set[str] = set()
    for u in _UNIVERSES.values():
        out.update(u["tickers"])
    return out


def default_universe_id() -> str:
    for uid, u in _UNIVERSES.items():
        if u["is_default"]:
            return uid
    # Fallback if no preset is marked default
    return next(iter(_UNIVERSES))


def gics_data_frames(
    dates,
    tickers: list[str],
):
    """Build (dates × tickers) string DataFrames for each GICS level.

    Used by the evaluator so expressions can reference ``sector``,
    ``industry_group``, etc. as data fields and feed them into the group_*
    operators.  Each cell is the ticker's GICS label string (constant across
    dates per ticker).  Tickers absent from the catalog get NaN, which the
    group operators treat as "exclude from groupby".
    """
    import pandas as _pd  # local import keeps universes.py framework-free at top

    gics = gics_for(tickers)
    out: dict[str, _pd.DataFrame] = {}
    for level in GICS_LEVELS:
        per_ticker = [gics[t].get(level) for t in tickers]
        # Broadcast the per-ticker label across all dates
        out[level] = _pd.DataFrame(
            [per_ticker] * len(dates),
            index=dates,
            columns=tickers,
        )
    return out


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
