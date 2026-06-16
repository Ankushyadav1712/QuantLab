"""FRED macro data loader.

Fetches a small curated set of free FRED daily series (VIX, Treasury yields,
credit / FX / commodity proxies), parses the CSV, caches as parquet, and
exposes them as pandas Series indexed by date.

Each series is broadcast to a (dates × tickers) matrix at lifespan-time so
operators don't have to special-case macro fields — `vix` looks identical to
`close` from the operators' point of view.

No API key required. FRED's public CSV endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}

Network failures degrade gracefully: any series that can't be fetched OR
loaded from cache is simply skipped.  The data-quality banner surfaces which
fields are present so the user knows their alpha can't reference a missing one.
"""

from __future__ import annotations

import logging
import time
import warnings
from io import StringIO
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from config import CACHE_DIR

CACHE_TTL_SECONDS = 24 * 60 * 60
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Map our user-facing field name -> FRED series id (and a short description
# used in the /api/operators metadata).  Daily series only — monthly series
# (like CPI) get awkward when forward-filled to a daily index.
FRED_SERIES: dict[str, tuple[str, str]] = {
    # Volatility
    "vix": ("VIXCLS", "CBOE VIX index — implied vol of S&P 500 options"),
    # Treasury yields
    "treasury_3m_yield": ("DGS3MO", "3-month Treasury constant-maturity yield"),
    "treasury_2y_yield": ("DGS2", "2-year Treasury constant-maturity yield"),
    "treasury_10y_yield": ("DGS10", "10-year Treasury constant-maturity yield"),
    # Credit
    "high_yield_spread": ("BAMLH0A0HYM2", "ICE BofA US High Yield Index OAS spread"),
    "baa_yield": ("DBAA", "Moody's seasoned Baa corporate bond yield (level)"),
    "aaa_yield": ("DAAA", "Moody's seasoned Aaa corporate bond yield (level)"),
    # FX / commodities
    "dxy": ("DTWEXBGS", "Broad trade-weighted dollar index"),
    "wti_oil": ("DCOILWTICO", "WTI crude oil spot price"),
    # NOTE: VVIXCLS (vvix) and GOLDAMGBD228NLBM (gold_price) were removed —
    # both series have been retired from FRED's free CSV endpoint and return
    # 404.  No clean free replacement on FRED for either.  yfinance has GLD
    # and ^VIX-VIX9D as alternatives if you want to add them via a separate
    # path later.
}

# Computed from base series after fetch.  Each entry: (output_name, fn(series_dict) -> Series).
# Keeps us from hitting FRED twice for the same data.
DERIVED_MACRO_BUILDERS: list[tuple[str, str]] = [
    # name, formula description (executed inline in compute_derived_macro)
    ("term_spread_10y_2y", "treasury_10y_yield - treasury_2y_yield"),
    ("term_spread_10y_3m", "treasury_10y_yield - treasury_3m_yield"),
    ("credit_spread_baa_aaa", "baa_yield - aaa_yield"),
]

# Combined surface used by parser/editor/FIELDS metadata
ALL_MACRO_FIELDS: tuple[str, ...] = (
    *FRED_SERIES.keys(),
    *(name for name, _ in DERIVED_MACRO_BUILDERS),
)

log = logging.getLogger("quantlab.macro")


def _cache_path(field: str) -> Path:
    return Path(CACHE_DIR) / f"macro__{field}.parquet"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _parse_fred_csv(text: str) -> pd.Series:
    """Parse a FRED CSV blob into a date-indexed Series.

    FRED's CSV uses ``observation_date`` (recent format) or ``DATE`` (older)
    in the first column and the series id in the second.  Missing values are
    encoded as ``.`` — coerce those to NaN.
    """
    df = pd.read_csv(StringIO(text))
    if df.shape[1] < 2:
        raise ValueError(f"FRED CSV has only {df.shape[1]} columns; expected 2")
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    # Coerce '.' (FRED's missing marker) and any other non-numeric to NaN
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    series = df.set_index(date_col)[value_col].sort_index()
    series.index.name = "date"
    series.name = None
    return series


def _download_one(field: str, series_id: str, *, timeout: float = 5.0) -> pd.Series | None:
    """Fetch one FRED series, returning None on any error."""
    url = FRED_CSV_URL.format(series_id=series_id)
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        warnings.warn(f"[macro:{field}] download failed ({series_id}): {exc}")
        return None
    try:
        return _parse_fred_csv(resp.text)
    except (ValueError, pd.errors.ParserError) as exc:
        warnings.warn(f"[macro:{field}] parse failed ({series_id}): {exc}")
        return None


def download_macro(*, fetch_fn=_download_one) -> dict[str, pd.Series]:
    """Fetch every base FRED series + compute derived spreads.

    Cache hit returns immediately; cache miss attempts a fresh download and
    falls back to a stale parquet if HTTP fails.  ``fetch_fn`` is injectable
    so tests can stub the network layer without monkeypatching httpx.
    """
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    series_dict: dict[str, pd.Series] = {}

    from concurrent.futures import ThreadPoolExecutor

    # Phase 1: Load from cache or download in parallel
    def _fetch_macro_field(field_name: str, sid: str) -> tuple[str, pd.Series | None]:
        path = _cache_path(field_name)
        if _is_cache_fresh(path):
            try:
                cached = pd.read_parquet(path)
                return field_name, cached.iloc[:, 0]
            except Exception as exc:
                warnings.warn(f"[macro:{field_name}] cache read failed ({exc}); re-downloading")

        fresh = fetch_fn(field_name, sid)
        if fresh is None:
            if path.exists():
                try:
                    cached = pd.read_parquet(path)
                    log.warning(f"macro {field_name} using stale cache")
                    return field_name, cached.iloc[:, 0]
                except Exception:
                    pass
            return field_name, None

        try:
            fresh.to_frame(name=field_name).to_parquet(path)
        except Exception as exc:
            warnings.warn(f"[macro:{field_name}] cache write failed: {exc}")

        return field_name, fresh

    with ThreadPoolExecutor(max_workers=len(FRED_SERIES)) as executor:
        futures = [
            executor.submit(_fetch_macro_field, field, sid)
            for field, (sid, _desc) in FRED_SERIES.items()
        ]
        for fut in futures:
            field, series = fut.result()
            if series is not None:
                series_dict[field] = series

    # Derived spreads — only if the base series are present
    for name, _formula in DERIVED_MACRO_BUILDERS:
        if name == "term_spread_10y_2y":
            if "treasury_10y_yield" in series_dict and "treasury_2y_yield" in series_dict:
                a, b = series_dict["treasury_10y_yield"], series_dict["treasury_2y_yield"]
                aligned_a, aligned_b = a.align(b, join="outer")
                series_dict[name] = aligned_a - aligned_b
        elif name == "term_spread_10y_3m":
            if "treasury_10y_yield" in series_dict and "treasury_3m_yield" in series_dict:
                a, b = series_dict["treasury_10y_yield"], series_dict["treasury_3m_yield"]
                aligned_a, aligned_b = a.align(b, join="outer")
                series_dict[name] = aligned_a - aligned_b
        elif name == "credit_spread_baa_aaa":
            if "baa_yield" in series_dict and "aaa_yield" in series_dict:
                a, b = series_dict["baa_yield"], series_dict["aaa_yield"]
                aligned_a, aligned_b = a.align(b, join="outer")
                series_dict[name] = aligned_a - aligned_b

    return series_dict


def broadcast_to_matrix(
    series: pd.Series, dates: pd.DatetimeIndex, tickers: list[str]
) -> pd.DataFrame:
    """Reindex ``series`` to ``dates`` and broadcast across every ticker column.

    Forward-fills across missing trading days (FRED holidays often differ
    from US equity market holidays).  Values before the series' first
    observation stay NaN — operators treat that as "data not yet available".
    """
    aligned = series.reindex(dates).ffill()
    return pd.DataFrame(
        np.broadcast_to(aligned.values[:, None], (len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
