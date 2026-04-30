"""Fama-French 5-factor daily returns.

Pulls Ken French's free CSV (daily returns of Mkt-RF, SMB, HML, RMW, CMA, RF
since 1963), parses it, and caches as a parquet file alongside the universe
data.  Cache is refreshed weekly — the academic factor data updates monthly
so daily refreshes would be wasted bandwidth.
"""

from __future__ import annotations

import io
import time
import warnings
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from config import CACHE_DIR

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_CACHE = CACHE_DIR / "factors_ff5_daily.parquet"
FF5_TTL_SECONDS = 7 * 24 * 60 * 60  # weekly refresh

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _is_fresh(path: Path, ttl: int = FF5_TTL_SECONDS) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


def _parse_ff5_csv(text: str) -> pd.DataFrame:
    """The Ken French CSV has a multi-line preamble before the data.  Walk
    until we find the row that starts the numeric data block (date in column 0
    in YYYYMMDD form), then parse with pandas.
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        upper = line.upper()
        if "MKT-RF" in upper and "SMB" in upper and "HML" in upper:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("FF5 CSV: could not locate header row")

    # The data block ends at the first line that doesn't start with a date.
    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        first_token = stripped.split(",")[0].strip()
        if not first_token.isdigit():
            break
        data_lines.append(line)

    body = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(body))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).rename_axis("date")
    # Ken French publishes values as percentages — convert to fractional.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def download_ff5_daily(force: bool = False) -> pd.DataFrame:
    """Return FF5 daily factor returns as a (date × factors) DataFrame.

    Uses a parquet cache with a 7-day TTL.  On network failure, falls back to
    whatever cached copy exists (even if stale).  Returns an empty DataFrame
    if neither network nor cache is available.
    """
    FF5_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if not force and _is_fresh(FF5_CACHE):
        try:
            return pd.read_parquet(FF5_CACHE)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"FF5 cache read failed ({exc}); re-downloading")

    try:
        resp = httpx.get(FF5_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv")),
                None,
            )
            if csv_name is None:
                raise ValueError("FF5 zip contained no CSV")
            text = zf.read(csv_name).decode("latin-1")
        df = _parse_ff5_csv(text)
        try:
            df.to_parquet(FF5_CACHE)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"FF5 cache write failed: {exc}")
        return df
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"FF5 download failed: {exc}")
        if FF5_CACHE.exists():
            try:
                return pd.read_parquet(FF5_CACHE)
            except Exception:
                pass
        return pd.DataFrame()
