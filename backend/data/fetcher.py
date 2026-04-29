from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from config import CACHE_DIR, DATA_END, DATA_START, UNIVERSE

CACHE_TTL_SECONDS = 24 * 60 * 60
PRICE_FIELDS = {"open", "high", "low", "close", "volume"}
DERIVED_FIELDS = {"returns", "vwap"}


class DataFetcher:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, pd.DataFrame] = {}

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"

        df["returns"] = df["close"].pct_change()
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
        return df

    def _download_one(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as exc:
            warnings.warn(f"[{ticker}] download failed: {exc}")
            return None
        if raw is None or raw.empty:
            warnings.warn(f"[{ticker}] no data returned")
            return None
        return self._normalize(raw)

    def download_universe(
        self,
        tickers: Iterable[str] = UNIVERSE,
        start: str = DATA_START,
        end: str = DATA_END,
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            path = self._cache_path(ticker)
            if self._is_cache_fresh(path):
                try:
                    df = pd.read_parquet(path)
                    result[ticker] = df
                    continue
                except Exception as exc:
                    warnings.warn(f"[{ticker}] cache read failed ({exc}); re-downloading")

            df = self._download_one(ticker, start, end)
            if df is None:
                continue
            try:
                df.to_parquet(path)
            except Exception as exc:
                warnings.warn(f"[{ticker}] cache write failed: {exc}")
            result[ticker] = df

        self._frames = result
        return result

    def get_data_matrix(self, field: str) -> pd.DataFrame:
        field = field.lower()
        if field not in PRICE_FIELDS and field not in DERIVED_FIELDS:
            raise ValueError(
                f"Unknown field '{field}'. Expected one of "
                f"{sorted(PRICE_FIELDS | DERIVED_FIELDS)}."
            )

        if not self._frames:
            self.download_universe()

        series = {}
        for ticker, df in self._frames.items():
            if field in df.columns:
                series[ticker] = df[field]

        if not series:
            return pd.DataFrame()

        matrix = pd.concat(series, axis=1).sort_index()
        matrix.columns.name = "ticker"
        return matrix
