from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from config import CACHE_DIR, DATA_END, DATA_START, UNIVERSE

CACHE_TTL_SECONDS = 24 * 60 * 60

# 7 fields stored per-ticker in {ticker}.parquet
BASE_FIELDS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "returns", "vwap",
)

# 25 fields computed from base matrices, cached at {field}.parquet
DERIVED_FIELDS: tuple[str, ...] = (
    # price structure
    "median_price", "weighted_close", "range_", "body",
    "upper_shadow", "lower_shadow", "gap",
    # return variants
    "log_returns", "abs_returns", "intraday_return",
    "overnight_return", "signed_volume",
    # volume & liquidity
    "dollar_volume", "adv20", "volume_ratio", "amihud",
    # volatility & risk
    "true_range", "atr", "realized_vol", "skewness", "kurtosis",
    # momentum & relative
    "momentum_5", "momentum_20", "close_to_high_252", "high_low_ratio",
)

ALL_FIELDS: tuple[str, ...] = BASE_FIELDS + DERIVED_FIELDS  # 32 total


class DataFetcher:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, pd.DataFrame] = {}
        self._matrix: dict[str, pd.DataFrame] = {}

    # ---------- paths / cache ----------

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def _derived_cache_path(self, field: str) -> Path:
        return self.cache_dir / f"{field}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS

    # ---------- normalize / download ----------

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
        compute_derived: bool = True,
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
        self._build_matrices(compute_derived=compute_derived)
        return result

    # ---------- matrix construction ----------

    def _build_matrices(self, compute_derived: bool = True) -> None:
        self._matrix.clear()
        if not self._frames:
            return

        for field in BASE_FIELDS:
            series = {
                t: df[field] for t, df in self._frames.items() if field in df.columns
            }
            if not series:
                continue
            m = pd.concat(series, axis=1).sort_index()
            m.columns.name = "ticker"
            self._matrix[field] = m

        if compute_derived:
            self._compute_derived_fields()

    def _compute_derived_fields(self) -> None:
        # Try cache first
        if self._load_derived_caches():
            return
        if not all(f in self._matrix for f in ("open", "high", "low", "close", "volume", "returns")):
            return

        o = self._matrix["open"]
        h = self._matrix["high"]
        l = self._matrix["low"]
        c = self._matrix["close"]
        v = self._matrix["volume"]
        r = self._matrix["returns"]
        idx, cols = c.index, c.columns

        def df_from_array(arr: np.ndarray) -> pd.DataFrame:
            out = pd.DataFrame(arr, index=idx, columns=cols)
            out.columns.name = "ticker"
            return out

        # ----- price structure -----
        self._matrix["median_price"] = (h + l) / 2.0
        self._matrix["weighted_close"] = (h + l + 2.0 * c) / 4.0
        self._matrix["range_"] = h - l
        self._matrix["body"] = (c - o).abs()
        # element-wise max/min of (open, close) — go via .values to dodge MultiIndex pitfalls
        oc_max = df_from_array(np.maximum(o.values, c.values))
        oc_min = df_from_array(np.minimum(o.values, c.values))
        self._matrix["upper_shadow"] = h - oc_max
        self._matrix["lower_shadow"] = oc_min - l
        self._matrix["gap"] = o - c.shift(1)

        # ----- return variants -----
        self._matrix["log_returns"] = np.log(c / c.shift(1))
        self._matrix["abs_returns"] = r.abs()
        self._matrix["intraday_return"] = (c - o) / o
        prev_close = c.shift(1)
        self._matrix["overnight_return"] = (o - prev_close) / prev_close
        self._matrix["signed_volume"] = v * np.sign(r)

        # ----- volume & liquidity -----
        dv = c * v
        self._matrix["dollar_volume"] = dv
        adv20 = v.rolling(20).mean()
        self._matrix["adv20"] = adv20
        self._matrix["volume_ratio"] = v / adv20.replace(0, np.nan)
        self._matrix["amihud"] = r.abs() / dv.replace(0, np.nan)

        # ----- volatility & risk -----
        hl = h - l
        hc = (h - prev_close).abs()
        lc = (l - prev_close).abs()
        tr_arr = np.maximum(np.maximum(hl.values, hc.values), lc.values)
        true_range = df_from_array(tr_arr)
        self._matrix["true_range"] = true_range
        self._matrix["atr"] = true_range.rolling(14).mean()
        self._matrix["realized_vol"] = r.rolling(20).std()
        self._matrix["skewness"] = r.rolling(60).skew()
        self._matrix["kurtosis"] = r.rolling(60).kurt()

        # ----- momentum & relative -----
        self._matrix["momentum_5"] = c / c.shift(5) - 1.0
        self._matrix["momentum_20"] = c / c.shift(20) - 1.0
        self._matrix["close_to_high_252"] = c / c.rolling(252).max()
        self._matrix["high_low_ratio"] = h / l

        self._save_derived_caches()

    def _load_derived_caches(self) -> bool:
        loaded: dict[str, pd.DataFrame] = {}
        for field in DERIVED_FIELDS:
            path = self._derived_cache_path(field)
            if not path.exists() or not self._is_cache_fresh(path):
                return False
            try:
                loaded[field] = pd.read_parquet(path)
            except Exception:
                return False

        # Sanity: derived caches must cover the same tickers as the current frames
        if self._frames:
            expected = set(self._frames.keys())
            for m in loaded.values():
                if set(m.columns) != expected:
                    return False
        self._matrix.update(loaded)
        return True

    def _save_derived_caches(self) -> None:
        for field in DERIVED_FIELDS:
            if field not in self._matrix:
                continue
            try:
                self._matrix[field].to_parquet(self._derived_cache_path(field))
            except Exception as exc:
                warnings.warn(f"[{field}] derived cache write failed: {exc}")

    # ---------- public lookup ----------

    def get_data_matrix(self, field: str) -> pd.DataFrame:
        field = field.lower()
        if field not in ALL_FIELDS:
            raise ValueError(
                f"Unknown field {field!r}. Expected one of {list(ALL_FIELDS)}."
            )
        if field in self._matrix:
            return self._matrix[field]
        if not self._frames:
            self.download_universe()
        return self._matrix.get(field, pd.DataFrame())

    # ---------- lazy expansion (custom universes) ----------

    def ensure_tickers(
        self,
        tickers: Iterable[str],
        start: str = DATA_START,
        end: str = DATA_END,
    ) -> list[str]:
        """Download any tickers not already in `_frames` and rebuild matrices.

        Returns the list of tickers that *failed* to load (yfinance returned
        nothing).  Callers can surface this so the user knows which symbols
        from a custom universe were silently dropped.

        Idempotent: tickers already present are skipped.  Rebuilds derived
        matrices since the column set has changed.
        """
        wanted = [t.upper().strip() for t in tickers if t and t.strip()]
        missing = [t for t in wanted if t not in self._frames]
        if not missing:
            return []

        failed: list[str] = []
        for t in missing:
            path = self._cache_path(t)
            if self._is_cache_fresh(path):
                try:
                    self._frames[t] = pd.read_parquet(path)
                    continue
                except Exception:
                    pass
            df = self._download_one(t, start, end)
            if df is None:
                failed.append(t)
                continue
            try:
                df.to_parquet(path)
            except Exception:
                pass
            self._frames[t] = df

        # Derived caches were keyed by the prior column set — recompute now.
        # Skip the per-field on-disk caches; they were stale the moment we
        # added a new ticker, and rewriting them on every custom-universe
        # request would just churn the cache for no benefit.
        self._matrix.clear()
        self._build_matrices(compute_derived=True)
        return failed
