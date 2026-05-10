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
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "vwap",
)

# 53 fields computed from base matrices, cached at {field}.parquet
DERIVED_FIELDS: tuple[str, ...] = (
    # price structure
    "median_price",
    "weighted_close",
    "range_",
    "body",
    "upper_shadow",
    "lower_shadow",
    "gap",
    # return variants
    "log_returns",
    "abs_returns",
    "intraday_return",
    "overnight_return",
    "signed_volume",
    # volume & liquidity
    "dollar_volume",
    "adv20",
    "volume_ratio",
    "amihud",
    # volatility & risk
    "true_range",
    "atr",
    "realized_vol",
    "skewness",
    "kurtosis",
    # momentum & relative
    "momentum_5",
    "momentum_20",
    "close_to_high_252",
    "high_low_ratio",
    # ----- Phase B: extended momentum (8) -----
    "momentum_3",
    "momentum_10",
    "momentum_60",
    "momentum_120",
    "momentum_252",
    "reversal_5",
    "reversal_20",
    "momentum_z_60",
    # ----- Phase B: extended volatility (6) -----
    "realized_vol_5",
    "realized_vol_60",
    "realized_vol_120",
    "vol_of_vol_20",
    "parkinson_vol",
    "garman_klass_vol",
    # ----- Phase B: microstructure (8) -----
    "roll_spread",
    "kyle_lambda",
    "vpin_proxy",
    "up_volume_ratio",
    "down_volume_ratio",
    "turnover_ratio",
    "dollar_amihud",
    "corwin_schultz",
    # ----- Phase B: extended range / candle structure (6) -----
    "atr_5",
    "atr_60",
    "range_z_20",
    "body_to_range",
    "consecutive_up",
    "consecutive_down",
)

ALL_FIELDS: tuple[str, ...] = BASE_FIELDS + DERIVED_FIELDS  # 32 total


def _streak_count(mask_df: pd.DataFrame) -> pd.DataFrame:
    """Per column, count consecutive True values; reset to 0 on any False.

    Vectorized trick (works column-wise without a Python loop):
      1. cumsum the integer mask
      2. wherever the mask is 0, record that cumsum value as the most recent
         "reset point" (forward-filled)
      3. subtract reset from cumsum → run length

    Example for a single column:
      mask    = [1, 1, 0, 1, 1, 1, 0]
      cum     = [1, 2, 2, 3, 4, 5, 5]
      reset   = [0, 0, 2, 2, 2, 2, 5]
      streak  = [1, 2, 0, 1, 2, 3, 0]
    """
    mask_int = mask_df.astype(int)
    cum = mask_int.cumsum()
    reset = cum.where(mask_int == 0).ffill().fillna(0)
    return (cum - reset).astype(float)


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
            series = {t: df[field] for t, df in self._frames.items() if field in df.columns}
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
        if not all(
            f in self._matrix for f in ("open", "high", "low", "close", "volume", "returns")
        ):
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

        # ----- Phase B: extended momentum (8) -----
        # Different lookback windows + risk-adjusted variants.  Researchers
        # routinely sweep momentum windows; precomputing the common ones means
        # alphas like `rank(momentum_60)` work without rolling-op composition.
        self._matrix["momentum_3"] = c / c.shift(3) - 1.0
        self._matrix["momentum_10"] = c / c.shift(10) - 1.0
        self._matrix["momentum_60"] = c / c.shift(60) - 1.0
        self._matrix["momentum_120"] = c / c.shift(120) - 1.0
        self._matrix["momentum_252"] = c / c.shift(252) - 1.0
        # Reversal signals are just the negative of momentum; precomputed for
        # convenience and so quants can write `rank(reversal_5)` directly.
        self._matrix["reversal_5"] = -self._matrix["momentum_5"]
        self._matrix["reversal_20"] = -self._matrix["momentum_20"]
        # Risk-adjusted momentum: 60-day return divided by 60-day return-vol.
        # Closer to a per-name Sharpe; less dominated by high-vol stocks.
        rv60 = r.rolling(60).std()
        self._matrix["momentum_z_60"] = self._matrix["momentum_60"] / rv60.replace(0, np.nan)

        # ----- Phase B: extended volatility (6) -----
        self._matrix["realized_vol_5"] = r.rolling(5).std()
        self._matrix["realized_vol_60"] = rv60
        self._matrix["realized_vol_120"] = r.rolling(120).std()
        # Vol-of-vol: how much realized_vol_20 itself moves over a month.
        # Surrogates for "vol regime change" alphas.
        self._matrix["vol_of_vol_20"] = self._matrix["realized_vol"].rolling(20).std()
        # Parkinson estimator: high-low range based vol.  More efficient than
        # close-to-close because it uses intraday extremes.
        # σ_P^2 = (1/(4·ln(2))) · (ln(H/L))^2  →  rolling-mean for stability.
        ln_hl_sq = (np.log(h / l)) ** 2
        self._matrix["parkinson_vol"] = np.sqrt(ln_hl_sq.rolling(20).mean() / (4.0 * np.log(2.0)))
        # Garman-Klass: combines OHLC for the most efficient single-day estimate.
        # σ_GK^2 = 0.5·(ln(H/L))^2 - (2·ln(2) - 1)·(ln(C/O))^2
        ln_co_sq = (np.log(c / o)) ** 2
        gk_var = 0.5 * ln_hl_sq - (2.0 * np.log(2.0) - 1.0) * ln_co_sq
        # Negative cells (numerically possible on flat days) → clip to 0
        self._matrix["garman_klass_vol"] = np.sqrt(gk_var.clip(lower=0).rolling(20).mean())

        # ----- Phase B: microstructure (8) -----
        # Roll's effective spread (1984): 2·sqrt(-cov(Δp_t, Δp_{t-1})).
        # Negative covariance is the signature of bid-ask bounce.  Cells with
        # positive cov produce NaN (formula undefined → not enough microstructure noise).
        dp = c.diff()
        cov_lag = dp.rolling(20).cov(dp.shift(1))
        # Take 2·sqrt(-cov) only where cov is negative
        self._matrix["roll_spread"] = 2.0 * np.sqrt((-cov_lag).clip(lower=0)).where(cov_lag < 0)
        # Kyle's lambda proxy: |returns| / sqrt(dollar_volume) — price impact per
        # unit of trade.  Higher = less liquid = more impact.
        self._matrix["kyle_lambda"] = (r.abs() / np.sqrt(dv.replace(0, np.nan))).rolling(20).mean()
        # VPIN proxy: |signed_volume| / volume — net order imbalance fraction.
        # Higher = more directional flow (toxic for market makers).
        self._matrix["vpin_proxy"] = (v * np.sign(r)).abs().rolling(20).sum() / v.rolling(
            20
        ).sum().replace(0, np.nan)
        # Up/down volume ratios — fraction of 20d volume traded on green days.
        up_vol = v.where(r > 0, 0.0)
        down_vol = v.where(r < 0, 0.0)
        total_vol_20 = v.rolling(20).sum().replace(0, np.nan)
        self._matrix["up_volume_ratio"] = up_vol.rolling(20).sum() / total_vol_20
        self._matrix["down_volume_ratio"] = down_vol.rolling(20).sum() / total_vol_20
        # Turnover ratio: today's volume vs the 60-day baseline.  Different
        # window from `volume_ratio` (which uses adv20) → captures longer-term shifts.
        adv60 = v.rolling(60).mean()
        self._matrix["turnover_ratio"] = v / adv60.replace(0, np.nan)
        # Dollar-Amihud: per-dollar-volume price impact, rolling 20.  Like the
        # raw `amihud` field but smoothed.
        self._matrix["dollar_amihud"] = (r.abs() / dv.replace(0, np.nan)).rolling(20).mean()
        # Corwin-Schultz (2012) high-low spread estimator.
        # β = (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2  (sum of two consecutive days)
        # γ = (ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1})))^2  (2-day high-low)
        # α = (sqrt(2β) - sqrt(β)) / (3 - 2sqrt(2)) - sqrt(γ / (3 - 2sqrt(2)))
        # S = 2(e^α - 1) / (1 + e^α)
        # Negative spread estimates are clipped to NaN per the original paper.
        ln_hl_today = ln_hl_sq  # already (ln H/L)^2 today
        ln_hl_yest = ln_hl_sq.shift(-1)  # next day's (ln H/L)^2
        beta = ln_hl_today + ln_hl_yest
        h_2d = pd.concat([h, h.shift(-1)]).groupby(level=0).max()
        l_2d = pd.concat([l, l.shift(-1)]).groupby(level=0).min()
        gamma = (np.log(h_2d / l_2d)) ** 2
        denom = 3.0 - 2.0 * np.sqrt(2.0)
        # sqrt(2β) - sqrt(β) on its own is well-defined for β >= 0
        beta_clipped = beta.clip(lower=0)
        gamma_clipped = gamma.clip(lower=0)
        alpha = (np.sqrt(2.0 * beta_clipped) - np.sqrt(beta_clipped)) / denom - np.sqrt(
            gamma_clipped / denom
        )
        cs_spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        # Per the paper, negative spread estimates indicate the formula breaks
        # down on that day — drop them rather than report negative spreads.
        self._matrix["corwin_schultz"] = cs_spread.where(cs_spread > 0)

        # ----- Phase B: extended range / candle structure (6) -----
        self._matrix["atr_5"] = true_range.rolling(5).mean()
        self._matrix["atr_60"] = true_range.rolling(60).mean()
        # z-score of today's range vs its 20-day distribution
        rng_mean_20 = self._matrix["range_"].rolling(20).mean()
        rng_std_20 = self._matrix["range_"].rolling(20).std()
        self._matrix["range_z_20"] = (self._matrix["range_"] - rng_mean_20) / rng_std_20.replace(
            0, np.nan
        )
        # Body as a fraction of the day's range — small body = indecision day
        self._matrix["body_to_range"] = self._matrix["body"] / self._matrix["range_"].replace(
            0, np.nan
        )
        # Consecutive up/down day counters — reset on any opposite move.  Useful
        # for "exhaustion" alphas after long streaks.  Vectorized streak count:
        # cumsum the mask, subtract the cumsum value at the most recent reset
        # (cell where the mask is 0).  Works column-wise on DataFrames.
        up_day = (r > 0).astype(int)
        down_day = (r < 0).astype(int)
        self._matrix["consecutive_up"] = _streak_count(up_day)
        self._matrix["consecutive_down"] = _streak_count(down_day)

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
            raise ValueError(f"Unknown field {field!r}. Expected one of {list(ALL_FIELDS)}.")
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
