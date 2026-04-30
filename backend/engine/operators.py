from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Time-series operators (axis=0, per ticker) ----------


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).mean()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).std()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).max()


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).sum()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).rank(pct=True)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x - x.shift(int(d))


def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(int(d))


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    if d <= 0:
        raise ValueError(f"decay_linear window must be positive, got {d}")
    weights = np.arange(1, d + 1, dtype=float)
    total = weights.sum()
    accum = None
    for i, w in enumerate(weights):
        # i=0 → oldest in window (shift d-1); i=d-1 → most recent (shift 0)
        term = x.shift(d - 1 - i) * w
        accum = term if accum is None else accum + term
    return accum / total


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    out = x.rolling(window=d, min_periods=d).corr(y)
    # pandas rolling.corr returns ±inf when a window's variance is numerically
    # near-zero; coerce those to NaN so downstream math stays finite.
    return out.replace([np.inf, -np.inf], np.nan)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    out = x.rolling(window=d, min_periods=d).cov(y)
    return out.replace([np.inf, -np.inf], np.nan)


# ---------- Cross-sectional operators (axis=1, per date) ----------


def rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std, axis=0)


def demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


def scale(x: pd.DataFrame) -> pd.DataFrame:
    denom = x.abs().sum(axis=1)
    return x.div(denom, axis=0)


def normalize(x: pd.DataFrame) -> pd.DataFrame:
    centered = demean(x)
    denom = centered.abs().sum(axis=1)
    return centered.div(denom, axis=0)


# ---------- Arithmetic / element-wise ----------


def op_abs(x):
    if isinstance(x, pd.DataFrame):
        return x.abs()
    return np.abs(x)


def op_log(x):
    if isinstance(x, pd.DataFrame):
        return np.log(x.where(x > 0))
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return out if arr.ndim else float(out)


def op_sign(x):
    if isinstance(x, pd.DataFrame):
        return np.sign(x)
    return np.sign(x)


def power(x, n):
    return x ** n


def _binary_elementwise(x, y, fn):
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        x_aligned, y_aligned = x.align(y, join="outer")
        return pd.DataFrame(
            fn(x_aligned.values, y_aligned.values),
            index=x_aligned.index,
            columns=x_aligned.columns,
        )
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(
            fn(x.values, np.asarray(y)),
            index=x.index,
            columns=x.columns,
        )
    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(
            fn(np.asarray(x), y.values),
            index=y.index,
            columns=y.columns,
        )
    return fn(x, y)


def op_max(x, y):
    return _binary_elementwise(x, y, np.maximum)


def op_min(x, y):
    return _binary_elementwise(x, y, np.minimum)


def if_else(cond, x, y):
    if isinstance(cond, pd.DataFrame):
        x_arr = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        y_arr = y.values if isinstance(y, pd.DataFrame) else np.asarray(y)
        return pd.DataFrame(
            np.where(cond.values, x_arr, y_arr),
            index=cond.index,
            columns=cond.columns,
        )
    return np.where(cond, x, y)
