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


# ---------- Phase A: additional time-series operators ----------


def ts_median(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).median()


def ts_skewness(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).skew()


def ts_kurtosis(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    return x.rolling(window=d, min_periods=d).kurt()


def ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    d = int(d)
    mean = x.rolling(window=d, min_periods=d).mean()
    std = x.rolling(window=d, min_periods=d).std()
    return (x - mean) / std.replace(0, np.nan)


def ts_quantile(x: pd.DataFrame, d: int, q: float = 0.5) -> pd.DataFrame:
    """Rolling q-th quantile (q in [0, 1])."""
    d = int(d)
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"ts_quantile q must be in [0,1], got {q}")
    return x.rolling(window=d, min_periods=d).quantile(q)


def ts_arg_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Days-ago index of the rolling max — 0 means today is the max, d-1 means
    the oldest day in the window was the max."""
    d = int(d)
    # In the rolling window array, position d-1 is "today" and 0 is "oldest".
    # Days-ago = (d-1) - argmax_position.
    def _argmax(a):
        return float(d - 1 - np.argmax(a))
    return x.rolling(window=d, min_periods=d).apply(_argmax, raw=True)


def ts_arg_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    def _argmin(a):
        return float(d - 1 - np.argmin(a))
    return x.rolling(window=d, min_periods=d).apply(_argmin, raw=True)


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling product over d periods."""
    d = int(d)
    # rolling.apply with raw=True is faster than .agg(np.prod)
    return x.rolling(window=d, min_periods=d).apply(np.prod, raw=True)


def ts_returns(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Multi-period simple return: x_t / x_{t-d} - 1."""
    d = int(d)
    return x / x.shift(d) - 1.0


def ts_decay_exp(x: pd.DataFrame, d: int, factor: float = 0.5) -> pd.DataFrame:
    """Exponentially-weighted MA over d periods. ``factor`` is the
    most-recent-day weight; older days decay geometrically by ``factor``."""
    d = int(d)
    factor = float(factor)
    if d <= 0:
        raise ValueError(f"ts_decay_exp window must be positive, got {d}")
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"ts_decay_exp factor must be in (0,1], got {factor}")
    # weights[0] = oldest, weights[-1] = newest = factor^0 = 1
    # weight at i days ago = factor^i
    weights = np.array([factor ** (d - 1 - i) for i in range(d)], dtype=float)
    total = weights.sum()
    accum = None
    for i, w in enumerate(weights):
        term = x.shift(d - 1 - i) * w
        accum = term if accum is None else accum + term
    return accum / total


def ts_partial_corr(
    x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, d: int
) -> pd.DataFrame:
    """Rolling partial correlation of x and y, controlling for z.
    Formula: (corr(x,y) - corr(x,z)*corr(y,z)) / sqrt((1-corr(x,z)^2)*(1-corr(y,z)^2))."""
    d = int(d)
    rxy = ts_corr(x, y, d)
    rxz = ts_corr(x, z, d)
    ryz = ts_corr(y, z, d)
    denom = ((1 - rxz ** 2) * (1 - ryz ** 2)).clip(lower=1e-12)
    out = (rxy - rxz * ryz) / np.sqrt(denom)
    return out.replace([np.inf, -np.inf], np.nan)


def ts_regression(y: pd.DataFrame, x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling OLS slope of ``y`` regressed on ``x``: cov(x,y) / var(x).
    Returns the slope (beta), which is the most useful single output for alphas."""
    d = int(d)
    cov = ts_cov(x, y, d)
    var = x.rolling(window=d, min_periods=d).var()
    out = cov / var.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def ts_min_max_diff(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling (max - min) — range of the window. Volatility proxy."""
    d = int(d)
    roll = x.rolling(window=d, min_periods=d)
    return roll.max() - roll.min()


def days_from_last_change(x: pd.DataFrame) -> pd.DataFrame:
    """For each cell, days since the value last changed.
    Useful for spotting stale signals or constant-price runs."""
    # diff != 0 marks a change; cumsum groups consecutive runs
    changed = (x.diff() != 0) | x.isna()
    # First row: changed=True (no prior, so it's a "first observation")
    changed.iloc[0] = True
    # Counter that resets each time changed is True
    group_id = changed.cumsum()
    # Within each group, count from 0 upward
    out = x.copy().astype(float)
    for col in x.columns:
        g = group_id[col]
        # Use groupby cumcount to get position within run
        out[col] = g.groupby(g).cumcount().astype(float)
    return out


def hump(x: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Smoothing operator: only update output when |x_t - prev_output| > threshold.
    Suppresses small noise-driven flips that drive turnover without alpha."""
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError(f"hump threshold must be >= 0, got {threshold}")
    out = x.copy().astype(float)
    for col in x.columns:
        series = x[col].values
        result = np.full_like(series, np.nan, dtype=float)
        last = np.nan
        for i, v in enumerate(series):
            if np.isnan(v):
                result[i] = last
                continue
            if np.isnan(last) or abs(v - last) > threshold:
                last = v
            result[i] = last
        out[col] = result
    return out


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


# ---------- Phase A: additional cross-sectional operators ----------


def winsorize(x: pd.DataFrame, std: float = 4.0) -> pd.DataFrame:
    """Clip each row at ±std·sigma around the row mean.
    Standard outlier control before ranking."""
    std = float(std)
    if std <= 0:
        raise ValueError(f"winsorize std must be positive, got {std}")
    mean = x.mean(axis=1)
    sigma = x.std(axis=1)
    upper = mean + std * sigma
    lower = mean - std * sigma
    return x.clip(lower=lower, upper=upper, axis=0)


def quantile(x: pd.DataFrame, q: float = 0.5) -> pd.DataFrame:
    """Cross-sectional q-th quantile per row, broadcast back over columns.
    Useful as a benchmark to subtract: ``x - quantile(x, 0.5)`` is
    median-relative."""
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile q must be in [0,1], got {q}")
    row_q = x.quantile(q, axis=1)
    return pd.DataFrame(
        np.broadcast_to(row_q.values[:, None], x.shape),
        index=x.index,
        columns=x.columns,
    )


def vector_neut(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Project x orthogonally to y, per row.
    Returns the residual of x after removing its component along y.
    Useful for "factor-neutral" alphas: ``vector_neut(rank(close), rank(volume))``
    gives close-rank with the volume-rank component removed."""
    # Per row: residual = x - (<x,y>/<y,y>) * y
    x_aligned, y_aligned = x.align(y, join="outer")
    # Per-row dot products
    xy = (x_aligned * y_aligned).sum(axis=1)
    yy = (y_aligned * y_aligned).sum(axis=1)
    beta = xy / yy.replace(0, np.nan)
    return x_aligned.sub(y_aligned.mul(beta, axis=0))


def regression_neut(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional OLS residual: regress x on y per row, return residuals.
    Like vector_neut but also subtracts the intercept (per-row mean).
    More appropriate when y has a non-zero mean."""
    x_aligned, y_aligned = x.align(y, join="outer")
    # Residualize: subtract row means first, then project
    x_dm = x_aligned.sub(x_aligned.mean(axis=1), axis=0)
    y_dm = y_aligned.sub(y_aligned.mean(axis=1), axis=0)
    xy = (x_dm * y_dm).sum(axis=1)
    yy = (y_dm * y_dm).sum(axis=1)
    beta = xy / yy.replace(0, np.nan)
    return x_dm.sub(y_dm.mul(beta, axis=0))


def bucket(x: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Discretize each row into n equal-frequency buckets in [0, n-1].
    Useful for portfolio sorts / quantile spreads."""
    n = int(n)
    if n < 2:
        raise ValueError(f"bucket n must be >= 2, got {n}")
    # Use rank-position division so buckets are even when N is divisible by n.
    # rank(method="first") gives integer ranks 1..N; subtract 1 → position 0..N-1.
    # bucket = floor(position * n / N).
    r = x.rank(axis=1, method="first")
    n_per_row = x.notna().sum(axis=1)
    pos = r.sub(1)
    out = pos.mul(n).div(n_per_row.replace(0, np.nan), axis=0).apply(np.floor)
    return out.clip(upper=n - 1)


def tail(x: pd.DataFrame, lower: float = -np.inf, upper: float = np.inf,
         replace: float = np.nan) -> pd.DataFrame:
    """Replace values outside [lower, upper] with ``replace`` (default NaN).
    Used to keep only extreme observations: ``tail(zscore(x), 2, inf)`` keeps
    only z-scores above 2."""
    out = x.where((x >= lower) & (x <= upper), other=replace)
    return out


def kth_element(x: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-row, return the k-th smallest value (k=1 is min) broadcast to all columns.
    Negative k indexes from the top (k=-1 is the row max)."""
    k = int(k)

    def _kth(row):
        clean = row.dropna().sort_values()
        n = len(clean)
        if n == 0:
            return np.nan
        if k > 0:
            idx = k - 1
        else:
            idx = n + k  # k=-1 → n-1 (last)
        if idx < 0 or idx >= n:
            return np.nan
        return clean.iloc[idx]

    per_row = x.apply(_kth, axis=1)
    return pd.DataFrame(
        np.broadcast_to(per_row.values[:, None], x.shape),
        index=x.index,
        columns=x.columns,
    )


def harmonic_mean(x: pd.DataFrame) -> pd.DataFrame:
    """Per-row harmonic mean broadcast across columns. Defined only for x > 0;
    non-positive entries get masked to NaN."""
    x_pos = x.where(x > 0)
    n = x_pos.notna().sum(axis=1)
    inv_sum = (1.0 / x_pos).sum(axis=1)
    hm = n / inv_sum.replace(0, np.nan)
    return pd.DataFrame(
        np.broadcast_to(hm.values[:, None], x.shape),
        index=x.index,
        columns=x.columns,
    )


def geometric_mean(x: pd.DataFrame) -> pd.DataFrame:
    """Per-row geometric mean broadcast across columns. x > 0 only."""
    x_pos = x.where(x > 0)
    log_x = np.log(x_pos)
    gm = np.exp(log_x.mean(axis=1))
    return pd.DataFrame(
        np.broadcast_to(gm.values[:, None], x.shape),
        index=x.index,
        columns=x.columns,
    )


def step(x: pd.DataFrame) -> pd.DataFrame:
    """Per row, return rank-position - mean-rank-position, scaled to [-1, 1].
    Linear ramp from -1 (lowest) to +1 (highest) — a sign-aware alternative to rank."""
    r = x.rank(axis=1, pct=False, method="average")
    n = x.notna().sum(axis=1)
    # rank goes 1..n; centered at (n+1)/2; scaled by (n-1)/2
    centered = r.sub((n + 1) / 2.0, axis=0)
    half = (n - 1) / 2.0
    return centered.div(half.replace(0, np.nan), axis=0)


# ---------- Phase A: group operators ----------
# Group operators take (x, group) where ``group`` is a (dates × tickers)
# DataFrame of label strings (e.g. the GICS `sector` field).  Tickers without
# a label (NaN/None) are excluded from the per-row groupby — their output
# cells stay NaN.  We assume the label is constant across dates per ticker
# (true for GICS), so the per-column labels are read from the first row.


def _column_labels(group: pd.DataFrame) -> pd.Series:
    """Pull per-ticker labels from the group frame.

    GICS labels are assumed time-invariant within a backtest window — taking
    the first non-NaN label per column avoids a per-date Python loop and lets
    pandas' ``groupby(axis=1)`` do the heavy lifting.
    """
    # bfill picks the first non-NaN label down each column
    first_valid = group.bfill().iloc[0] if len(group) > 0 else pd.Series(dtype=object)
    return pd.Series(first_valid.values, index=group.columns)


def _grouped_transform(
    x: pd.DataFrame, labels: pd.Series, transform: str
) -> pd.DataFrame:
    """Apply ``transform`` to x's columns grouped by ``labels``.

    pandas 2.x removed ``DataFrame.groupby(axis=1)``, so we transpose first.
    Result is transposed back and reindexed to x's columns so NaN-labeled
    columns (excluded by ``dropna=True``) come back as NaN cells.
    """
    grouped = x.T.groupby(labels, dropna=True, group_keys=False)
    if transform == "rank":
        out = grouped.rank(pct=True)
    else:
        out = grouped.transform(transform)
    return out.T.reindex(columns=x.columns)


def group_rank(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Per row, percentile-rank x within each group label."""
    return _grouped_transform(x, _column_labels(group), "rank")


def group_zscore(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Per row, z-score x within each group: (x - group_mean) / group_std."""
    labels = _column_labels(group)
    mean = _grouped_transform(x, labels, "mean")
    std = _grouped_transform(x, labels, "std")
    return (x - mean) / std.replace(0, np.nan)


def group_neutralize(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Per row, subtract the group mean from each member.
    Produces a sector-neutral signal without rescaling."""
    labels = _column_labels(group)
    return x - _grouped_transform(x, labels, "mean")


def group_mean(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Per row, broadcast the group mean to each member."""
    return _grouped_transform(x, _column_labels(group), "mean")


def group_sum(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _grouped_transform(x, _column_labels(group), "sum")


def group_count(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _grouped_transform(x, _column_labels(group), "count")


def group_max(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _grouped_transform(x, _column_labels(group), "max")


def group_min(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    return _grouped_transform(x, _column_labels(group), "min")


def group_normalize(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Within each group: demean then scale so |sum| = 1 inside the group."""
    labels = _column_labels(group)
    centered = x - _grouped_transform(x, labels, "mean")
    abs_sum = _grouped_transform(centered.abs(), labels, "sum")
    return centered / abs_sum.replace(0, np.nan)


def group_scale(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Within each group: scale so |sum| = 1 (no demean)."""
    labels = _column_labels(group)
    abs_sum = _grouped_transform(x.abs(), labels, "sum")
    return x / abs_sum.replace(0, np.nan)


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


# ---------- Phase A: arithmetic / element-wise additions ----------


def op_exp(x):
    if isinstance(x, pd.DataFrame):
        return np.exp(x)
    return np.exp(x)


def op_sqrt(x):
    """sqrt with NaN for negative inputs (no domain error)."""
    if isinstance(x, pd.DataFrame):
        return np.sqrt(x.where(x >= 0))
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = arr >= 0
    out[mask] = np.sqrt(arr[mask])
    return out if arr.ndim else float(out)


def op_mod(x, y):
    """x mod y, element-wise. Division-by-zero cells return NaN."""
    return _binary_elementwise(
        x, y,
        lambda a, b: np.where(np.asarray(b) != 0, np.mod(a, np.where(b == 0, 1, b)), np.nan),
    )


def clip(x, lo: float = -np.inf, hi: float = np.inf):
    """Element-wise clip to [lo, hi]. Either bound can be infinite."""
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        raise ValueError(f"clip: lo ({lo}) must be <= hi ({hi})")
    if isinstance(x, pd.DataFrame):
        return x.clip(lower=lo, upper=hi)
    return np.clip(x, lo, hi)


def signed_power(x, n):
    """sign(x) * |x|^n — preserves sign, exponentiates magnitude.
    Useful for nonlinear ranking transforms that don't flip sign."""
    n = float(n)
    if isinstance(x, pd.DataFrame):
        return np.sign(x) * (x.abs() ** n)
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * (np.abs(arr) ** n)


def sigmoid(x):
    """Logistic sigmoid: 1 / (1 + exp(-x)). Squashes to (0, 1)."""
    if isinstance(x, pd.DataFrame):
        return 1.0 / (1.0 + np.exp(-x))
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def replace(x, old, new):
    """Replace cells equal to ``old`` with ``new``. NaN-safe via mask."""
    old = float(old)
    new = float(new)
    if isinstance(x, pd.DataFrame):
        return x.mask(x == old, new)
    arr = np.asarray(x, dtype=float)
    out = arr.copy()
    out[arr == old] = new
    return out if arr.ndim else float(out)


def isnan(x):
    """1.0 where the cell is NaN, else 0.0. Useful inside if_else conditions."""
    if isinstance(x, pd.DataFrame):
        return x.isna().astype(float)
    return np.isnan(np.asarray(x, dtype=float)).astype(float)


def op_equal(x, y):
    """Element-wise equality, returns 1.0 / 0.0 (so it composes with arithmetic)."""
    return _binary_elementwise(x, y, lambda a, b: np.equal(a, b).astype(float))


def op_less(x, y):
    """Element-wise x < y, returns 1.0 / 0.0.  Composable with arithmetic
    (you can multiply by it to mask) and with conditional ops (trade_when, when)."""
    return _binary_elementwise(x, y, lambda a, b: np.less(a, b).astype(float))


def op_greater(x, y):
    """Element-wise x > y, returns 1.0 / 0.0."""
    return _binary_elementwise(x, y, lambda a, b: np.greater(a, b).astype(float))


def op_less_eq(x, y):
    """Element-wise x <= y, returns 1.0 / 0.0."""
    return _binary_elementwise(x, y, lambda a, b: np.less_equal(a, b).astype(float))


def op_greater_eq(x, y):
    """Element-wise x >= y, returns 1.0 / 0.0."""
    return _binary_elementwise(x, y, lambda a, b: np.greater_equal(a, b).astype(float))


def op_not_equal(x, y):
    """Element-wise x != y, returns 1.0 / 0.0."""
    return _binary_elementwise(x, y, lambda a, b: np.not_equal(a, b).astype(float))


def where(cond, x, y):
    """Alias for if_else with a more pandas-native name. Cond is treated as
    boolean (any non-zero numeric cell is True)."""
    if isinstance(cond, pd.DataFrame):
        cond_bool = cond.astype(bool)
    else:
        cond_bool = np.asarray(cond).astype(bool)
    return if_else(cond_bool, x, y)


# ---------- Phase A: conditional / state operators ----------


def trade_when(cond: pd.DataFrame, x: pd.DataFrame, exit_cond=None) -> pd.DataFrame:
    """Take position x only when cond is True; otherwise carry the previous
    position forward (or NaN until the first activation).  When ``exit_cond``
    is provided, the position is dropped to NaN at exit rows and *stays* NaN
    (no carry-forward) until cond fires again.

    Cuts turnover by suppressing trades while cond is false.  Implemented as
    a per-column state machine because the "exit zeroes future carry" semantics
    don't compose with a single forward-fill."""
    cond_bool = cond.astype(bool)
    exit_bool = exit_cond.astype(bool) if exit_cond is not None else None

    out = x.copy().astype(float)
    for col in x.columns:
        c_arr = cond_bool[col].values
        v_arr = x[col].values
        e_arr = exit_bool[col].values if exit_bool is not None else None
        result = np.full(len(v_arr), np.nan, dtype=float)
        last = np.nan
        for i in range(len(v_arr)):
            if e_arr is not None and e_arr[i]:
                last = np.nan
            elif c_arr[i]:
                last = float(v_arr[i])
            result[i] = last
        out[col] = result
    return out


def when(cond: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Take x where cond is True, else NaN.  No carry-forward (unlike trade_when).
    Useful for conditional sub-signals: ``when(realized_vol > 0.02, momentum_20)``."""
    cond_bool = cond.astype(bool)
    return x.where(cond_bool)


def mask(x: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    """Replace cells where ``cond`` is True with NaN.  Inverse of ``when``."""
    cond_bool = cond.astype(bool)
    return x.mask(cond_bool)


def keep(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per row, keep only the n largest-magnitude entries; zero out the rest.
    Concentrates the alpha into its strongest names — a manual truncation."""
    n = int(n)
    if n < 1:
        raise ValueError(f"keep n must be >= 1, got {n}")
    abs_x = x.abs()
    # rank by |x| descending; top n keep, others drop
    rank_desc = abs_x.rank(axis=1, ascending=False, method="first")
    keep_mask = rank_desc <= n
    return x.where(keep_mask, other=0.0)


def pasteurize(x: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf and NaN with 0.  Use after divisions that may produce
    inf/NaN you don't want propagating into sizing."""
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
