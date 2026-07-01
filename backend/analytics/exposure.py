"""Portfolio exposure analytics — sector + size factor.

Both are diagnostic lenses for *hidden bets*: an alpha that looks
market-neutral by topline can still be a thematic sector wager or a
systematic size tilt, and that's what these functions surface.

- ``compute_sector_exposure`` — group the per-day weight matrix by GICS
  sector and summarise the per-sector net + gross exposure plus the
  most-concentrated long/short headline.

- ``compute_size_exposure`` — daily Pearson correlation of weights against
  ``log(market_cap)``.  Positive → size factor long (mega-caps),
  negative → small-cap tilt.  Falls back to ``log(close)`` as an
  approximation when ``market_cap`` is unavailable.

- ``compute_market_cap_distribution`` — per-decile dollar exposure on the
  long side vs short side.  Visualises the size tilt that
  ``compute_size_exposure`` summarises to a single scalar: e.g. a "longs
  in deciles 8-10, shorts in deciles 1-3" pattern = systematic large-vs-
  small carry that the headline Sharpe won't expose.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Sector exposure
# ----------------------------------------------------------------------


def compute_sector_exposure(
    weights: pd.DataFrame,
    gics_map: dict[str, dict[str, str | None]] | None,
    level: str = "sector",
) -> dict[str, Any] | None:
    """Aggregate per-day position weights by GICS sector (or any GICS level).

    Returns
    -------
    None if ``weights`` is empty or ``gics_map`` is None.  Otherwise a dict
    with shape::

        {
          "level": "sector",
          "by_sector": {
            "Information Technology": {
              "avg_net":   0.12,     # mean over t of sum(weights[t, sector])
              "avg_gross": 0.25,     # mean over t of sum(|weights[t, sector]|)
              "n_tickers": 8,
            },
            ...
          },
          "headline": {
            "max_long_sector":  "Information Technology",
            "max_long_exposure": 0.12,
            "max_short_sector": "Energy",
            "max_short_exposure": -0.05,
          }
        }

    Tickers missing from the GICS map (or missing this level) are bucketed
    as ``"Unknown"`` — better than silently dropping them.
    """
    if weights is None or weights.empty or not gics_map:
        return None

    def _lookup(ticker: str) -> str:
        row = gics_map.get(ticker)
        if row is None:
            return "Unknown"
        val = row.get(level)
        return val if val else "Unknown"

    # Group columns by sector
    groups: dict[str, list[str]] = {}
    for ticker in weights.columns:
        sector = _lookup(ticker)
        groups.setdefault(sector, []).append(ticker)

    by_sector: dict[str, dict[str, float]] = {}
    for sector, cols in groups.items():
        sub = weights[cols]
        # Net = signed sum across the sector's tickers, then mean over time.
        # Gross = sum of absolute values — concentration regardless of sign.
        net_series = sub.sum(axis=1, skipna=True)
        gross_series = sub.abs().sum(axis=1, skipna=True)
        by_sector[sector] = {
            "avg_net": float(net_series.mean()),
            "avg_gross": float(gross_series.mean()),
            "n_tickers": int(len(cols)),
        }

    # Headline: the sector with the biggest signed net exposure on each side.
    long_items = [(s, v["avg_net"]) for s, v in by_sector.items() if v["avg_net"] > 0]
    short_items = [(s, v["avg_net"]) for s, v in by_sector.items() if v["avg_net"] < 0]
    max_long = max(long_items, key=lambda kv: kv[1], default=(None, 0.0))
    max_short = min(short_items, key=lambda kv: kv[1], default=(None, 0.0))

    return {
        "level": level,
        "by_sector": by_sector,
        "headline": {
            "max_long_sector": max_long[0],
            "max_long_exposure": float(max_long[1]),
            "max_short_sector": max_short[0],
            "max_short_exposure": float(max_short[1]),
        },
    }


def compute_sector_exposure_timeseries(
    weights: pd.DataFrame,
    gics_map: dict[str, dict[str, str | None]] | None,
    level: str = "sector",
    max_buckets: int = 120,
) -> dict[str, Any] | None:
    """Per-date per-sector net exposure — the dataset for a heatmap.

    Returns the same per-day net (signed) exposure that
    ``compute_sector_exposure`` averages into a single number, but kept
    as a time series so the frontend can render a sector × time heatmap
    that reveals sector *timing* (e.g. "this alpha was long Tech in 2021,
    short Tech in 2022").

    To keep the payload bounded on long backtests, the time axis is
    downsampled to at most ``max_buckets`` evenly-spaced periods using a
    mean aggregation.  Default 120 ≈ monthly resolution over 10 years.

    Returns
    -------
    None if inputs are empty.  Otherwise::

        {
          "level":   "sector",
          "sectors": ["Information Technology", "Energy", …],   # row labels
          "dates":   ["2019-01-31", "2019-02-28", …],           # column labels (downsampled)
          "matrix":  [[0.12, 0.08, …], [-0.03, -0.05, …], …],   # rows × dates, signed net
          "n_periods_per_bucket": 21,                           # so the UI can show "~monthly"
        }
    """
    if weights is None or weights.empty or not gics_map:
        return None

    def _lookup(ticker: str) -> str:
        row = gics_map.get(ticker)
        if row is None:
            return "Unknown"
        val = row.get(level)
        return val if val else "Unknown"

    # Build {sector → list of ticker columns}
    groups: dict[str, list[str]] = {}
    for ticker in weights.columns:
        sector = _lookup(ticker)
        groups.setdefault(sector, []).append(ticker)

    # Sort sectors alphabetically for a stable row order (heatmap legends
    # otherwise re-shuffle between runs and look like a regression).
    sectors = sorted(groups.keys())

    # Per-sector net exposure series, one column per sector
    per_day_net = pd.DataFrame(
        {s: weights[groups[s]].sum(axis=1, skipna=True) for s in sectors},
        index=weights.index,
    )

    n_days = len(per_day_net)
    if n_days == 0:
        return None

    # Downsample: bucket every k consecutive days, mean within each.
    # For n_days <= max_buckets we keep daily resolution (k=1).
    bucket_size = max(1, math.ceil(n_days / max_buckets))
    bucket_ids = np.arange(n_days) // bucket_size
    grouped = per_day_net.groupby(bucket_ids)
    bucketed = grouped.mean()
    # Bucket label = last date in the bucket (more intuitive than first)
    bucket_end_dates = grouped.apply(lambda g: g.index[-1])

    matrix = bucketed.T.values.tolist()  # rows = sectors, cols = buckets

    return {
        "level": level,
        "sectors": sectors,
        "dates": [d.strftime("%Y-%m-%d") for d in bucket_end_dates],
        # Coerce NaN → None so JSON is valid
        "matrix": [
            [float(v) if not (isinstance(v, float) and math.isnan(v)) else None for v in row]
            for row in matrix
        ],
        "n_periods_per_bucket": bucket_size,
    }


# ----------------------------------------------------------------------
# Size factor exposure
# ----------------------------------------------------------------------


def compute_size_exposure(
    weights: pd.DataFrame,
    size_field: pd.DataFrame | None,
    *,
    is_approximation: bool = False,
) -> dict[str, Any] | None:
    """Daily Pearson correlation between weights and ``log(size_field)``.

    ``size_field`` should be the per-day market_cap matrix.  If only ``close``
    is available the caller can pass that and set ``is_approximation=True``
    so the UI can flag the metric as a proxy.

    Returns
    -------
    None if either input is empty / unaligned.  Otherwise::

        {
          "size_corr":         0.23,   # avg daily corr(weights, log(size))
          "size_corr_std":     0.18,   # std of the daily corr — stability
          "n_days":            1041,
          "is_approximation":  False,
        }

    Interpretation: positive → systematic long large-caps / short small-caps
    (size factor exposure).  Near zero → size-neutral.  Magnitudes >0.3 are
    typically considered a meaningful tilt worth flagging.
    """
    if weights is None or weights.empty or size_field is None or size_field.empty:
        return None

    cols = weights.columns.intersection(size_field.columns)
    if len(cols) < 3:
        return None  # too few tickers for a meaningful daily correlation

    # Align to weights' date index — size_field may extend beyond the
    # backtest window.  Reindex returns NaN for missing dates which we then
    # drop in the per-row correlation.
    size_aligned = size_field[cols].reindex(weights.index)
    # log() blows up at 0/negative; replace non-positive with NaN.
    size_log = size_aligned.where(size_aligned > 0, other=np.nan).apply(np.log)
    w = weights[cols]

    daily_corrs: list[float] = []
    for date in w.index:
        x = w.loc[date].to_numpy(dtype=float)
        y = size_log.loc[date].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 3:
            continue
        xv = x[mask]
        yv = y[mask]
        sx = xv.std(ddof=1)
        sy = yv.std(ddof=1)
        if sx <= 0 or sy <= 0:
            continue
        c = float(((xv - xv.mean()) * (yv - yv.mean())).sum() / ((len(xv) - 1) * sx * sy))
        if not math.isnan(c):
            daily_corrs.append(c)

    if not daily_corrs:
        return None

    arr = np.asarray(daily_corrs)
    return {
        "size_corr": float(arr.mean()),
        "size_corr_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_days": int(len(arr)),
        "is_approximation": bool(is_approximation),
    }


# ----------------------------------------------------------------------
# Market-cap distribution (PDF Section 5.3.4)
# ----------------------------------------------------------------------


def compute_market_cap_distribution(
    weights: pd.DataFrame,
    size_field: pd.DataFrame | None,
    *,
    n_buckets: int = 10,
    is_approximation: bool = False,
) -> dict[str, Any] | None:
    """Per-decile dollar exposure on the long side vs the short side.

    For each backtest day:
      1. Bucket every ticker by its cross-sectional market-cap percentile
         (decile 0 = smallest 10 %, decile 9 = largest 10 %)
      2. Sum the long-side dollar weight and short-side |dollar weight| in
         each bucket
    Then average across days to give a stable distribution.

    The point: a size-neutral L/S strategy spreads its longs and shorts
    evenly across the deciles.  A size-biased strategy concentrates longs in
    the top deciles and shorts in the bottom (or vice versa) — visible
    immediately as a tilted side-by-side bar chart.

    Returns
    -------
    None if either input is unusable.  Otherwise::

        {
          "n_buckets": 10,
          "long_per_bucket":  [0.02, 0.04, 0.05, …],   # mean fraction of long book
          "short_per_bucket": [0.10, 0.08, 0.07, …],   # mean fraction of short book
          "long_avg_decile":  6.8,
          "short_avg_decile": 3.2,
          "decile_tilt":      3.6,                     # long - short avg decile
          "n_days":           1041,
          "is_approximation": False,
        }
    """
    if weights is None or weights.empty or size_field is None or size_field.empty:
        return None

    cols = weights.columns.intersection(size_field.columns)
    # Need at least n_buckets tickers per day to bucket meaningfully
    if len(cols) < n_buckets:
        return None

    size_aligned = size_field[cols].reindex(weights.index)
    w = weights[cols]

    # Per-day per-bucket totals; rows are dates, columns are buckets 0..n-1
    long_bucket_sums: list[np.ndarray] = []
    short_bucket_sums: list[np.ndarray] = []
    long_decile_means: list[float] = []
    short_decile_means: list[float] = []

    for date in w.index:
        size_row = size_aligned.loc[date].to_numpy(dtype=float)
        w_row = w.loc[date].to_numpy(dtype=float)
        # Need both signal and a positive market cap to bucket
        mask = (~np.isnan(size_row)) & (size_row > 0) & (~np.isnan(w_row))
        if mask.sum() < n_buckets:
            continue
        size_v = size_row[mask]
        w_v = w_row[mask]
        # qcut would be ideal but ties + duplicate edges fail at small N.
        # Fall back to argsort-based percentile bucketing (always works).
        ranks = np.argsort(np.argsort(size_v))
        buckets = (ranks * n_buckets // len(size_v)).astype(int)
        buckets = np.clip(buckets, 0, n_buckets - 1)

        longs = np.where(w_v > 0, w_v, 0.0)
        shorts = np.where(w_v < 0, -w_v, 0.0)  # absolute short dollars
        long_total = longs.sum()
        short_total = shorts.sum()
        if long_total <= 0 and short_total <= 0:
            continue

        long_per = np.zeros(n_buckets)
        short_per = np.zeros(n_buckets)
        for b in range(n_buckets):
            in_bucket = buckets == b
            long_per[b] = longs[in_bucket].sum()
            short_per[b] = shorts[in_bucket].sum()

        # Normalise so each side sums to 1; a 0-sided book contributes 0s
        # (won't affect the mean of the other side).
        if long_total > 0:
            long_bucket_sums.append(long_per / long_total)
            # Weighted-avg decile of the long book
            long_decile_means.append(float((np.arange(n_buckets) * (long_per / long_total)).sum()))
        if short_total > 0:
            short_bucket_sums.append(short_per / short_total)
            short_decile_means.append(
                float((np.arange(n_buckets) * (short_per / short_total)).sum())
            )

    if not long_bucket_sums and not short_bucket_sums:
        return None

    long_avg = np.vstack(long_bucket_sums).mean(axis=0) if long_bucket_sums else np.zeros(n_buckets)
    short_avg = (
        np.vstack(short_bucket_sums).mean(axis=0) if short_bucket_sums else np.zeros(n_buckets)
    )
    long_avg_decile = float(np.mean(long_decile_means)) if long_decile_means else 0.0
    short_avg_decile = float(np.mean(short_decile_means)) if short_decile_means else 0.0

    return {
        "n_buckets": n_buckets,
        "long_per_bucket": [float(v) for v in long_avg],
        "short_per_bucket": [float(v) for v in short_avg],
        "long_avg_decile": long_avg_decile,
        "short_avg_decile": short_avg_decile,
        # Positive tilt → longs land in higher deciles than shorts (large-cap
        # long carry).  Negative → small-cap long carry.  Magnitudes >2 are
        # typically a meaningful concentration worth flagging.
        "decile_tilt": long_avg_decile - short_avg_decile,
        "n_days": max(len(long_bucket_sums), len(short_bucket_sums)),
        "is_approximation": bool(is_approximation),
    }
