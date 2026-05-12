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
