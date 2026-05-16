"""PnL attribution — Brinson-style decomposition of the backtest's daily PnL.

The headline Sharpe answers "did the alpha make money?".  Attribution
answers the diagnostic question: *where did the money come from?*  Two
candidate alphas with identical Sharpe can differ wildly in attribution —
one might be a thinly-disguised sector bet (high allocation, low selection),
the other a true cross-sectional stock picker (low allocation, high
selection).  The PDF flags this as Section 5.2.

Decomposition (no benchmark required — works directly on a dollar-neutral
L/S book using the universe itself as the implicit benchmark):

    For each sector s, each day t:
      w_net[s,t]    = sum_{i in s} weights[i,t]              # net $ in sector
      r_avg[s,t]    = mean_{i in s} returns[i,t]             # sector avg return
      allocation[s,t] = w_net[s,t] * r_avg[s,t]              # PnL from sector bet
      selection[s,t] = sum_{i in s} w[i,t]*r[i,t]            # actual sector PnL
                      - allocation[s,t]                       # minus the avg-return contribution

The identity ``sum_s (allocation[s] + selection[s])`` exactly equals the
day's total PnL — no residual, no interaction term.  We aggregate across
days by simple summation (linear PnL is additive, unlike returns).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_pnl_attribution(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame | None,
    gics_map: dict[str, dict[str, str | None]] | None,
    level: str = "sector",
) -> dict[str, Any] | None:
    """Decompose total backtest PnL into sector allocation + stock selection.

    Parameters
    ----------
    weights
        Per-day dollar positions per ticker (date × ticker).  Same matrix
        used for L/S exposure calculations.
    forward_returns
        Per-stock daily returns aligned to the same grid.  Each cell is the
        return realised by that stock on that date — what was earned by
        holding it.
    gics_map
        ticker → {sector, industry, ...}.  Tickers missing from the map
        bucket into ``"Unknown"``.
    level
        Which GICS level to attribute across.  Default ``"sector"`` (11
        buckets).  Pass ``"industry_group"`` (24) or ``"industry"`` (~70)
        for a finer cut.

    Returns
    -------
    None if any input is unusable.  Otherwise::

        {
          "level":          "sector",
          "by_sector": {
            "Information Technology": {
              "allocation":   123_456.78,    # $ PnL from sector net exposure
              "selection":    -45_678.90,    # $ PnL from within-sector picking
              "total":         77_777.88,    # = allocation + selection
              "n_tickers":     8,
            },
            ...
          },
          "totals": {
            "allocation":     400_000.00,    # sum across sectors
            "selection":      280_000.00,
            "total_pnl":      680_000.00,    # = allocation + selection
            "allocation_pct": 58.8,          # share of |allocation| vs |total|
            "selection_pct":  41.2,
          }
        }
    """
    if (
        weights is None
        or weights.empty
        or forward_returns is None
        or forward_returns.empty
        or not gics_map
    ):
        return None

    # Restrict to dates + tickers present in both matrices — avoids silent
    # broadcast errors when one has been trimmed
    cols = weights.columns.intersection(forward_returns.columns)
    if len(cols) == 0:
        return None
    idx = weights.index.intersection(forward_returns.index)
    if len(idx) == 0:
        return None

    w = weights.loc[idx, cols]
    r = forward_returns.loc[idx, cols]

    def _lookup(ticker: str) -> str:
        row = gics_map.get(ticker)
        if row is None:
            return "Unknown"
        val = row.get(level)
        return val if val else "Unknown"

    groups: dict[str, list[str]] = {}
    for ticker in cols:
        sector = _lookup(ticker)
        groups.setdefault(sector, []).append(ticker)

    by_sector: dict[str, dict[str, float]] = {}
    total_allocation = 0.0
    total_selection = 0.0

    for sector, ticks in groups.items():
        w_sec = w[ticks]
        r_sec = r[ticks]
        # Per-day net weight + sector-average return.  Using the equal-weighted
        # mean of returns gives the "sector index" return; the dot product
        # with the net weight is what the strategy would've earned holding
        # the sector basket at its actual net exposure.
        w_net = w_sec.sum(axis=1, skipna=True)
        # nanmean across stocks to ignore tickers that weren't trading that day
        r_avg = r_sec.mean(axis=1, skipna=True)
        # Actual per-day per-sector PnL = sum_i w[i] * r[i]
        actual_pnl = (w_sec * r_sec).sum(axis=1, skipna=True)
        # Allocation = what the net sector exposure earned at the sector mean.
        # Selection = the residual — i.e. the value-add from picking specific
        # stocks within the sector rather than holding an equal-weight basket.
        allocation = (w_net * r_avg).sum()
        selection = actual_pnl.sum() - allocation

        # Floats can hold NaN; coerce to 0 so downstream summing is clean.
        if np.isnan(allocation):
            allocation = 0.0
        if np.isnan(selection):
            selection = 0.0

        by_sector[sector] = {
            "allocation": float(allocation),
            "selection": float(selection),
            "total": float(allocation + selection),
            "n_tickers": len(ticks),
        }
        total_allocation += float(allocation)
        total_selection += float(selection)

    total_pnl = total_allocation + total_selection
    # Percentage split uses absolute values so a +allocation/-selection mix
    # (sector bet helped, stock-picking hurt) shows the right magnitude of
    # each contribution.  Total of pct ≈ 100 (within rounding).
    abs_total = abs(total_allocation) + abs(total_selection)
    if abs_total > 0:
        allocation_pct = 100.0 * abs(total_allocation) / abs_total
        selection_pct = 100.0 * abs(total_selection) / abs_total
    else:
        allocation_pct = 0.0
        selection_pct = 0.0

    return {
        "level": level,
        "by_sector": by_sector,
        "totals": {
            "allocation": total_allocation,
            "selection": total_selection,
            "total_pnl": total_pnl,
            "allocation_pct": allocation_pct,
            "selection_pct": selection_pct,
        },
    }
