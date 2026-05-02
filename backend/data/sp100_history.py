"""Point-in-time S&P 100 membership.

The default ``UNIVERSE`` in ``config.py`` is the *current* S&P 100 snapshot.
Backtesting on it inherits two biases:

  1. **Survivorship** — names that were in the index in 2019 but were removed
     by 2024 (delistings, M&A, index reshuffles) are absent.  Fixing this
     requires paid PIT data (CRSP / Norgate / Sharadar) and is out of scope.
  2. **Anachronism** — names that *were not yet in the index* during part of
     the backtest window are still treated as tradeable.  TSLA, for instance,
     was added to the S&P 500 on 2020-12-21 and to the S&P 100 shortly after.
     Trading TSLA in a "S&P 100 universe" backtest in 2019–2020 is fictional.

This module addresses (2) only.  It exposes a curated ``SP100_INCLUSION_DATES``
mapping and a ``build_membership_mask`` helper that returns a (dates × tickers)
boolean DataFrame the backtester can use to NaN out alpha values for any
ticker that wasn't an index member on a given date.

Adding a ticker to the dict immediately makes the backtester respect that
ticker's join date when ``point_in_time_universe=True``.  Tickers absent from
the dict are assumed to have been members for the entire window.
"""

from __future__ import annotations

import pandas as pd

# Dates here are conservative best-effort.  S&P does not publish a clean
# machine-readable index history; these come from public S&P announcements
# and press releases.  Refine when better sources become available.
SP100_INCLUSION_DATES: dict[str, str] = {
    # TSLA joined the S&P 500 on 2020-12-21.  S&P 100 inclusion lagged by a
    # few quarters but using the S&P 500 date is a conservative *earliest*
    # bound — pre-2020-12-21 TSLA was definitively not in the S&P 100.
    "TSLA": "2020-12-21",
}


def is_member_on(ticker: str, date: pd.Timestamp) -> bool:
    """True iff ``ticker`` was an S&P 100 member on ``date``.

    Tickers absent from ``SP100_INCLUSION_DATES`` are assumed always-in.
    """
    join = SP100_INCLUSION_DATES.get(ticker)
    if join is None:
        return True
    return date >= pd.to_datetime(join)


def build_membership_mask(
    dates: pd.DatetimeIndex, tickers: list[str]
) -> pd.DataFrame:
    """Return a (dates × tickers) bool DataFrame: True where ticker is a member.

    Used by the backtester to mask out anachronistic positions when
    ``point_in_time_universe`` is enabled.
    """
    mask = pd.DataFrame(True, index=dates, columns=tickers)
    for t in tickers:
        join = SP100_INCLUSION_DATES.get(t)
        if join is None:
            continue
        mask.loc[mask.index < pd.to_datetime(join), t] = False
    return mask


def membership_summary(
    dates: pd.DatetimeIndex, tickers: list[str]
) -> dict:
    """Diagnostic summary the API can surface in the data-quality banner."""
    affected: list[dict] = []
    for t in tickers:
        join = SP100_INCLUSION_DATES.get(t)
        if join is None:
            continue
        join_ts = pd.to_datetime(join)
        n_before = int((dates < join_ts).sum())
        if n_before <= 0:
            continue
        affected.append(
            {
                "ticker": t,
                "join_date": join,
                "days_masked": n_before,
            }
        )
    return {
        "total_known_changes": len(SP100_INCLUSION_DATES),
        "tickers_affected": affected,
    }
