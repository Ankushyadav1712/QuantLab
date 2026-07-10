# Backtest settings

Every backtest is driven by a single `SimulationConfig` dataclass
(`backend/engine/backtester.py`). When you run an alpha through the API, the
request's `settings` object is mapped onto that dataclass by `_make_config`
in `backend/main.py`. This page is the reference for each field: its default,
and the one-line effect it has on the simulation.

See [Concepts](concepts.md) for the underlying ideas (neutralization,
turnover, cost realism, out-of-sample validation) that these settings expose.

## How settings become a config

`_make_config(settings)` reads each key off the request `settings` dict,
falling back to the dataclass-equivalent default when the key is absent:

```python
SimulationConfig(
    universe=tickers,
    start_date=s.get("start_date") or DATA_START,
    end_date=s.get("end_date") or DATA_END,
    neutralization=s.get("neutralization", "market"),
    truncation=float(s.get("truncation", 0.05)),
    booksize=float(s.get("booksize", DEFAULT_BOOKSIZE)),
    ...
)
```

Anything you omit gets the default in the table below, so a bare backtest
runs market-neutral, 5 bps flat cost, one-day-lagged (`execution_lag_days=1`)
execution, with a 70/30 in-sample / out-of-sample split.

## Window and universe

| Setting | Default | Effect |
| --- | --- | --- |
| `start_date` | `DATA_START` (`"2020-01-01"` in production, `"2019-01-01"` otherwise) | First date kept; the alpha matrix is filtered to `index >= start_date`. |
| `end_date` | `DATA_END` (env `DATA_END`, else ~7 days before today) | Last date kept; the alpha matrix is filtered to `index <= end_date`. |

Both are ISO date strings. `_make_config` uses `or` fallbacks, so an empty
string also falls back to the `DATA_START` / `DATA_END` defaults. After the
date/universe filter the engine also drops any row with >50% NaN alpha.

## Portfolio construction

| Setting | Default | Effect |
| --- | --- | --- |
| `neutralization` | `"market"` | Cross-sectional demeaning mode. `"none"` = raw; `"market"` = subtract the per-day mean; `"sector"`, `"industry_group"`, `"industry"`, `"sub_industry"` = demean within each GICS group at that level. |
| `decay` | `0` | If >0, applies a `decay_linear(alpha, decay)` linearly-weighted moving average before neutralization, smoothing the signal over `decay` days. `0` means no decay. |
| `truncation` | `0.05` | After normalizing to fractional weights (sum of abs ≈ 1 per row), each weight is clipped to `±truncation`, capping any single name at 5% of the book. |
| `renormalize_truncation` | `False` | Brain-parity truncation. A plain clip leaves `sum(\|w\|)` below 1 whenever the cap binds, silently shrinking gross exposure below booksize. When `True`, clipped weight is redistributed across uncapped names via exact per-side water-filling (longs and shorts rescaled to their own pre-clip masses, so a market-neutral book *stays* neutral), keeping the book fully invested like WorldQuant Brain. Applied by the **Brain parity** preset (`GET /api/presets/brain`), which also sets `truncation=0.08` and zero costs. |
| `booksize` | `20_000_000` (`DEFAULT_BOOKSIZE`) | Dollar gross book. Fractional weights are scaled to dollar positions by `weights * booksize`; daily returns are `net_pnl / booksize`. |

Neutralization order in the pipeline: point-in-time gating → ADV gating →
decay → neutralization → normalize → truncation → position sizing.

## Execution timing

| Setting | Default | Effect |
| --- | --- | --- |
| `execution_lag_days` | `1` | Shift between signal and PnL. `1` (the original one-day-lag convention) = the signal computed at `close[t-1]` earns the `close[t-1]→close[t]` return on day `t`. `2` models the realistic T+1 case where you can't trade at the close you just saw, shifting realization one more day to capture the close-to-open gap (the ~10–30 bps of slippage a flat-bps backtest ignores). `_make_config` floors this at `1` via `max(1, int(...))`. |

Internally the daily PnL is `positions.shift(exec_lag) * returns`, where
`exec_lag = max(1, int(execution_lag_days))`.

## Universe gating (liquidity and survivorship)

| Setting | Default | Effect |
| --- | --- | --- |
| `point_in_time_universe` | `False` | When `True`, alpha for any `(date, ticker)` where the ticker was not yet an S&P 100 member on that date is zeroed out before sizing — so e.g. TSLA can't be traded in 2019 just because it is in the index today. Off by default to keep saved-alpha headline numbers stable. |
| `min_adv_dollars` | `0.0` | When >0, on each date any ticker whose 20-day average dollar volume is below this threshold is NaN-ed out before sizing, so the backtest can't book PnL from stocks it couldn't have traded. Pass `1_000_000.0` for Brain's US3000-style `adv20 > $1M` parity. `0.0` = no filter. |

## Transaction and holding costs

Costs decompose into four independent streams — flat, spread, impact, and
borrow — that are summed per day and subtracted from gross PnL. All but the
flat charge are off by default to preserve existing backtest numbers.

| Setting | Default | Effect |
| --- | --- | --- |
| `transaction_cost_bps` | `5.0` | Flat, commission-style bps charged on every trade dollar: `|Δ$| * bps / 10_000`. Always applied. |
| `cost_model` | `"flat"` | `"flat"` keeps only the `bps × turnover` charge. `"sqrt_impact"` adds an Almgren-Chriss permanent-impact term on top: `impact_$ = impact_coefficient · σ_daily · |trade$| · √(participation)`, where `participation = |trade$| / dollar_volume`. |
| `impact_coefficient` | `0.1` | Multiplier `c` in the `sqrt_impact` term above. Ignored when `cost_model="flat"`. |
| `spread_model` | `"none"` | Bid/ask spread cost added on top of `transaction_cost_bps`. `"none"` = no spread charge; `"flat"` = charge `half_spread_bps` on every trade; `"corwin_schultz"` = per-day, per-ticker half-spread from the high/low-based Corwin-Schultz estimator. |
| `half_spread_bps` | `2.5` | Half-spread in bps charged per trade — used only when `spread_model="flat"`. |
| `borrow_cost_bps_annual` | `0.0` | Annual borrow rate charged daily on the absolute value of short positions: `daily = |short$| × bps / (10_000 × 252)`. `0.0` = free shorting; pass `50` for an IB-style large-cap baseline. |

Note on spread: because `spread_model` cost is added on top of the flat
charge, set `transaction_cost_bps` to `0` if you want spread cost in
isolation. Missing derived fields (e.g. `corwin_schultz`, `dollar_volume`,
`realized_vol`) degrade gracefully to a zero charge rather than erroring.

## Out-of-sample split

| Setting | Default | Effect |
| --- | --- | --- |
| `run_oos` | `True` | When `True`, the window is split into an in-sample block and a held-out out-of-sample block, returned as `(is_result, oos_result)`. When `False`, the whole window is treated as in-sample and the OOS slot is `None`. |
| `oos_split` | `0.3` | Fraction of dates (by row count, not calendar year) held out as the final OOS block. `0.3` → last 30% of dates. The split is skipped and treated as full-window IS if it would be degenerate (split index ≤ 0 or ≥ n). |

## Walk-forward analysis

Rolling train→test windows slid across the full history — the "does this
generalize?" test that removes the luck of which regime lands in a single OOS
block. Off by default since it costs roughly 20× a single backtest.

| Setting | Default | Effect |
| --- | --- | --- |
| `run_walk_forward` | `False` | Master switch for walk-forward. Off by default. |
| `walk_forward_train_days` | `252` | Length in days of each training window (~1 trading year). |
| `walk_forward_test_days` | `63` | Length in days of each out-of-sample test window (~1 quarter). |
| `walk_forward_step_days` | `63` | How far the window advances between iterations (~1 quarter). |

Each window yields the train period, test period, train Sharpe, test Sharpe,
test total return, and test cumulative PnL. Windows too short for even one
`train + test` span produce an empty result.