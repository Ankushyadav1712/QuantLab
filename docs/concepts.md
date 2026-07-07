# How QuantLab works

This page traces a single alpha expression through QuantLab's pipeline, from a
string you type in the editor to the scored performance report. Every stage
below maps to a real class or function in the `backend/` codebase.

The pipeline has four stages:

```
expression string
   │  Parser.parse            →  AST
   ▼
AlphaEvaluator.evaluate       →  signal matrix (dates × tickers)
   │  Backtester.run
   ▼  _run_pipeline: gate → decay → neutralize → normalize → truncate → scale
positions / daily PnL         →  BacktestResult
   │  PerformanceAnalytics.compute
   ▼
metrics report (Sharpe, IC, drawdown, …)
```

## 1. Parse the expression into an AST

The entry point is `Parser.parse` in `backend/engine/parser.py`. It runs a
`Tokenizer` over the raw string, then a recursive-descent parser builds an
abstract syntax tree.

```
rank(close - open) / adv(20)
```

Tokenizing recognizes numbers, identifiers (`[A-Za-z_][A-Za-z0-9_]*`), the four
arithmetic operators `+ - * /`, parentheses, and commas. The parser enforces
standard precedence through its grammar levels:

- `parse_expression` handles `+` / `-`
- `parse_term` handles `*` / `/`
- `parse_factor` handles unary `+` / `-`
- `parse_atom` handles numbers, parenthesized sub-expressions, data fields, and
  function calls

The AST is built from five dataclass node types: `BinaryOp`, `UnaryOp`,
`FunctionCall`, `DataField`, and `Literal`.

An identifier becomes a `FunctionCall` if it is followed by `(`; otherwise it
must be a known data field. The parser checks the name against the `DATA_FIELDS`
set (open/high/low/close, returns, momentum and volatility variants,
microstructure fields, GICS labels like `sector`, FRED macro series, and
fundamentals ratios) plus the `DATA_FIELD_ALIASES` map (e.g. `range` →
`range_`). An unknown bare identifier raises `ValueError` — it is neither a data
field nor a function call. The parser preserves your original spelling in the
`DataField` node; aliases are resolved later at evaluation time.

For the full list of fields and operators you can name here, see the
[Operator reference](reference/operators.md) and [Writing alphas](writing-alphas.md).

## 2. Evaluate the AST into a signal matrix

`AlphaEvaluator.evaluate` in `backend/engine/evaluator.py` parses the expression
and then walks the tree with `_eval`. The evaluator is constructed with a
`data` dict mapping each field name to a `(dates × tickers)` `pandas.DataFrame`.

Node handling in `_eval`:

- `Literal` returns its scalar value.
- `DataField` resolves aliases via `DATA_FIELD_ALIASES`, then looks the field up
  in `self.data`, raising `ValueError` if it isn't loaded.
- `UnaryOp` negates (`-`) or passes through (`+`) its operand.
- `BinaryOp` applies `+ - * /` directly to its evaluated operands. Because the
  operands are usually DataFrames, these are element-wise pandas operations that
  broadcast across the whole `(dates × tickers)` grid.
- `FunctionCall` is dispatched to the operator library.

Function names are resolved by `_resolve_function`, which applies the
`_FUNCTION_NAME_REMAP` table (`abs` → `op_abs`, `log` → `op_log`, `max` →
`op_max`, comparison ops like `less`/`greater` → `op_less`/`op_greater`, etc.)
and then looks the name up as an attribute of `engine.operators`. Comparison
operators return `1.0`/`0.0` so they compose with arithmetic and slot into
conditional operators.

Two functions are handled specially by `_eval_data_aware` because they need to
reach into `self.data` for a field the user didn't name:

- `adv(d)` is not a plain operator — it computes `ops.ts_mean(dollar_volume, d)`,
  a rolling mean of the `dollar_volume` field over window `d`.
- `cap_weight(x)` calls `ops.cap_weight(x, market_cap)`, pulling in the
  `market_cap` field.

Both raise a clear `ValueError` if their required field isn't loaded. The result
of evaluation is a single `(dates × tickers)` DataFrame: the **signal matrix**.

## 3. Turn the signal into positions and PnL

`Backtester.run` in `backend/engine/backtester.py` takes the signal matrix
(`alpha_matrix`) and a `SimulationConfig`, and returns a
`tuple[BacktestResult, BacktestResult | None]` — in-sample and out-of-sample.

Before the core pipeline, `run` applies filters that affect both splits
identically:

1. **Universe + date filter** — keep only tickers in `config.universe` that
   appear as columns, then restrict rows to `[start_date, end_date]`.
2. **Sparse-row drop** — any date whose alpha is more than 50% NaN across
   tickers is dropped.
3. **In/out-of-sample split** — when `config.run_oos` is true, the surviving
   dates are split by row count at `1 - oos_split` (default `0.3` held out as the
   last block). Each half is run through `_run_pipeline` independently. Degenerate
   splits fall back to a single full-window run.

### The `_run_pipeline` stages

`_run_pipeline` is where a signal matrix becomes dollar positions and PnL. In
order:

1. **Point-in-time gating** (optional, `point_in_time_universe`) — via
   `build_membership_mask`, alpha for a `(date, ticker)` where the ticker wasn't
   yet an S&P 100 member is set to `0.0`, so e.g. a stock can't be traded before
   it joined the index.
2. **ADV liquidity gating** (optional, `min_adv_dollars > 0`) — cells whose
   20-day average dollar volume is below the threshold are set to `NaN`, so the
   backtest can't claim PnL from stocks it couldn't have traded. Uses the `adv20`
   field, falling back to a rolling mean of `dollar_volume`.
3. **Decay** (optional, `decay > 0`) — `ops.decay_linear(alpha, decay)` applies a
   linear-weighted moving average so today's position blends in prior days'
   signals, smoothing turnover.
4. **Neutralization** — `_neutralize` removes unwanted common exposure.
5. **Signal snapshot** — the *post-neutralization* alpha is copied and stored as
   `signal_matrix`. This snapshot (taken before normalization/truncation, which
   would distort ranks) is what IC and alpha-decay analytics later correlate
   against forward returns.
6. **Normalize to weights** — divide each row by the sum of its absolute values,
   so per-day gross weight ≈ 1; empty rows fill to `0.0`.
7. **Truncation** — clip each fractional weight to `±config.truncation` (default
   `0.05`) so no single name dominates the book.
8. **Book-size scaling** — `positions = weights * config.booksize` (default
   `$20M`) converts fractional weights to dollar positions.

Neutralization modes (`_neutralize`, default `market`):

- `none` — pass through unchanged.
- `market` — subtract each day's cross-sectional mean (dollar-neutral).
- `sector` / `industry_group` / `industry` / `sub_industry` — cross-sectional
  demean *within* each GICS group at that level, via `_neutralize_by_gics`.
  Tickers with no classification (or absent from the GICS map) are bucketed as
  `"Unknown"` rather than dropped.

### Daily PnL and costs

PnL follows the convention that **yesterday's dollar position earns today's
return**: `stock_pnl = positions.shift(exec_lag) * returns`, where `exec_lag =
max(1, execution_lag_days)`. Setting `execution_lag_days` to `2` shifts
realization a further day to model the T+1 "can't trade the close you just saw"
friction. Per-stock PnL is summed across tickers into `gross_pnl`. (The
`returns` field is required; its absence raises `ValueError`.)

Transaction costs are decomposed into four streams, all built from the per-stock
traded-dollar matrix `delta_pos = |positions - positions.shift(1)|`:

- **Flat bps** — `delta_pos * transaction_cost_bps / 10_000`, a commission-style
  charge on every traded dollar.
- **Spread** (`spread_model`) — `none` (default), `flat` (charge
  `half_spread_bps`), or `corwin_schultz` (per-day per-ticker fractional spread
  from the `corwin_schultz` field, at half the quoted spread).
- **Impact** (`cost_model = "sqrt_impact"`) — an Almgren-Chriss permanent-impact
  term `impact_coefficient · σ_daily · |trade$| · √participation`, where
  `participation = |trade$| / dollar_volume` and σ is daily `realized_vol`.
  Missing fields degrade gracefully to zero.
- **Borrow** (`borrow_cost_bps_annual > 0`) — a daily charge on the absolute
  short notional held: `|short$| × borrow_bps_annual / (10_000 × 252)`.

`net_pnl = gross_pnl - total_cost_per_day`, `cumulative = net_pnl.cumsum()`, and
`daily_returns = net_pnl / booksize`. Turnover per day is `delta_pos` summed
across stocks.

The stage returns a `BacktestResult` carrying `dates`, `daily_pnl`,
`cumulative_pnl`, `daily_returns`, `weights`, `turnover`, `positions`, the
`signal_matrix` snapshot, per-stock `forward_returns`, and the four
`cost_components` streams.

### Walk-forward (optional)

When `run_walk_forward` is set, `Backtester.walk_forward` slides a
`(walk_forward_train_days → walk_forward_test_days)` window forward by
`walk_forward_step_days`, running `_run_pipeline` on each train and test slice
and reporting each window's train/test Sharpe (via the local `_quick_sharpe`)
plus test return and cumulative PnL. It answers "does this alpha generalize
across regimes?" rather than relying on a single split.

See [Backtest settings](backtest-settings.md) for how each `SimulationConfig`
field is exposed in the editor.

## 4. Score the result

`PerformanceAnalytics.compute` in `backend/analytics/performance.py` takes a
`BacktestResult` and produces the metrics dictionary the UI renders.

From `daily_returns` and `daily_pnl` it computes the headline risk/return
figures:

- **Sharpe** — `mean / std × √252` (annualized, `ddof=1`).
- **Annual return (CAGR)** — from `(1 + total_return)^(1/years) - 1` over
  `years = n / 252`, and **annual vol** = `std × √252`.
- **Max drawdown** — on the equity curve `1 + daily_returns.cumsum()` relative to
  its running max; **Calmar** = annual return / |max drawdown|.
- **Sortino** — like Sharpe but using downside deviation only.
- **Win rate**, **profit factor** (positive PnL sum / |negative PnL sum|), and
  turnover (dollar and as a fraction of the book proxy).
- **Fitness** — `sharpe · √|annual_return| · max(0, 1 - turnover_frac)`, plus
  `fitness_wq`, the WorldQuant Brain composite
  `sign(returns) · √(|annual_return| / max(turnover, 0.125)) · sharpe`.

When a `benchmark_returns` series is supplied it adds **beta** (covariance /
benchmark variance) and the **information ratio**. Time-series outputs include a
63-day **rolling Sharpe**, monthly and per-year returns, and the drawdown series.

**Signal-quality metrics** are computed only when the result carries a
`signal_matrix` and `forward_returns` (older hand-built results degrade to
`None`):

- **IC** summary via `compute_ic_summary` — the information coefficient plus
  `icir`, IC t-stat, and percent-positive.
- **Alpha decay** via `compute_alpha_decay` — IC by horizon, half-life, and R².
- **Rank stability**, a daily **IC series**, and **quintile returns** (stocks
  bucketed by signal, forward returns averaged per bucket).

Further blocks: tail ratio, positive-months %, drawdown durations, a
**deflated Sharpe** (Bailey & López de Prado, adjusting the headline Sharpe for
running `n_trials` candidates), long/short/gross/net **exposure** as fractions of
the book, a **cost breakdown** summing the four cost streams (with cost as a
percent of gross PnL), and a **stress test** over the daily returns. When a
`gics_map` and/or `size_field` are passed, it additionally reports sector/size
exposure, market-cap distribution, and PnL attribution.

Finally, `PerformanceAnalytics.compare_is_oos` diagnoses overfitting by comparing
in-sample and out-of-sample metrics: it computes Sharpe/return decay and emits a
label and `overfitting_flag` (raised when Sharpe decays more than 50% or OOS
Sharpe goes negative), ranging from "Robust" through "Severe overfit".

## Where to go next

- [Writing alphas](writing-alphas.md) — the expression syntax and available fields.
- [Operator reference](reference/operators.md) — every operator and data field.
- [Backtest settings](backtest-settings.md) — configuring neutralization, decay,
  truncation, costs, and walk-forward.