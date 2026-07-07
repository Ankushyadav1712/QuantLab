# Writing alphas (the DSL)

An alpha in QuantLab is a single expression. You combine **data fields** with **operators** and ordinary **arithmetic**, and the engine evaluates that expression into a `(dates × tickers)` matrix of signal values — one number per stock per day. Those cross-sectional values become your portfolio weights after neutralization and scaling.

This page explains how expressions are built, how to think cross-sectionally, and how the linter protects you from look-ahead bias. For the exhaustive catalogs, see the [Operator reference](reference/operators.md) and [Field reference](reference/fields.md).

## The building blocks

An expression is made of four kinds of terms:

- **Data fields** — bare identifiers like `close`, `volume`, `momentum_20`, `pe_ratio`, `vix`, `sector`. Every field the parser accepts is enumerated in `DATA_FIELDS`; see the [Field reference](reference/fields.md). An identifier that is not a known field and is not followed by `(` is rejected as an unknown identifier.
- **Operators** — function calls like `rank(x)`, `ts_mean(x, 20)`, `group_neutralize(x, sector)`.
- **Arithmetic** — `+`, `-`, `*`, `/` with the usual precedence (`*` and `/` bind tighter than `+` and `-`), plus parentheses for grouping.
- **Unary minus** — a leading `-` negates a term, e.g. `-zscore(pe_ratio)`. (A leading unary `+` is accepted and is a no-op.)

Numbers may be integers or decimals (`20`, `0.001`, `0.5`). A few user-facing aliases resolve to canonical field names — for example `range` resolves to the internal `range_` field.

```text
decay_linear(rank(momentum_20), 20)
rank(reversal_5) / (realized_vol + 0.001)
zscore(roe) + zscore(momentum_60)
```

## Operator categories

Rather than list every operator here, here is the map. The full signatures and semantics live in the [Operator reference](reference/operators.md).

### Time-series operators (`ts_*`, `delta`, `delay`, `decay_*`)

These roll **along time, per ticker** (down each column) and look backward — they never reach into the future. The windowed rolling operators (`ts_mean`, `ts_std`, and the rest built on `.rolling(window=d, min_periods=d)`) require `min_periods=d`, so the first `d-1` days of any window are NaN — no partial windows. The shift-based operators (`delta`, `delay`, `ts_returns`) and the decay operators (`decay_linear`, `ts_decay_exp`) are built from `.shift(...)` instead and produce NaN wherever the shifted history isn't available.

Examples: `ts_mean`, `ts_std`, `ts_min`, `ts_max`, `ts_sum`, `ts_rank`, `ts_median`, `ts_zscore`, `ts_returns`, `ts_corr`, `ts_cov`, `ts_regression`, `decay_linear`, `ts_decay_exp`. Two shift operators, `delta(x, d)` (`x - x.shift(d)`) and `delay(x, d)` (`x.shift(d)`), move data backward in time.

### Cross-sectional operators (`rank`, `zscore`, `scale`, …)

These act **across tickers, per day** (along each row). This is the heart of cross-sectional alpha research: on each date they compare every stock against every other stock.

- `rank(x)` — percentile rank per day, in `[0, 1]`.
- `zscore(x)` — `(x - row_mean) / row_std` per day.
- `demean(x)`, `scale(x)` (divide by the row's sum of absolute values), `normalize(x)` (demean then scale).
- Plus outlier/shape tools like `winsorize`, `quantile`, `bucket`, `tail`, `step`, and orthogonalizers `vector_neut` and `regression_neut`.

### Group operators (`group_*`) — sector/industry neutralization

Group operators take a second argument that is a classification **field** — one of the GICS labels `sector`, `industry_group`, `industry`, `sub_industry`. The engine resolves that label into a `(dates × tickers)` frame of group-label strings, and the operator applies its transform **within each group, per day**.

- `group_rank`, `group_zscore`, `group_neutralize` (subtract the group mean → sector-neutral), `group_mean`, `group_normalize`, `group_scale`, and more.
- `neutralize(x, g)` is an alias for `group_neutralize` (matching the WorldQuant Brain spelling).

Tickers with no label (NaN) are excluded and their output cells stay NaN.

### Conditional / state operators (`trade_when`, `when`, `if_else`)

These gate or switch signals based on a condition. Conditions are usually built from the comparison operators, which return `1.0`/`0.0` so they compose with everything else: `less`, `greater`, `less_eq`, `greater_eq`, `equal`, `not_equal`.

- `when(cond, x)` — take `x` where `cond` is true, else **NaN** (no position, no carry-forward).
- `trade_when(cond, x, exit_cond=None)` — take `x` when `cond` fires, otherwise **carry the previous position forward**; an optional `exit_cond` drops to NaN and stays there until `cond` fires again. Cuts turnover.
- `if_else(cond, x, y)` — pick `x` where `cond` is true, else `y`. `where(cond, x, y)` is a pandas-native alias.

### Arithmetic / element-wise helpers

Beyond `+ - * /`, there are element-wise helpers: `abs`, `log`, `sqrt`, `exp`, `sign`, `signed_power`, `power`, `clip`, `sigmoid`, `pasteurize` (replace ±inf/NaN with 0), `max`/`min` (element-wise pairwise max/min of two inputs), `mod`, `keep` (keep the `n` largest-magnitude names per row, zero the rest), and more.

## Thinking cross-sectionally

The single most important mental shift: **`rank`, `zscore`, and the `group_*` operators compare stocks against each other on the same day**, not a stock against its own past. `rank(momentum_20)` does not ask "is this stock's momentum high for it?" — it asks "is this stock's momentum high **relative to every other stock today**?" The top-ranked names get the largest long weight; the bottom-ranked get shorted (after neutralization/scaling).

Time-series operators are the opposite axis: `ts_zscore(close, 5)` measures where today's close sits relative to *this same stock's* last five days.

A great deal of alpha design is choosing the right axis and composing the two. `rank(ts_rank(dollar_volume, 20))` first asks "how high is today's dollar volume in this stock's own 20-day range?" (time-series), then ranks *that* across stocks (cross-sectional).

### Neutralization

Neutralization removes exposures you don't want to bet on so the signal isolates stock-specific alpha. Two layers:

- **In the expression** — `group_neutralize(x, sector)` subtracts each sector's mean, so you bet on the best name *within* a sector rather than the best sector. `vector_neut(x, y)` / `regression_neut(x, y)` orthogonalize `x` against another factor `y`.
- **As a backtest setting** — each example ships a `recommended_settings.neutralization` (`"none"`, `"market"`, or `"sector"`) applied by the backtest engine on top of the expression.

## Avoiding look-ahead bias (what the linter flags)

Before the evaluator runs an expression, `engine/lint.py` walks its AST and reports diagnostics with severity `error` or `warning`. An **error** means the engine should refuse to run — a look-ahead or invalid-window bug would otherwise silently inflate your Sharpe. The rules:

**Negative shifts are look-ahead bias → error.** The shift operators `delta`, `delay`, and `ts_returns` take a shift count as their second argument. A **negative** shift reaches into the future, so it is flagged as an error:

```text
delay(close, -5)      # ERROR: peeks 5 days into the future
delta(close, -1)      # ERROR: same look-ahead failure mode
```

A **zero** shift is flagged as a `warning` — `delay(x, 0)` is the identity and `delta(x, 0)` is always zero, so you probably meant a positive shift.

**Non-positive rolling windows → error.** Every windowed `ts_*`/`decay` operator (`ts_mean`, `ts_std`, `ts_rank`, `ts_zscore`, `decay_linear`, `ts_corr`, `ts_cov`, `ts_regression`, and the rest listed in `WINDOWED_OPERATORS`) requires a **positive** window. A window `<= 0` is an error:

```text
ts_mean(close, 0)     # ERROR: requires a positive rolling window
ts_std(returns, -10)  # ERROR
```

**Very long windows → warning.** A window greater than 504 days (~2 trading years) is usually a typo, so it is flagged as a warning, not blocked.

**Wrong argument count → error.** If a windowed or shift operator is called with too few arguments to even have its window/shift argument, the linter reports an error.

The linter unwraps unary signs, so it catches `delta(x, -1)` even though the parser represents `-1` as a unary-minus over the literal `1`. Note it only checks **literal** window/shift values; if you pass a computed expression it cannot statically verify it.

## Worked examples

These are real entries from the Load Example dropdown (`data/example_alphas.py`). Each ships an expression plus recommended settings.

### 1. Sector-neutral momentum

```text
group_neutralize(rank(momentum_60), sector)
```

Read it inside-out. `rank(momentum_60)` percentile-ranks 60-day momentum across all tickers each day. `group_neutralize(..., sector)` then subtracts the mean rank of each GICS sector, so the signal is zero-centered *within* every sector. The net effect: you bet on the strongest-momentum name **in each sector** rather than tilting into whichever sector is hottest — cleaner stock-specific exposure at the cost of giving up sector-rotation signal. This is the canonical use of a `group_*` operator with a GICS label field.

### 2. P/E mean reversion (the value factor)

```text
-zscore(pe_ratio)
```

`zscore(pe_ratio)` standardizes P/E cross-sectionally each day. The **leading unary minus** flips the sign so that *low* P/E (cheap) names score high and *high* P/E (expensive) names score negative — the classic value tilt. Note the sign convention: because P/E is a price/earnings *level* where low is good, it must be negated; contrast `rank(fcf_yield)`, where FCF yield is already a yield (higher = cheaper) and ranks directly with no negation. Fundamentals are lagged ~1 quarter as a point-in-time proxy, so this won't react to fresh earnings until the next quarter.

### 3. Momentum, only when VIX is calm

```text
trade_when(less(vix, 20), rank(momentum_20))
```

`less(vix, 20)` returns a `1.0`/`0.0` condition — true on days the VIX macro field is below 20. `trade_when` takes the `rank(momentum_20)` position **only when that condition fires**, and otherwise **carries the previous position forward** rather than flattening. The result: you hold momentum through calm regimes and freeze your book when volatility spikes, trading fewer whipsaws for some missed continuations. Swap `trade_when` for `when` and the off-regime days go NaN (fully out of the market) instead of carrying forward — see the yield-curve-reversal example for that pattern.

## Where to go next

- [Operator reference](reference/operators.md) — every operator, its arguments, and exact semantics.
- [Field reference](reference/fields.md) — the full list of data fields (price, volume, volatility, momentum, GICS labels, macro, fundamentals).
- [Alpha cookbook](reference/cookbook.md) — more complete, annotated example expressions to adapt.