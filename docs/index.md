# QuantLab — overview & quickstart

QuantLab is a WorldQuant-Brain-style **cross-sectional alpha backtester** for US equities. You write an alpha as an expression in a small DSL, and the platform pipes it through neutralization → truncation → sizing → PnL, returning Sharpe / drawdown / turnover charts back in under a second.

A representative expression:

```
-rank(delta(close, 5)) * ts_std(returns, 20)
```

Under the hood, a recursive-descent parser builds an AST, an evaluator walks it, and a vectorised pandas backtester runs the pipeline. The engine ships **80+ operators** across time-series, cross-sectional, group (sector-aware), conditional, and arithmetic categories.

## Core workflow

The research loop is: **write an expression → backtest → inspect metrics → save / compare.**

1. **Write** an expression in the editor. You get syntax highlighting, in-editor autocomplete (operators + fields), a look-ahead-bias linter that catches things like `delay(x, -1)` before the backtest runs, and validate-without-running.
2. **Backtest** it. Choose a universe preset, a neutralization mode (`none` / `market` / `sector` / `industry_group` / `industry` / `sub_industry`), truncation, booksize sizing, transaction costs in bps, and optional decay. Defaults are `market` neutralization, 0.05 truncation, a $20M booksize, and 5 bps of transaction cost.
3. **Inspect metrics.** Sharpe, CAGR, Sortino, Calmar, max drawdown, turnover, fitness, win rate, profit factor, beta vs SPY, information ratio, rolling 63-day Sharpe, and a monthly returns heatmap. Every backtest also surfaces a **Fama-French 5-factor decomposition** (residual alpha + factor loadings with t-stats), a **Deflated Sharpe Ratio** (Bailey & López de Prado multi-trial bias correction), and an **IS/OOS split** with an overfitting verdict.
4. **Save / compare.** Persist alphas to SQLite, run side-by-side comparison of 2–4 alphas, blend multiple alphas with an MV optimizer (equal / inverse-variance / Markowitz tangency / risk parity), and compute pairwise correlation across saved alphas.

You can also drive parameter sweeps with `{a..b}` / `{a..b:s}` syntax, rendered as 1D bars, a 2D heatmap, or an N-D table.

## Data

Three sources, parquet-cached (24 h TTL on prices, 1 week on fundamentals):

- **OHLCV** (60 fields) — yfinance: 7 base + 53 derived (momentum, realized vol, intraday structure, microstructure proxies).
- **Macro** (12 fields) — FRED daily: VIX, Treasury curve (3M / 2Y / 10Y), credit spreads (HY, BAA, AAA), DXY, WTI oil, plus 3 computed term/credit spreads.
- **Fundamentals** (31 fields) — yfinance quarterly with a 1-quarter lag PIT proxy: 16 raw line items + 15 ratios (P/E, P/B, EV/EBITDA, ROE, ROA, margins, FCF yield). At 50–100 ticker scale yfinance's free API typically returns these mostly-NaN, so fundamentals are hidden from the operator catalog and rejected in validation when coverage drops below 20%.

Universes include 4 curated built-in presets (S&P 100 top-50, S&P 100 extended ~75, NASDAQ-100 subset ~45, Tech/Comm Services focus), plus larger on-demand universes (S&P 500, Russell 1000) loaded lazily on first use. A shared GICS catalog and optional point-in-time S&P 100 membership gating back these.

## Quick start (Docker)

Requires Docker Desktop running.

```bash
docker compose up --build
```

- Frontend: http://localhost
- Backend API: http://localhost:8000  (Swagger UI at /docs)

The backend's parquet cache is bind-mounted from `backend/data/cache/`, so the first boot's yfinance round-trip (~30 s for 50 tickers) survives container restarts.

## Quick start (local, no Docker)

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py            # ~30 s, populates parquet cache
uvicorn main:app --reload                  # http://localhost:8000

# frontend (separate shell)
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

## Next steps

- [Concepts](concepts.md) — the pipeline (neutralization → truncation → sizing → PnL), data model, and analytics.
- [Writing alphas](writing-alphas.md) — the DSL, operators, fields, and the look-ahead-bias linter.
- [Alpha cookbook](reference/cookbook.md) — worked example expressions with recommended settings.