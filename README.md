# QuantLab

A quantitative backtesting platform for cross-sectional alpha research on US equities. Write expressions in a small DSL (`-rank(delta(close, 5)) * ts_std(returns, 20)`), pipe them through neutralization → truncation → sizing → PnL, get Sharpe / drawdown / turnover charts back in under a second.

## Project overview

- **Universe** — picker over built-in presets (S&P 100 top-50, S&P 100 extended ~75, NASDAQ-100 subset, Tech/Comm Services focus) plus a custom-tickers mode that lazy-loads from yfinance. Full GICS catalog (sector / industry group / industry / sub-industry).
- **Data** — daily OHLCV from yfinance, parquet-cached (24h TTL). 32 fields available in expressions: 7 base (open/high/low/close/volume/returns/vwap) + 25 derived (momentum, vol, liquidity, candle structure, etc.).
- **Engine** — recursive-descent parser → AST evaluator → vectorised pandas backtester. 23 operators (rolling, cross-sectional, arithmetic, conditional).
- **Backtest** — neutralization across all 4 GICS levels (`none`/`market`/`sector`/`industry_group`/`industry`/`sub_industry`), truncation, booksize sizing, transaction costs in bps, optional decay.
- **Analytics** — Sharpe, CAGR, Sortino, Calmar, max drawdown, turnover, fitness, win rate, profit factor, beta vs SPY, information ratio, rolling 63-day Sharpe, monthly returns heatmap, **Fama-French 5-factor decomposition** (residual alpha + factor loadings with t-stats), **IS/OOS split** with overfitting verdict.
- **Persistence** — SQLite for saved alphas + cached results; multi-blend and pairwise-correlation endpoints across saved alphas.

## Architecture

```
                ┌────────────────────────────────────────┐
                │          Browser (Vite app)            │
                │  editor • dashboard • charts • sidebar │
                └────────────────┬───────────────────────┘
                          fetch  │ JSON (CORS)
                                 ▼
       ┌──────────────────────────────────────────────────────┐
       │                FastAPI  (uvicorn :8000)              │
       │  /api/simulate  /api/validate  /api/operators        │
       │  /api/alphas (CRUD, multi-blend, correlations)       │
       │  /api/data/preview  /api/universe   /health          │
       └────┬─────────────────┬───────────────────┬───────────┘
            │                 │                   │
            ▼                 ▼                   ▼
   engine/parser.py    engine/backtester.py   db/database.py
   engine/evaluator.py analytics/perf.py      (SQLite alphas)
   engine/operators.py
            │
            ▼
   data/fetcher.py  ──▶  yfinance  →  data/cache/*.parquet
                          (50 tickers × 25 derived fields)
```

## Tech stack

| Layer | Tool |
|---|---|
| Backend | FastAPI, uvicorn (Python 3.11) |
| Numerics | pandas, numpy, pyarrow |
| Data | yfinance, parquet cache |
| Persistence | SQLite via aiosqlite |
| Frontend | Vite + Vanilla JS (no framework) |
| Charts | TradingView Lightweight Charts (CDN, v4.2.3) + Canvas + CSS grid |
| Testing | pytest (169 tests), GitHub Actions CI |
| Container | Docker (multi-stage), nginx for static frontend |
| Deploy | Render (web + static services), Vercel (frontend) |

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

## API endpoints

| Method | Path | What it does |
|---|---|---|
| GET  | `/health` | liveness probe |
| GET  | `/api/universe` | legacy: default preset's tickers + sector map (kept for v1 clients) |
| GET  | `/api/universes` | all built-in presets + their available neutralization modes |
| GET  | `/api/operators` | 23 operators + 32 fields with descriptions |
| POST | `/api/validate` | `{expression}` → `{valid, error}` (pure parse check) |
| POST | `/api/simulate` | `{expression, settings}` → IS/OOS metrics + timeseries + monthly_returns + factor decomposition + data-quality disclosure |
| POST | `/api/alphas` | save expression + run results to SQLite |
| GET  | `/api/alphas` | list saved alphas (newest first) |
| GET  | `/api/alphas/{id}` | full record incl. parsed `result_json` |
| DELETE | `/api/alphas/{id}` | delete |
| POST | `/api/alphas/multi-blend` | `[{expression, weight}…]` → simulate the weighted-sum alpha |
| POST | `/api/alphas/correlations` | pairwise Pearson on saved alphas' daily returns |
| GET  | `/api/data/preview?ticker=AAPL` | last 30 rows for that ticker, all 32 fields |

Browse the auto-generated Swagger UI at `/docs` for full schemas and a "Try it out" button.

## Example alphas

All measured on the 50-ticker S&P 100 universe, 2019-01 → 2024-12, $20M booksize, 5 bps transaction cost. `decay_linear(...)` is a left-weighted moving average that lowers turnover.

| Expression | Neutralization | Sharpe | AnnRet | MaxDD | Turnover/day |
|---|---|---|---|---|---|
| `decay_linear(rank(momentum_20), 20)` | none | **1.02** | 13.7% | -24% | $0.74M |
| `rank(momentum_20) / (realized_vol + 0.001)` | none | 0.96 | 12.3% | -23% | $2.86M |
| `rank(close_to_high_252)` | none | 0.88 | 12.6% | -27% | $1.87M |
| `decay_linear(rank(close_to_high_252), 20)` | none | 0.89 | 12.9% | -29% | $0.39M |
| `rank(delta(close, 5))` | market | -0.90 | -7.9% | -40% | $11.1M |
| `-rank(delta(close, 5))` (5d reversal) | market | -1.05 | -9.5% | -51% | $11.1M |
| `rank(volume) * rank(returns)` | market | -1.33 | -13.2% | -62% | (high) |

The pattern is: **long-only momentum** clears costs on this universe + window; **market-neutral cross-sectional** rank alphas systematically short the mega-caps that dominated 2019-2024 and bleed.

## Deployment

### Render (recommended, free tier works)

1. Push the repo to GitHub.
2. In Render, "New" → "Blueprint" → point at the repo.
3. `render.yaml` provisions two services automatically:
   - `quantlab-backend` (Python web service)
   - `quantlab-frontend` (static site)
4. After the backend deploys, copy its URL (e.g. `https://quantlab-backend.onrender.com`) into the frontend service's `VITE_API_URL` env var, then trigger a manual rebuild so Vite re-bakes the URL into the bundle.

Free-tier note: services sleep after 15 min idle. First request after sleep takes ~30 s while yfinance re-downloads (fresh container = empty cache, no host volume).

### Docker on a single host

```bash
# Custom backend URL baked into the frontend bundle
VITE_API_URL=https://api.example.com docker compose build
docker compose up -d
```

### Manual

Anything that runs `pip install + uvicorn` and serves a static `dist/` directory works. The `data/cache/` directory must be writable; mount it as a volume for persistence.

## Repository layout

```
QuantLab/
├── .github/workflows/ci.yml         pytest on every push to main
├── backend/
│   ├── main.py                      FastAPI app + endpoints
│   ├── config.py                    UNIVERSE, SECTOR_MAP, env-aware paths
│   ├── data/
│   │   ├── fetcher.py               yfinance + 32-field matrix builder
│   │   ├── factors.py               Fama-French 5 factor download + cache
│   │   ├── universes.py             preset registry + GICS catalog (sector/industry/sub-industry)
│   │   └── universe.py
│   ├── engine/
│   │   ├── parser.py                tokenizer + recursive-descent parser
│   │   ├── operators.py             23 ops (ts/cs/arithmetic/conditional)
│   │   ├── evaluator.py             AST walker
│   │   ├── lint.py                  look-ahead-bias linter
│   │   └── backtester.py            run() pipeline + IS/OOS split
│   ├── analytics/
│   │   ├── performance.py           Sharpe, CAGR, Sortino, IS/OOS comparison
│   │   └── factor_decomp.py         FF5 OLS regression + factor loadings
│   ├── models/schemas.py            Pydantic request/response shapes
│   ├── db/                          aiosqlite, migrations
│   ├── scripts/
│   │   ├── download_data.py         pre-populate parquet cache
│   │   └── verify_alphas.py         CLI run of canonical expressions
│   ├── tests/                       pytest (130 tests)
│   ├── Dockerfile + .dockerignore
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── main.js                  layout, welcome modal, wiring
│   │   ├── api.js                   fetch wrappers
│   │   ├── ui/toast.js              toast + confirmDialog (replaces alert/confirm)
│   │   ├── styles/index.css         design system
│   │   └── components/
│   │       ├── editor.js            highlight + validate + settings panel
│   │       ├── dashboard.js         metrics + IS/OOS panel + factor decomp
│   │       ├── charts.js            equity, drawdown, sharpe, hist, heatmap
│   │       ├── sidebar.js           saved alphas, blend, correlations
│   │       └── correlation.js       pairwise heatmap
│   ├── nginx.conf                   SPA fallback config
│   ├── Dockerfile + .dockerignore
│   └── package.json
├── docker-compose.yml
├── render.yaml
└── README.md
```

## Tests

```bash
backend/.venv/bin/pytest backend/tests/ -v
```

169 tests covering parser (operators + invalid syntax + precedence), operators (synthetic 100×10 fixture against pandas), backtester (constant-alpha invariants, turnover ≥ 0, cost-vs-zero-cost, IS/OOS partitioning, T+1 execution lag), the look-ahead-bias linter, the Fama-French 5-factor decomposition (synthetic-truth recovery), Deflated Sharpe Ratio formula sanity, point-in-time universe gating, the universe registry + GICS-mode neutralization (all 4 levels), and the API (TestClient-based, including the full save → load → correlate cycle).

CI runs the same suite on every push to `main` via `.github/workflows/ci.yml`.

## Known limitations / drawbacks

These are honest gaps, not promises.  The platform's metrics should be read in light of them.

- **Survivorship bias.** Every preset universe is a *current* index snapshot, not point-in-time membership.  Names that were in the index in 2019 but got delisted, acquired, or removed by 2024 are absent.  Estimated Sharpe inflation: **0.1–0.3** for typical strategies.  Proper fix needs paid PIT data (CRSP / Norgate / Sharadar).  The dashboard surfaces this as a banner on every backtest.
- **Naïve transaction cost model.** Flat bps regardless of trade size or stock liquidity.  Real cost ≈ half-spread + market impact (∝ √(trade_size / ADV)) + commission + borrow.  The platform has `adv20` as a field but doesn't use it for cost modeling — high-turnover alphas look better here than they would live.
- **No execution-lag model.** Assumes signal-at-close = trade-at-close.  Real desks sign at close, trade at next open.  Adds 10–30 bps of slippage we ignore.
- **No borrow cost on shorts.** Market-neutral strategies have a 1–3 % per-year cost we don't charge.
- **Single 70/30 IS/OOS split**, not walk-forward.  One regime in OOS can pass or fail you by chance.  Walk-forward (rolling 12-month OOS windows) is the proper test.
- **Daily resolution only.** Can't capture intraday signals or do anything tick-level.
- **No fundamentals or alternative data.** Limits the ceiling of what alphas can be expressed.

## Mitigations already shipped

- ✅ **IS/OOS split** on every backtest with overfitting-decay verdict ("Robust" / "Moderate decay" / "High decay" / "Severe overfit" / "Negative OOS").
- ✅ **Look-ahead-bias linter** — `delay(x, -1)` and `delta(x, -5)` are caught at parse time and rejected by `/api/simulate` with a 400.
- ✅ **Fama-French 5-factor decomposition** on every backtest. Surfaces residual alpha (intercept), per-factor loadings (`market`, `size`, `value`, `profitability`, `investment`) with t-stats, R², and the share of variance explained by factor exposure. Tells you how much of your headline Sharpe is just market beta.
- ✅ **Data-quality banner** on the dashboard so the survivorship-bias caveat is visible alongside every reported Sharpe.
- ✅ **GitHub Actions CI** running pytest on every push, including an integration test of the full save → load → correlate cycle to prevent shape-rename regressions.
