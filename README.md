# QuantLab

A quantitative backtesting platform for cross-sectional alpha research on US equities. Write expressions in a small DSL (`-rank(delta(close, 5)) * ts_std(returns, 20)`), pipe them through neutralization → truncation → sizing → PnL, get Sharpe / drawdown / turnover charts back in under a second.

## Project overview

- **Universe** — picker over 4 built-in presets (S&P 100 top-50, S&P 100 extended ~75, NASDAQ-100 subset, Tech/Comm Services focus). Full GICS catalog (sector / industry group / industry / sub-industry). Optional point-in-time S&P 100 membership gating (e.g. blocks TSLA before its 2020-12-21 inclusion).
- **Data** — three sources, parquet-cached (24 h TTL on prices, 1 week on fundamentals):
  - **OHLCV** (60 fields) — yfinance, 7 base + 53 derived (momentum, realized vol, intraday structure, microstructure proxies)
  - **Macro** (13 fields) — FRED daily: VIX, Treasury curve (3M / 2Y / 10Y), credit spreads (HY, BAA, AAA), DXY, WTI oil, plus computed term/credit spreads
  - **Fundamentals** (26 fields) — yfinance quarterly with 1-quarter lag PIT proxy: 15 raw line items + 11 ratios (P/E, P/B, EV/EBITDA, ROE, ROA, margins, FCF yield)
- **Engine** — recursive-descent parser → AST evaluator → vectorised pandas backtester. **80+ operators** across time-series, cross-sectional, group (sector-aware), conditional, volatility, momentum, and microstructure categories.
- **Backtest** — neutralization across all 4 GICS levels (`none`/`market`/`sector`/`industry_group`/`industry`/`sub_industry`), truncation, booksize sizing, transaction costs in bps, optional decay.
- **Analytics** — Sharpe, CAGR, Sortino, Calmar, max drawdown, turnover, fitness, win rate, profit factor, beta vs SPY, information ratio, rolling 63-day Sharpe, monthly returns heatmap, **Fama-French 5-factor decomposition** (residual alpha + factor loadings with t-stats), **Deflated Sharpe Ratio** (Bailey & López de Prado, multi-trial bias correction), **IS/OOS split** with overfitting verdict.
- **Research workflow** — curated alpha library (10 examples with recommended settings), parameter sweep with `{a..b:s}` syntax (1D bars / 2D heatmap / N-D table), side-by-side comparison of 2–4 alphas, multi-alpha blending with **MV optimizer** (equal / inverse-variance / Markowitz tangency / risk parity).
- **Editor** — syntax highlight, in-editor autocomplete (operators + fields), look-ahead-bias linter (catches `delay(x, -1)` before the backtest runs), validate-without-running.
- **Persistence** — SQLite for saved alphas + cached results; pairwise-correlation endpoint across saved alphas.

## Architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    Browser (Vite app)                        │
   │  editor + autocomplete • dashboard • charts • sweep •        │
   │  comparison • correlation heatmap • sidebar (examples /      │
   │  saved alphas / blend)                                       │
   └────────────────────────┬─────────────────────────────────────┘
                       fetch│ JSON (CORS)
                            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                   FastAPI (uvicorn :8000)                    │
   │  Single-alpha:  /simulate  /validate  /sweep  /compare       │
   │  Catalog:       /operators  /universes  /examples            │
   │  Persistence:   /alphas (CRUD)  /alphas/correlations         │
   │                 /alphas/multi-blend  (MV optimizer)          │
   │  Misc:          /data/preview   /universe   /health          │
   └────┬─────────────┬───────────────┬──────────────┬────────────┘
        │             │               │              │
        ▼             ▼               ▼              ▼
   engine/      engine/         analytics/      db/database.py
   parser.py    backtester.py   performance.py  (aiosqlite,
   evaluator.py + lint.py       factor_decomp   saved alphas)
   operators.py + sweep.py      deflated_sharpe
                                mv_optimizer
        │
        ▼
   data/{fetcher,macro,fundamentals,sp100_history,universes}.py
        │
        ▼
   yfinance + FRED  →  data/cache/*.parquet
                       (60 OHLCV + 13 macro + 26 fundamentals fields)
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
| Testing | pytest (318 tests, 83% coverage), vitest (frontend), GitHub Actions CI |
| Lint / typecheck | ruff (lint + format), mypy, pip-audit |
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

17 endpoints. Browse the auto-generated Swagger UI at `/docs` for full schemas and a "Try it out" button.

**Catalog & metadata**

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/api/universe` | legacy: default preset's tickers + sector map (kept for v1 clients) |
| GET | `/api/universes` | all built-in presets + their available neutralization modes |
| GET | `/api/operators` | every operator + field with category/description/args |
| GET | `/api/examples` | curated alpha library (10 examples with recommended settings) |
| GET | `/api/examples/{id}` | one example by id |
| GET | `/api/data/preview?ticker=AAPL` | last 30 rows for that ticker, every available field |

**Single-alpha research**

| Method | Path | What it does |
|---|---|---|
| POST | `/api/validate` | `{expression}` → `{valid, error}` (parse + lint check) |
| POST | `/api/simulate` | `{expression, settings, n_trials}` → IS/OOS metrics + timeseries + monthly returns + factor decomposition + Deflated Sharpe + data-quality disclosure |
| POST | `/api/sweep` | `{expression, settings, max_combinations}` — expand `{a..b:s}` tokens into a parameter grid, backtest each cell, return ranked results |
| POST | `/api/compare` | `{expressions[2-4], settings}` — IS-only side-by-side overlay |

**Saved alphas**

| Method | Path | What it does |
|---|---|---|
| POST | `/api/alphas` | save expression + run results to SQLite (auth required) |
| GET | `/api/alphas` | list saved alphas (newest first) |
| GET | `/api/alphas/{id}` | full record incl. parsed `result_json` |
| DELETE | `/api/alphas/{id}` | delete (auth required) |
| POST | `/api/alphas/multi-blend` | `{alphas, weight_method, target_vol}` — blend with equal / inverse-variance / Markowitz tangency / risk-parity weights |
| POST | `/api/alphas/correlations` | pairwise Pearson on saved alphas' daily returns |

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
├── .github/
│   ├── workflows/ci.yml             6 parallel jobs: lint, test+coverage,
│   │                                mypy, pip-audit, frontend, docker build
│   └── dependabot.yml               weekly pip / npm / actions / docker bumps
├── pyproject.toml                   ruff + mypy + coverage config
├── backend/
│   ├── main.py                      FastAPI app + 17 endpoints + OPERATORS metadata
│   ├── config.py                    UNIVERSE, SECTOR_MAP, env-aware paths
│   ├── conftest.py                  pytest fixtures
│   ├── analytics/
│   │   ├── performance.py           Sharpe, CAGR, Sortino, IS/OOS comparison
│   │   ├── factor_decomp.py         FF5 OLS regression + factor loadings
│   │   ├── deflated_sharpe.py       Bailey & López de Prado DSR (multi-trial bias)
│   │   └── mv_optimizer.py          equal / inv-vol / Markowitz tangency / risk parity
│   ├── data/
│   │   ├── fetcher.py               yfinance OHLCV + 53 derived fields
│   │   ├── factors.py               Fama-French 5 factor download + cache
│   │   ├── macro.py                 FRED daily series (10 base + 3 derived spreads)
│   │   ├── fundamentals.py          yfinance quarterlies + 1Q PIT lag + 11 ratios
│   │   ├── sp100_history.py         point-in-time S&P 100 inclusion dates
│   │   ├── universes.py             preset registry + GICS catalog
│   │   ├── universe.py              legacy single-universe accessor
│   │   └── example_alphas.py        curated library (10 examples for the UI)
│   ├── engine/
│   │   ├── parser.py                tokenizer + recursive-descent parser
│   │   ├── operators.py             80+ operator implementations
│   │   ├── evaluator.py             AST walker
│   │   ├── lint.py                  look-ahead-bias linter
│   │   ├── sweep.py                 {a..b:s} expansion → cartesian grid
│   │   └── backtester.py            run() pipeline + IS/OOS split
│   ├── models/schemas.py            Pydantic request/response shapes
│   ├── db/
│   │   ├── database.py              aiosqlite connection wrapper
│   │   └── migrations.py            schema init
│   ├── scripts/
│   │   ├── download_data.py         pre-populate parquet cache
│   │   └── verify_alphas.py         CLI run of canonical expressions
│   ├── tests/                       15 files, 222 test funcs / 318 test runs
│   ├── Dockerfile + .dockerignore
│   ├── requirements.txt             prod deps (fastapi, pandas, yfinance, …)
│   └── requirements-dev.txt         + ruff, mypy, pytest-cov, pip-audit
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── main.js                  layout, welcome modal, wiring
│   │   ├── api.js                   fetch wrappers (all 17 endpoints)
│   │   ├── ui/toast.js              toast + confirmDialog (replaces alert/confirm)
│   │   ├── styles/index.css         design system
│   │   └── components/
│   │       ├── editor.js            syntax highlight + validate + settings panel
│   │       ├── autocomplete.js      operator/field dropdown, arrow nav, auto-paren
│   │       ├── dashboard.js         metrics + IS/OOS panel + factor decomp + DSR
│   │       ├── charts.js            equity, drawdown, sharpe, hist, heatmap
│   │       ├── sweep.js             1D bars / 2D heatmap / N-D sortable table
│   │       ├── comparison.js        2-4 alpha overlay (SVG, color-coded)
│   │       ├── sidebar.js           saved alphas, examples, blend, correlations
│   │       └── correlation.js       pairwise heatmap
│   ├── tests/                       vitest (api.test.js + toast.test.js)
│   ├── nginx.conf                   SPA fallback config
│   ├── Dockerfile + .dockerignore
│   └── package.json
├── docker-compose.yml
├── render.yaml
└── README.md
```

## Tests

```bash
# backend
backend/.venv/bin/pytest backend/tests/ -v

# frontend
cd frontend && npm test
```

**Backend** — 222 test functions across 15 files → 318 test runs (parametrized). 83% line coverage. Per-area breakdown:

| Area | File | Tests |
|---|---|---|
| Parser | `test_parser.py` | 15 |
| Operators (synthetic 100×10 vs pandas) | `test_operators.py` | 83 |
| Backtester (constant-alpha invariants, T+1 lag, cost) | `test_backtester.py` | 20 |
| Look-ahead-bias linter | `test_lint.py` | 6 |
| API (TestClient, save→load→correlate cycle) | `test_api.py` | 17 |
| Universe registry + GICS neutralization | `test_universes.py` | 11 |
| Point-in-time S&P 100 gating | `test_pit_universe.py` | 7 |
| FF5 factor decomposition | `test_factor_decomp.py` | 4 |
| Deflated Sharpe formula | `test_deflated_sharpe.py` | 9 |
| MV optimizer (4 weighting methods) | `test_mv_optimizer.py` | 11 |
| Parameter sweep expansion | `test_sweep.py` | 17 |
| Phase B OHLCV-derived fields | `test_phase_b_fields.py` | 22 |
| FRED macro fetch + spreads | `test_macro.py` | 13 |
| Fundamentals + 1Q PIT lag + ratios | `test_fundamentals.py` | 16 |
| Curated example library | `test_examples.py` | 11 |

**Frontend (11 tests)** — api.js fetch wrappers (method/path/auth/error handling), toast UI primitive.

## CI/CD

`.github/workflows/ci.yml` runs **6 parallel jobs** on every push and PR to `main`:

| Job | Tool | Blocking? |
|---|---|---|
| `ruff (backend)` | ruff check + format | yes |
| `pytest (backend)` | pytest + coverage → sticky PR comment | yes |
| `mypy (backend)` | mypy (lenient config in `pyproject.toml`) | yes |
| `pip-audit (backend)` | CVE scan on pinned deps | **no** (advisory) |
| `vite build + vitest` | vitest + production build | yes |
| `docker build` | both Dockerfiles build cleanly (no push) | yes |

Concurrency-gated: a fresh push cancels the in-flight run on the same branch. The data-fetcher parquet cache is shared across runs via `actions/cache`, so yfinance is hit at most once per `config.py` / `fetcher.py` change.

**Coverage** — `pytest-cov` writes `coverage.xml`, which is rendered into a markdown table by `irongut/CodeCoverageSummary` and posted as a sticky PR comment by `marocchino/sticky-pull-request-comment`. Thresholds: red <60, yellow 60–80, green ≥80.

**Dependabot** — `.github/dependabot.yml` opens weekly PRs (Mondays) for pip / npm / GitHub Actions / Docker base images. Minor + patch grouped per ecosystem, majors stay separate.

**Branch protection** — enable in **Settings → Branches → Add rule** with `main` as the pattern. Required status checks: the 5 blocking jobs above (don't add `pip-audit` — it's `continue-on-error`).

Running CI tools locally:

```bash
ruff check backend/ --fix    # auto-fix lint
ruff format backend/         # apply formatter
mypy backend/                # type-check
pip-audit -r backend/requirements.txt   # CVE scan
pytest backend/tests/ --cov  # tests + coverage
```

## Known limitations / drawbacks

These are honest gaps, not promises.  The platform's metrics should be read in light of them.

- **Survivorship bias.** Every preset universe is a *current* index snapshot, not point-in-time membership.  Names that were in the index in 2019 but got delisted, acquired, or removed by 2024 are absent.  Estimated Sharpe inflation: **0.1–0.3** for typical strategies.  Proper fix needs paid PIT data (CRSP / Norgate / Sharadar).  The dashboard surfaces this as a banner on every backtest.
- **Naïve transaction cost model.** Flat bps regardless of trade size or stock liquidity.  Real cost ≈ half-spread + market impact (∝ √(trade_size / ADV)) + commission + borrow.  The platform has `adv20` as a field but doesn't use it for cost modeling — high-turnover alphas look better here than they would live.
- **No execution-lag model.** Assumes signal-at-close = trade-at-close.  Real desks sign at close, trade at next open.  Adds 10–30 bps of slippage we ignore.
- **No borrow cost on shorts.** Market-neutral strategies have a 1–3 % per-year cost we don't charge.
- **Single 70/30 IS/OOS split**, not walk-forward.  One regime in OOS can pass or fail you by chance.  Walk-forward (rolling 12-month OOS windows) is the proper test.
- **Daily resolution only.** Can't capture intraday signals or do anything tick-level.
- **Fundamentals are rate-limited.** yfinance's free fundamentals API silently truncated to the 5 most-recent quarters in 2025 and rate-limits at universe sizes > ~30 tickers. The 26 fundamentals fields (`roe`, `pe_ratio`, `revenue`, …) exist in the codebase but typically come back mostly-NaN at our 50–100 ticker scale. The platform detects this at startup, **hides fundamentals from `/api/operators` when coverage < 20%, and rejects fundamentals expressions in `/api/validate`** with a clear "fundamentals unavailable" error — so the user can't get a confusing "all dates dropped" failure later. Real fundamentals research needs a paid source (Polygon.io / WRDS / Compustat).
- **No alternative data.** No news sentiment, options flow, ETF holdings, satellite, etc. Limits the ceiling of what alphas can be expressed.

## Mitigations already shipped

- ✅ **IS/OOS split** on every backtest with overfitting-decay verdict ("Robust" / "Moderate decay" / "High decay" / "Severe overfit" / "Negative OOS").
- ✅ **Deflated Sharpe Ratio** (Bailey & López de Prado) on every `/api/simulate`. Pass `n_trials` to declare the size of the search you ran; the DSR p-value tells you whether your headline Sharpe survives the multiple-comparisons penalty.
- ✅ **Point-in-time S&P 100 membership gating** — opt-in flag that masks tickers before their inclusion date (e.g. TSLA → no trades before 2020-12-21). Catches anachronisms; Sharpe inflation from this alone is often 0.1–0.3.
- ✅ **Look-ahead-bias linter** — `delay(x, -1)` and `delta(x, -5)` are caught at parse time and rejected by `/api/simulate` with a 400. Also runs inside `/api/sweep` and `/api/compare` per cell.
- ✅ **Fundamentals 1-quarter lag** — yfinance reports are forward-shifted by 90 days as a coarse PIT proxy. Surfaces in the data-quality banner so users know it's a proxy, not real PIT.
- ✅ **Fama-French 5-factor decomposition** on every backtest. Surfaces residual alpha (intercept), per-factor loadings (`market`, `size`, `value`, `profitability`, `investment`) with t-stats, R², and the share of variance explained by factor exposure. Tells you how much of your headline Sharpe is just market beta.
- ✅ **Data-quality banner** on the dashboard so the survivorship-bias caveat is visible alongside every reported Sharpe.
- ✅ **GitHub Actions CI** — 6 parallel jobs on every push: ruff lint, pytest with coverage (posted as PR comment), mypy type-check, pip-audit CVE scan, frontend build + vitest, and Docker image build. Includes an integration test of the full save → load → correlate cycle.
