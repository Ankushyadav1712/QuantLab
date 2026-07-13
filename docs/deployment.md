# Deployment

QuantLab ships as two deployable pieces: a **FastAPI backend** (uvicorn, Python 3.11) and a **static frontend** (Vite build served as plain files). This page covers running both locally with `docker-compose` and deploying to Render.com via the checked-in `render.yaml`, plus the environment variables you need to wire the two together and to keep a small instance from running out of memory.

See the [Overview](index.md) for what the platform does.

## Local (docker-compose)

`docker-compose.yml` brings up both services together. Requires Docker Desktop running.

```bash
docker compose up --build
```

- Frontend: `http://localhost` (nginx on port 80)
- Backend API: `http://localhost:8000` (Swagger UI at `/docs`)

What the compose file does:

- **backend** builds from `backend/Dockerfile`, publishes port `8000:8000`, and runs with `ENVIRONMENT=development`. Its parquet cache directory is bind-mounted (`./backend/data/cache:/app/data/cache`), so the first boot's yfinance download survives container restarts instead of re-fetching every time.
- **frontend** builds from `frontend/Dockerfile` and publishes port `80:80`. It waits for the backend's healthcheck (`condition: service_healthy`) before starting.
- The backend healthcheck polls `http://127.0.0.1:8000/health` every 10s with a 30s start period (timeout 3s, 12 retries), so the frontend does not come up until the API is live.

### Pointing the frontend at a different backend URL

`VITE_API_URL` is baked into the frontend bundle **at build time** (Vite inlines `VITE_*` vars). The compose file defaults it to `http://localhost:8000`. To bake in a different URL, set it on the `build` step:

```bash
VITE_API_URL=https://api.example.com docker compose build
docker compose up -d
```

Changing this variable requires a **rebuild** — it is not read at runtime.

## Render.com (render.yaml)

`render.yaml` is a Render Blueprint that provisions **two services**. In Render, choose "New" → "Blueprint" and point it at the repo.

### Service 1 — `quantlab-backend` (Python web service)

| Field | Value |
|---|---|
| `type` | `web` |
| `runtime` | `python` |
| `plan` | `free` |
| `buildCommand` | `pip install -r backend/requirements.txt && cd backend && python scripts/preseed_cache.py` |
| `startCommand` | `uvicorn main:app --app-dir backend --host 0.0.0.0 --port $PORT` |
| `healthCheckPath` | `/health` |

The build step pre-seeds the parquet data cache so the first request after a cold start does not have to hit the network. The start command uses `--app-dir backend` (rather than `backend.main:app`) because the repo's internal imports are flat (`from data.fetcher import ...`).

Env vars set in `render.yaml`:

- `ENVIRONMENT=production`
- `PYTHON_VERSION=3.11.10`
- `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` (`sync: false`) — see below

### Durable persistence (Turso)

Render's free disk is **ephemeral**: it is wiped on every restart, redeploy, and idle spin-down. The saved-alpha database is a SQLite file on that disk, so **without Turso your saved alphas disappear** the next time the instance cycles. Turso is hosted [libSQL](https://turso.tech) (SQLite-compatible) with a free tier; the backend talks to it over its HTTP protocol, so the data lives off the ephemeral disk.

**It's opt-in.** With no Turso env vars the backend uses the local SQLite file exactly as before — fine for local dev and the test suite. Set both vars and it switches to Turso automatically. Confirm which backend is live at any time via `GET /health` → `{"status":"ok","persistence":"turso"}` (or `"local"`).

One-time setup:

1. Install the CLI and sign up: `curl -sSfL https://get.tur.so/install.sh | bash`, then `turso auth signup`.
2. Create a database: `turso db create quantlab`.
3. Get its URL: `turso db show quantlab --url` → a `libsql://…turso.io` value.
4. Mint a token: `turso db tokens create quantlab`.
5. In the Render dashboard (backend service → Environment), set `TURSO_DATABASE_URL` to the URL from step 3 and `TURSO_AUTH_TOKEN` to the token from step 4, then save (Render redeploys).

The schema is created automatically on first boot (`init_db` runs the same `CREATE TABLE IF NOT EXISTS` + additive migrations against Turso). Nothing to migrate by hand. `libsql://` URLs are converted to `https://` internally; pass the URL exactly as `turso db show` prints it.

### Service 2 — `quantlab-frontend` (static site)

| Field | Value |
|---|---|
| `type` | `web` |
| `runtime` | `static` |
| `buildCommand` | `cd frontend && npm install && npm run build` |
| `staticPublishPath` | `frontend/dist` |

A rewrite route sends all paths (`/*`) to `/index.html` for SPA-style routing:

```yaml
routes:
  - type: rewrite
    source: /*
    destination: /index.html
```

### Wiring VITE_API_URL (required after first deploy)

The frontend service declares `VITE_API_URL` with `sync: false`, meaning Render does **not** auto-populate it — you set it manually. Because the value is baked into the Vite bundle at build time, the order matters:

1. Deploy the blueprint. The backend service (`quantlab-backend`) gets an auto-generated URL, e.g. `https://quantlab-backend.onrender.com`.
2. Copy that URL into the frontend service's `VITE_API_URL` environment variable.
3. Trigger a manual rebuild of the frontend so Vite re-bakes the URL into the bundle.

Until step 3 runs, the frontend bundle will not carry the backend URL.

## Environment variables

| Variable | Service | Default | Effect |
|---|---|---|---|
| `VITE_API_URL` | frontend | `http://localhost:8000` (compose) / unset (Render) | Backend base URL, inlined into the Vite bundle **at build time**. Change requires a rebuild. |
| `QUANTLAB_API_TOKEN` | backend | *(empty)* | Bearer-token gate on alpha-mutating endpoints (`POST /api/alphas`, `DELETE /api/alphas/{id}`, and `POST /api/alphas/{id}/rollback/{version}`). When unset, auth is bypassed (frictionless local dev). Set it on a public deployment to prevent drive-by deletion of saved alphas. |
| `QUANTLAB_MAX_CONCURRENT_BACKTESTS` | backend | `1` | Semaphore cap on how many heavy backtest endpoints run at once. |
| `QUANTLAB_MAX_WORKERS` | backend | `1` | Cap on parallel backtest workers inside `run_batch` (the parameter sweep and batch-simulate paths). |
| `QUANTLAB_BACKTEST_QUEUE_TIMEOUT` | backend | `90` | Seconds a queued request waits for a free backtest slot before it is shed with a `429`. |

### Backtest concurrency knobs (why the defaults are 1)

A full-universe backtest is memory-heavy. Running two at once roughly doubles peak RAM and can OOM-kill the single worker on a small host — for example Render's 512 MB free tier — which 502s every in-flight request.

- `QUANTLAB_MAX_CONCURRENT_BACKTESTS` (default `1`) gates heavy work behind a `BoundedSemaphore` so only N run at once. The gate wraps `/api/simulate`, `/api/sweep`, `/api/compare`, `/api/batch_simulate`, and `/api/alphas/multi-blend`. With the default, these run strictly **sequentially**. A queued request that cannot get a slot within `QUANTLAB_BACKTEST_QUEUE_TIMEOUT` (default 90s) is shed with a `429` (`Retry-After: 5`) rather than tying up a worker thread. Bump this only on a larger instance where the RAM headroom exists.
- `QUANTLAB_MAX_WORKERS` (default `1`) caps parallel workers in `run_batch` (used by the parameter sweep and batch-simulate). Each worker holds a full backtest's memory, so N workers ≈ N× peak RAM. The code deliberately does **not** trust `os.cpu_count()` as a default, because on shared platforms that reports the host's cores, not the container's throttled slice. Raise it only where RAM allows.

### Free-tier reality

The Render free tier (and any single-worker / 512 MB instance) runs one uvicorn worker. With the default knobs above, backtests are **serialized** — one at a time — specifically so a small instance cannot OOM. Free-tier services also sleep after idle, so the first request after a wake takes longer while the fresh container warms up.

## Docker images (reference)

Both Dockerfiles are multi-stage:

- **`backend/Dockerfile`** — a `python:3.11-slim` builder installs deps with `pip install --user`, the runtime stage (also `python:3.11-slim`) copies `/root/.local`, pre-seeds the cache with `ENVIRONMENT=production python scripts/preseed_cache.py`, `EXPOSE 8000`, and runs `uvicorn main:app --host 0.0.0.0 --port 8000`.
- **`frontend/Dockerfile`** — a `node:20-alpine` builder runs `npm ci` + `npm run build` (Vite 8 requires Node ≥20.19), reading `VITE_API_URL` via `ARG`/`ENV` at build time; the runtime stage is `nginx:alpine` serving `/usr/share/nginx/html`, with the project's `nginx.conf` copied to `/etc/nginx/conf.d/default.conf`, `EXPOSE 80`.