# QuantLab

A quantitative backtesting platform: FastAPI backend + Vite (vanilla JS) frontend.

## Layout

```
QuantLab/
├── backend/          FastAPI service, data layer, scripts
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   ├── fetcher.py
│   │   └── universe.py
│   ├── scripts/
│   │   └── download_data.py
│   └── requirements.txt
├── frontend/         Vite + Vanilla JS app
└── README.md
```

## Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py     # populates data/cache/*.parquet
uvicorn main:app --reload           # http://127.0.0.1:8000
```

Health check: `GET http://127.0.0.1:8000/health` → `{"status": "ok"}`.

### Data layer

- `config.UNIVERSE` — 50 liquid S&P 100 tickers.
- `config.SECTOR_MAP` — GICS sector for each ticker.
- `data.fetcher.DataFetcher`
  - `download_universe(tickers, start, end)` — yfinance OHLCV, parquet-cached per ticker (24h TTL).
  - `get_data_matrix(field)` — `(dates × tickers)` matrix for `open/high/low/close/volume/returns/vwap`.
- `data.universe.UniverseManager` — `get_tickers()`, `get_sector(ticker)`.

## Frontend

```bash
cd frontend
npm install
npm run dev
```
