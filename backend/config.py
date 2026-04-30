import os
from pathlib import Path

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "AVGO", "COST", "PEP", "KO", "ADBE", "CSCO", "CRM", "ACN", "MCD",
    "TMO", "ABT", "NFLX", "LIN", "DHR", "TXN", "NKE", "VZ", "AMGN", "PM",
    "MS", "GS", "RTX", "INTC", "SPGI", "BLK", "MDLZ", "ADP", "GILD", "T",
]

SECTOR_MAP = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOG": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Information Technology",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "UNH": "Health Care",
    "XOM": "Energy",
    "LLY": "Health Care",
    "JNJ": "Health Care",
    "WMT": "Consumer Staples",
    "MA": "Financials",
    "PG": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "CVX": "Energy",
    "MRK": "Health Care",
    "ABBV": "Health Care",
    "AVGO": "Information Technology",
    "COST": "Consumer Staples",
    "PEP": "Consumer Staples",
    "KO": "Consumer Staples",
    "ADBE": "Information Technology",
    "CSCO": "Information Technology",
    "CRM": "Information Technology",
    "ACN": "Information Technology",
    "MCD": "Consumer Discretionary",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "NFLX": "Communication Services",
    "LIN": "Materials",
    "DHR": "Health Care",
    "TXN": "Information Technology",
    "NKE": "Consumer Discretionary",
    "VZ": "Communication Services",
    "AMGN": "Health Care",
    "PM": "Consumer Staples",
    "MS": "Financials",
    "GS": "Financials",
    "RTX": "Industrials",
    "INTC": "Information Technology",
    "SPGI": "Financials",
    "BLK": "Financials",
    "MDLZ": "Consumer Staples",
    "ADP": "Industrials",
    "GILD": "Health Care",
    "T": "Communication Services",
}

# Production uses a smaller window to keep cold-start data downloads fast on
# free-tier hosts; development gets the full 6-year history.
DATA_START = "2020-01-01" if ENVIRONMENT == "production" else "2019-01-01"
DATA_END = "2024-12-31"

CACHE_DIR = Path(__file__).parent / "data" / "cache"

DEFAULT_CAPITAL = 10_000_000
DEFAULT_BOOKSIZE = 20_000_000

# CORS allow-list. In development we additionally permit "*" so that loading
# the bundle from any host (e.g. localhost variants, network LAN) works.
# In production set the env var ALLOWED_ORIGINS to a comma-separated list,
# e.g. ALLOWED_ORIGINS="https://quantlab.vercel.app,https://www.example.com".
_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]
_env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _DEFAULT_ALLOWED_ORIGINS
)
