from pathlib import Path

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

DATA_START = "2019-01-01"
DATA_END = "2024-12-31"

CACHE_DIR = Path(__file__).parent / "data" / "cache"

DEFAULT_CAPITAL = 10_000_000
DEFAULT_BOOKSIZE = 20_000_000
