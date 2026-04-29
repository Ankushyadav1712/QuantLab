from __future__ import annotations

from db.database import connect

CREATE_ALPHAS_TABLE = """
CREATE TABLE IF NOT EXISTS alphas (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    expression    TEXT    NOT NULL,
    notes         TEXT    DEFAULT '',
    sharpe        REAL,
    annual_return REAL,
    max_drawdown  REAL,
    turnover      REAL,
    fitness       REAL,
    created_at    TEXT    NOT NULL,
    result_json   TEXT
);
"""


async def init_db() -> None:
    async with connect() as db:
        await db.execute(CREATE_ALPHAS_TABLE)
        await db.commit()
