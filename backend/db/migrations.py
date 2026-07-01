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
    result_json   TEXT,
    code_signature TEXT,
    data_signature TEXT,
    git_hash      TEXT
);
"""

# Forward-only migrations: each row is (column_name, ddl-for-add-column).
# We probe each column with PRAGMA table_info before ALTER so re-runs are
# idempotent.  No down-migrations — this is a single-user research project,
# not a multi-tenant production schema.
_ADDITIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("code_signature", "ALTER TABLE alphas ADD COLUMN code_signature TEXT"),
    ("data_signature", "ALTER TABLE alphas ADD COLUMN data_signature TEXT"),
    ("git_hash", "ALTER TABLE alphas ADD COLUMN git_hash TEXT"),
)


async def init_db() -> None:
    async with connect() as db:
        await db.execute(CREATE_ALPHAS_TABLE)
        # Apply additive migrations for existing databases that predate the
        # provenance columns.  PRAGMA table_info gives us [(cid, name, ...), …].
        cursor = await db.execute("PRAGMA table_info(alphas)")
        existing = {row[1] for row in await cursor.fetchall()}
        for col_name, ddl in _ADDITIVE_COLUMNS:
            if col_name not in existing:
                await db.execute(ddl)
        await db.commit()
