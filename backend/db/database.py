from __future__ import annotations

from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quantlab.db"


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> aiosqlite.Connection:
    """Return a new aiosqlite connection (caller must `async with`)."""
    _ensure_dir()
    return aiosqlite.connect(DB_PATH)
