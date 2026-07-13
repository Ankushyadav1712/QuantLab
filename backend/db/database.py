from __future__ import annotations

import base64
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

import aiosqlite
import httpx

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quantlab.db"

# ---------------------------------------------------------------------------
# Persistence backend selection
# ---------------------------------------------------------------------------
# Render's free tier has an EPHEMERAL disk — the local SQLite file is wiped on
# every restart / redeploy / idle spin-down, so saved alphas evaporate.  When
# TURSO_DATABASE_URL + TURSO_AUTH_TOKEN are set, the DB layer instead talks to
# Turso (hosted libSQL) over its HTTP pipeline protocol, which survives
# restarts.  Absent those env vars — local dev, the test suite, or any deploy
# that hasn't configured Turso — it falls back to the local aiosqlite file, so
# nothing changes for existing setups.
#
# Both backends expose the exact subset of the aiosqlite API the app uses:
#   async with connect() as db:
#       db.row_factory = aiosqlite.Row      # accepted (Turso rows are always
#                                           # name-and-index addressable)
#       cur = await db.execute(sql, params)
#       row = await cur.fetchone()          # Row | None
#       rows = await cur.fetchall()         # list[Row]
#       new_id = cur.lastrowid
#       await db.commit()


def _turso_env() -> tuple[str, str]:
    return (
        os.getenv("TURSO_DATABASE_URL", "").strip(),
        os.getenv("TURSO_AUTH_TOKEN", "").strip(),
    )


def turso_enabled() -> bool:
    url, token = _turso_env()
    return bool(url and token)


def active_backend() -> str:
    """`"turso"` or `"local"` — for /health and startup logging."""
    return "turso" if turso_enabled() else "local"


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> Any:
    """Return an async DB connection usable as ``async with connect() as db``.

    Turso (remote libSQL) when configured, else local aiosqlite — see the
    module docstring for the shared API contract.
    """
    if turso_enabled():
        url, token = _turso_env()
        return TursoConnection(url, token)
    _ensure_dir()
    return aiosqlite.connect(DB_PATH)


# ---------------------------------------------------------------------------
# libSQL value codec
# ---------------------------------------------------------------------------
# libSQL's HTTP protocol tags every value with its SQLite storage class.
# Integers travel as *strings* so a full i64 survives JSON's float mantissa;
# blobs as base64.  We mirror that on the way in and undo it on the way out.


def _encode_arg(v: Any) -> dict[str, Any]:
    if v is None:
        return {"type": "null"}
    # bool is an int subclass — check it first so True/False don't become 1/0
    # silently under a different branch order.
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray)):
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode("ascii")}
    return {"type": "text", "value": str(v)}


def _decode_value(cell: dict[str, Any]) -> Any:
    t = cell.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(cell["value"])
    if t == "float":
        return float(cell["value"])
    if t == "blob":
        return base64.b64decode(cell.get("base64", ""))
    return cell.get("value")  # text


class Row(Mapping):
    """Mimics ``aiosqlite.Row``: addressable by column index *and* name, and
    convertible via ``dict(row)`` (Mapping supplies that from keys + items)."""

    __slots__ = ("_cols", "_vals", "_map")

    def __init__(self, cols: list[str], vals: list[Any]) -> None:
        self._cols = cols
        self._vals = vals
        self._map = dict(zip(cols, vals))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._vals[key]
        return self._map[key]

    def keys(self) -> list[str]:  # type: ignore[override]
        return self._cols

    def __iter__(self) -> Iterator[str]:
        return iter(self._cols)

    def __len__(self) -> int:
        return len(self._cols)


class TursoCursor:
    """Holds a statement's fully-materialized result set.  libSQL returns every
    row in one HTTP response, so ``fetchone``/``fetchall`` just walk the buffer
    (kept async to match the aiosqlite call sites)."""

    def __init__(self, rows: list[Row], lastrowid: int | None, rowcount: int = -1) -> None:
        self._rows = rows
        self.lastrowid = lastrowid
        # DELETE/UPDATE call sites read .rowcount to distinguish "affected N"
        # from "affected nothing" (→ 404).  libSQL reports affected_row_count;
        # -1 is the DB-API "unknown" sentinel aiosqlite uses for SELECTs.
        self.rowcount = rowcount
        self._i = 0

    async def fetchone(self) -> Row | None:
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    async def fetchall(self) -> list[Row]:
        rest = self._rows[self._i :]
        self._i = len(self._rows)
        return rest


ClientFactory = Callable[[], httpx.AsyncClient]


class TursoConnection:
    """Thin async wrapper over libSQL's HTTP ``/v2/pipeline`` endpoint.

    Each ``execute`` is one stateless pipeline request (``[execute, close]``),
    which libSQL runs in autocommit.  That's sufficient here because no save
    flow performs multiple *interdependent* writes — save/rollback each do
    reads then a single INSERT, delete is one DELETE — so per-statement
    autocommit is indistinguishable from aiosqlite's connection-scoped
    transaction for this app.  ``commit()`` is therefore a no-op.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        # Turso hands out libsql:// (and occasionally ws[s]://) URLs; the HTTP
        # pipeline lives on the https:// origin.
        http_url = url
        for prefix, repl in (
            ("libsql://", "https://"),
            ("wss://", "https://"),
            ("ws://", "http://"),
        ):
            if http_url.startswith(prefix):
                http_url = repl + http_url[len(prefix) :]
                break
        self._url = http_url.rstrip("/") + "/v2/pipeline"
        self._token = token
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=30.0))
        self._client: httpx.AsyncClient | None = None
        # Assigned by call sites (`db.row_factory = aiosqlite.Row`); accepted
        # and ignored — Turso rows are always Row instances.
        self.row_factory: Any = None

    async def __aenter__(self) -> TursoConnection:
        self._client = self._client_factory()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def commit(self) -> None:
        # libSQL autocommits each statement; nothing to flush.
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> TursoCursor:
        if self._client is None:
            # Support use outside `async with` (defensive; call sites always
            # use the context manager).
            self._client = self._client_factory()
        args = [_encode_arg(p) for p in (params or ())]
        body = {
            "requests": [
                {"type": "execute", "stmt": {"sql": sql, "args": args}},
                {"type": "close"},
            ]
        }
        resp = await self._client.post(
            self._url,
            json=body,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        resp.raise_for_status()
        payload = resp.json()

        # The server returns one result per request in order; results[0] is our
        # execute.  HTTP 4xx/5xx (auth, server errors) already raised above; a
        # single-statement failure surfaces as a per-result error entry.
        results = payload.get("results") or []
        if not results:
            raise RuntimeError("libSQL returned no results")
        first = results[0]
        if first.get("type") == "error":
            raise RuntimeError(_libsql_err(first.get("error")))

        result = (first.get("response") or {}).get("result") or {}
        cols = [c["name"] for c in result.get("cols", [])]
        rows = [
            Row(cols, [_decode_value(cell) for cell in raw_row])
            for raw_row in result.get("rows", [])
        ]
        last = result.get("last_insert_rowid")
        lastrowid = int(last) if last is not None else None
        affected = result.get("affected_row_count")
        rowcount = int(affected) if affected is not None else -1
        return TursoCursor(rows, lastrowid, rowcount)


def _libsql_err(err: Any) -> str:
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err or "libSQL error")
