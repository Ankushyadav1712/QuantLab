"""Turso (hosted libSQL) persistence adapter.

The adapter speaks libSQL's HTTP ``/v2/pipeline`` protocol and mirrors the
slice of the aiosqlite API the app relies on.  Tests drive it against an
``httpx.MockTransport`` so no real Turso instance is needed — they assert the
request serialization (typed args, Bearer auth, URL shape) and the response
decoding (typed values, rows, lastrowid, errors) are correct.
"""

import asyncio
import base64
import json

import httpx
import pytest
from db import database
from db.database import Row, TursoConnection, active_backend, connect, turso_enabled


def run(coro):
    return asyncio.run(coro)


def _pipeline_response(*, cols=None, rows=None, last_insert_rowid=None, affected=0, error=None):
    """Build a libSQL pipeline HTTP body: one execute result + one close."""
    if error is not None:
        first = {"type": "error", "error": {"message": error}}
    else:
        first = {
            "type": "ok",
            "response": {
                "type": "execute",
                "result": {
                    "cols": [{"name": c} for c in (cols or [])],
                    "rows": rows or [],
                    "affected_row_count": affected,
                    "last_insert_rowid": last_insert_rowid,
                },
            },
        }
    return {
        "baton": None,
        "base_url": None,
        "results": [first, {"type": "ok", "response": {"type": "close"}}],
    }


def _conn(handler, url="libsql://db-org.turso.io", token="tok"):
    """A TursoConnection whose HTTP client is backed by a mock transport."""
    captured: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(wrapped))  # noqa: E731
    conn = TursoConnection(url, token, client_factory=factory)
    return conn, captured


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_connect_uses_local_when_no_env(monkeypatch):
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    assert turso_enabled() is False
    assert active_backend() == "local"
    c = connect()
    assert not isinstance(c, TursoConnection)  # aiosqlite context manager


def test_connect_uses_turso_when_env_set(monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://x-y.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "secret")
    assert turso_enabled() is True
    assert active_backend() == "turso"
    assert isinstance(connect(), TursoConnection)


def test_partial_env_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://x-y.turso.io")
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    assert turso_enabled() is False


# ---------------------------------------------------------------------------
# URL + request shape
# ---------------------------------------------------------------------------


def test_libsql_url_becomes_https_pipeline():
    conn, cap = _conn(lambda r: httpx.Response(200, json=_pipeline_response()))

    async def go():
        async with conn as db:
            await db.execute("SELECT 1")

    run(go())
    assert str(cap[0].url) == "https://db-org.turso.io/v2/pipeline"
    assert cap[0].headers["Authorization"] == "Bearer tok"


def test_typed_arg_encoding():
    conn, cap = _conn(lambda r: httpx.Response(200, json=_pipeline_response()))

    async def go():
        async with conn as db:
            await db.execute(
                "INSERT INTO t VALUES (?,?,?,?,?,?)",
                (5, 1.5, None, "hi", True, b"\x00\x01"),
            )

    run(go())
    args = json.loads(cap[0].content)["requests"][0]["stmt"]["args"]
    assert args[0] == {"type": "integer", "value": "5"}  # i64 as string
    assert args[1] == {"type": "float", "value": 1.5}
    assert args[2] == {"type": "null"}
    assert args[3] == {"type": "text", "value": "hi"}
    # bool is checked before int → stays integer 1, not mis-typed
    assert args[4] == {"type": "integer", "value": "1"}
    assert args[5] == {"type": "blob", "base64": base64.b64encode(b"\x00\x01").decode()}


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------


def test_row_value_decoding_all_types():
    resp = _pipeline_response(
        cols=["i", "f", "t", "n", "b"],
        rows=[
            [
                {"type": "integer", "value": "9007199254740993"},  # > 2^53, must stay exact
                {"type": "float", "value": 2.25},
                {"type": "text", "value": "x"},
                {"type": "null"},
                {"type": "blob", "base64": base64.b64encode(b"ab").decode()},
            ]
        ],
    )
    conn, _ = _conn(lambda r: httpx.Response(200, json=resp))

    async def go():
        async with conn as db:
            cur = await db.execute("SELECT *")
            return await cur.fetchone()

    row = run(go())
    assert row["i"] == 9007199254740993
    assert row["f"] == 2.25
    assert row["t"] == "x"
    assert row["n"] is None
    assert row["b"] == b"ab"


def test_lastrowid_parsed_as_int():
    conn, _ = _conn(lambda r: httpx.Response(200, json=_pipeline_response(last_insert_rowid="42")))

    async def go():
        async with conn as db:
            cur = await db.execute("INSERT INTO t DEFAULT VALUES")
            return cur.lastrowid

    assert run(go()) == 42


def test_lastrowid_none_on_select():
    conn, _ = _conn(lambda r: httpx.Response(200, json=_pipeline_response(cols=["x"], rows=[])))

    async def go():
        async with conn as db:
            cur = await db.execute("SELECT x FROM t")
            return cur.lastrowid

    assert run(go()) is None


def test_fetchone_then_exhausts():
    resp = _pipeline_response(cols=["x"], rows=[[{"type": "integer", "value": "1"}]])
    conn, _ = _conn(lambda r: httpx.Response(200, json=resp))

    async def go():
        async with conn as db:
            cur = await db.execute("SELECT x")
            first = await cur.fetchone()
            second = await cur.fetchone()
            return first, second

    first, second = run(go())
    assert first["x"] == 1
    assert second is None


def test_fetchall_returns_remaining():
    resp = _pipeline_response(
        cols=["x"],
        rows=[[{"type": "integer", "value": str(i)}] for i in range(3)],
    )
    conn, _ = _conn(lambda r: httpx.Response(200, json=resp))

    async def go():
        async with conn as db:
            cur = await db.execute("SELECT x")
            await cur.fetchone()  # consume the first
            return await cur.fetchall()

    rest = run(go())
    assert [r["x"] for r in rest] == [1, 2]


# ---------------------------------------------------------------------------
# Row addressing + errors + multi-statement flow
# ---------------------------------------------------------------------------


def test_row_index_name_and_dict():
    r = Row(["id", "name"], [7, "alpha"])
    assert r[0] == 7
    assert r["name"] == "alpha"
    assert dict(r) == {"id": 7, "name": "alpha"}
    assert list(r.keys()) == ["id", "name"]
    assert len(r) == 2


def test_error_result_raises():
    conn, _ = _conn(
        lambda r: httpx.Response(200, json=_pipeline_response(error="no such table: t"))
    )

    async def go():
        async with conn as db:
            await db.execute("SELECT * FROM t")

    with pytest.raises(RuntimeError, match="no such table"):
        run(go())


def test_read_then_write_flow_shares_connection():
    # Mirrors save_alpha: SELECT head version, then INSERT, then commit().
    def handler(request):
        sql = json.loads(request.content)["requests"][0]["stmt"]["sql"]
        if sql.strip().upper().startswith("SELECT"):
            return httpx.Response(
                200,
                json=_pipeline_response(
                    cols=["id", "version"],
                    rows=[[{"type": "integer", "value": "3"}, {"type": "integer", "value": "2"}]],
                ),
            )
        return httpx.Response(200, json=_pipeline_response(last_insert_rowid="4"))

    conn, cap = _conn(handler)

    async def go():
        async with conn as db:
            db.row_factory = object()  # accepted + ignored
            head = await (await db.execute("SELECT id, version FROM alphas")).fetchone()
            version = int(head["version"]) + 1
            cur = await db.execute("INSERT INTO alphas (version) VALUES (?)", (version,))
            await db.commit()  # no-op, must not raise
            return version, cur.lastrowid

    version, new_id = run(go())
    assert version == 3
    assert new_id == 4
    assert len(cap) == 2  # two statements, one shared connection


def test_rowcount_from_affected_row_count():
    # delete_alpha reads cursor.rowcount to choose 200-vs-404 — it MUST exist
    # and reflect affected_row_count (this path 500'd before the fix).
    conn, _ = _conn(lambda r: httpx.Response(200, json=_pipeline_response(affected=1)))

    async def go():
        async with conn as db:
            cur = await db.execute("DELETE FROM alphas WHERE id = ?", (5,))
            return cur.rowcount

    assert run(go()) == 1


def test_rowcount_zero_when_nothing_deleted():
    conn, _ = _conn(lambda r: httpx.Response(200, json=_pipeline_response(affected=0)))

    async def go():
        async with conn as db:
            cur = await db.execute("DELETE FROM alphas WHERE id = ?", (999,))
            return cur.rowcount

    assert run(go()) == 0


def test_empty_results_raises_clean_error():
    conn, _ = _conn(lambda r: httpx.Response(200, json={"results": []}))

    async def go():
        async with conn as db:
            await db.execute("SELECT 1")

    with pytest.raises(RuntimeError, match="no results"):
        run(go())


def test_raises_on_http_error_status():
    conn, _ = _conn(lambda r: httpx.Response(401, json={"error": "unauthorized"}))

    async def go():
        async with conn as db:
            await db.execute("SELECT 1")

    with pytest.raises(httpx.HTTPStatusError):
        run(go())


def test_module_exposes_backend_helpers():
    # Guards the import surface main.py depends on.
    assert callable(database.connect)
    assert callable(database.active_backend)
    assert callable(database.turso_enabled)
