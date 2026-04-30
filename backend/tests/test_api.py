"""HTTP-level tests for the FastAPI app via TestClient.

Wraps the app in `with TestClient(...)` so the lifespan handler runs
(loading OHLCV from cache + initializing SQLite). Uses a module-scoped
fixture so we don't pay that cost per test.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_validate_valid_expression(client):
    r = client.post("/api/validate", json={"expression": "rank(delta(close, 5))"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["error"] is None


def test_validate_invalid_expression(client):
    r = client.post("/api/validate", json={"expression": "rank("})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert isinstance(body["error"], str) and len(body["error"]) > 0


def test_operators_returns_at_least_20(client):
    r = client.get("/api/operators")
    assert r.status_code == 200
    body = r.json()
    assert "operators" in body
    assert len(body["operators"]) >= 20
    # Each entry should have a name
    assert all("name" in op for op in body["operators"])


def test_universe_has_tickers_and_sectors(client):
    r = client.get("/api/universe")
    assert r.status_code == 200
    body = r.json()
    assert "tickers" in body
    assert "sectors" in body
    assert len(body["tickers"]) > 0
    assert len(body["sectors"]) > 0
    # Sectors map covers every ticker
    assert all(t in body["sectors"] for t in body["tickers"])
