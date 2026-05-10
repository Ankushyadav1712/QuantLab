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


# ---------- /api/simulate response shape ----------


def test_simulate_returns_full_is_oos_shape(client):
    """Lock the /api/simulate response shape so the next IS/OOS / factor /
    data_quality refactor can't silently drop a field that the frontend or a
    consumer relies on."""
    r = client.post(
        "/api/simulate",
        json={
            "expression": "rank(close)",
            "settings": {"start_date": "2020-01-01", "end_date": "2024-12-31"},
            "run_oos": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Top-level keys
    expected = {
        "is_metrics", "oos_metrics", "is_timeseries", "oos_timeseries",
        "overfitting_analysis", "factor_decomposition", "monthly_returns",
        "expression", "settings", "data_quality", "diagnostics",
    }
    missing = expected - set(body.keys())
    assert not missing, f"missing top-level keys: {missing}"

    # IS/OOS metric blocks both populated when run_oos is true
    for k in ("sharpe", "annual_return", "max_drawdown", "fitness", "win_rate"):
        assert k in body["is_metrics"], f"is_metrics missing {k}"
        assert k in body["oos_metrics"], f"oos_metrics missing {k}"

    # Time-series have parallel structure
    for ts_key in ("is_timeseries", "oos_timeseries"):
        ts = body[ts_key]
        assert ts is not None
        for sub in ("dates", "cumulative_pnl", "daily_returns",
                    "drawdown", "rolling_sharpe", "turnover"):
            assert sub in ts, f"{ts_key} missing {sub}"

    # IS/OOS dates partition the window — no overlap, IS ends before OOS starts
    is_dates = body["is_timeseries"]["dates"]
    oos_dates = body["oos_timeseries"]["dates"]
    assert is_dates and oos_dates
    assert is_dates[-1] < oos_dates[0]


def test_simulate_with_run_oos_false(client):
    """run_oos=false → oos fields are null, is_* fields cover the full window."""
    r = client.post(
        "/api/simulate",
        json={"expression": "rank(close)", "run_oos": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["oos_metrics"] is None
    assert body["oos_timeseries"] is None
    assert body["overfitting_analysis"] is None
    assert body["is_metrics"]["sharpe"] is not None


def test_simulate_rejects_lookahead_expression(client):
    """The linter must block negative shifts at the simulate boundary."""
    r = client.post(
        "/api/simulate",
        json={"expression": "rank(delay(close, -1))"},
    )
    assert r.status_code == 400
    assert "look-ahead" in r.json()["detail"] or "future" in r.json()["detail"]


# ---------- /api/alphas full save → load → correlate → delete cycle ----------
#
# Both bugs that bit us today (KeyError on `metrics`, empty correlation matrix)
# would have been caught by these tests.  Anything that breaks the
# save → read pipeline now fails CI before it lands.


def _save_alpha(client, name: str, expression: str) -> int:
    r = client.post(
        "/api/alphas",
        json={"name": name, "expression": expression, "notes": "ci"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == name
    assert body["expression"] == expression
    assert isinstance(body["sharpe"], (float, int)) or body["sharpe"] is None
    assert isinstance(body["id"], int)
    return body["id"]


def test_alpha_full_lifecycle(client):
    """Save 2 alphas, list them, fetch each by id, correlate, delete."""
    id1 = _save_alpha(client, "_ci_alpha_a", "rank(close)")
    id2 = _save_alpha(client, "_ci_alpha_b", "rank(volume)")

    try:
        # LIST — must include both
        r = client.get("/api/alphas")
        assert r.status_code == 200
        listed = {row["id"] for row in r.json()}
        assert id1 in listed and id2 in listed

        # GET by id — full record incl. parsed result_json
        r = client.get(f"/api/alphas/{id1}")
        assert r.status_code == 200
        record = r.json()
        assert record["id"] == id1
        assert record["expression"] == "rank(close)"
        # `result` should be the parsed simulate response with the full IS shape
        assert record.get("result") is not None
        assert "is_metrics" in record["result"]
        assert "is_timeseries" in record["result"]

        # CORRELATIONS — the bug we fixed today: empty matrix even with valid ids
        r = client.post(
            "/api/alphas/correlations",
            json={"alpha_ids": [id1, id2]},
        )
        assert r.status_code == 200
        body = r.json()
        # Must produce a real 2×2 matrix, not the silently-empty `{tickers:[],matrix:[]}`
        assert len(body["tickers"]) == 2
        assert len(body["matrix"]) == 2
        assert len(body["matrix"][0]) == 2
        # Diagonal == 1.0
        assert body["matrix"][0][0] == pytest.approx(1.0)
        assert body["matrix"][1][1] == pytest.approx(1.0)
        # Off-diagonal symmetry
        assert body["matrix"][0][1] == pytest.approx(body["matrix"][1][0])

        # NOT FOUND — both endpoints
        assert client.get("/api/alphas/99999999").status_code == 404
        assert client.delete("/api/alphas/99999999").status_code == 404

    finally:
        # Always clean up both rows so re-runs aren't polluted
        for aid in (id1, id2):
            client.delete(f"/api/alphas/{aid}")


def test_universes_endpoint_lists_presets(client):
    r = client.get("/api/universes")
    assert r.status_code == 200
    body = r.json()
    assert "universes" in body and "default" in body
    ids = [u["id"] for u in body["universes"]]
    assert "sp100_50" in ids
    # Each preset must surface its available neutralizations + ticker count
    for u in body["universes"]:
        assert isinstance(u["ticker_count"], int) and u["ticker_count"] > 0
        assert "available_neutralizations" in u
        assert "none" in u["available_neutralizations"]
        assert "market" in u["available_neutralizations"]
    assert body["default"] in ids


def test_simulate_accepts_universe_id_and_echoes_it(client):
    r = client.post(
        "/api/simulate",
        json={
            "expression": "rank(close)",
            "settings": {"universe_id": "sp100_50", "run_oos": False},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings"].get("universe_id") == "sp100_50"
    assert body["data_quality"]["universe"]["id"] == "sp100_50"


def test_simulate_rejects_unknown_universe_id(client):
    r = client.post(
        "/api/simulate",
        json={
            "expression": "rank(close)",
            "settings": {"universe_id": "totally-fake-universe", "run_oos": False},
        },
    )
    assert r.status_code == 400
    assert "unknown universe" in r.json()["detail"].lower()


def test_simulate_with_industry_neutralization(client):
    """End-to-end: a built-in universe + the new 'industry' neutralization
    mode runs without error and the response echoes the chosen mode."""
    r = client.post(
        "/api/simulate",
        json={
            "expression": "rank(close)",
            "settings": {
                "universe_id": "sp100_50",
                "neutralization": "industry",
                "run_oos": False,
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings"]["neutralization"] == "industry"


def test_compare_returns_per_alpha_overlay(client):
    """Two valid alphas should come back with per-alpha labels A/B and the
    full IS metrics + timeseries each."""
    r = client.post(
        "/api/compare",
        json={
            "expressions": ["rank(close)", "rank(volume)"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["alphas"]) == 2
    labels = [a["label"] for a in body["alphas"]]
    assert labels == ["A", "B"]
    for a in body["alphas"]:
        assert "metrics" in a, f"Compare entry has no metrics: {a}"
        assert "timeseries" in a
        assert "sharpe" in a["metrics"]
        # Compare is IS-only, so no overfitting block at the per-alpha level
        assert "oos_metrics" not in a


def test_compare_per_alpha_lint_failure_does_not_kill_others(client):
    """A look-ahead expression should produce an error for that cell only —
    the valid expressions must still come back with metrics."""
    r = client.post(
        "/api/compare",
        json={
            "expressions": ["delay(close, -5)", "rank(close)"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["alphas"]) == 2
    # First alpha (look-ahead) → error; second → metrics
    assert "error" in body["alphas"][0]
    assert "metrics" in body["alphas"][1]


def test_compare_rejects_under_two_or_over_four(client):
    too_few = client.post("/api/compare", json={"expressions": ["rank(close)"]})
    assert too_few.status_code == 422  # pydantic validation error
    too_many = client.post(
        "/api/compare",
        json={"expressions": ["rank(close)"] * 5},
    )
    assert too_many.status_code == 422


def test_multi_blend_returns_simulate_shape(client):
    """Multi-blend must produce a full IS/OOS response, same shape as /simulate."""
    r = client.post(
        "/api/alphas/multi-blend",
        json={
            "alphas": [
                {"expression": "rank(close)", "weight": 0.6},
                {"expression": "rank(volume)", "weight": 0.4},
            ],
            "settings": {"run_oos": True},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "is_metrics" in body
    assert "is_timeseries" in body
    assert body["expression"] == "multi-blend"
    # Settings echo includes the per-alpha breakdown
    assert "alphas" in body["settings"]
    assert len(body["settings"]["alphas"]) == 2
