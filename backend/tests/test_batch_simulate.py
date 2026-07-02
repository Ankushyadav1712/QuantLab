"""HTTP tests for POST /api/batch_simulate — the multi-alpha ranked batch.

Uses the shared session-scoped ``client`` fixture (conftest) so the app's
lifespan (data load + SQLite init) runs once. Expressions are kept simple and
settings omitted so the default universe + cached date range are used.
"""

import pytest

_METRIC_KEYS = ("sharpe", "ic", "ic_tstat", "annual_return", "max_drawdown", "avg_turnover")


def test_batch_returns_ranked_results_and_correlation(client):
    alphas = [
        {"id": "px", "expression": "rank(close)"},
        {"id": "vol", "expression": "rank(volume)"},
        {"id": "mom", "expression": "rank(ts_mean(close, 10))"},
    ]
    r = client.post("/api/batch_simulate", json={"alphas": alphas})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["n_alphas"] == 3
    assert body["n_ok"] == 3
    assert len(body["results"]) == 3
    # Results come back in input order so the frontend can rank client-side.
    assert [x["id"] for x in body["results"]] == ["px", "vol", "mom"]
    for row in body["results"]:
        assert "metrics" in row, row
        for k in _METRIC_KEYS:
            assert k in row["metrics"], f"metrics missing {k}"

    corr = body["correlation_matrix"]
    assert corr["labels"] == ["px", "vol", "mom"]
    matrix = corr["matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    # Unit diagonal + symmetric.
    for i in range(3):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-6)
        for j in range(3):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-6)

    assert isinstance(body["elapsed_sec"], (int, float))


def test_batch_auto_assigns_ids_when_missing(client):
    r = client.post(
        "/api/batch_simulate",
        json={"alphas": [{"expression": "rank(close)"}, {"expression": "rank(volume)"}]},
    )
    assert r.status_code == 200, r.text
    assert [x["id"] for x in r.json()["results"]] == ["alpha_1", "alpha_2"]


def test_batch_partial_failure_isolates_bad_alpha(client):
    """One bad expression yields an error row; the rest still succeed and the
    bad one is excluded from the correlation matrix."""
    alphas = [
        {"id": "good", "expression": "rank(close)"},
        {"id": "bad", "expression": "rank("},  # parse error
    ]
    r = client.post("/api/batch_simulate", json={"alphas": alphas})
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {x["id"]: x for x in body["results"]}
    assert "metrics" in rows["good"]
    assert "error" in rows["bad"]
    assert body["n_ok"] == 1
    assert "bad" not in body["correlation_matrix"]["labels"]


def test_batch_rejects_empty(client):
    r = client.post("/api/batch_simulate", json={"alphas": []})
    assert r.status_code == 400


def test_batch_rejects_oversized(client):
    alphas = [{"expression": "rank(close)"} for _ in range(51)]
    r = client.post("/api/batch_simulate", json={"alphas": alphas})
    assert r.status_code == 422


def test_batch_rejects_missing_expression(client):
    r = client.post("/api/batch_simulate", json={"alphas": [{"id": "x"}]})
    assert r.status_code == 400


def test_batch_rejects_duplicate_ids(client):
    alphas = [
        {"id": "dup", "expression": "rank(close)"},
        {"id": "dup", "expression": "rank(volume)"},
    ]
    r = client.post("/api/batch_simulate", json={"alphas": alphas})
    assert r.status_code == 400
