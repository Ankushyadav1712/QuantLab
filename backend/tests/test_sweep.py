"""Parameter sweep — tokenizer + cartesian expansion + endpoint."""

import pytest
from engine.sweep import combo_for_index, expand_sweeps, has_sweep_syntax
from fastapi.testclient import TestClient
from main import app

# ---------- Detection ----------


def test_has_sweep_syntax_true_for_simple_range():
    assert has_sweep_syntax("rank(momentum_{5..30})")


def test_has_sweep_syntax_true_for_step_range():
    assert has_sweep_syntax("decay_linear(rank(close), {10..40:5})")


def test_has_sweep_syntax_false_for_normal_expression():
    assert not has_sweep_syntax("rank(delta(close, 5))")


def test_has_sweep_syntax_false_for_braces_without_range():
    assert not has_sweep_syntax("foo{bar}baz")
    assert not has_sweep_syntax("{}")


# ---------- Single-dim expansion ----------


def test_expand_sweeps_simple_integer_range():
    out = expand_sweeps("rank(momentum_{5..15:5})")
    assert out["total"] == 3
    assert out["expressions"] == [
        "rank(momentum_5)",
        "rank(momentum_10)",
        "rank(momentum_15)",
    ]
    assert out["dimensions"][0]["values"] == [5, 10, 15]


def test_expand_sweeps_default_step_one():
    out = expand_sweeps("ts_mean(close, {1..3})")
    assert out["total"] == 3
    assert out["expressions"] == [
        "ts_mean(close, 1)",
        "ts_mean(close, 2)",
        "ts_mean(close, 3)",
    ]


def test_expand_sweeps_includes_endpoint():
    out = expand_sweeps("ts_mean(close, {10..40:10})")
    assert out["dimensions"][0]["values"] == [10, 20, 30, 40]


def test_expand_sweeps_float_range_keeps_floats():
    out = expand_sweeps("scale(close * {1.0..2.0:0.5})")
    assert out["dimensions"][0]["values"] == [1.0, 1.5, 2.0]


# ---------- Multi-dim cartesian product ----------


def test_expand_sweeps_two_dims_cartesian():
    out = expand_sweeps("decay_linear(rank(momentum_{5..10:5}), {10..30:10})")
    assert out["total"] == 6
    expr_set = set(out["expressions"])
    for inner in (5, 10):
        for outer in (10, 20, 30):
            assert f"decay_linear(rank(momentum_{inner}), {outer})" in expr_set


def test_combo_for_index_round_trip():
    """Every cell index must map to a unique (token → value) dict that matches
    the substituted expression."""
    out = expand_sweeps("op({1..3}, {10..20:5})")
    seen = set()
    for i, _expr in enumerate(out["expressions"]):
        params = combo_for_index(i, out)
        key = tuple(sorted(params.items()))
        assert key not in seen
        seen.add(key)
    assert len(seen) == out["total"]


# ---------- Errors ----------


def test_expand_sweeps_raises_when_no_sweep_tokens():
    with pytest.raises(ValueError, match="no sweep tokens"):
        expand_sweeps("rank(close)")


def test_expand_sweeps_caps_at_max_combinations():
    with pytest.raises(ValueError, match="exceeds max_combinations"):
        expand_sweeps("op({1..100}, {1..100})", max_combinations=50)


def test_expand_sweeps_rejects_descending_range():
    with pytest.raises(ValueError, match="end must be >= start"):
        expand_sweeps("op(close, {30..5:5})")


def test_expand_sweeps_rejects_zero_step():
    with pytest.raises(ValueError, match="step must be positive"):
        expand_sweeps("op(close, {5..30:0})")


# ---------- Endpoint ----------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_sweep_endpoint_runs_each_combination(client):
    """A 2-dim sweep returns one cell per cartesian combination, each
    carrying IS metrics or an error."""
    r = client.post(
        "/api/sweep",
        json={
            "expression": "decay_linear(rank(momentum_{5..10:5}), {10..30:10})",
            "max_combinations": 50,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_combinations"] == 6
    assert len(body["cells"]) == 6
    assert len(body["dimensions"]) == 2
    for cell in body["cells"]:
        assert "expression" in cell
        assert "params" in cell
        assert set(cell["params"].keys()) == {"{5..10:5}", "{10..30:10}"}
        assert "sharpe" in cell or "error" in cell


def test_sweep_endpoint_rejects_no_sweep_tokens(client):
    r = client.post("/api/sweep", json={"expression": "rank(close)"})
    assert r.status_code == 400
    assert "no sweep tokens" in r.json()["detail"].lower()


def test_sweep_endpoint_rejects_too_many_combinations(client):
    r = client.post(
        "/api/sweep",
        json={
            "expression": "op({1..20}, {1..20})",
            "max_combinations": 50,
        },
    )
    assert r.status_code == 400
    assert "exceeds max_combinations" in r.json()["detail"].lower()
