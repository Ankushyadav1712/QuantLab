"""Curated example alphas — parse + lint + endpoint shape.

These guard against drift: the catalog uses real operator + field names, so
if someone removes an operator without updating an example, this catches it
at CI time instead of at "user clicks Load Example."
"""

from __future__ import annotations

import pytest
from data.example_alphas import EXAMPLE_ALPHAS, get_example, list_examples
from engine.lint import lint_ast
from engine.parser import Parser
from fastapi.testclient import TestClient
from main import app

# ---------- Catalog shape ----------


def test_examples_have_required_fields():
    required = {
        "id",
        "name",
        "category",
        "expression",
        "description",
        "recommended_settings",
        "teaches",
    }
    for e in EXAMPLE_ALPHAS:
        missing = required - set(e.keys())
        assert not missing, f"Example {e.get('id', '?')} missing fields: {missing}"


def test_example_ids_unique():
    ids = [e["id"] for e in EXAMPLE_ALPHAS]
    assert len(ids) == len(set(ids))


def test_example_settings_use_known_keys():
    """Recommended settings must be keys the simulate endpoint understands —
    typos would silently no-op when applied."""
    known = {
        "neutralization",
        "decay",
        "transaction_cost_bps",
        "booksize",
        "truncation",
        "start_date",
        "end_date",
        "run_oos",
        "cost_model",
        "impact_coefficient",
        "execution_lag_days",
        "point_in_time_universe",
    }
    for e in EXAMPLE_ALPHAS:
        unknown = set(e["recommended_settings"].keys()) - known
        assert not unknown, f"Example {e['id']} has unknown setting keys: {unknown}"


# ---------- Every expression parses cleanly ----------


def test_every_example_parses():
    """If an example references a removed operator or field, this catches it."""
    parser = Parser()
    for e in EXAMPLE_ALPHAS:
        try:
            parser.parse(e["expression"])
        except ValueError as exc:
            pytest.fail(f"Example {e['id']} fails to parse: {exc}\n  Expression: {e['expression']}")


def test_every_example_lints_clean():
    """No look-ahead errors — examples should be exemplars, not warnings."""
    parser = Parser()
    for e in EXAMPLE_ALPHAS:
        ast = parser.parse(e["expression"])
        diagnostics = lint_ast(ast)
        errors = [d for d in diagnostics if d["severity"] == "error"]
        assert not errors, f"Example {e['id']} has lint errors: {errors}"


# ---------- Helpers ----------


def test_get_example_returns_copy():
    """Mutating the returned dict must not affect the source catalog."""
    e = get_example(EXAMPLE_ALPHAS[0]["id"])
    assert e is not None
    e["name"] = "MUTATED"
    fresh = get_example(EXAMPLE_ALPHAS[0]["id"])
    assert fresh["name"] != "MUTATED"


def test_get_example_unknown_returns_none():
    assert get_example("nonexistent_id") is None


def test_list_examples_returns_all():
    assert len(list_examples()) == len(EXAMPLE_ALPHAS)


# ---------- Endpoints ----------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_get_examples_returns_catalog(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    body = r.json()
    assert "examples" in body
    assert len(body["examples"]) == len(EXAMPLE_ALPHAS)
    # Each entry is self-contained
    for e in body["examples"]:
        assert "id" in e
        assert "expression" in e
        assert "recommended_settings" in e


def test_get_example_by_id(client):
    target = EXAMPLE_ALPHAS[0]["id"]
    r = client.get(f"/api/examples/{target}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == target


def test_get_example_404_on_unknown(client):
    r = client.get("/api/examples/this_does_not_exist")
    assert r.status_code == 404
