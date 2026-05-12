"""Fundamentals coverage gate — /api/operators filtering + /api/validate lint.

When yfinance fundamentals coverage is too low (default threshold 20% non-NaN
on the `revenue` canary), the platform hides fundamentals fields from the
catalog so the autocomplete doesn't suggest broken fields, and surfaces a
clear validate-time error if users reference them anyway.

These tests directly manipulate ``_state["fundamentals_available"]`` so we
can exercise both branches (available + unavailable) without needing to
mock the yfinance fetch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from main import _FUNDAMENTALS_CATEGORIES, _state, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _set_fundamentals(available: bool, coverage_pct: float = 0.0):
    """Patch the gate state for the duration of one test."""
    _state["fundamentals_available"] = available
    _state["fundamentals_coverage_pct"] = coverage_pct


# ---------- /api/operators filtering ----------


def test_operators_hides_fundamentals_when_unavailable(client):
    _set_fundamentals(False, 5.0)
    r = client.get("/api/operators")
    assert r.status_code == 200
    body = r.json()
    assert body["fundamentals_available"] is False
    cats = {f.get("category") for f in body["fields"]}
    assert _FUNDAMENTALS_CATEGORIES.isdisjoint(cats)
    # The metadata still tells callers how many fields were hidden
    assert body["fundamentals_field_count"] > 0


def test_operators_shows_fundamentals_when_available(client):
    _set_fundamentals(True, 50.0)
    r = client.get("/api/operators")
    body = r.json()
    assert body["fundamentals_available"] is True
    cats = {f.get("category") for f in body["fields"]}
    # At least one fundamentals category should be present
    assert _FUNDAMENTALS_CATEGORIES & cats


def test_operators_metadata_reports_coverage_pct(client):
    _set_fundamentals(False, 12.5)
    r = client.get("/api/operators")
    body = r.json()
    assert body["fundamentals_coverage_pct"] == pytest.approx(12.5)


# ---------- /api/validate gate ----------


def test_validate_rejects_fundamentals_when_unavailable(client):
    _set_fundamentals(False, 8.0)
    r = client.post("/api/validate", json={"expression": "zscore(roe) + zscore(momentum_60)"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["error"] is not None
    assert "roe" in body["error"]
    rules = [d.get("rule") for d in body["diagnostics"]]
    assert "fundamentals-unavailable" in rules


def test_validate_rejects_ratio_field_too(client):
    """`pe_ratio` is in the `fundamentals_ratio` category; should also gate."""
    _set_fundamentals(False, 0.0)
    r = client.post("/api/validate", json={"expression": "rank(pe_ratio)"})
    body = r.json()
    assert body["valid"] is False
    assert "pe_ratio" in body["error"]


def test_validate_allows_fundamentals_when_available(client):
    _set_fundamentals(True, 95.0)
    r = client.post("/api/validate", json={"expression": "rank(roe)"})
    body = r.json()
    assert body["valid"] is True


def test_validate_allows_non_fundamentals_when_unavailable(client):
    """Non-fundamentals expressions should pass through unchanged."""
    _set_fundamentals(False, 0.0)
    r = client.post("/api/validate", json={"expression": "rank(close) + rank(momentum_60)"})
    body = r.json()
    assert body["valid"] is True
    rules = [d.get("rule") for d in body["diagnostics"]]
    assert "fundamentals-unavailable" not in rules


def test_validate_finds_nested_fundamentals_reference(client):
    """Fundamentals inside nested function calls should still trigger the gate."""
    _set_fundamentals(False, 0.0)
    r = client.post(
        "/api/validate",
        json={"expression": "rank(ts_mean(zscore(net_income), 20))"},
    )
    body = r.json()
    assert body["valid"] is False
    assert "net_income" in body["error"]


def test_validate_lists_multiple_fundamentals_refs(client):
    """When several fundamentals fields are used, the error names them."""
    _set_fundamentals(False, 0.0)
    r = client.post(
        "/api/validate",
        json={"expression": "rank(roe) + rank(pe_ratio) + rank(net_income)"},
    )
    body = r.json()
    assert body["valid"] is False
    # Sorted preview in the message — first three fields shown
    msg = body["error"]
    for name in ("net_income", "pe_ratio", "roe"):
        assert name in msg


def test_validate_existing_lint_rules_still_fire(client):
    """Existing look-ahead lint rules shouldn't be displaced by the new gate."""
    _set_fundamentals(False, 0.0)
    # Negative shift is the canonical look-ahead bug; lint should still error
    r = client.post("/api/validate", json={"expression": "delay(close, -1)"})
    body = r.json()
    assert body["valid"] is False
    # Existing lint uses `severity` + `message` (no `rule` key); just confirm
    # the look-ahead message is present alongside any fundamentals gate.
    errors = [d for d in body["diagnostics"] if d["severity"] == "error"]
    assert any("look-ahead" in d["message"].lower() for d in errors)
