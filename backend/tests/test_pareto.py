"""Pareto frontier across saved alphas — synthetic dominance fixtures."""

from __future__ import annotations

import pytest
from analytics.pareto import compute_pareto, pareto_frontier_only


def _a(id_: int, sharpe: float, turnover: float, name: str | None = None) -> dict:
    """Minimal saved-alpha shape."""
    return {
        "id": id_,
        "name": name or f"alpha_{id_}",
        "sharpe": sharpe,
        "turnover": turnover,
    }


# ---------- Dominance basics ----------


def test_single_alpha_is_pareto():
    out = compute_pareto([_a(1, 1.0, 100.0)])
    assert len(out) == 1
    assert out[0]["is_pareto"] is True
    assert out[0]["dominated_by"] == []


def test_clear_dominator():
    """A=(2.0 sharpe, $1M turn) dominates B=(1.0, $2M) and C=(0.5, $3M)."""
    out = compute_pareto([_a(1, 2.0, 1_000_000), _a(2, 1.0, 2_000_000), _a(3, 0.5, 3_000_000)])
    by_id = {a["id"]: a for a in out}
    assert by_id[1]["is_pareto"] is True
    assert by_id[2]["is_pareto"] is False
    assert by_id[3]["is_pareto"] is False
    # Both 2 and 3 are dominated by 1
    assert 1 in by_id[2]["dominated_by"]
    assert 1 in by_id[3]["dominated_by"]


def test_clean_tradeoff_curve_all_pareto():
    """Three alphas, each best on a different point — no dominance."""
    out = compute_pareto(
        [
            _a(1, 0.5, 500_000),  # cheapest, lowest sharpe
            _a(2, 1.0, 2_000_000),  # middle
            _a(3, 2.0, 8_000_000),  # priciest, highest sharpe
        ]
    )
    for a in out:
        assert a["is_pareto"] is True, f"{a['id']} unexpectedly dominated"


def test_negative_sharpe_always_dominated():
    """A loss-maker is never on the frontier, regardless of turnover."""
    out = compute_pareto([_a(1, 1.0, 1_000_000), _a(2, -0.5, 100)])  # cheap but negative
    by_id = {a["id"]: a for a in out}
    assert by_id[1]["is_pareto"] is True
    assert by_id[2]["is_pareto"] is False


def test_negative_sharpe_low_turnover_still_disqualified():
    """Even the cheapest alpha can't be Pareto if it loses money."""
    out = compute_pareto([_a(1, -0.1, 1)])  # only alpha, but negative
    assert out[0]["is_pareto"] is False


# ---------- Edge cases ----------


def test_empty_list():
    assert compute_pareto([]) == []


def test_missing_sharpe_or_turnover_disqualified():
    """Missing metrics → can't place on the plane → not on frontier."""
    out = compute_pareto(
        [
            {"id": 1, "sharpe": None, "turnover": 100},
            {"id": 2, "sharpe": 1.0, "turnover": None},
            {"id": 3, "sharpe": 1.0, "turnover": 100},
        ]
    )
    by_id = {a["id"]: a for a in out}
    assert by_id[1]["is_pareto"] is False
    assert by_id[2]["is_pareto"] is False
    assert by_id[3]["is_pareto"] is True  # only valid one


def test_nan_sharpe_disqualified():
    """NaN coerces to None → same disqualification as missing."""
    out = compute_pareto([{"id": 1, "sharpe": float("nan"), "turnover": 1.0}])
    assert out[0]["is_pareto"] is False


def test_input_order_preserved():
    """Output order must match input order — frontend expects deterministic indexing."""
    input_alphas = [_a(99, 1.0, 100), _a(7, 0.5, 50), _a(42, 2.0, 200)]
    out = compute_pareto(input_alphas)
    assert [a["id"] for a in out] == [99, 7, 42]


def test_extra_keys_preserved():
    """Pass-through of caller's auxiliary fields (name, expression, etc.)."""
    out = compute_pareto(
        [{"id": 1, "name": "myalpha", "sharpe": 1.0, "turnover": 100, "fitness": 0.3}]
    )
    assert out[0]["name"] == "myalpha"
    assert out[0]["fitness"] == 0.3


def test_tied_on_both_axes_both_pareto():
    """Two alphas with identical (sharpe, turnover) are both on the frontier
    — neither *strictly* dominates the other."""
    out = compute_pareto([_a(1, 1.0, 100), _a(2, 1.0, 100)])
    assert out[0]["is_pareto"] is True
    assert out[1]["is_pareto"] is True


def test_tied_sharpe_better_turnover_dominates():
    """Same sharpe, lower turnover → strictly better → dominates."""
    out = compute_pareto([_a(1, 1.0, 100), _a(2, 1.0, 200)])
    by_id = {a["id"]: a for a in out}
    assert by_id[1]["is_pareto"] is True  # lower turnover wins
    assert by_id[2]["is_pareto"] is False
    assert 1 in by_id[2]["dominated_by"]


# ---------- Convenience wrapper ----------


def test_frontier_only_returns_subset_sorted():
    out = pareto_frontier_only(
        [_a(1, 0.5, 500_000), _a(2, 2.0, 8_000_000), _a(3, 1.0, 2_000_000), _a(4, -0.5, 100)]
    )
    # 1, 2, 3 are all Pareto; 4 disqualified.  Sorted by turnover ascending.
    assert [a["id"] for a in out] == [1, 3, 2]


def test_frontier_only_empty_when_all_dominated():
    """When all alphas are negative-Sharpe → empty frontier."""
    out = pareto_frontier_only([_a(1, -0.5, 100), _a(2, -1.0, 50)])
    assert out == []
