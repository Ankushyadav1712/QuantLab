"""Alpha combiner tests: engine/combine.py units + the ic_weighted /
orthogonalize paths on POST /api/alphas/multi-blend (and equal-path regression)."""

import pytest
from engine.combine import greedy_orthogonal_select, ic_weights

# ---------- unit: ic_weights ----------


def test_ic_weights_proportional():
    assert list(ic_weights([0.02, 0.04])) == pytest.approx([1 / 3, 2 / 3])


def test_ic_weights_clamps_negative_to_zero():
    assert list(ic_weights([0.03, -0.01])) == pytest.approx([1.0, 0.0])


def test_ic_weights_all_nonpositive_falls_back_to_equal():
    assert list(ic_weights([-0.01, -0.02, 0.0])) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_ic_weights_handles_none_and_nan():
    assert list(ic_weights([0.02, None, float("nan")])) == pytest.approx([1.0, 0.0, 0.0])


def test_ic_weights_empty():
    assert list(ic_weights([])) == []


# ---------- unit: greedy_orthogonal_select ----------


def test_orthogonal_keeps_all_when_uncorrelated():
    corr = [[1.0, 0.1], [0.1, 1.0]]
    assert greedy_orthogonal_select(corr, [1.0, 2.0], max_rho=0.7) == [0, 1]


def test_orthogonal_drops_redundant_keeps_higher_score():
    corr = [[1.0, 0.9], [0.9, 1.0]]
    # idx1 has the higher score → kept; idx0 is 0.9-correlated to it → dropped.
    assert greedy_orthogonal_select(corr, [1.0, 2.0], max_rho=0.7) == [1]


def test_orthogonal_nan_treated_as_not_redundant():
    corr = [[1.0, float("nan")], [float("nan"), 1.0]]
    assert greedy_orthogonal_select(corr, [1.0, 2.0]) == [0, 1]


def test_orthogonal_three_way_drops_only_the_duplicate():
    corr = [
        [1.0, 0.1, 0.95],
        [0.1, 1.0, 0.1],
        [0.95, 0.1, 1.0],
    ]
    # scores → order [2, 1, 0]: keep 2, keep 1 (0.1 to 2), drop 0 (0.95 to 2).
    assert greedy_orthogonal_select(corr, [0.5, 1.0, 2.0], max_rho=0.7) == [1, 2]


# ---------- HTTP: multi-blend new paths ----------


def test_multi_blend_ic_weighted(client):
    r = client.post(
        "/api/alphas/multi-blend",
        json={
            "alphas": [{"expression": "rank(close)"}, {"expression": "rank(volume)"}],
            "weight_method": "ic_weighted",
        },
    )
    assert r.status_code == 200, r.text
    s = r.json()["settings"]
    assert s["weight_method"] == "ic_weighted"
    assert len(s["alphas"]) == 2
    assert all("ic" in a and "weight" in a for a in s["alphas"])
    assert sum(a["weight"] for a in s["alphas"]) == pytest.approx(1.0, abs=1e-6)


def test_multi_blend_orthogonalize_drops_duplicate(client):
    # rank(close) and rank(ts_mean(close,10)) are ~0.99 return-correlated → one dropped.
    r = client.post(
        "/api/alphas/multi-blend",
        json={
            "alphas": [
                {"expression": "rank(close)"},
                {"expression": "rank(ts_mean(close, 10))"},
            ],
            "weight_method": "equal",
            "orthogonalize": True,
        },
    )
    assert r.status_code == 200, r.text
    s = r.json()["settings"]
    assert s["orthogonalize"] is True
    assert s["effective_n"] == 1
    assert len(s["dropped_alphas"]) == 1


def test_multi_blend_orthogonalize_keeps_diverse(client):
    # rank(close) vs rank(volume) are weakly correlated → both kept.
    r = client.post(
        "/api/alphas/multi-blend",
        json={
            "alphas": [{"expression": "rank(close)"}, {"expression": "rank(volume)"}],
            "weight_method": "equal",
            "orthogonalize": True,
        },
    )
    assert r.status_code == 200, r.text
    s = r.json()["settings"]
    assert s["effective_n"] == 2
    assert s["dropped_alphas"] == []


def test_multi_blend_equal_backwards_compatible(client):
    # The existing equal path is unchanged: user weights 1:3 → 0.25 / 0.75.
    r = client.post(
        "/api/alphas/multi-blend",
        json={
            "alphas": [
                {"expression": "rank(close)", "weight": 1.0},
                {"expression": "rank(volume)", "weight": 3.0},
            ],
            "weight_method": "equal",
        },
    )
    assert r.status_code == 200, r.text
    s = r.json()["settings"]
    assert s["weight_method"] == "equal_user_supplied"
    assert [a["weight"] for a in s["alphas"]] == pytest.approx([0.25, 0.75])
