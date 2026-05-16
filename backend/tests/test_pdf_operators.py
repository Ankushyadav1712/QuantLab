"""Tests for the WorldQuant-Brain-spec operators added in the PDF audit:

- ``neutralize(x, g)``  — alias for group_neutralize
- ``correlation(x, y)`` — per-day cross-sectional Pearson
- ``cap_weight(x)``     — market-cap-weighted variant (data-aware op)
- ``adv(d)``            — rolling mean of dollar_volume (data-aware op)

Pure unit tests on the operators + evaluator-level tests proving the
data-aware dispatch wires up correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from engine.evaluator import AlphaEvaluator
from engine.operators import (
    cap_weight,
    correlation,
    group_neutralize,
    neutralize,
)

# ---------- neutralize alias ----------


def test_neutralize_is_group_neutralize():
    """Just an alias — confirming they're the *same* function so behaviour
    is identical, not just equivalent."""
    assert neutralize is group_neutralize


def test_neutralize_alias_subtracts_group_mean():
    # Two groups of two stocks; the per-group mean should net to zero.
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    x = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [3.0, 4.0, 5.0],
            "C": [10.0, 20.0, 30.0],
            "D": [12.0, 22.0, 32.0],
        },
        index=idx,
    )
    groups = pd.DataFrame(
        {"A": ["g1"] * 3, "B": ["g1"] * 3, "C": ["g2"] * 3, "D": ["g2"] * 3},
        index=idx,
    )
    out = neutralize(x, groups)
    # Within group g1: mean of (1,3)=2 → A becomes -1, B becomes +1
    assert out.iloc[0]["A"] == pytest.approx(-1.0)
    assert out.iloc[0]["B"] == pytest.approx(1.0)
    assert out.iloc[0]["C"] == pytest.approx(-1.0)
    assert out.iloc[0]["D"] == pytest.approx(1.0)


# ---------- correlation(x, y) cross-sectional ----------


def test_correlation_perfect_positive():
    """When y = 2x + 3, per-row correlation is exactly +1."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = list("ABCDE")
    x = pd.DataFrame(np.arange(20).reshape(4, 5).astype(float), index=idx, columns=cols)
    y = 2.0 * x + 3.0
    c = correlation(x, y)
    assert c.shape == x.shape
    # All cells in each row carry the same scalar (broadcast); each row's
    # correlation must be +1 to within floating tolerance.
    for row in c.itertuples(index=False):
        assert all(v == pytest.approx(1.0) for v in row)


def test_correlation_perfect_negative():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = list("ABCD")
    x = pd.DataFrame(np.arange(12).reshape(3, 4).astype(float), index=idx, columns=cols)
    y = -x
    c = correlation(x, y)
    for row in c.itertuples(index=False):
        assert all(v == pytest.approx(-1.0) for v in row)


def test_correlation_broadcasts_one_scalar_per_day():
    """Each row of the output should be constant across columns."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    cols = list("ABCD")
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.standard_normal((5, 4)), index=idx, columns=cols)
    y = pd.DataFrame(rng.standard_normal((5, 4)), index=idx, columns=cols)
    c = correlation(x, y)
    for i in range(len(idx)):
        # All values in row i are equal
        first = c.iloc[i, 0]
        assert (c.iloc[i] == first).all()


def test_correlation_aligns_mismatched_columns():
    """When x and y have different column sets, only the intersection is used —
    no silent broadcast errors."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    x = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}, index=idx)
    y = pd.DataFrame({"B": [1.0, 2.0], "C": [3.0, 4.0]}, index=idx)
    # Should not raise; should compute on the BC intersection
    c = correlation(x, y)
    assert "A" not in c.columns
    assert set(c.columns) == {"B", "C"}


# ---------- cap_weight(x) — pure-function form ----------


def test_cap_weight_returns_bounded_rank():
    """Output is a rank, so values must live in [0, 1]."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = list("ABCD")
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.standard_normal((4, 4)), index=idx, columns=cols)
    mc = pd.DataFrame(rng.uniform(1e9, 1e12, (4, 4)), index=idx, columns=cols)
    out = cap_weight(x, mc)
    assert out.min().min() >= 0.0
    assert out.max().max() <= 1.0


def test_cap_weight_largest_market_cap_dominates():
    """If x is constant (no info) but market_cap varies, the highest-cap
    stock should land at the highest rank.

    rank() uses pandas pct=True, so values land in [1/n, 1], not [0, 1].
    For n=3: ranks become 1/3, 2/3, 1.0.
    """
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    x = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0], "C": [1.0, 1.0]}, index=idx)
    mc = pd.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0], "C": [1.0, 1.0]}, index=idx)
    out = cap_weight(x, mc)
    # Highest market cap (A) → rank 1.0; lowest (C) → rank 1/3
    assert out.iloc[0]["A"] == pytest.approx(1.0)
    assert out.iloc[0]["B"] == pytest.approx(2.0 / 3.0)
    assert out.iloc[0]["C"] == pytest.approx(1.0 / 3.0)
    # Ordering check (the bit researchers actually care about)
    assert out.iloc[0]["A"] > out.iloc[0]["B"] > out.iloc[0]["C"]


# ---------- Evaluator integration: cap_weight + adv as data-aware ops ----------


def _make_data() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(100 + rng.standard_normal((10, 3)).cumsum(axis=0), index=idx, columns=cols)
    volume = pd.DataFrame(rng.uniform(1e6, 1e7, (10, 3)), index=idx, columns=cols)
    dollar_volume = close * volume
    market_cap = pd.DataFrame(rng.uniform(1e9, 1e12, (10, 3)), index=idx, columns=cols)
    return {
        "close": close,
        "volume": volume,
        "dollar_volume": dollar_volume,
        "market_cap": market_cap,
    }


def test_evaluator_adv_dispatches_to_dollar_volume():
    """`adv(5)` in an expression should compute ts_mean(dollar_volume, 5)."""
    data = _make_data()
    ev = AlphaEvaluator(data)
    out = ev.evaluate("adv(5)")
    # Same shape as the underlying data
    assert out.shape == data["close"].shape
    # First 4 rows are NaN (window=5), 5th onward is finite
    assert out.iloc[:4].isna().all().all()
    assert not out.iloc[4:].isna().any().any()


def test_evaluator_adv_rejects_zero_or_negative_window():
    data = _make_data()
    ev = AlphaEvaluator(data)
    with pytest.raises(ValueError, match="positive integer"):
        ev.evaluate("adv(0)")


def test_evaluator_adv_raises_when_dollar_volume_missing():
    data = _make_data()
    data.pop("dollar_volume")
    ev = AlphaEvaluator(data)
    with pytest.raises(ValueError, match="dollar_volume"):
        ev.evaluate("adv(5)")


def test_evaluator_cap_weight_dispatches_with_market_cap():
    """`cap_weight(rank(close))` should ingest the data dict's market_cap."""
    data = _make_data()
    ev = AlphaEvaluator(data)
    out = ev.evaluate("cap_weight(rank(close))")
    assert out.shape == data["close"].shape
    # cap_weight ends with rank() → bounded [0, 1] modulo NaNs
    assert out.min().min() >= 0.0
    assert out.max().max() <= 1.0


def test_evaluator_cap_weight_raises_when_market_cap_missing():
    data = _make_data()
    data.pop("market_cap")
    ev = AlphaEvaluator(data)
    with pytest.raises(ValueError, match="market_cap"):
        ev.evaluate("cap_weight(rank(close))")


def test_evaluator_neutralize_alias_works_in_expression():
    """The `neutralize` name should resolve to group_neutralize through the
    evaluator's normal function dispatch (no special-case needed)."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D"]
    data = {
        "close": pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [3.0, 4.0, 5.0],
                "C": [10.0, 20.0, 30.0],
                "D": [12.0, 22.0, 32.0],
            },
            index=idx,
        ),
        "sector": pd.DataFrame(
            {"A": ["g1"] * 3, "B": ["g1"] * 3, "C": ["g2"] * 3, "D": ["g2"] * 3},
            index=idx,
            columns=cols,
        ),
    }
    ev = AlphaEvaluator(data)
    out = ev.evaluate("neutralize(close, sector)")
    # Same as the group_neutralize test above — within g1, mean is 2, so A=-1
    assert out.iloc[0]["A"] == pytest.approx(-1.0)
    assert out.iloc[0]["C"] == pytest.approx(-1.0)
