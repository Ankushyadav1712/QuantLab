"""Universe registry, GICS map, and the new neutralization modes."""

import numpy as np
import pandas as pd
import pytest

from data.universes import (
    GICS_LEVELS,
    all_tickers,
    available_neutralizations,
    default_universe_id,
    get_universe,
    gics_for,
    list_universes,
)
from engine.backtester import Backtester, SimulationConfig


# ---------- Registry ----------


def test_list_universes_includes_default():
    items = list_universes()
    assert any(u["is_default"] for u in items)
    # Default id resolution must agree with the flag
    default_id = default_universe_id()
    matched = [u for u in items if u["id"] == default_id]
    assert matched and matched[0]["is_default"]


def test_get_universe_known_id_returns_tickers_and_gics():
    u = get_universe("sp100_50")
    assert len(u["tickers"]) == 50
    # Every ticker in a built-in universe must have a GICS row (else
    # neutralization quietly buckets it as Unknown — bug for built-ins)
    for t in u["tickers"]:
        row = u["gics"][t]
        for level in GICS_LEVELS:
            assert row.get(level), f"{t} missing {level}"


def test_get_universe_unknown_id_raises():
    with pytest.raises(KeyError, match="Unknown universe"):
        get_universe("not-a-real-universe")


def test_all_tickers_is_union_of_universes():
    union = all_tickers()
    for u in list_universes():
        sub = set(get_universe(u["id"])["tickers"])
        assert sub.issubset(union)


# ---------- gics_for / available_neutralizations ----------


def test_gics_for_returns_none_for_unknown_tickers():
    rows = gics_for(["AAPL", "XYZ_NOT_REAL"])
    assert rows["AAPL"]["sector"] == "Information Technology"
    assert rows["XYZ_NOT_REAL"]["sector"] is None
    for level in GICS_LEVELS:
        assert rows["XYZ_NOT_REAL"][level] is None


def test_available_neutralizations_full_universe_has_all_levels():
    u = get_universe("sp100_50")
    modes = available_neutralizations(u["gics"])
    assert "none" in modes and "market" in modes
    # 50 megacaps span every GICS level — all 4 group modes should be available
    for level in GICS_LEVELS:
        assert level in modes


def test_available_neutralizations_strips_levels_with_one_group():
    # Force a one-sector universe → only sector with 1 group → not usable
    fake_gics = {
        "AAPL": {"sector": "Tech", "industry_group": "X", "industry": "Y", "sub_industry": "Z"},
        "MSFT": {"sector": "Tech", "industry_group": "X", "industry": "Y", "sub_industry": "Z"},
    }
    modes = available_neutralizations(fake_gics)
    assert modes == ["none", "market"]


def test_available_neutralizations_unknown_only_supports_none_and_market():
    fake_gics = {"AAA": {level: None for level in GICS_LEVELS}}
    assert available_neutralizations(fake_gics) == ["none", "market"]


# ---------- Backtester GICS-mode neutralization ----------


@pytest.fixture
def four_sector_data():
    """4 tickers across 2 sectors × 2 industries — enough to test any GICS mode."""
    dates = pd.date_range("2020-01-02", periods=30, freq="B")
    tickers = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (30, 4)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return {"close": closes, "returns": closes.pct_change()}


@pytest.fixture
def four_sector_gics():
    return {
        "A": {"sector": "Tech", "industry_group": "Software", "industry": "App SW", "sub_industry": "App SW"},
        "B": {"sector": "Tech", "industry_group": "Software", "industry": "App SW", "sub_industry": "App SW"},
        "C": {"sector": "Energy", "industry_group": "Oil", "industry": "E&P", "sub_industry": "E&P"},
        "D": {"sector": "Energy", "industry_group": "Oil", "industry": "E&P", "sub_industry": "E&P"},
    }


@pytest.mark.parametrize("mode", ["sector", "industry_group", "industry", "sub_industry"])
def test_gics_neutralization_demeans_per_group(mode, four_sector_data, four_sector_gics):
    """For a constant per-group alpha, GICS-level neutralization must zero
    every weight (within-group demean of a constant is 0)."""
    closes = four_sector_data["close"]
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(four_sector_data, gics_map=four_sector_gics)
    cfg = SimulationConfig(
        universe=list(closes.columns),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        neutralization=mode,
        run_oos=False,
    )
    result, _ = bt.run(alpha, cfg)
    assert (result.weights.abs().to_numpy() < 1e-12).all()


def test_unknown_neutralization_mode_raises(four_sector_data, four_sector_gics):
    bt = Backtester(four_sector_data, gics_map=four_sector_gics)
    cfg = SimulationConfig(
        universe=list(four_sector_data["close"].columns),
        start_date="2020-01-02",
        end_date="2020-02-12",
        neutralization="bogus_mode",  # type: ignore[arg-type]
        run_oos=False,
    )
    rng = np.random.default_rng(1)
    alpha = pd.DataFrame(
        rng.standard_normal(four_sector_data["close"].shape),
        index=four_sector_data["close"].index,
        columns=four_sector_data["close"].columns,
    )
    with pytest.raises(ValueError, match="Unknown neutralization mode"):
        bt.run(alpha, cfg)


def test_gics_mode_buckets_unknowns_as_unknown(four_sector_data):
    """Tickers absent from gics_map must not be silently dropped — they go
    into an 'Unknown' bucket and demean among themselves."""
    closes = four_sector_data["close"]
    # Only A and B have GICS; C and D should bucket together as 'Unknown'
    partial_gics = {
        "A": {"sector": "Tech", "industry_group": "X", "industry": "Y", "sub_industry": "Z"},
        "B": {"sector": "Tech", "industry_group": "X", "industry": "Y", "sub_industry": "Z"},
    }
    bt = Backtester(four_sector_data, gics_map=partial_gics)
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)
    cfg = SimulationConfig(
        universe=list(closes.columns),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        neutralization="industry",
        run_oos=False,
    )
    result, _ = bt.run(alpha, cfg)
    # Constant-alpha-within-group → all weights zero
    assert (result.weights.abs().to_numpy() < 1e-12).all()
