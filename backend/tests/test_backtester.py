import numpy as np
import pandas as pd
import pytest

from engine.backtester import Backtester, BacktestResult, SimulationConfig


@pytest.fixture
def simple_data():
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    tickers = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (20, 4)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return {"close": closes, "returns": closes.pct_change()}


@pytest.fixture
def sector_map():
    return {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}


def _config(universe, dates, **overrides):
    cfg = dict(
        universe=universe,
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
    )
    cfg.update(overrides)
    return SimulationConfig(**cfg)


def test_constant_alpha_zero_pnl_after_market_neutralization(simple_data, sector_map):
    closes = simple_data["close"]
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(simple_data, sector_map)
    cfg = _config(list(closes.columns), closes.index, neutralization="market")
    result = bt.run(alpha, cfg)

    assert isinstance(result, BacktestResult)
    assert all(abs(v) < 1e-6 for v in result.daily_pnl)
    assert all(abs(v) < 1e-6 for v in result.cumulative_pnl)
    # weights should also be all zero
    assert (result.weights.abs().to_numpy() < 1e-12).all()


def test_constant_alpha_zero_pnl_after_sector_neutralization(simple_data, sector_map):
    closes = simple_data["close"]
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(simple_data, sector_map)
    cfg = _config(list(closes.columns), closes.index, neutralization="sector")
    result = bt.run(alpha, cfg)

    assert all(abs(v) < 1e-6 for v in result.daily_pnl)


def test_turnover_non_negative(simple_data, sector_map):
    closes = simple_data["close"]
    rng = np.random.default_rng(0)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(simple_data, sector_map)
    cfg = _config(list(closes.columns), closes.index)
    result = bt.run(alpha, cfg)

    assert all(t >= 0 for t in result.turnover)
    assert len(result.turnover) == len(result.dates)


def test_cumulative_pnl_is_cumsum_of_daily(simple_data, sector_map):
    closes = simple_data["close"]
    rng = np.random.default_rng(123)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(simple_data, sector_map)
    cfg = _config(list(closes.columns), closes.index)
    result = bt.run(alpha, cfg)

    expected = np.cumsum(result.daily_pnl)
    np.testing.assert_allclose(result.cumulative_pnl, expected, rtol=0, atol=1e-9)


def test_truncation_caps_fractional_weights(simple_data, sector_map):
    closes = simple_data["close"]
    # alpha that would put all weight into one stock
    alpha = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    alpha["A"] = 1.0  # only A has signal

    bt = Backtester(simple_data, sector_map)
    cfg = _config(
        list(closes.columns),
        closes.index,
        neutralization="none",
        truncation=0.1,
    )
    result = bt.run(alpha, cfg)

    # |w_i| should never exceed 0.1
    assert (result.weights.abs().to_numpy() <= 0.1 + 1e-12).all()
    # corresponding dollar positions never exceed truncation * booksize
    assert (
        result.positions.abs().to_numpy() <= cfg.truncation * cfg.booksize + 1e-6
    ).all()


def test_universe_and_date_filter(simple_data, sector_map):
    closes = simple_data["close"]
    alpha = pd.DataFrame(
        np.ones(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(simple_data, sector_map)
    cfg = SimulationConfig(
        universe=["A", "B"],
        start_date=str(closes.index[5].date()),
        end_date=str(closes.index[14].date()),
        neutralization="none",
    )
    result = bt.run(alpha, cfg)

    assert list(result.weights.columns) == ["A", "B"]
    assert len(result.dates) == 10  # rows 5..14 inclusive
