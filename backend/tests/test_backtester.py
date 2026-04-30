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
    # Existing tests assume a single full-window backtest.  Default run_oos=False
    # so they keep that semantics; IS/OOS-specific tests opt in below.
    cfg = dict(
        universe=universe,
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        run_oos=False,
    )
    cfg.update(overrides)
    return SimulationConfig(**cfg)


def test_constant_alpha_zero_pnl_after_market_neutralization(simple_data, sector_map):
    closes = simple_data["close"]
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(simple_data, sector_map)
    cfg = _config(list(closes.columns), closes.index, neutralization="market")
    result, _ = bt.run(alpha, cfg)

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
    result, _ = bt.run(alpha, cfg)

    assert all(abs(v) < 1e-6 for v in result.daily_pnl)


def test_turnover_non_negative(simple_data, sector_map):
    closes = simple_data["close"]
    rng = np.random.default_rng(0)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(simple_data, sector_map)
    cfg = _config(list(closes.columns), closes.index)
    result, _ = bt.run(alpha, cfg)

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
    result, _ = bt.run(alpha, cfg)

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
    result, _ = bt.run(alpha, cfg)

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
        run_oos=False,
    )
    result, _ = bt.run(alpha, cfg)

    assert list(result.weights.columns) == ["A", "B"]
    assert len(result.dates) == 10  # rows 5..14 inclusive


# ---------- Spec-required additions on a 100×10 synthetic fixture ----------


@pytest.fixture
def synth_data():
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    tickers = [f"T{i}" for i in range(10)]
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (100, 10)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return {"close": closes, "returns": closes.pct_change()}


@pytest.fixture
def synth_sector_map():
    return {f"T{i}": ("A" if i < 5 else "B") for i in range(10)}


def _full_universe_config(data, **overrides):
    closes = data["close"]
    cfg = dict(
        universe=list(closes.columns),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        run_oos=False,
    )
    cfg.update(overrides)
    return SimulationConfig(**cfg)


def test_constant_alpha_zero_weights_after_market_neutralization(
    synth_data, synth_sector_map
):
    closes = synth_data["close"]
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(synth_data, synth_sector_map)
    cfg = _full_universe_config(synth_data, neutralization="market")
    result, _ = bt.run(alpha, cfg)

    # Spec: weights are all-zero (after neutralization the alpha is identically 0,
    # then the divide-by-zero guard turns weights into zeros).
    assert (result.weights.abs().to_numpy() < 1e-12).all()
    assert (result.positions.abs().to_numpy() < 1e-9).all()


def test_first_turnover_is_zero(synth_data, synth_sector_map):
    """No prior position on day 0 → turnover is 0 (NaN difference is skipped)."""
    closes = synth_data["close"]
    rng = np.random.default_rng(11)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(synth_data, synth_sector_map)
    result, _ = bt.run(alpha, _full_universe_config(synth_data))

    assert result.turnover[0] == 0.0
    # Subsequent turnovers should be strictly positive (random alpha → real trades)
    assert any(t > 0 for t in result.turnover[1:])


def test_final_cum_pnl_is_sum_of_daily(synth_data, synth_sector_map):
    closes = synth_data["close"]
    rng = np.random.default_rng(99)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(synth_data, synth_sector_map)
    result, _ = bt.run(alpha, _full_universe_config(synth_data))

    assert result.cumulative_pnl[-1] == pytest.approx(sum(result.daily_pnl), abs=1e-9)


def test_costs_reduce_pnl_vs_zero_cost(synth_data, synth_sector_map):
    closes = synth_data["close"]
    rng = np.random.default_rng(2026)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(synth_data, synth_sector_map)
    free, _ = bt.run(alpha, _full_universe_config(synth_data, transaction_cost_bps=0.0))
    paid, _ = bt.run(alpha, _full_universe_config(synth_data, transaction_cost_bps=10.0))

    # Same gross PnL, but the paid run trades into costs every day → strictly lower
    # final cumulative PnL.  And the cost magnitude must equal sum(turnover) * bps.
    assert paid.cumulative_pnl[-1] < free.cumulative_pnl[-1]

    expected_cost = sum(paid.turnover) * 10.0 / 10_000.0
    actual_cost = free.cumulative_pnl[-1] - paid.cumulative_pnl[-1]
    assert actual_cost == pytest.approx(expected_cost, rel=1e-9, abs=1e-6)


# ---------- IS/OOS split ----------


def test_run_oos_false_returns_none_in_oos_slot(synth_data, synth_sector_map):
    closes = synth_data["close"]
    rng = np.random.default_rng(1)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(synth_data, synth_sector_map)
    is_result, oos_result = bt.run(alpha, _full_universe_config(synth_data))

    assert isinstance(is_result, BacktestResult)
    assert oos_result is None
    # IS covers the full window when run_oos=False
    assert len(is_result.dates) == len(closes)


def test_is_oos_split_partitions_dates_70_30(synth_data, synth_sector_map):
    closes = synth_data["close"]
    rng = np.random.default_rng(2)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(synth_data, synth_sector_map)
    cfg = _full_universe_config(synth_data, run_oos=True, oos_split=0.3)
    is_result, oos_result = bt.run(alpha, cfg)

    assert oos_result is not None
    # 100 rows × 0.7 = 70 IS, 30 OOS — IS+OOS partitions the universe
    assert len(is_result.dates) == 70
    assert len(oos_result.dates) == 30
    # No date overlap between halves
    assert set(is_result.dates).isdisjoint(set(oos_result.dates))
    # IS ends strictly before OOS begins
    assert is_result.dates[-1] < oos_result.dates[0]


def test_is_oos_obeys_custom_split_fraction(synth_data, synth_sector_map):
    closes = synth_data["close"]
    rng = np.random.default_rng(3)
    alpha = pd.DataFrame(
        rng.standard_normal(closes.shape), index=closes.index, columns=closes.columns
    )

    bt = Backtester(synth_data, synth_sector_map)
    cfg = _full_universe_config(synth_data, run_oos=True, oos_split=0.5)
    is_result, oos_result = bt.run(alpha, cfg)

    assert len(is_result.dates) == 50
    assert len(oos_result.dates) == 50


def test_compare_is_oos_robust_label():
    from analytics.performance import PerformanceAnalytics

    perf = PerformanceAnalytics()
    is_m = {"sharpe": 1.0, "annual_return": 0.10, "start_date": "2020-01-02", "end_date": "2022-12-31"}
    oos_m = {"sharpe": 0.95, "annual_return": 0.09, "start_date": "2023-01-01", "end_date": "2024-12-31"}
    out = perf.compare_is_oos(is_m, oos_m)
    assert out["sharpe_decay"] == pytest.approx(0.05, rel=1e-9)
    assert out["overfitting_flag"] is False
    assert out["severity"] == "robust"
    assert out["is_period"] == {"start": "2020-01-02", "end": "2022-12-31"}
    assert out["oos_period"] == {"start": "2023-01-01", "end": "2024-12-31"}


def test_compare_is_oos_severe_overfit():
    from analytics.performance import PerformanceAnalytics

    perf = PerformanceAnalytics()
    is_m = {"sharpe": 2.0, "annual_return": 0.20}
    oos_m = {"sharpe": -0.1, "annual_return": -0.02}
    out = perf.compare_is_oos(is_m, oos_m)
    # Sharpe dropped from 2.0 to -0.1 → decay = 1.05
    assert out["sharpe_decay"] > 0.6
    assert out["overfitting_flag"] is True
    assert out["severity"] == "severe"
