"""Tests for the ADV liquidity filter (PDF Section 9.2).

The filter NaN-s out alpha cells where 20-day average dollar volume is
below a threshold, ensuring the backtester can't claim PnL from stocks
it couldn't actually have traded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from engine.backtester import Backtester, SimulationConfig


def _build_data(n_days: int = 30) -> dict[str, pd.DataFrame]:
    """Two tickers: LIQ (huge volume, always tradeable) and ILLIQ (tiny
    volume, below any reasonable ADV threshold)."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = pd.DataFrame({"LIQ": [100.0] * n_days, "ILLIQ": [100.0] * n_days}, index=dates)
    # Volume: LIQ = 1M shares @ $100 = $100M/day; ILLIQ = 100 shares @ $100 = $10k/day
    volume = pd.DataFrame({"LIQ": [1_000_000] * n_days, "ILLIQ": [100] * n_days}, index=dates)
    dollar_volume = close * volume
    # Returns: ILLIQ returns 1% every day (would be very profitable if tradeable)
    returns = pd.DataFrame({"LIQ": [0.0] * n_days, "ILLIQ": [0.01] * n_days}, index=dates)
    return {
        "close": close,
        "volume": volume,
        "dollar_volume": dollar_volume,
        "returns": returns,
    }


def _config(min_adv_dollars: float = 0.0) -> SimulationConfig:
    return SimulationConfig(
        universe=["LIQ", "ILLIQ"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        neutralization="none",
        run_oos=False,
        min_adv_dollars=min_adv_dollars,
    )


def test_adv_filter_default_is_off():
    """Default `min_adv_dollars=0.0` must mean the filter is a no-op —
    backwards compatible with every existing saved alpha."""
    data = _build_data()
    bt = Backtester(data)
    # Long ILLIQ → it should produce PnL because the filter is off
    alpha = pd.DataFrame({"LIQ": [0.0] * 30, "ILLIQ": [1.0] * 30}, index=data["close"].index)
    is_result, _ = bt.run(alpha, _config())
    # ILLIQ returns 1%/day → should have non-zero PnL
    assert sum(is_result.daily_pnl) > 0


def test_adv_filter_masks_illiquid_ticker():
    """With min_adv_dollars=$1M, ILLIQ (only $10k ADV) should be NaN-ed
    out → 100% of book lands in LIQ (which has 0% return → zero PnL)."""
    data = _build_data()
    bt = Backtester(data)
    alpha = pd.DataFrame({"LIQ": [0.0] * 30, "ILLIQ": [1.0] * 30}, index=data["close"].index)
    is_result, _ = bt.run(alpha, _config(min_adv_dollars=1_000_000.0))
    # ILLIQ filtered → no positions to hold → zero PnL (LIQ alpha is 0)
    assert sum(is_result.daily_pnl) == pytest.approx(0.0, abs=1e-6)


def test_adv_filter_keeps_liquid_ticker():
    """LIQ ($100M ADV) should sail through a $1M threshold and trade
    normally."""
    data = _build_data()
    # LIQ has 0% return in _build_data — flip it to 1% so we can check PnL
    data["returns"] = pd.DataFrame(
        {"LIQ": [0.01] * 30, "ILLIQ": [0.0] * 30}, index=data["close"].index
    )
    bt = Backtester(data)
    alpha = pd.DataFrame({"LIQ": [1.0] * 30, "ILLIQ": [0.0] * 30}, index=data["close"].index)
    is_result, _ = bt.run(alpha, _config(min_adv_dollars=1_000_000.0))
    # LIQ passes the filter → should accumulate the 1%/day PnL
    assert sum(is_result.daily_pnl) > 0


def test_adv_filter_uses_existing_adv20_field_when_present():
    """If adv20 is in the data dict (the API loads it from fetcher), the
    filter should use it directly instead of recomputing."""
    data = _build_data()
    # Pre-load adv20 = dollar_volume rolling mean (matches what the fetcher does)
    data["adv20"] = data["dollar_volume"].rolling(20, min_periods=1).mean()
    bt = Backtester(data)
    alpha = pd.DataFrame({"LIQ": [0.0] * 30, "ILLIQ": [1.0] * 30}, index=data["close"].index)
    is_result, _ = bt.run(alpha, _config(min_adv_dollars=1_000_000.0))
    # Same result as the recompute path
    assert sum(is_result.daily_pnl) == pytest.approx(0.0, abs=1e-6)


def test_adv_filter_silently_no_ops_without_dollar_volume():
    """If neither adv20 nor dollar_volume is loaded, the filter should be
    a graceful no-op (rather than crashing) — important for test contexts
    and custom data feeds."""
    data = _build_data()
    data.pop("dollar_volume", None)
    bt = Backtester(data)
    alpha = pd.DataFrame({"LIQ": [0.0] * 30, "ILLIQ": [1.0] * 30}, index=data["close"].index)
    # Should not raise; behaves as if filter were off
    is_result, _ = bt.run(alpha, _config(min_adv_dollars=1_000_000.0))
    assert sum(is_result.daily_pnl) > 0  # ILLIQ still trades


def test_adv_filter_high_threshold_silently_zeros_pnl():
    """Threshold above every ticker's ADV → every cell NaN-ed → normalization
    produces all-zero weights → zero PnL.  We prefer this silent degradation
    over a crash so a researcher tuning the threshold doesn't have to wrap
    each run in a try/except."""
    data = _build_data()
    bt = Backtester(data)
    alpha = pd.DataFrame({"LIQ": [1.0] * 30, "ILLIQ": [1.0] * 30}, index=data["close"].index)
    # $10B threshold — way above LIQ's $100M ADV
    is_result, _ = bt.run(alpha, _config(min_adv_dollars=10_000_000_000.0))
    assert sum(is_result.daily_pnl) == pytest.approx(0.0, abs=1e-9)
    # Weights should be all-zero too — a researcher debugging would see this
    assert float(is_result.weights.abs().sum().sum()) == 0.0


def test_adv_config_threading_via_api_make_config():
    """Make sure the API's _make_config plumbs min_adv_dollars through
    from the settings dict to the SimulationConfig."""
    from main import _make_config

    cfg = _make_config({"min_adv_dollars": 5_000_000.0})
    assert cfg.min_adv_dollars == 5_000_000.0

    cfg_default = _make_config({})
    assert cfg_default.min_adv_dollars == 0.0


# Touch numpy so the import isn't flagged as unused — used implicitly via pandas
_ = np
