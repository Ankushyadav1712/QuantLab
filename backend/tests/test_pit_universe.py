"""Point-in-time S&P 100 membership gating."""

import numpy as np
import pandas as pd
import pytest
from data.sp100_history import (
    SP100_INCLUSION_DATES,
    build_membership_mask,
    is_member_on,
    membership_summary,
)
from engine.backtester import Backtester, SimulationConfig


def test_is_member_on_default_true_for_unknown_ticker():
    # Tickers absent from the curated dict are assumed always-in
    assert is_member_on("AAPL", pd.Timestamp("2010-01-01"))
    assert is_member_on("AAPL", pd.Timestamp("2024-12-31"))


def test_is_member_on_respects_known_inclusion_dates():
    # TSLA is the canonical example in the curated history
    assert "TSLA" in SP100_INCLUSION_DATES
    join = pd.to_datetime(SP100_INCLUSION_DATES["TSLA"])
    assert not is_member_on("TSLA", join - pd.Timedelta(days=1))
    assert is_member_on("TSLA", join)
    assert is_member_on("TSLA", join + pd.Timedelta(days=365))


def test_build_membership_mask_shape_and_values():
    dates = pd.date_range("2019-01-02", "2022-01-31", freq="B")
    tickers = ["AAPL", "TSLA", "MSFT"]
    mask = build_membership_mask(dates, tickers)
    assert mask.shape == (len(dates), 3)
    # AAPL/MSFT are always-in
    assert mask["AAPL"].all() and mask["MSFT"].all()
    # TSLA is False before the join date and True after
    join = pd.to_datetime(SP100_INCLUSION_DATES["TSLA"])
    assert (~mask.loc[mask.index < join, "TSLA"]).all()
    assert mask.loc[mask.index >= join, "TSLA"].all()


def test_membership_summary_lists_affected_tickers():
    # Window that straddles TSLA's inclusion date
    dates = pd.date_range("2019-01-02", "2022-01-31", freq="B")
    summary = membership_summary(dates, ["AAPL", "TSLA"])
    assert summary["total_known_changes"] == len(SP100_INCLUSION_DATES)
    affected = {a["ticker"]: a for a in summary["tickers_affected"]}
    assert "TSLA" in affected
    assert affected["TSLA"]["days_masked"] > 0


def test_membership_summary_skips_unaffected_window():
    # Window starts after TSLA's inclusion date — nothing to mask
    dates = pd.date_range("2022-01-03", "2024-12-31", freq="B")
    summary = membership_summary(dates, ["AAPL", "TSLA"])
    assert summary["tickers_affected"] == []


# ---------- Backtester integration ----------


@pytest.fixture
def pit_data():
    """Synthetic data fixture with a TSLA column so PIT gating bites."""
    dates = pd.date_range("2019-01-02", periods=750, freq="B")  # ~3 years
    tickers = ["AAPL", "TSLA", "MSFT"]
    rng = np.random.default_rng(42)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (750, 3)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return {"close": closes, "returns": closes.pct_change()}


def test_pit_gating_zeros_out_tsla_pre_inclusion(pit_data):
    """With PIT on, TSLA must contribute zero PnL before its inclusion date."""
    closes = pit_data["close"]
    sector_map = {"AAPL": "Tech", "TSLA": "Auto", "MSFT": "Tech"}
    # All-ones alpha — without gating, TSLA would carry equal weight throughout.
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(pit_data, sector_map)
    cfg = SimulationConfig(
        universe=list(closes.columns),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        neutralization="none",
        run_oos=False,
        point_in_time_universe=True,
    )
    result, _ = bt.run(alpha, cfg)

    # TSLA's weight series should be zero for every date before the join
    join = pd.to_datetime(SP100_INCLUSION_DATES["TSLA"])
    pre = result.weights.index < join
    assert (result.weights.loc[pre, "TSLA"].abs() < 1e-12).all()
    # And non-zero once it's in
    post = result.weights.index >= join
    assert (result.weights.loc[post, "TSLA"].abs() > 0).any()


def test_pit_off_keeps_tsla_pre_inclusion_active(pit_data):
    """Without PIT, the legacy behavior — TSLA trades from day 1 — must hold."""
    closes = pit_data["close"]
    sector_map = {"AAPL": "Tech", "TSLA": "Auto", "MSFT": "Tech"}
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)

    bt = Backtester(pit_data, sector_map)
    cfg = SimulationConfig(
        universe=list(closes.columns),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        neutralization="none",
        run_oos=False,
        point_in_time_universe=False,
    )
    result, _ = bt.run(alpha, cfg)

    # TSLA carries weight on day 1 (legacy / "current snapshot" behavior)
    assert abs(result.weights.iloc[0]["TSLA"]) > 0
