"""Brain-parity features (WorldQuant Brain convention alignment):

* ``renormalize_truncation`` — Brain redistributes clipped weight so the book
  stays fully invested; a plain clip leaves gross exposure below booksize.
* Brain-convention metrics — ``margin_bps`` (PnL per $ traded),
  ``avg_turnover_frac`` (turnover / booksize), ``annual_return_arith``
  (mean daily x 252), and ``fitness_wq`` with Sharpe carrying the sign.
* ``GET /api/presets/brain`` — the canonical parity settings blob.
"""

import math

import numpy as np
import pandas as pd
import pytest
from analytics.performance import PerformanceAnalytics, _fitness_wq
from engine.backtester import Backtester, BacktestResult, SimulationConfig

# ---------- renormalize_truncation ----------


@pytest.fixture
def wide_data():
    # 30 names so a 0.08 cap can absorb redistribution (30 x 0.08 = 2.4 > 1).
    dates = pd.date_range("2024-01-02", periods=15, freq="B")
    tickers = [f"T{i:02d}" for i in range(30)]
    rng = np.random.default_rng(7)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (15, 30)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return {"close": closes, "returns": closes.pct_change()}


def _dominant_alpha(data):
    # One name dwarfs the rest -> the +/-cap binds hard on it.
    closes = data["close"]
    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)
    alpha.iloc[:, 0] = 100.0
    return alpha


def _cfg(data, **overrides):
    closes = data["close"]
    cfg = dict(
        universe=list(closes.columns),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        neutralization="none",
        run_oos=False,
    )
    cfg.update(overrides)
    return SimulationConfig(**cfg)


def test_plain_clip_leaves_gross_below_one(wide_data):
    # Baseline behavior (renormalize off): the dominant name's weight
    # (100/129 ~ 0.78) is clipped to 0.08 and the clipped mass is simply lost.
    bt = Backtester(wide_data, {})
    result, _ = bt.run(_dominant_alpha(wide_data), _cfg(wide_data, truncation=0.08))
    gross = result.weights.abs().sum(axis=1)
    assert (gross < 0.5).all()


def test_renormalize_restores_full_investment(wide_data):
    bt = Backtester(wide_data, {})
    cfg = _cfg(wide_data, truncation=0.08, renormalize_truncation=True)
    result, _ = bt.run(_dominant_alpha(wide_data), cfg)
    w = result.weights
    gross = w.abs().sum(axis=1)
    np.testing.assert_allclose(gross.to_numpy(), 1.0, atol=1e-6)
    # The cap is still respected after redistribution.
    assert (w.abs().to_numpy() <= 0.08 + 1e-9).all()


def test_renormalize_saturated_book_terminates(wide_data):
    # 30 names x 0.01 cap = 0.30 max gross — full reinvestment is impossible.
    # The bounded loop must stop with every live name pinned at the cap
    # rather than iterating forever chasing sum(|w|)=1.
    bt = Backtester(wide_data, {})
    cfg = _cfg(wide_data, truncation=0.01, renormalize_truncation=True)
    result, _ = bt.run(_dominant_alpha(wide_data), cfg)
    np.testing.assert_allclose(result.weights.abs().to_numpy(), 0.01, atol=1e-9)


def test_renormalize_preserves_dollar_neutrality(wide_data):
    # The attack that killed the first (whole-row) implementation: a zero-mean
    # alpha whose LONG side is concentrated (cap binds) while the short side
    # is diffuse (cap doesn't).  Whole-row rescaling redistributed the clipped
    # long mass onto the SHORT side, doubling net exposure; per-side
    # water-filling must keep net at 0 and gross at 1.
    closes = wide_data["close"]
    n = len(closes.columns)  # 30
    alpha = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    alpha.iloc[:, 0:2] = 10.0  # 2 huge longs — will be capped
    alpha.iloc[:, 2:12] = 1.0  # 10 small longs
    alpha.iloc[:, 12:n] = -30.0 / (n - 12)  # diffuse shorts, sum = -30
    assert float(alpha.iloc[0].sum()) == pytest.approx(0.0)

    bt = Backtester(wide_data, {})
    cfg = _cfg(wide_data, truncation=0.05, renormalize_truncation=True)
    result, _ = bt.run(alpha, cfg)
    w = result.weights
    net = w.sum(axis=1)
    gross = w.abs().sum(axis=1)
    np.testing.assert_allclose(net.to_numpy(), 0.0, atol=1e-9)
    np.testing.assert_allclose(gross.to_numpy(), 1.0, atol=1e-9)
    assert (w.abs().to_numpy() <= 0.05 + 1e-9).all()

    # Plain clip on the same alpha loses the capped long mass: net goes
    # visibly short — the exact distortion redistribution is meant to avoid.
    plain, _ = bt.run(alpha, _cfg(wide_data, truncation=0.05))
    assert (plain.weights.sum(axis=1) < -0.2).all()


def test_renormalize_exact_on_skewed_alphas():
    # Skewed lognormal magnitudes near the feasibility boundary — the case
    # where the old fixed-budget iterative loop exited ~1% short of full
    # gross.  The exact water-filling solution has no convergence budget.
    rng = np.random.default_rng(11)
    n_days, n_names, cap = 10, 76, 0.02  # per side ~38 names x 0.02 = 0.76
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    tickers = [f"S{i:03d}" for i in range(n_names)]
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (n_days, n_names)), axis=0)),
        index=dates,
        columns=tickers,
    )
    data = {"close": closes, "returns": closes.pct_change()}
    signs = np.where(np.arange(n_names) % 2 == 0, 1.0, -1.0)
    alpha = pd.DataFrame(
        rng.lognormal(0.0, 1.5, (n_days, n_names)) * signs,
        index=dates,
        columns=tickers,
    )
    bt = Backtester(data, {})
    cfg = _cfg(data, truncation=cap, renormalize_truncation=True)
    result, _ = bt.run(alpha, cfg)
    w = result.weights[list(alpha.columns)].to_numpy()
    assert (np.abs(w) <= cap + 1e-9).all()
    # Exact invariant, no convergence tolerance: each side lands precisely on
    # its pre-clip mass, or on m_side*cap when that mass is unreachable under
    # the cap (heavy lognormal tails CAN legitimately saturate a side).
    pre = (
        alpha.div(alpha.abs().sum(axis=1), axis=0)
        .loc[result.weights.index, list(alpha.columns)]
        .to_numpy()
    )
    exp_long = np.minimum(np.where(pre > 0, pre, 0.0).sum(axis=1), (pre > 0).sum(axis=1) * cap)
    exp_short = np.minimum(np.where(pre < 0, -pre, 0.0).sum(axis=1), (pre < 0).sum(axis=1) * cap)
    np.testing.assert_allclose(np.where(w > 0, w, 0.0).sum(axis=1), exp_long, atol=1e-9)
    np.testing.assert_allclose(np.where(w < 0, -w, 0.0).sum(axis=1), exp_short, atol=1e-9)


def test_renormalize_off_by_default_matches_plain_clip(wide_data):
    bt = Backtester(wide_data, {})
    alpha = _dominant_alpha(wide_data)
    r_default, _ = bt.run(alpha, _cfg(wide_data, truncation=0.08))
    r_explicit, _ = bt.run(alpha, _cfg(wide_data, truncation=0.08, renormalize_truncation=False))
    pd.testing.assert_frame_equal(r_default.weights, r_explicit.weights)


# ---------- Brain-convention metrics ----------


def _result_from(daily_pnl, turnover, booksize=20_000_000.0):
    n = len(daily_pnl)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    weights = pd.DataFrame({"A": [0.5] * n, "B": [-0.5] * n}, index=idx)
    return BacktestResult(
        dates=[d.strftime("%Y-%m-%d") for d in idx],
        daily_pnl=list(daily_pnl),
        cumulative_pnl=list(np.cumsum(daily_pnl)),
        daily_returns=[p / booksize for p in daily_pnl],
        weights=weights,
        turnover=list(turnover),
        positions=weights * booksize,
        booksize=booksize,
    )


def test_margin_bps_is_pnl_per_dollar_traded():
    pnl = [1000.0, -400.0, 600.0, 200.0]
    to = [100_000.0, 50_000.0, 150_000.0, 100_000.0]
    m = PerformanceAnalytics().compute(_result_from(pnl, to))
    expected = sum(pnl) / sum(to) * 10_000.0  # 1400 / 400k x 1e4 = 35 bps
    assert m["margin_bps"] == pytest.approx(expected)


def test_margin_bps_none_when_nothing_traded():
    m = PerformanceAnalytics().compute(_result_from([0.0, 0.0], [0.0, 0.0]))
    assert m["margin_bps"] is None


def test_avg_turnover_frac_divides_by_booksize():
    m = PerformanceAnalytics().compute(
        _result_from([100.0] * 5, [2_000_000.0] * 5, booksize=20_000_000.0)
    )
    assert m["avg_turnover_frac"] == pytest.approx(0.1)


def test_annual_return_arith_uses_half_book_invested_amount():
    # Brain's Returns divide annualized PnL by the INVESTED amount — half the
    # book ($10M long + $10M short on $20M) — while Turnover uses the full
    # book.  daily_returns are net_pnl / booksize, hence the x2.
    booksize = 20_000_000.0
    pnl = [2000.0, -1000.0, 3000.0]
    m = PerformanceAnalytics().compute(_result_from(pnl, [1e6] * 3, booksize))
    mean_dr = float(np.mean([p / booksize for p in pnl]))
    assert m["annual_return_arith"] == pytest.approx(mean_dr * 252 * 2.0)


def test_fitness_wq_sign_carried_by_sharpe_not_doubled():
    # A coherent losing alpha has negative Sharpe AND negative annual return.
    # Brain's formula lets Sharpe carry the sign; the old sign(return) factor
    # double-negated this case into a bogus positive score.
    f = _fitness_wq(-1.2, -0.08, 0.5)
    assert f is not None and f < 0
    assert f == pytest.approx(-1.2 * math.sqrt(0.08 / 0.5))


# ---------- /api/presets/brain + end-to-end settings echo ----------


def test_brain_preset_endpoint(client):
    r = client.get("/api/presets/brain")
    assert r.status_code == 200
    body = r.json()
    s = body["settings"]
    assert s["transaction_cost_bps"] == 0.0
    assert s["truncation"] == 0.08
    assert s["renormalize_truncation"] is True
    assert body["notes"]


def test_simulate_accepts_renormalize_truncation(client):
    r = client.post(
        "/api/simulate",
        json={
            "expression": "rank(close)",
            "settings": {
                "renormalize_truncation": True,
                "truncation": 0.08,
                "transaction_cost_bps": 0.0,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["settings"]["renormalize_truncation"] is True
    m = body["is_metrics"]
    assert m["margin_bps"] is not None
    assert m["avg_turnover_frac"] is not None and m["avg_turnover_frac"] > 0
    assert m["annual_return_arith"] is not None
