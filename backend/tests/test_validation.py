"""Brain validator: local-vs-external return correlation + Sharpe-gap check.

Pure-function tests over analytics.validation — no app, no data fixture, fast.
"""

import numpy as np
import pandas as pd
from analytics.validation import validate_correlation


def _dates(n):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-01", periods=n)]


def test_identical_series_is_a_pass():
    rng = np.random.default_rng(0)
    n = 250
    dates = _dates(n)
    r = rng.normal(0.0005, 0.01, n).tolist()
    out = validate_correlation(dates, r, dates, r)
    assert out["ok"]
    assert out["n_overlap"] == n
    assert out["correlation"] > 0.999
    assert abs(out["sharpe_diff"]) < 1e-9
    assert out["within_tolerance"] is True
    assert out["verdict"] == "pass"


def test_scale_invariance_pnl_vs_returns():
    # External as dollar PnL (returns × booksize) must still correlate ~1.0 and
    # match Sharpe, since both metrics are scale-invariant.
    rng = np.random.default_rng(1)
    n = 200
    dates = _dates(n)
    r = rng.normal(0.0003, 0.008, n)
    pnl = (r * 1_000_000).tolist()
    out = validate_correlation(dates, r.tolist(), dates, pnl)
    assert out["correlation"] > 0.999
    assert abs(out["sharpe_diff"]) < 1e-6
    assert out["verdict"] == "pass"


def test_independent_series_fails_on_correlation():
    rng = np.random.default_rng(2)
    n = 250
    dates = _dates(n)
    a = rng.normal(0, 0.01, n).tolist()
    b = rng.normal(0, 0.01, n).tolist()  # independent draw
    out = validate_correlation(dates, a, dates, b)
    assert out["ok"]
    assert abs(out["correlation"]) < 0.3
    assert out["verdict"] == "fail"


def test_no_overlap_is_not_ok():
    a_dates = _dates(100)
    b_dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2030-01-01", periods=100)]
    vals = [0.01, -0.01] * 50
    out = validate_correlation(a_dates, vals, b_dates, vals)
    assert out["ok"] is False
    assert out["n_overlap"] == 0


def test_flat_external_sharpe_leaves_tolerance_undecided():
    rng = np.random.default_rng(3)
    n = 250
    dates = _dates(n)
    local = rng.normal(0.001, 0.01, n).tolist()
    ext = [0.01, -0.01] * (n // 2)  # mean ~0 → Sharpe ~0
    out = validate_correlation(dates, local, dates, ext)
    assert out["ok"]
    assert out["within_tolerance"] is None
    assert out["sharpe_diff_pct"] is None


def test_mismatched_lengths_and_bad_rows_are_tolerated():
    dates = _dates(60)
    returns = [0.01] * 60
    # Extra dates with no matching returns + a garbage date row must not crash.
    out = validate_correlation(dates + ["not-a-date"], returns, dates, returns)
    assert out["ok"]
    assert out["n_overlap"] == 60


def test_endpoint_parses_nested_series_and_returns_verdict():
    # HTTP-layer wiring: nested ReturnSeries parsing + endpoint result. No
    # lifespan (no `with`), so no market-data load — this endpoint needs none.
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    dates = _dates(40)
    r = ([0.01, -0.004, 0.006] * 14)[:40]
    resp = client.post(
        "/api/validate_correlation",
        json={
            "local": {"dates": dates, "returns": r},
            "external": {"dates": dates, "returns": r},
            "sharpe_tolerance_pct": 3.0,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["correlation"] > 0.999
    assert body["verdict"] == "pass"
