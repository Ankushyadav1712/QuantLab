from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analytics.factor_decomp import FactorDecomposition
from analytics.performance import PerformanceAnalytics, _safe_float, _safe_list
from config import (
    ALLOWED_ORIGINS,
    DATA_END,
    DATA_START,
    DEFAULT_BOOKSIZE,
    ENVIRONMENT,
    SECTOR_MAP,
    UNIVERSE,
)
from data.factors import download_ff5_daily
from data.fetcher import ALL_FIELDS, DataFetcher
from db.database import connect
from db.migrations import init_db
from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator
from engine.lint import lint_ast
from engine.parser import Parser
from models.schemas import (
    AlphaSaveRequest,
    CorrelationRequest,
    MultiAlphaRequest,
    SimulationRequest,
    ValidateRequest,
)


OPERATORS = [
    {"name": "ts_mean", "args": "(x, d)", "description": "Rolling mean of x over d periods"},
    {"name": "ts_std", "args": "(x, d)", "description": "Rolling stdev of x over d periods"},
    {"name": "ts_min", "args": "(x, d)", "description": "Rolling min over d periods"},
    {"name": "ts_max", "args": "(x, d)", "description": "Rolling max over d periods"},
    {"name": "ts_sum", "args": "(x, d)", "description": "Rolling sum over d periods"},
    {"name": "ts_rank", "args": "(x, d)", "description": "Rolling percentile rank in [0,1] over d periods"},
    {"name": "delta", "args": "(x, d)", "description": "x - x.shift(d)"},
    {"name": "delay", "args": "(x, d)", "description": "x.shift(d)"},
    {"name": "decay_linear", "args": "(x, d)", "description": "Linearly weighted MA over d periods"},
    {"name": "ts_corr", "args": "(x, y, d)", "description": "Rolling Pearson correlation over d periods"},
    {"name": "ts_cov", "args": "(x, y, d)", "description": "Rolling covariance over d periods"},
    {"name": "rank", "args": "(x)", "description": "Cross-sectional percentile rank in [0,1] per date"},
    {"name": "zscore", "args": "(x)", "description": "Cross-sectional z-score per date"},
    {"name": "demean", "args": "(x)", "description": "Subtract cross-sectional mean per date"},
    {"name": "scale", "args": "(x)", "description": "Scale so |sum|=1 per date"},
    {"name": "normalize", "args": "(x)", "description": "Demean then scale to unit |sum|"},
    {"name": "abs", "args": "(x)", "description": "Element-wise absolute value"},
    {"name": "log", "args": "(x)", "description": "Element-wise natural log (NaN for x<=0)"},
    {"name": "sign", "args": "(x)", "description": "Element-wise sign (-1, 0, +1)"},
    {"name": "power", "args": "(x, n)", "description": "x ** n element-wise"},
    {"name": "max", "args": "(x, y)", "description": "Element-wise max"},
    {"name": "min", "args": "(x, y)", "description": "Element-wise min"},
    {"name": "if_else", "args": "(cond, x, y)", "description": "Element-wise conditional"},
]

# Rich field metadata: every entry is {name, category, description}.  Returned
# from /api/operators so the frontend can list & autocomplete all 32 fields.
FIELDS: list[dict[str, str]] = [
    # ----- price (7 originals) -----
    {"name": "open", "category": "price", "description": "Opening price"},
    {"name": "high", "category": "price", "description": "Daily high price"},
    {"name": "low", "category": "price", "description": "Daily low price"},
    {"name": "close", "category": "price", "description": "Closing price"},
    {"name": "volume", "category": "price", "description": "Daily share volume"},
    {"name": "returns", "category": "price", "description": "Daily simple return (close.pct_change)"},
    {"name": "vwap", "category": "price", "description": "Typical price (high + low + close) / 3"},
    # ----- price structure (7) -----
    {"name": "median_price", "category": "price_structure", "description": "Average of high and low; cleaner price estimate than close"},
    {"name": "weighted_close", "category": "price_structure", "description": "Close-weighted typical price; smoother than simple average"},
    {"name": "range_", "category": "price_structure", "description": "High minus low; daily volatility proxy (alias: range)"},
    {"name": "body", "category": "price_structure", "description": "Absolute candle body size; small body indicates indecision"},
    {"name": "upper_shadow", "category": "price_structure", "description": "Rejection of higher prices; potential bearish signal"},
    {"name": "lower_shadow", "category": "price_structure", "description": "Rejection of lower prices; potential bullish signal"},
    {"name": "gap", "category": "price_structure", "description": "Overnight price gap; captures after-hours sentiment"},
    # ----- return variants (5) -----
    {"name": "log_returns", "category": "return_variants", "description": "Log of price ratio; symmetric and better for compounding"},
    {"name": "abs_returns", "category": "return_variants", "description": "Absolute daily return; volatility proxy"},
    {"name": "intraday_return", "category": "return_variants", "description": "Close minus open over open; pure intraday momentum"},
    {"name": "overnight_return", "category": "return_variants", "description": "Open minus prior close; captures news/earnings reaction"},
    {"name": "signed_volume", "category": "return_variants", "description": "Volume signed by return direction; money flow proxy"},
    # ----- volume & liquidity (4) -----
    {"name": "dollar_volume", "category": "volume_liquidity", "description": "Price times volume; true liquidity measure"},
    {"name": "adv20", "category": "volume_liquidity", "description": "20-day average daily volume; liquidity baseline"},
    {"name": "volume_ratio", "category": "volume_liquidity", "description": "Volume divided by adv20; values above 1 indicate unusual activity"},
    {"name": "amihud", "category": "volume_liquidity", "description": "Absolute return over dollar volume; Amihud illiquidity ratio"},
    # ----- volatility & risk (5) -----
    {"name": "true_range", "category": "volatility_risk", "description": "Volatility accounting for gaps; better than simple high-low range"},
    {"name": "atr", "category": "volatility_risk", "description": "14-day average true range; standard volatility measure"},
    {"name": "realized_vol", "category": "volatility_risk", "description": "20-day rolling return standard deviation"},
    {"name": "skewness", "category": "volatility_risk", "description": "60-day rolling return skewness; negative values indicate crash risk"},
    {"name": "kurtosis", "category": "volatility_risk", "description": "60-day rolling return kurtosis; high values indicate fat tails"},
    # ----- momentum & relative (4) -----
    {"name": "momentum_5", "category": "momentum_relative", "description": "5-day price momentum"},
    {"name": "momentum_20", "category": "momentum_relative", "description": "20-day price momentum"},
    {"name": "close_to_high_252", "category": "momentum_relative", "description": "Ratio of close to 52-week high; distance from recent peak"},
    {"name": "high_low_ratio", "category": "momentum_relative", "description": "High over low; intraday volatility as a ratio"},
]

# Plain-name list for callers that only want field names.
DATA_FIELDS: list[str] = [f["name"] for f in FIELDS]
assert sorted(DATA_FIELDS) == sorted(ALL_FIELDS), "FIELDS and ALL_FIELDS must agree"


# ---------- Lifespan: load data + init DB ----------


_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()

    fetcher = DataFetcher()
    fetcher.download_universe()  # idempotent — fast on cache hit

    spy_fetcher = DataFetcher()
    spy_frames = spy_fetcher.download_universe(
        tickers=["SPY"], compute_derived=False
    )
    spy_returns = (
        spy_frames["SPY"]["returns"] if "SPY" in spy_frames else pd.Series(dtype=float)
    )

    # Fama-French 5 factor returns (Mkt-RF, SMB, HML, RMW, CMA + RF) cached
    # weekly.  Network failure is non-fatal — factor decomposition just gets
    # skipped in /api/simulate if the DataFrame is empty.
    ff5 = download_ff5_daily()

    _state["fetcher"] = fetcher
    _state["spy_returns"] = spy_returns
    _state["ff5"] = ff5
    _state["data"] = {field: fetcher.get_data_matrix(field) for field in DATA_FIELDS}
    yield


app = FastAPI(title="QuantLab", lifespan=lifespan)

# Dev keeps "*" so a browser can hit the API from any localhost variant or
# from a LAN box; production tightens to the configured allow-list.
_cors_origins = ["*"] if ENVIRONMENT != "production" else ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helpers ----------


def _make_config(
    settings: dict | None, *, run_oos: bool = True
) -> SimulationConfig:
    s = settings or {}
    return SimulationConfig(
        universe=s.get("universe") or UNIVERSE,
        start_date=s.get("start_date") or DATA_START,
        end_date=s.get("end_date") or DATA_END,
        neutralization=s.get("neutralization", "market"),
        truncation=float(s.get("truncation", 0.05)),
        booksize=float(s.get("booksize", DEFAULT_BOOKSIZE)),
        transaction_cost_bps=float(s.get("transaction_cost_bps", 5.0)),
        decay=int(s.get("decay", 0)),
        oos_split=float(s.get("oos_split", 0.3)),
        run_oos=bool(s.get("run_oos", run_oos)),
    )


def _config_to_dict(cfg: SimulationConfig) -> dict[str, Any]:
    return {
        "universe": cfg.universe,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "neutralization": cfg.neutralization,
        "truncation": cfg.truncation,
        "booksize": cfg.booksize,
        "transaction_cost_bps": cfg.transaction_cost_bps,
        "decay": cfg.decay,
        "oos_split": cfg.oos_split,
        "run_oos": cfg.run_oos,
    }


def _evaluate(expression: str) -> pd.DataFrame:
    try:
        evaluator = AlphaEvaluator(_state["data"])
        result = evaluator.evaluate(expression)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression error: {e}")
    if not isinstance(result, pd.DataFrame):
        raise HTTPException(
            status_code=400,
            detail="Expression must produce a (dates × tickers) matrix, not a scalar",
        )
    return result


_METRIC_KEYS = (
    "sharpe", "annual_return", "annual_vol", "max_drawdown",
    "calmar_ratio", "sortino_ratio", "avg_turnover", "fitness",
    "win_rate", "profit_factor", "beta", "information_ratio",
)


def _compute_perf_pack(result, perf: PerformanceAnalytics, *, spy: pd.Series | None):
    """Run analytics on a BacktestResult and shape the (metrics, timeseries) pair."""
    bench = None
    if isinstance(spy, pd.Series) and not spy.empty:
        bench = spy.reindex(pd.to_datetime(result.dates))
    full = perf.compute(result, benchmark_returns=bench)

    metrics = {k: full[k] for k in _METRIC_KEYS}
    # Pass period info into compare_is_oos through the metrics dict
    metrics["start_date"] = full.get("start_date")
    metrics["end_date"] = full.get("end_date")

    timeseries = {
        "dates": list(result.dates),
        "cumulative_pnl": _safe_list(result.cumulative_pnl),
        "daily_returns": _safe_list(result.daily_returns),
        "drawdown": full["drawdown_series"],
        "rolling_sharpe": full["rolling_sharpe"],
        "turnover": _safe_list(result.turnover),
    }
    return metrics, timeseries, full["monthly_returns"]


def _build_response(
    expression: str, alpha_matrix: pd.DataFrame, cfg: SimulationConfig
) -> dict[str, Any]:
    bt = Backtester(_state["data"], SECTOR_MAP)
    try:
        is_result, oos_result = bt.run(alpha_matrix, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Backtest error: {e}")

    spy = _state.get("spy_returns")
    perf = PerformanceAnalytics()

    is_metrics, is_timeseries, is_monthly = _compute_perf_pack(
        is_result, perf, spy=spy
    )

    oos_metrics: dict[str, Any] | None = None
    oos_timeseries: dict[str, Any] | None = None
    overfitting: dict[str, Any] | None = None
    monthly_returns = is_monthly  # default: only IS data

    if oos_result is not None:
        oos_metrics, oos_timeseries, oos_monthly = _compute_perf_pack(
            oos_result, perf, spy=spy
        )
        overfitting = perf.compare_is_oos(is_metrics, oos_metrics)
        # Combined monthly-returns heatmap spans both halves
        monthly_returns = is_monthly + oos_monthly

    # Fama-French 5-factor decomposition on the full backtest window (IS+OOS
    # if both, else just IS).  Tells the user how much of their headline
    # Sharpe is just market-beta + size + value + profitability + investment.
    ff5 = _state.get("ff5")
    decomp_input_dates = list(is_timeseries["dates"])
    decomp_input_returns = list(is_timeseries["daily_returns"])
    if oos_timeseries:
        decomp_input_dates += list(oos_timeseries["dates"])
        decomp_input_returns += list(oos_timeseries["daily_returns"])
    factor_decomp = (
        FactorDecomposition().compute(decomp_input_returns, decomp_input_dates, ff5)
        if ff5 is not None and not ff5.empty
        else None
    )

    return {
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "is_timeseries": is_timeseries,
        "oos_timeseries": oos_timeseries,
        "overfitting_analysis": overfitting,
        "factor_decomposition": factor_decomp,
        "monthly_returns": monthly_returns,
        "expression": expression,
        "settings": _config_to_dict(cfg),
        # Honest disclosure: this universe is *current* S&P 100, not
        # point-in-time membership.  Backtests inherit survivorship bias
        # because dropouts (delistings, M&A, index removals) aren't represented.
        "data_quality": {
            "survivorship_bias": True,
            "universe_kind": "current_snapshot",
            "expected_sharpe_inflation": 0.2,
            "notes": [
                "Universe is the *current* S&P 100, not point-in-time index "
                "membership. Names that were in the index in 2019 but got "
                "delisted, acquired, or removed by 2024 are absent from the "
                "backtest.",
                "This survivorship bias systematically inflates Sharpe by an "
                "estimated 0.1–0.3 vs. a survivorship-free backtest. Treat "
                "all reported metrics as upper bounds.",
                "Fix requires paid PIT data (CRSP / Norgate / Sharadar). "
                "Documented in README; see Drawbacks section.",
            ],
        },
    }


# ---------- Endpoints ----------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/universe")
def get_universe():
    return {"tickers": list(UNIVERSE), "sectors": dict(SECTOR_MAP)}


@app.get("/api/operators")
def get_operators():
    return {
        "operators": OPERATORS,
        "fields": FIELDS,
        # Kept for backwards compatibility with any caller that read the old
        # flat-list shape; new code should prefer `fields`.
        "data_fields": DATA_FIELDS,
    }


@app.post("/api/validate")
def validate(req: ValidateRequest):
    try:
        ast = Parser().parse(req.expression)
    except ValueError as e:
        return {"valid": False, "error": str(e), "diagnostics": []}

    diagnostics = lint_ast(ast)
    has_lint_errors = any(d["severity"] == "error" for d in diagnostics)
    return {
        "valid": not has_lint_errors,
        "error": (
            next(d["message"] for d in diagnostics if d["severity"] == "error")
            if has_lint_errors
            else None
        ),
        # Full list of warnings + errors so the editor can render them inline.
        "diagnostics": diagnostics,
    }


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    # Lint before evaluating so we catch look-ahead bias before any
    # inflated-Sharpe number ever leaves the server.
    try:
        ast = Parser().parse(req.expression)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Expression error: {e}")

    diagnostics = lint_ast(ast)
    errors = [d for d in diagnostics if d["severity"] == "error"]
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"Lint error: {errors[0]['message']}",
        )

    alpha = _evaluate(req.expression)
    # The body's `run_oos` flag wins over anything inside `settings` so
    # callers can toggle the split without rebuilding their settings dict.
    cfg = _make_config(req.settings, run_oos=req.run_oos)
    response = _build_response(req.expression, alpha, cfg)
    # Surface any non-fatal lint warnings (long windows, zero shifts) so the
    # frontend can show them next to the dashboard.
    response["diagnostics"] = diagnostics
    return response


@app.post("/api/alphas/multi-blend")
def multi_blend(req: MultiAlphaRequest):
    if not req.alphas:
        raise HTTPException(status_code=400, detail="No alphas provided")

    raw_weights = np.array([float(a.get("weight", 1.0)) for a in req.alphas])
    total = float(np.abs(raw_weights).sum())
    if total == 0:
        raise HTTPException(status_code=400, detail="Weights sum to zero")
    weights = raw_weights / total

    combined: pd.DataFrame | None = None
    items: list[dict[str, Any]] = []
    for w, item in zip(weights, req.alphas):
        expr = item.get("expression")
        if not expr:
            raise HTTPException(status_code=400, detail="Each alpha needs an 'expression'")
        matrix = _evaluate(expr)
        items.append({"expression": expr, "weight": float(w)})
        weighted = matrix * float(w)
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)

    cfg = _make_config(req.settings)
    response = _build_response("multi-blend", combined, cfg)
    response["expression"] = "multi-blend"
    response["settings"]["alphas"] = items
    return response


@app.post("/api/alphas")
async def save_alpha(req: AlphaSaveRequest):
    alpha = _evaluate(req.expression)
    cfg = _make_config(req.settings)
    response = _build_response(req.expression, alpha, cfg)
    metrics = response["metrics"]
    created_at = datetime.now(timezone.utc).isoformat()

    payload = json.dumps(response, default=str)

    async with connect() as db:
        cursor = await db.execute(
            """
            INSERT INTO alphas
                (name, expression, notes, sharpe, annual_return, max_drawdown,
                 turnover, fitness, created_at, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.name,
                req.expression,
                req.notes,
                metrics.get("sharpe"),
                metrics.get("annual_return"),
                metrics.get("max_drawdown"),
                metrics.get("avg_turnover"),
                metrics.get("fitness"),
                created_at,
                payload,
            ),
        )
        await db.commit()
        row_id = cursor.lastrowid

    return {
        "id": row_id,
        "name": req.name,
        "expression": req.expression,
        "notes": req.notes,
        "sharpe": metrics.get("sharpe"),
        "created_at": created_at,
    }


@app.get("/api/alphas")
async def list_alphas():
    async with connect() as db:
        db.row_factory = __import__("aiosqlite").Row
        cursor = await db.execute(
            """
            SELECT id, name, expression, notes, sharpe, annual_return,
                   max_drawdown, turnover, fitness, created_at
            FROM alphas
            ORDER BY id DESC
            """
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


@app.get("/api/alphas/{alpha_id}")
async def get_alpha(alpha_id: int):
    async with connect() as db:
        db.row_factory = __import__("aiosqlite").Row
        cursor = await db.execute("SELECT * FROM alphas WHERE id = ?", (alpha_id,))
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Alpha not found")

    record = dict(row)
    raw = record.pop("result_json", None)
    if raw:
        try:
            record["result"] = json.loads(raw)
        except json.JSONDecodeError:
            record["result"] = None
    else:
        record["result"] = None
    return record


@app.delete("/api/alphas/{alpha_id}")
async def delete_alpha(alpha_id: int):
    async with connect() as db:
        cursor = await db.execute("DELETE FROM alphas WHERE id = ?", (alpha_id,))
        await db.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alpha not found")
    return {"deleted": alpha_id}


@app.post("/api/alphas/correlations")
async def alphas_correlations(req: CorrelationRequest):
    if not req.alpha_ids:
        raise HTTPException(status_code=400, detail="No alpha_ids provided")

    placeholders = ",".join("?" for _ in req.alpha_ids)
    async with connect() as db:
        db.row_factory = __import__("aiosqlite").Row
        cursor = await db.execute(
            f"SELECT id, name, result_json FROM alphas WHERE id IN ({placeholders})",
            tuple(req.alpha_ids),
        )
        rows = await cursor.fetchall()

    series: dict[str, pd.Series] = {}
    labels: list[str] = []
    for r in rows:
        raw = r["result_json"]
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = payload.get("timeseries") or {}
        dates = ts.get("dates") or []
        rets = ts.get("daily_returns") or []
        if not dates or not rets:
            continue
        label = (r["name"] or f"alpha_{r['id']}")
        # If duplicate names, disambiguate by id
        if label in series:
            label = f"{label}#{r['id']}"
        series[label] = pd.Series(rets, index=pd.to_datetime(dates))
        labels.append(label)

    if not series:
        return {"tickers": [], "matrix": []}

    df = pd.DataFrame(series)
    corr = df.corr().reindex(index=labels, columns=labels)
    matrix = [[_safe_float(v) for v in row] for row in corr.values.tolist()]
    return {"tickers": labels, "matrix": matrix}


@app.get("/api/data/preview")
def data_preview(ticker: str):
    fetcher: DataFetcher = _state["fetcher"]

    # Start with the per-ticker frame (the 7 base fields).
    base = fetcher._frames.get(ticker)
    if base is None:
        path = fetcher._cache_path(ticker)
        if path.exists():
            try:
                base = pd.read_parquet(path)
            except Exception:
                base = None
    if base is None or base.empty:
        raise HTTPException(status_code=404, detail=f"No cached data for {ticker}")

    # Pull this ticker's column out of every derived (dates × tickers) matrix
    # and join it onto the per-ticker frame, so the preview shows all 32 fields.
    columns = {f["name"]: base[f["name"]] for f in FIELDS if f["name"] in base.columns}
    for field, matrix in fetcher._matrix.items():
        if field in columns or matrix is None or matrix.empty:
            continue
        if ticker in matrix.columns:
            columns[field] = matrix[ticker]
    enriched = pd.DataFrame(columns)

    # Order columns to match FIELDS so the response is predictable.
    ordered = [f["name"] for f in FIELDS if f["name"] in enriched.columns]
    enriched = enriched[ordered]

    last30 = enriched.tail(30).copy()
    last30.index = pd.to_datetime(last30.index).strftime("%Y-%m-%d")
    last30 = last30.reset_index().rename(columns={"index": "date"})
    rows: list[dict[str, Any]] = []
    for record in last30.to_dict(orient="records"):
        clean = {}
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        rows.append(clean)
    return {"ticker": ticker, "rows": rows}
