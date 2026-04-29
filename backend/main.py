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

from analytics.performance import PerformanceAnalytics, _safe_float, _safe_list
from config import (
    DATA_END,
    DATA_START,
    DEFAULT_BOOKSIZE,
    SECTOR_MAP,
    UNIVERSE,
)
from data.fetcher import DataFetcher
from db.database import connect
from db.migrations import init_db
from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator
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

DATA_FIELDS = ["open", "high", "low", "close", "volume", "returns", "vwap"]


# ---------- Lifespan: load data + init DB ----------


_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()

    fetcher = DataFetcher()
    fetcher.download_universe()  # idempotent — fast on cache hit

    spy_fetcher = DataFetcher()
    spy_frames = spy_fetcher.download_universe(tickers=["SPY"])
    spy_returns = (
        spy_frames["SPY"]["returns"] if "SPY" in spy_frames else pd.Series(dtype=float)
    )

    _state["fetcher"] = fetcher
    _state["spy_returns"] = spy_returns
    _state["data"] = {field: fetcher.get_data_matrix(field) for field in DATA_FIELDS}
    yield


app = FastAPI(title="QuantLab", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helpers ----------


def _make_config(settings: dict | None) -> SimulationConfig:
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


def _build_response(
    expression: str, alpha_matrix: pd.DataFrame, cfg: SimulationConfig
) -> dict[str, Any]:
    bt = Backtester(_state["data"], SECTOR_MAP)
    try:
        result = bt.run(alpha_matrix, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Backtest error: {e}")

    spy = _state.get("spy_returns")
    bench = None
    if isinstance(spy, pd.Series) and not spy.empty:
        bench = spy.reindex(pd.to_datetime(result.dates))

    perf = PerformanceAnalytics().compute(result, benchmark_returns=bench)

    metric_keys = (
        "sharpe", "annual_return", "annual_vol", "max_drawdown",
        "calmar_ratio", "sortino_ratio", "avg_turnover", "fitness",
        "win_rate", "profit_factor", "beta", "information_ratio",
    )

    return {
        "metrics": {k: perf[k] for k in metric_keys},
        "timeseries": {
            "dates": list(result.dates),
            "cumulative_pnl": _safe_list(result.cumulative_pnl),
            "daily_returns": _safe_list(result.daily_returns),
            "drawdown": perf["drawdown_series"],
            "rolling_sharpe": perf["rolling_sharpe"],
            "turnover": _safe_list(result.turnover),
        },
        "monthly_returns": perf["monthly_returns"],
        "expression": expression,
        "settings": _config_to_dict(cfg),
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
    return {"operators": OPERATORS, "data_fields": DATA_FIELDS}


@app.post("/api/validate")
def validate(req: ValidateRequest):
    try:
        Parser().parse(req.expression)
        return {"valid": True, "error": None}
    except ValueError as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    alpha = _evaluate(req.expression)
    cfg = _make_config(req.settings)
    return _build_response(req.expression, alpha, cfg)


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
    frame = fetcher._frames.get(ticker)
    if frame is None:
        path = fetcher._cache_path(ticker)
        if path.exists():
            try:
                frame = pd.read_parquet(path)
            except Exception:
                frame = None
    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail=f"No cached data for {ticker}")

    last30 = frame.tail(30).copy()
    last30.index = last30.index.strftime("%Y-%m-%d")
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
