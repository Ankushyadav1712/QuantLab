"""Multi-alpha batch runner.

Runs N alpha expressions through the standard IS-only pipeline
(evaluate → backtest → performance) concurrently and returns a compact,
rankable metrics row per alpha plus a pairwise correlation matrix of their
daily-return series.

Kept intentionally lean: the batch view is for scanning and ranking many
candidates fast, not deep analysis. Click a row and run ``/api/simulate`` for
the full IS/OOS tearsheet.

Pure module — depends only on ``engine`` + ``analytics`` + ``config``, never on
``main`` — so it can be unit-tested in isolation and imported without triggering
the app's data load.

Thread-safety: workers only READ the shared price/GICS matrices (pandas reads
are safe under threads) and each builds its own alpha signal + result objects.
The caller MUST resolve the universe and snapshot the data dicts before calling
``run_batch``, since universe resolution can mutate the shared data pool.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from analytics.performance import PerformanceAnalytics, _safe_float
from config import SECTOR_MAP

from engine.backtester import Backtester, SimulationConfig
from engine.evaluator import AlphaEvaluator
from engine.lint import lint_ast
from engine.parser import Parser

# Headline metrics surfaced in the ranked comparison table. Every key here is a
# top-level field in PerformanceAnalytics.compute()'s output.
BATCH_METRIC_KEYS = (
    "sharpe",
    "ic",
    "ic_tstat",
    "icir",
    "annual_return",
    "max_drawdown",
    "avg_turnover",
    "fitness",
    "win_rate",
)

# |ρ| at or above this flags a redundant pair in the correlation matrix.
REDUNDANT_RHO = 0.7

# Cap on parallel backtest workers. Each worker holds a full backtest's memory,
# so N workers ≈ N× peak RAM → OOM on a small host. os.cpu_count() reports the
# HOST's cores on shared platforms (e.g. Render's free tier), not the
# container's throttled slice, so we never trust it as the default. Gate via
# QUANTLAB_MAX_WORKERS (default 1 → sequential; raise only where RAM allows).
_MAX_WORKERS_CAP = max(1, int(os.getenv("QUANTLAB_MAX_WORKERS", "1")))


def _run_one(
    item: dict[str, Any],
    cfg: SimulationConfig,
    data: dict[str, pd.DataFrame],
    gics_data: dict[str, pd.DataFrame],
    gics_map: dict[str, dict[str, str | None]] | None,
    spy: pd.Series | None,
    size_field: pd.DataFrame | None,
) -> dict[str, Any]:
    """Evaluate + backtest a single alpha.

    Never raises: every failure mode is returned as an
    ``{"id", "expression", "error"}`` row so one bad alpha can't sink the batch.
    On success returns ``{"id", "expression", "metrics", "_returns"}`` — the
    private ``_returns`` series is consumed by the correlation step and stripped
    before the response leaves ``run_batch``.
    """
    aid = item.get("id")
    expr = item.get("expression", "")

    try:
        ast = Parser().parse(expr)
    except ValueError as exc:
        return {"id": aid, "expression": expr, "error": f"parse error: {exc}"}

    errors = [d for d in lint_ast(ast) if d["severity"] == "error"]
    if errors:
        return {"id": aid, "expression": expr, "error": errors[0]["message"]}

    # Fresh eval_data dict per task (mirrors main._evaluate): the evaluator may
    # inject data-aware helpers into the dict, so we never share one across
    # threads. The DataFrame values themselves are shared read-only.
    eval_data = {**data, **gics_data}
    try:
        signal = AlphaEvaluator(eval_data).evaluate(expr)
    except ValueError as exc:
        return {"id": aid, "expression": expr, "error": f"evaluation error: {exc}"}
    if not isinstance(signal, pd.DataFrame):
        return {
            "id": aid,
            "expression": expr,
            "error": "expression produced a scalar, not a (dates × tickers) matrix",
        }

    try:
        is_result, _ = Backtester(data, sector_map=SECTOR_MAP, gics_map=gics_map).run(signal, cfg)
    except ValueError as exc:
        return {"id": aid, "expression": expr, "error": f"backtest error: {exc}"}

    bench = None
    if isinstance(spy, pd.Series) and not spy.empty:
        bench = spy.reindex(pd.to_datetime(is_result.dates))
    full = PerformanceAnalytics().compute(
        is_result,
        benchmark_returns=bench,
        n_trials=1,
        gics_map=gics_map,
        size_field=size_field,
    )
    metrics = {k: _safe_float(full.get(k)) for k in BATCH_METRIC_KEYS}
    returns = pd.Series(is_result.daily_returns, index=pd.to_datetime(is_result.dates))
    return {"id": aid, "expression": expr, "metrics": metrics, "_returns": returns}


def _correlation(series: dict[str, pd.Series], labels: list[str]) -> dict[str, Any]:
    """Pairwise Pearson correlation of daily-return series.

    Matches the existing ``/api/alphas/correlations`` convention (returns, not
    raw signals) so the frontend heatmap and this share one mental model.
    """
    if not series:
        return {"labels": [], "matrix": [], "redundant_pairs": []}
    df = pd.DataFrame(series)
    corr = df.corr().reindex(index=labels, columns=labels)
    matrix = [[_safe_float(v) for v in row] for row in corr.values.tolist()]
    redundant: list[dict[str, Any]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rho = matrix[i][j]
            if rho is not None and abs(rho) >= REDUNDANT_RHO:
                redundant.append({"a": labels[i], "b": labels[j], "rho": rho})
    return {"labels": labels, "matrix": matrix, "redundant_pairs": redundant}


def run_batch(
    alphas: list[dict[str, Any]],
    cfg: SimulationConfig,
    *,
    data: dict[str, pd.DataFrame],
    gics_data: dict[str, pd.DataFrame] | None = None,
    gics_map: dict[str, dict[str, str | None]] | None = None,
    spy: pd.Series | None = None,
    size_field: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run every alpha concurrently; return ranked metrics + correlation.

    ``alphas`` is a list of ``{"id": str, "expression": str}``. ``cfg`` should
    be IS-only (``run_oos=False``) — the batch view is a fast scan. Results come
    back in input order; a failed alpha carries an ``error`` instead of
    ``metrics``.
    """
    gics_data = gics_data or {}
    if max_workers is None:
        max_workers = max(1, min(len(alphas) or 1, _MAX_WORKERS_CAP))

    def _task(item: dict[str, Any]) -> dict[str, Any]:
        try:
            return _run_one(item, cfg, data, gics_data, gics_map, spy, size_field)
        except Exception as exc:  # one bad alpha must never sink the whole batch
            return {
                "id": item.get("id"),
                "expression": item.get("expression", ""),
                "error": f"internal error: {exc}",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_task, alphas))

    # Correlate the daily-return series of the alphas that ran successfully.
    series: dict[str, pd.Series] = {}
    labels: list[str] = []
    for r in results:
        returns = r.pop("_returns", None)
        if returns is not None and not returns.empty:
            label = str(r.get("id"))
            series[label] = returns
            labels.append(label)

    return {"results": results, "correlation_matrix": _correlation(series, labels)}
