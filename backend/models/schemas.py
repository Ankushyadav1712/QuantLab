from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    expression: str
    settings: dict[str, Any] | None = None
    # When True (default), the backtester splits the date range into a 70/30
    # IS/OOS pair and the response carries both halves.  When False, only the
    # IS slot is populated (covering the full window) and OOS fields are null.
    run_oos: bool = True
    # Number of distinct alpha expressions the researcher has tried in this
    # session.  Fed into the Deflated Sharpe Ratio to discount the headline
    # for selection bias — picking the best of many trials inflates Sharpe.
    n_trials: int = 1


class SimulationResponse(BaseModel):
    # Primary IS/OOS shape.  is_* fields are always populated; oos_* are null
    # when run_oos was False (or the split was degenerate).
    is_metrics: dict[str, Any]
    oos_metrics: dict[str, Any] | None = None
    is_timeseries: dict[str, Any]
    oos_timeseries: dict[str, Any] | None = None
    overfitting_analysis: dict[str, Any] | None = None
    factor_decomposition: dict[str, Any] | None = None
    walk_forward: list[dict[str, Any]] | None = None

    monthly_returns: list[list[Any]]
    expression: str
    settings: dict[str, Any]


class ValidateRequest(BaseModel):
    expression: str


class AlphaSaveRequest(BaseModel):
    expression: str
    name: str
    notes: str = ""
    settings: dict[str, Any] | None = None


class AlphaRecord(BaseModel):
    id: int
    name: str
    expression: str
    notes: str = ""
    sharpe: float | None = None
    created_at: str


class MultiAlphaRequest(BaseModel):
    alphas: list[dict[str, Any]] = Field(
        ..., description="Each item: {expression: str, weight: float}"
    )
    settings: dict[str, Any] | None = None


class CorrelationRequest(BaseModel):
    alpha_ids: list[int]
