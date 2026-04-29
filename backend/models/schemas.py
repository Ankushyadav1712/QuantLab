from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    expression: str
    settings: dict[str, Any] | None = None


class SimulationResponse(BaseModel):
    metrics: dict[str, Any]
    timeseries: dict[str, Any]
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
