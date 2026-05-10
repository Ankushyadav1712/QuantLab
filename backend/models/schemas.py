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
        ..., description="Each item: {expression: str, weight: float}. "
                         "When weight_method != 'equal', user-supplied weights "
                         "are ignored and the optimizer's output is used instead."
    )
    settings: dict[str, Any] | None = None
    # New: covariance-aware weighting (defaults to "equal" for backwards compat).
    # See analytics/mv_optimizer.py for the supported methods.
    weight_method: str = "equal"
    # Only used by mv_optimal — annualized portfolio vol target.  Ignored
    # otherwise.  None → no scaling, weights normalize to sum=1.
    target_vol: float | None = None


class CorrelationRequest(BaseModel):
    alpha_ids: list[int]


class SweepRequest(BaseModel):
    """Run a parameter sweep — expand ``{a..b:s}`` tokens in the expression
    into a cartesian product, run each cell IS-only, return a flat grid."""
    expression: str
    settings: dict[str, Any] | None = None
    # Hard ceiling on the cartesian product so a misconfigured sweep can't
    # blow up the worker.  The endpoint enforces this; clients see a 400.
    max_combinations: int = 50


class CompareRequest(BaseModel):
    """Run 2-4 expressions through the IS-only pipeline and overlay the results.

    No OOS, no walk-forward, no factor decomposition — pure visual diff.
    Validation against the chosen-best is left to running each winner through
    /api/simulate separately.
    """
    expressions: list[str] = Field(
        ..., description="2 to 4 alpha expressions to compare side-by-side",
        min_length=2, max_length=4,
    )
    settings: dict[str, Any] | None = None
    n_trials: int = 1
