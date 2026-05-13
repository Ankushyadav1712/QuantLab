"""Pareto frontier across saved alphas on the (Sharpe, Turnover) plane.

When a researcher has accumulated 20+ saved alphas, comparing them
pairwise becomes impractical.  The Pareto frontier identifies the subset
of alphas where no other alpha is strictly better on *both* axes — those
are the candidates worth keeping; everything else is dominated.

Conventions:
- Higher Sharpe is better, lower turnover is better.
- An alpha with negative Sharpe is never on the frontier — no rational
  portfolio includes a loss-maker regardless of cost.  We tag it
  ``is_pareto=False`` up front rather than letting it sneak onto the
  frontier when turnover is also low (negative-Sharpe + low turnover
  isn't a useful trade-off — it's just a low-turnover way to lose money).
- Ties: an alpha that ties another on one axis and beats on the other
  is NOT dominated by the tied alpha (it has a strictly-better axis).
  Two alphas that tie on both axes are both on the frontier.
"""

from __future__ import annotations

import math
from typing import Any


def _coerce_metric(value: Any) -> float | None:
    """Best-effort float coercion that maps invalid / missing to None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def compute_pareto(alphas: list[dict]) -> list[dict]:
    """Annotate each alpha with ``is_pareto: bool`` on the Sharpe-vs-turnover
    Pareto frontier.

    Parameters
    ----------
    alphas : list[dict]
        Each must have at least ``sharpe`` and ``turnover`` numeric fields.
        Other keys are passed through unchanged.  Missing or non-numeric
        sharpe/turnover → alpha is tagged ``is_pareto=False`` (can't be
        placed on the frontier without a valid axis pair).

    Returns
    -------
    list[dict]
        Same alphas, same order, each augmented with:
          - ``is_pareto``: bool
          - ``dominated_by``: list[int] — ids of alphas that strictly
            dominate this one (empty if Pareto-optimal).  Only populated
            when input alphas have an ``id`` field; otherwise omitted.

    O(n²) comparison.  At <1000 saved alphas that's still trivial; if the
    library grows much past that, switch to a sort+sweep (O(n log n)).
    """
    out: list[dict] = []
    # First pass: extract & coerce the two metrics we care about, plus the
    # "always-False" disqualifications (missing data, negative Sharpe).
    points: list[tuple[float | None, float | None, dict]] = []
    for a in alphas:
        sharpe = _coerce_metric(a.get("sharpe"))
        turnover = _coerce_metric(a.get("turnover"))
        points.append((sharpe, turnover, a))

    # Second pass: per-alpha dominance check.
    for i, (sh_i, tn_i, alpha_i) in enumerate(points):
        record = {**alpha_i}
        # Disqualifications: missing axis or negative Sharpe → not on frontier
        if sh_i is None or tn_i is None or sh_i < 0:
            record["is_pareto"] = False
            if "id" in alpha_i:
                record["dominated_by"] = []
            out.append(record)
            continue

        dominators: list[int] = []
        for j, (sh_j, tn_j, alpha_j) in enumerate(points):
            if i == j:
                continue
            if sh_j is None or tn_j is None or sh_j < 0:
                continue
            # j dominates i iff j is weakly better on BOTH axes and
            # strictly better on at least one.
            weakly_better = (sh_j >= sh_i) and (tn_j <= tn_i)
            strictly_better_on_one = (sh_j > sh_i) or (tn_j < tn_i)
            if weakly_better and strictly_better_on_one:
                if "id" in alpha_j:
                    dominators.append(int(alpha_j["id"]))
                else:
                    # No id available — just mark dominated and stop early
                    dominators.append(-1)
                    break

        record["is_pareto"] = len(dominators) == 0
        if "id" in alpha_i:
            record["dominated_by"] = dominators
        out.append(record)

    return out


def pareto_frontier_only(alphas: list[dict]) -> list[dict]:
    """Convenience wrapper that returns only the Pareto-optimal subset,
    sorted by turnover ascending (so consumers can draw the frontier line)."""
    annotated = compute_pareto(alphas)
    frontier = [a for a in annotated if a.get("is_pareto")]
    # Sort by turnover ascending for clean plotting.  Missing turnover
    # (which shouldn't happen post-filter) sorts to end.
    frontier.sort(key=lambda a: _coerce_metric(a.get("turnover")) or float("inf"))
    return frontier
