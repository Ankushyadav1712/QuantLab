"""Alpha-combination helpers: IC-weighting and greedy orthogonal selection.

These complement ``analytics/mv_optimizer.py`` (equal / inverse-variance /
mean-variance / risk-parity, all of which weight by the *return* series). The
two functions here need signal-quality inputs the return-based optimizers don't
have:

- ``ic_weights`` weights each alpha by its Information Coefficient — a
  signal-quality measure — rather than by return statistics.
- ``greedy_orthogonal_select`` prunes near-duplicate alphas before weighting,
  so a book of ten variations on one idea doesn't masquerade as ten
  independent bets.

Both are pure functions (numpy in, numpy/list out) — no app state, unit-testable
in isolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ic_weights(ics: list[Any]) -> np.ndarray:
    """Weight each alpha in proportion to ``max(0, IC)``, normalized to sum 1.

    Alphas with negative, NaN, or missing IC get zero weight — a negative-IC
    alpha predicts the *wrong* direction, so it shouldn't contribute to the
    blend. If every alpha has non-positive IC (nothing to weight toward), fall
    back to equal weights rather than returning an all-zero vector.
    """
    vals: list[float] = []
    for x in ics:
        try:
            v = float(x)
        except (TypeError, ValueError):
            v = 0.0
        if not np.isfinite(v) or v < 0:
            v = 0.0
        vals.append(v)
    arr = np.array(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([])
    total = arr.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return arr / total


def greedy_orthogonal_select(
    corr: Any,
    scores: list[Any],
    max_rho: float = 0.7,
) -> list[int]:
    """Pick a low-redundancy subset of alphas. Returns kept indices, sorted.

    Walks alphas in descending ``scores`` order (e.g. IC t-stat) and keeps one
    only if its absolute correlation to every already-kept alpha is at or below
    ``max_rho``. This is the classic greedy set-cover heuristic for breadth:
    the strongest alpha is always kept; each subsequent alpha earns a slot only
    if it adds something the kept set doesn't already have.

    ``corr`` is an N×N correlation matrix (any array-like). NaN correlations are
    treated as 0 (undefined → not redundant). Alphas with NaN/missing scores
    sort last.
    """
    matrix = np.asarray(corr, dtype=float)
    n = len(scores)

    def _key(i: int) -> float:
        try:
            s = float(scores[i])
        except (TypeError, ValueError):
            return -np.inf
        return s if np.isfinite(s) else -np.inf

    order = sorted(range(n), key=_key, reverse=True)
    kept: list[int] = []
    for i in order:
        redundant = False
        for j in kept:
            if i < matrix.shape[0] and j < matrix.shape[1]:
                rho = matrix[i, j]
                if np.isnan(rho):
                    rho = 0.0
                if abs(rho) > max_rho:
                    redundant = True
                    break
        if not redundant:
            kept.append(i)
    return sorted(kept)
