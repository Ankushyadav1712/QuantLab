"""Parameter sweep expansion.

A sweep token has the form ``{a..b}`` (step 1) or ``{a..b:s}`` (explicit step
``s``).  Tokens are expanded to a list of values; multiple tokens form a
cartesian product.  Each combination is substituted back into the expression
to produce a concrete, parser-valid alpha string.

Examples:
    "rank(momentum_{5..30:5})"
        → 6 expressions, sweep over [5, 10, 15, 20, 25, 30]
    "decay_linear(rank(momentum_{5..20:5}), {10..40:10})"
        → 4 × 4 = 16 expressions

The expansion happens at the *string* level — no parser changes required.
This keeps the parser unchanged and makes the feature trivially composable
with any existing expression syntax.  The downside is that sweep tokens are
substituted blindly; the DSL has no string literals so there's no ambiguity
in practice.
"""

from __future__ import annotations

import re
from itertools import product
from typing import Any

# Matches {a..b} and {a..b:s} where a, b, s are signed integers or decimals.
SWEEP_RE = re.compile(
    r"\{(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)(?::(-?\d+(?:\.\d+)?))?\}"
)


def _generate_values(start: float, end: float, step: float) -> list[int | float]:
    """Inclusive range from ``start`` to ``end`` with ``step``.

    Returns ints if all three inputs are integer-valued, else floats rounded
    to 10 decimals (kills accumulated float drift in the comparison).
    """
    if step <= 0:
        raise ValueError(f"Sweep step must be positive (got {step})")
    if end < start:
        raise ValueError(f"Sweep end must be >= start (got {start}..{end})")

    all_int = (
        float(start).is_integer() and float(end).is_integer() and float(step).is_integer()
    )
    values: list[int | float] = []
    v = start
    # Add a small epsilon to absorb float rounding so the endpoint is included
    while v <= end + 1e-9:
        values.append(int(round(v)) if all_int else round(v, 10))
        v += step
    return values


def has_sweep_syntax(expression: str) -> bool:
    """Quick check used by the frontend to switch between Run/Sweep buttons."""
    return SWEEP_RE.search(expression) is not None


def expand_sweeps(expression: str, max_combinations: int = 50) -> dict[str, Any]:
    """Expand sweep tokens in ``expression`` into a list of concrete strings.

    Returns:
        ``{expressions, dimensions, total}`` where:
          - ``expressions`` is the list of concrete expressions to run
          - ``dimensions`` is the per-sweep metadata (token, values, position)
          - ``total`` is len(expressions) — convenience for the API

    Raises:
        ValueError if no sweep tokens are present, if any sweep is degenerate,
        or if the cartesian product would exceed ``max_combinations``.
    """
    matches = list(SWEEP_RE.finditer(expression))
    if not matches:
        raise ValueError(
            "Expression has no sweep tokens. Use {a..b} or {a..b:s} syntax."
        )

    dimensions: list[dict[str, Any]] = []
    value_lists: list[list[int | float]] = []
    for i, m in enumerate(matches):
        a = float(m.group(1))
        b = float(m.group(2))
        s = float(m.group(3)) if m.group(3) else 1.0
        values = _generate_values(a, b, s)
        if not values:
            raise ValueError(f"Sweep {m.group(0)} produced no values")
        dimensions.append(
            {
                "index": i,
                "token": m.group(0),
                "start": a if not float(a).is_integer() else int(a),
                "end": b if not float(b).is_integer() else int(b),
                "step": s if not float(s).is_integer() else int(s),
                "values": values,
            }
        )
        value_lists.append(values)

    combos = list(product(*value_lists))
    if len(combos) > max_combinations:
        raise ValueError(
            f"Sweep expands to {len(combos)} combinations, exceeds "
            f"max_combinations={max_combinations}. Tighten the ranges or "
            f"increase the limit."
        )

    expanded: list[str] = []
    for combo in combos:
        # Walk the original string once, splicing each match span with the
        # corresponding combo value.  Done in order so the ranges line up.
        out: list[str] = []
        last_end = 0
        for m, v in zip(matches, combo):
            out.append(expression[last_end : m.start()])
            out.append(str(v))
            last_end = m.end()
        out.append(expression[last_end:])
        expanded.append("".join(out))

    return {
        "expressions": expanded,
        "dimensions": dimensions,
        "total": len(expanded),
    }


def combo_for_index(i: int, expansion: dict) -> dict[str, Any]:
    """Recover the per-dimension values for the i-th expansion.

    ``itertools.product`` orders combos so the rightmost dimension varies
    fastest (nested-for-loop reading order).  This re-derives that ordering
    from the dimension value lists so the API can return a clean
    ``{token: value}`` dict per cell without threading combo tuples through.
    """
    dims = expansion["dimensions"]
    if not dims:
        return {}
    sizes = [len(d["values"]) for d in dims]
    idx = i
    out: dict[str, Any] = {}
    for d, size in zip(reversed(dims), reversed(sizes)):
        out[d["token"]] = d["values"][idx % size]
        idx //= size
    return out
