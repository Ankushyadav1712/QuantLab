"""Static analysis of parsed alpha expressions.

Catches the silent-failure cases that ruin retail backtests: negative shifts
into the future (`delay(x, -5)`, `delta(x, -1)`) and non-positive rolling
windows.  Runs against the AST before the evaluator touches it, so a flagged
expression never produces an inflated Sharpe.
"""

from __future__ import annotations

from typing import Any

from engine.parser import (
    BinaryOp,
    FunctionCall,
    Literal,
    UnaryOp,
)


# Operators where one of the args MUST be positive (a rolling window length).
# Maps op name -> argument index (0-based) of the window arg.
WINDOWED_OPERATORS: dict[str, int] = {
    "ts_mean": 1,
    "ts_std": 1,
    "ts_min": 1,
    "ts_max": 1,
    "ts_sum": 1,
    "ts_rank": 1,
    "decay_linear": 1,
    "ts_corr": 2,
    "ts_cov": 2,
}

# Operators where the second arg is a *shift* (not a window).  Negative values
# peek into the future and are flat-out look-ahead bias.
SHIFT_OPERATORS: dict[str, int] = {
    "delta": 1,
    "delay": 1,
}


def lint_ast(node) -> list[dict[str, Any]]:
    """Walk the AST and return a list of diagnostics.

    Each diagnostic is `{severity, message, op}`.  Severity is one of:
      - 'error'   — the engine should refuse to run the expression
      - 'warning' — the engine will run, but the user should be aware
    """
    diagnostics: list[dict[str, Any]] = []
    _walk(node, diagnostics)
    return diagnostics


def _walk(node, diagnostics: list[dict[str, Any]]) -> None:
    if isinstance(node, FunctionCall):
        _check_function(node, diagnostics)
        for arg in node.args:
            _walk(arg, diagnostics)
    elif isinstance(node, BinaryOp):
        _walk(node.left, diagnostics)
        _walk(node.right, diagnostics)
    elif isinstance(node, UnaryOp):
        _walk(node.operand, diagnostics)
    # Literal / DataField — leaves


def _literal_int(node) -> int | None:
    """If `node` resolves to a literal integer (possibly under a unary +/-),
    return it.  Otherwise None.  The parser turns ``-5`` into a
    ``UnaryOp('-', Literal(5))`` AST, so we unwrap that here — without this
    the linter silently misses every negative-shift look-ahead.
    """
    if isinstance(node, UnaryOp):
        if node.op == "-":
            inner = _literal_int(node.operand)
            return -inner if inner is not None else None
        if node.op == "+":
            return _literal_int(node.operand)
        return None
    if not isinstance(node, Literal):
        return None
    try:
        return int(node.value)
    except (TypeError, ValueError):
        return None


def _check_function(node: FunctionCall, diagnostics: list[dict[str, Any]]) -> None:
    name = node.name

    if name in SHIFT_OPERATORS:
        idx = SHIFT_OPERATORS[name]
        if idx >= len(node.args):
            diagnostics.append({
                "severity": "error",
                "op": name,
                "message": f"{name}() expects {idx + 1} arguments, got {len(node.args)}.",
            })
            return
        n = _literal_int(node.args[idx])
        if n is None:
            return  # Non-literal arg; can't statically check
        if n < 0:
            diagnostics.append({
                "severity": "error",
                "op": name,
                "message": (
                    f"{name}(x, {n}) peeks {abs(n)} day(s) into the future — "
                    f"this is look-ahead bias and would invalidate the backtest. "
                    f"Use a positive shift."
                ),
            })
        elif n == 0:
            diagnostics.append({
                "severity": "warning",
                "op": name,
                "message": (
                    f"{name}(x, 0) is a no-op "
                    f"({'identity' if name == 'delay' else 'always zero'}). "
                    f"Did you mean a positive shift?"
                ),
            })
        return

    if name in WINDOWED_OPERATORS:
        idx = WINDOWED_OPERATORS[name]
        if idx >= len(node.args):
            diagnostics.append({
                "severity": "error",
                "op": name,
                "message": f"{name}() expects {idx + 1} arguments, got {len(node.args)}.",
            })
            return
        d = _literal_int(node.args[idx])
        if d is None:
            return
        if d <= 0:
            diagnostics.append({
                "severity": "error",
                "op": name,
                "message": (
                    f"{name}(x, {d}) requires a positive rolling window. Got {d}."
                ),
            })
        elif d > 504:  # > ~2 trading years — usually a typo
            diagnostics.append({
                "severity": "warning",
                "op": name,
                "message": (
                    f"{name}(x, {d}) — window is very long ({d} days ≈ "
                    f"{d / 252:.1f} trading years). Make sure this is intentional."
                ),
            })
