from __future__ import annotations

import pandas as pd

from engine import operators as ops
from engine.parser import (
    DATA_FIELD_ALIASES,
    BinaryOp,
    DataField,
    FunctionCall,
    Literal,
    Parser,
    UnaryOp,
)

_FUNCTION_NAME_REMAP = {
    "abs": "op_abs",
    "log": "op_log",
    "sign": "op_sign",
    "max": "op_max",
    "min": "op_min",
    # Phase A arithmetic additions — Python builtins or pandas method names
    # we don't want to shadow at module scope.
    "exp": "op_exp",
    "sqrt": "op_sqrt",
    "mod": "op_mod",
    "equal": "op_equal",
    # Comparison operators — return 1.0/0.0 so they compose with arithmetic
    # and slot into trade_when/when/where as boolean conditions.
    "less": "op_less",
    "greater": "op_greater",
    "less_eq": "op_less_eq",
    "greater_eq": "op_greater_eq",
    "not_equal": "op_not_equal",
}

# Functions handled directly by the evaluator (not dispatched to operators.py)
# because they need access to ``self.data`` to fetch a named field:
#   * adv(d)        → ts_mean(dollar_volume, d)
#   * cap_weight(x) → operators.cap_weight(x, market_cap)
# Kept here so operators.py stays a pure-function module with no awareness
# of where data is stored.
_DATA_AWARE_FUNCTIONS = frozenset({"adv", "cap_weight"})


def _resolve_function(name: str):
    actual = _FUNCTION_NAME_REMAP.get(name, name)
    fn = getattr(ops, actual, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Unknown function: {name!r}")
    return fn


class AlphaEvaluator:
    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data
        self._parser = Parser()

    def evaluate(self, expression: str) -> pd.DataFrame:
        ast = self._parser.parse(expression)
        return self._eval(ast)

    def _eval(self, node):
        if isinstance(node, Literal):
            return node.value

        if isinstance(node, DataField):
            # Resolve user-facing aliases (e.g. `range` -> `range_`).
            actual = DATA_FIELD_ALIASES.get(node.name, node.name)
            if actual not in self.data:
                raise ValueError(
                    f"Unknown data field: {node.name!r} (available: {sorted(self.data.keys())})"
                )
            return self.data[actual]

        if isinstance(node, UnaryOp):
            value = self._eval(node.operand)
            if node.op == "-":
                return -value
            if node.op == "+":
                return value
            raise ValueError(f"Unknown unary operator: {node.op!r}")

        if isinstance(node, BinaryOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                return left / right
            raise ValueError(f"Unknown binary operator: {node.op!r}")

        if isinstance(node, FunctionCall):
            # Data-aware ops need a field from self.data injected — handle them
            # before the generic dispatch so operators.py stays pure.
            if node.name in _DATA_AWARE_FUNCTIONS:
                return self._eval_data_aware(node)
            fn = _resolve_function(node.name)
            args = [self._eval(a) for a in node.args]
            return fn(*args)

        raise ValueError(f"Unknown AST node type: {type(node).__name__}")

    def _eval_data_aware(self, node: FunctionCall):
        """Dispatch for ops that need a data-dict lookup.

        Kept as a separate method (not extra entries in _FUNCTION_NAME_REMAP)
        because these ops have non-standard signatures from the user's
        perspective: ``adv(20)`` looks like a one-arg call but is actually a
        rolling-mean of dollar_volume, and ``cap_weight(x)`` looks one-arg
        but secretly needs market_cap.  Trying to model that in the regular
        dispatcher would leak data-layer concerns into operators.py.
        """
        if node.name == "adv":
            if len(node.args) != 1:
                raise ValueError(f"adv() takes 1 argument (window d), got {len(node.args)}")
            d = self._eval(node.args[0])
            if not isinstance(d, int | float) or int(d) < 1:
                raise ValueError(f"adv(d): d must be a positive integer, got {d!r}")
            dv = self.data.get("dollar_volume")
            if dv is None:
                raise ValueError("adv(d) requires the 'dollar_volume' field, which is not loaded")
            return ops.ts_mean(dv, int(d))

        if node.name == "cap_weight":
            if len(node.args) != 1:
                raise ValueError(f"cap_weight() takes 1 argument (signal), got {len(node.args)}")
            x = self._eval(node.args[0])
            mc = self.data.get("market_cap")
            if mc is None:
                raise ValueError(
                    "cap_weight(x) requires the 'market_cap' field, which is not loaded"
                )
            return ops.cap_weight(x, mc)

        # Defensive — node.name should always be in _DATA_AWARE_FUNCTIONS to
        # have reached this method.
        raise ValueError(f"Internal error: no data-aware handler for {node.name!r}")
