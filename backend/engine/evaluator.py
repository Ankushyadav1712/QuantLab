from __future__ import annotations

import pandas as pd

from engine import operators as ops
from engine.parser import (
    BinaryOp,
    DataField,
    DATA_FIELD_ALIASES,
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
}


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
                    f"Unknown data field: {node.name!r} "
                    f"(available: {sorted(self.data.keys())})"
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
            fn = _resolve_function(node.name)
            args = [self._eval(a) for a in node.args]
            return fn(*args)

        raise ValueError(f"Unknown AST node type: {type(node).__name__}")
