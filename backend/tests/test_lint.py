import pytest

from engine.lint import lint_ast
from engine.parser import Parser


def _lint(expr: str):
    return lint_ast(Parser().parse(expr))


# ---------- look-ahead errors (the headline reason for the linter) ----------


@pytest.mark.parametrize(
    "expr,op",
    [
        ("delay(close, -1)", "delay"),
        ("delay(close, -5)", "delay"),
        ("delta(close, -1)", "delta"),
        ("delta(returns, -10)", "delta"),
        ("rank(delay(close, -1))", "delay"),  # nested
        ("delta(rank(close), -3) * volume", "delta"),  # under a binary op
    ],
)
def test_negative_shift_is_an_error(expr, op):
    diags = _lint(expr)
    errs = [d for d in diags if d["severity"] == "error"]
    assert any(d["op"] == op for d in errs), f"Expected {op} look-ahead error in {expr!r}: {diags}"


# ---------- non-positive windows on rolling ops ----------


@pytest.mark.parametrize(
    "expr,op",
    [
        ("ts_mean(close, 0)", "ts_mean"),
        ("ts_mean(close, -5)", "ts_mean"),
        ("ts_std(returns, 0)", "ts_std"),
        ("decay_linear(close, 0)", "decay_linear"),
        ("ts_corr(close, volume, 0)", "ts_corr"),
        ("ts_corr(close, volume, -3)", "ts_corr"),
    ],
)
def test_non_positive_window_is_an_error(expr, op):
    diags = _lint(expr)
    errs = [d for d in diags if d["severity"] == "error"]
    assert any(d["op"] == op for d in errs)


# ---------- zero-shift warnings (technically allowed but a no-op) ----------


@pytest.mark.parametrize("expr,op", [("delay(close, 0)", "delay"), ("delta(close, 0)", "delta")])
def test_zero_shift_is_a_warning(expr, op):
    diags = _lint(expr)
    warns = [d for d in diags if d["severity"] == "warning"]
    assert any(d["op"] == op for d in warns)


# ---------- very-long-window warnings ----------


def test_excessively_long_window_warns():
    diags = _lint("ts_mean(close, 5000)")
    warns = [d for d in diags if d["severity"] == "warning"]
    assert any(d["op"] == "ts_mean" for d in warns)


# ---------- valid expressions — no diagnostics ----------


@pytest.mark.parametrize(
    "expr",
    [
        "rank(close)",
        "delta(close, 5)",
        "delay(close, 1)",
        "ts_mean(close, 20)",
        "ts_corr(rank(close), rank(volume), 60)",
        "decay_linear(rank(momentum_20), 20)",
        "-rank(delta(close, 5)) * ts_std(returns, 20)",
    ],
)
def test_valid_expressions_have_no_diagnostics(expr):
    diags = _lint(expr)
    assert diags == [], f"Expected clean lint, got {diags}"


# ---------- non-literal window args are not flagged ----------


def test_non_literal_window_is_not_checked():
    """If the user computes the window dynamically the linter can't check it
    statically; should NOT produce a false-positive."""
    # Note: we don't actually support dynamic windows in the engine, but
    # syntactically `ts_mean(close, ts_mean(volume, 5))` parses, and the linter
    # should silently ignore it rather than misfire.
    diags = _lint("ts_mean(close, ts_mean(volume, 5))")
    # The inner ts_mean(volume, 5) is fine; the outer's window is non-literal.
    # Neither should trigger an error.
    assert all(d["severity"] != "error" for d in diags), diags
