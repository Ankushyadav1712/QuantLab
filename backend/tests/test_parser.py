import pytest
from engine.parser import (
    BinaryOp,
    DataField,
    FunctionCall,
    Literal,
    Parser,
    Tokenizer,
    TokenType,
    UnaryOp,
)


def test_tokenize_basic():
    toks = Tokenizer("rank(close, 5)").tokenize()
    types = [t.type for t in toks]
    assert types == [
        TokenType.IDENT,
        TokenType.LPAREN,
        TokenType.IDENT,
        TokenType.COMMA,
        TokenType.NUMBER,
        TokenType.RPAREN,
        TokenType.EOF,
    ]
    assert toks[0].value == "rank"
    assert toks[2].value == "close"
    assert toks[4].value == 5


def test_parse_data_field():
    ast = Parser().parse("close")
    assert isinstance(ast, DataField) and ast.name == "close"


def test_parse_simple_function():
    ast = Parser().parse("rank(close)")
    assert isinstance(ast, FunctionCall) and ast.name == "rank"
    assert len(ast.args) == 1
    assert isinstance(ast.args[0], DataField) and ast.args[0].name == "close"


def test_parse_rank_delta_close_5():
    ast = Parser().parse("rank(delta(close, 5))")
    assert isinstance(ast, FunctionCall) and ast.name == "rank"
    inner = ast.args[0]
    assert isinstance(inner, FunctionCall) and inner.name == "delta"
    assert isinstance(inner.args[0], DataField) and inner.args[0].name == "close"
    assert isinstance(inner.args[1], Literal) and inner.args[1].value == 5


def test_parse_unary_minus_times_volume():
    # -rank(close) * volume  ==  (-(rank(close))) * volume
    ast = Parser().parse("-rank(close)*volume")
    assert isinstance(ast, BinaryOp) and ast.op == "*"
    assert isinstance(ast.left, UnaryOp) and ast.left.op == "-"
    assert isinstance(ast.left.operand, FunctionCall)
    assert ast.left.operand.name == "rank"
    assert isinstance(ast.right, DataField) and ast.right.name == "volume"


def test_parse_nested_functions():
    ast = Parser().parse("ts_mean(rank(delta(close, 1)), 10)")
    assert isinstance(ast, FunctionCall) and ast.name == "ts_mean"
    assert isinstance(ast.args[1], Literal) and ast.args[1].value == 10
    inner = ast.args[0]
    assert isinstance(inner, FunctionCall) and inner.name == "rank"
    inner2 = inner.args[0]
    assert isinstance(inner2, FunctionCall) and inner2.name == "delta"


def test_parse_full_example_expression():
    ast = Parser().parse("-rank(delta(close, 5)) * ts_std(returns, 20)")
    assert isinstance(ast, BinaryOp) and ast.op == "*"
    assert isinstance(ast.left, UnaryOp) and ast.left.op == "-"
    inner = ast.left.operand
    assert isinstance(inner, FunctionCall) and inner.name == "rank"
    right = ast.right
    assert isinstance(right, FunctionCall) and right.name == "ts_std"


def test_parse_arithmetic_precedence():
    # 1 + 2 * 3  ->  BinaryOp(+, 1, BinaryOp(*, 2, 3))
    ast = Parser().parse("1 + 2 * 3")
    assert isinstance(ast, BinaryOp) and ast.op == "+"
    assert isinstance(ast.left, Literal) and ast.left.value == 1
    assert isinstance(ast.right, BinaryOp) and ast.right.op == "*"


def test_parse_parentheses_change_precedence():
    ast = Parser().parse("(1 + 2) * 3")
    assert isinstance(ast, BinaryOp) and ast.op == "*"
    assert isinstance(ast.left, BinaryOp) and ast.left.op == "+"


def test_parse_decimal_literal():
    ast = Parser().parse("0.5 * close")
    assert isinstance(ast, BinaryOp) and ast.op == "*"
    assert isinstance(ast.left, Literal) and ast.left.value == 0.5


@pytest.mark.parametrize(
    "expr",
    [
        "rank(",
        "rank(close,",
        "rank close)",
        "1 +",
        "()",
        "rank()*",
        "@invalid",
        "1.2.3",
        "rank(close,)",
        "* close",
        "",  # empty string
        "   ",  # whitespace only
        "rank(close",  # missing closing paren
        "ts_mean(close, 20",  # missing closing paren (multi-arg)
        "((close)",  # unbalanced extra paren
    ],
)
def test_invalid_syntax_raises(expr):
    with pytest.raises(ValueError):
        Parser().parse(expr)


# ---------- Spec-required additions ----------


# Each canonical-call expression should parse cleanly.  These cover all 23
# operators exposed by the engine; the parser itself doesn't validate that the
# name is a known operator, but each one must round-trip as a FunctionCall.
ALL_OPERATOR_EXPRESSIONS = [
    "ts_mean(close, 20)",
    "ts_std(returns, 20)",
    "ts_min(close, 20)",
    "ts_max(close, 20)",
    "ts_sum(volume, 20)",
    "ts_rank(close, 20)",
    "delta(close, 5)",
    "delay(close, 1)",
    "decay_linear(close, 10)",
    "ts_corr(close, volume, 20)",
    "ts_cov(close, volume, 20)",
    "rank(close)",
    "zscore(close)",
    "demean(close)",
    "scale(close)",
    "normalize(close)",
    "abs(close)",
    "log(close)",
    "sign(close)",
    "power(close, 2)",
    "max(close, open)",
    "min(close, open)",
    "if_else(close, open, high)",
]


@pytest.mark.parametrize("expr", ALL_OPERATOR_EXPRESSIONS)
def test_all_operators_parse(expr):
    ast = Parser().parse(expr)
    assert isinstance(ast, FunctionCall)
    assert ast.name == expr.split("(", 1)[0]


def test_precedence_2_plus_3_times_4():
    # "2 + 3 * 4"  ->  BinaryOp(+, 2, BinaryOp(*, 3, 4))
    ast = Parser().parse("2 + 3 * 4")
    assert isinstance(ast, BinaryOp) and ast.op == "+"
    assert isinstance(ast.left, Literal) and ast.left.value == 2
    assert isinstance(ast.right, BinaryOp) and ast.right.op == "*"
    assert ast.right.left.value == 3 and ast.right.right.value == 4


def test_unary_minus_rank_close():
    ast = Parser().parse("-rank(close)")
    assert isinstance(ast, UnaryOp) and ast.op == "-"
    assert isinstance(ast.operand, FunctionCall) and ast.operand.name == "rank"
    assert isinstance(ast.operand.args[0], DataField)
    assert ast.operand.args[0].name == "close"


def test_nested_ts_corr_with_two_ranks():
    ast = Parser().parse("ts_corr(rank(close), rank(volume), 10)")
    assert isinstance(ast, FunctionCall) and ast.name == "ts_corr"
    assert len(ast.args) == 3

    a, b, d = ast.args
    assert isinstance(a, FunctionCall) and a.name == "rank"
    assert isinstance(a.args[0], DataField) and a.args[0].name == "close"
    assert isinstance(b, FunctionCall) and b.name == "rank"
    assert isinstance(b.args[0], DataField) and b.args[0].name == "volume"
    assert isinstance(d, Literal) and d.value == 10
