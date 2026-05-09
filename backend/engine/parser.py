from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

DATA_FIELDS = {
    # Original (7)
    "open", "high", "low", "close", "volume", "returns", "vwap",
    # Price structure (7)
    "median_price", "weighted_close", "range_", "body",
    "upper_shadow", "lower_shadow", "gap",
    # Return variants (5)
    "log_returns", "abs_returns", "intraday_return",
    "overnight_return", "signed_volume",
    # Volume & liquidity (4)
    "dollar_volume", "adv20", "volume_ratio", "amihud",
    # Volatility & risk (5)
    "true_range", "atr", "realized_vol", "skewness", "kurtosis",
    # Momentum & relative (4)
    "momentum_5", "momentum_20", "close_to_high_252", "high_low_ratio",
    # ----- Phase B: extended momentum (8) -----
    "momentum_3", "momentum_10", "momentum_60", "momentum_120", "momentum_252",
    "reversal_5", "reversal_20", "momentum_z_60",
    # ----- Phase B: extended volatility (6) -----
    "realized_vol_5", "realized_vol_60", "realized_vol_120",
    "vol_of_vol_20", "parkinson_vol", "garman_klass_vol",
    # ----- Phase B: microstructure (8) -----
    "roll_spread", "kyle_lambda", "vpin_proxy",
    "up_volume_ratio", "down_volume_ratio", "turnover_ratio",
    "dollar_amihud", "corwin_schultz",
    # ----- Phase B: extended range / candle structure (6) -----
    "atr_5", "atr_60", "range_z_20", "body_to_range",
    "consecutive_up", "consecutive_down",
    # GICS classification labels (4) — string-valued, used as the second arg
    # of group_* operators.  The evaluator resolves them to (dates × tickers)
    # frames where every cell is the ticker's group label string.
    "sector", "industry_group", "industry", "sub_industry",
    # ----- Phase C: FRED macro (broadcast to every ticker per day) -----
    "vix",
    "treasury_3m_yield", "treasury_2y_yield", "treasury_10y_yield",
    "term_spread_10y_2y", "term_spread_10y_3m",
    "high_yield_spread", "baa_yield", "aaa_yield", "credit_spread_baa_aaa",
    "dxy", "wti_oil",
    # ----- Phase C: yfinance fundamentals (raw, lagged 1Q) -----
    "revenue", "gross_profit", "operating_income", "net_income", "ebitda", "eps",
    "total_assets", "total_debt", "total_equity", "cash",
    "current_assets", "current_liabilities",
    "operating_cash_flow", "capex", "free_cash_flow",
    # ----- Phase C: computed fundamentals ratios -----
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa", "debt_to_equity", "current_ratio",
    "gross_margin", "operating_margin", "fcf_yield",
}

# User-facing aliases that resolve to a canonical field name.  `range` is the
# obvious user spelling, but the canonical field is `range_` to avoid shadowing
# Python's builtin in any consumer that uses getattr-based dispatch.
DATA_FIELD_ALIASES = {"range": "range_"}


class TokenType(Enum):
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: Any = None


@dataclass
class BinaryOp:
    op: str
    left: "ASTNode"
    right: "ASTNode"


@dataclass
class UnaryOp:
    op: str
    operand: "ASTNode"


@dataclass
class FunctionCall:
    name: str
    args: list = field(default_factory=list)


@dataclass
class DataField:
    name: str


@dataclass
class Literal:
    value: float


ASTNode = Union[BinaryOp, UnaryOp, FunctionCall, DataField, Literal]


_SINGLE_CHAR_TOKENS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
}


class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def next_token(self) -> Token:
        self._skip_whitespace()
        if self.pos >= len(self.text):
            return Token(TokenType.EOF)

        ch = self.text[self.pos]

        if ch in _SINGLE_CHAR_TOKENS:
            self.pos += 1
            return Token(_SINGLE_CHAR_TOKENS[ch])

        if ch.isdigit() or ch == ".":
            start = self.pos
            seen_dot = False
            while self.pos < len(self.text) and (
                self.text[self.pos].isdigit() or self.text[self.pos] == "."
            ):
                if self.text[self.pos] == ".":
                    if seen_dot:
                        raise ValueError(
                            f"Invalid number at position {start}: multiple decimal points"
                        )
                    seen_dot = True
                self.pos += 1
            num_text = self.text[start : self.pos]
            if num_text == ".":
                raise ValueError(f"Invalid number '.' at position {start}")
            try:
                value = float(num_text) if seen_dot else int(num_text)
            except ValueError as exc:
                raise ValueError(f"Invalid number '{num_text}' at position {start}") from exc
            return Token(TokenType.NUMBER, value)

        if ch.isalpha() or ch == "_":
            start = self.pos
            while self.pos < len(self.text) and (
                self.text[self.pos].isalnum() or self.text[self.pos] == "_"
            ):
                self.pos += 1
            return Token(TokenType.IDENT, self.text[start : self.pos])

        raise ValueError(f"Unexpected character {ch!r} at position {self.pos}")

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type is TokenType.EOF:
                break
        return tokens


class Parser:
    def __init__(self, tokens: list[Token] | None = None):
        self.tokens: list[Token] = tokens or []
        self.pos: int = 0

    def parse(self, expression: str) -> ASTNode:
        self.tokens = Tokenizer(expression).tokenize()
        self.pos = 0
        node = self.parse_expression()
        if self._peek().type is not TokenType.EOF:
            raise ValueError(
                f"Unexpected token {self._peek().type.name} after expression"
            )
        return node

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _consume(self, expected: TokenType | None = None) -> Token:
        tok = self.tokens[self.pos]
        if expected is not None and tok.type is not expected:
            raise ValueError(
                f"Expected {expected.name}, got {tok.type.name}"
            )
        self.pos += 1
        return tok

    def parse_expression(self) -> ASTNode:
        node = self.parse_term()
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            tok = self._consume()
            right = self.parse_term()
            op = "+" if tok.type is TokenType.PLUS else "-"
            node = BinaryOp(op, node, right)
        return node

    def parse_term(self) -> ASTNode:
        node = self.parse_factor()
        while self._peek().type in (TokenType.STAR, TokenType.SLASH):
            tok = self._consume()
            right = self.parse_factor()
            op = "*" if tok.type is TokenType.STAR else "/"
            node = BinaryOp(op, node, right)
        return node

    def parse_factor(self) -> ASTNode:
        tok = self._peek()
        if tok.type is TokenType.MINUS:
            self._consume()
            return UnaryOp("-", self.parse_factor())
        if tok.type is TokenType.PLUS:
            self._consume()
            return self.parse_factor()
        return self.parse_atom()

    def parse_atom(self) -> ASTNode:
        tok = self._peek()

        if tok.type is TokenType.NUMBER:
            self._consume()
            return Literal(tok.value)

        if tok.type is TokenType.LPAREN:
            self._consume()
            node = self.parse_expression()
            self._consume(TokenType.RPAREN)
            return node

        if tok.type is TokenType.IDENT:
            self._consume()
            name = tok.value
            if self._peek().type is TokenType.LPAREN:
                self._consume()
                args = self.parse_args()
                self._consume(TokenType.RPAREN)
                return FunctionCall(name, args)
            if name in DATA_FIELDS or name in DATA_FIELD_ALIASES:
                # Preserve the user's spelling in the AST; the evaluator
                # resolves aliases at lookup time.
                return DataField(name)
            raise ValueError(
                f"Unknown identifier {name!r}: not a data field; "
                f"expected '(' for function call"
            )

        raise ValueError(
            f"Unexpected token {tok.type.name} at position {self.pos}"
        )

    def parse_args(self) -> list[ASTNode]:
        args: list[ASTNode] = []
        if self._peek().type is TokenType.RPAREN:
            return args
        args.append(self.parse_expression())
        while self._peek().type is TokenType.COMMA:
            self._consume()
            args.append(self.parse_expression())
        return args
