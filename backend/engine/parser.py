from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

DATA_FIELDS = {"open", "high", "low", "close", "volume", "returns", "vwap"}


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
            if name in DATA_FIELDS:
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
