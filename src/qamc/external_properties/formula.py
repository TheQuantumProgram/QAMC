"""Temporal property formula syntax for external QAMC properties."""

from __future__ import annotations

from dataclasses import dataclass
import re


class TemporalFormula:
    """Base class for temporal property formulas."""


@dataclass(frozen=True, slots=True)
class TrueFormula(TemporalFormula):
    def __str__(self) -> str:
        return "true"


@dataclass(frozen=True, slots=True)
class AtomFormula(TemporalFormula):
    property_id: str

    def __str__(self) -> str:
        return self.property_id


@dataclass(frozen=True, slots=True)
class InlineAtomFormula(TemporalFormula):
    source: str

    def __str__(self) -> str:
        return f"[{self.source}]"


@dataclass(frozen=True, slots=True)
class AnchorFormula(TemporalFormula):
    anchor_id: str
    operand: TemporalFormula

    def __str__(self) -> str:
        return f"@{self.anchor_id} {_format_unary_operand(self.operand)}"


@dataclass(frozen=True, slots=True)
class NotFormula(TemporalFormula):
    operand: TemporalFormula

    def __str__(self) -> str:
        return f"!{_format_unary_operand(self.operand)}"


@dataclass(frozen=True, slots=True)
class AndFormula(TemporalFormula):
    left: TemporalFormula
    right: TemporalFormula

    def __str__(self) -> str:
        return f"{_format_binary_operand(self.left)} && {_format_binary_operand(self.right)}"


@dataclass(frozen=True, slots=True)
class NextFormula(TemporalFormula):
    operand: TemporalFormula

    def __str__(self) -> str:
        return f"X {_format_unary_operand(self.operand)}"


@dataclass(frozen=True, slots=True)
class UntilFormula(TemporalFormula):
    left: TemporalFormula
    right: TemporalFormula

    def __str__(self) -> str:
        return f"{_format_binary_operand(self.left)} U {_format_binary_operand(self.right)}"


@dataclass(frozen=True, slots=True)
class EventuallyFormula(TemporalFormula):
    operand: TemporalFormula

    def __str__(self) -> str:
        return f"F {_format_unary_operand(self.operand)}"


@dataclass(frozen=True, slots=True)
class AlwaysFormula(TemporalFormula):
    operand: TemporalFormula

    def __str__(self) -> str:
        return f"G {_format_unary_operand(self.operand)}"


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNARY_OPERATORS = {"!", "not", "X", "F", "G"}
_AND_OPERATORS = {"&&", "and"}
_UNTIL_OPERATORS = {"U", "until"}


def parse_temporal_formula(source: str) -> TemporalFormula:
    """Parse a compact temporal formula over base property identifiers."""

    tokens = _tokenize(source)
    parser = _FormulaParser(tokens)
    formula = parser.parse()
    if parser.peek() is not None:
        raise ValueError(f"Unexpected token in temporal formula: {parser.peek()!r}.")
    return formula


def _tokenize(source: str) -> tuple[str, ...]:
    tokens: list[str] = []
    offset = 0
    while offset < len(source):
        if source[offset].isspace():
            offset += 1
            continue
        if not source[offset:].strip():
            break
        if source.startswith("&&", offset):
            tokens.append("&&")
            offset += 2
            continue
        char = source[offset]
        if char in {"(", ")", "!"}:
            tokens.append(char)
            offset += 1
            continue
        if char == "@":
            match = _IDENT_RE.match(source, offset + 1)
            if match is None:
                raise ValueError(f"Expected anchor identifier near {source[offset:]!r}.")
            tokens.append(f"@{match.group(0)}")
            offset = match.end()
            continue
        if char == "[":
            end = source.find("]", offset + 1)
            if end < 0:
                raise ValueError("Unclosed inline atomic property in temporal formula.")
            inline_source = source[offset + 1 : end].strip()
            if not inline_source:
                raise ValueError("Inline atomic property must not be empty.")
            tokens.append(f"[{inline_source}]")
            offset = end + 1
            continue
        match = _IDENT_RE.match(source, offset)
        if match is None:
            raise ValueError(f"Invalid token in temporal formula near {source[offset:]!r}.")
        tokens.append(match.group(0))
        offset = match.end()
    if not tokens:
        raise ValueError("Temporal formula must not be empty.")
    return tuple(tokens)


class _FormulaParser:
    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> TemporalFormula:
        return self.parse_until()

    def parse_until(self) -> TemporalFormula:
        left = self.parse_and()
        while self.peek() in _UNTIL_OPERATORS:
            self.advance()
            right = self.parse_until()
            left = UntilFormula(left, right)
        return left

    def parse_and(self) -> TemporalFormula:
        left = self.parse_unary()
        while self.peek() in _AND_OPERATORS:
            self.advance()
            right = self.parse_unary()
            left = AndFormula(left, right)
        return left

    def parse_unary(self) -> TemporalFormula:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of temporal formula.")
        if token in _UNARY_OPERATORS:
            self.advance()
            operand = self.parse_unary()
            if token in {"!", "not"}:
                return NotFormula(operand)
            if token == "X":
                return NextFormula(operand)
            if token == "F":
                return EventuallyFormula(operand)
            if token == "G":
                return AlwaysFormula(operand)
        if token.startswith("@"):
            self.advance()
            return AnchorFormula(token[1:], self.parse_unary())
        if token.startswith("[") and token.endswith("]"):
            self.advance()
            return InlineAtomFormula(token[1:-1])
        if token == "(":
            self.advance()
            inner = self.parse_until()
            self.expect(")")
            return inner
        if token == "true":
            self.advance()
            return TrueFormula()
        if token in {")", "&&", "and", "U", "until"}:
            raise ValueError(f"Unexpected token in temporal formula: {token!r}.")
        self.advance()
        return AtomFormula(token)

    def expect(self, token: str) -> None:
        actual = self.peek()
        if actual != token:
            raise ValueError(f"Expected {token!r}, got {actual!r}.")
        self.advance()

    def peek(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def advance(self) -> str:
        token = self.tokens[self.position]
        self.position += 1
        return token


def _format_unary_operand(formula: TemporalFormula) -> str:
    if isinstance(formula, (AtomFormula, InlineAtomFormula, TrueFormula)):
        return str(formula)
    return f"({formula})"


def _format_binary_operand(formula: TemporalFormula) -> str:
    if isinstance(formula, (AtomFormula, InlineAtomFormula, TrueFormula, NotFormula, NextFormula, EventuallyFormula, AlwaysFormula, AnchorFormula)):
        return str(formula)
    return f"({formula})"
