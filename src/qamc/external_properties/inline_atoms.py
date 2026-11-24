"""Translate inline atomic property formulas into QAMC property specs."""

from __future__ import annotations

import re

from qamc.ir.enums import EvaluationMode, PropertyKind, ScopeKind
from qamc.ir.property import ObservableSpec, PropertySpec


_DIST_RE = re.compile(r"^dist\((?P<args>[^)]*)\)\s*==\s*\{(?P<body>.*)\}$")
_CDIST_RE = re.compile(r"^cdist\((?P<args>[^)]*)\)\s*==\s*\{(?P<body>.*)\}$")
_REL_RE = re.compile(r"^rel\((?P<args>[^)]*)\)\s*==\s*(?P<relation>[A-Za-z_][A-Za-z0-9_]*)$")
_CLASSICAL_IN_RE = re.compile(r"^(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+\{(?P<body>.*)\}$")
_CLASSICAL_EQ_RE = re.compile(r"^(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*==\s*(?P<value>.+)$")
_BASIS_NAMES = {"X", "Y", "Z"}


def inline_atom_to_property(
    source: str,
    *,
    property_id: str,
    anchor_id: str,
) -> PropertySpec:
    """Translate one inline atom into a temporary property specification."""

    normalized = source.strip()
    dist_match = _DIST_RE.match(normalized)
    if dist_match is not None:
        qubits, basis = _parse_qubits_and_basis(dist_match.group("args"))
        return PropertySpec(
            id=property_id,
            kind=PropertyKind.DISTRIBUTION,
            scope_kind=ScopeKind.ANCHORS,
            scope_value=(anchor_id,),
            mode=EvaluationMode.AGGREGATE,
            observable=ObservableSpec(kind="quantum", qubits=qubits, basis=basis),
            expectation=_parse_distribution(dist_match.group("body")),
        )

    cdist_match = _CDIST_RE.match(normalized)
    if cdist_match is not None:
        classical_vars = tuple(
            item.strip()
            for item in _split_top_level(cdist_match.group("args"))
            if item.strip()
        )
        if not classical_vars:
            raise ValueError("Classical distribution observable requires at least one variable.")
        return PropertySpec(
            id=property_id,
            kind=PropertyKind.DISTRIBUTION,
            scope_kind=ScopeKind.ANCHORS,
            scope_value=(anchor_id,),
            mode=EvaluationMode.AGGREGATE,
            observable=ObservableSpec(kind="classical", classical_vars=classical_vars),
            expectation=_parse_distribution(cdist_match.group("body")),
        )

    rel_match = _REL_RE.match(normalized)
    if rel_match is not None:
        qubits, basis = _parse_qubits_and_basis(rel_match.group("args"))
        return PropertySpec(
            id=property_id,
            kind=PropertyKind.RELATION,
            scope_kind=ScopeKind.ANCHORS,
            scope_value=(anchor_id,),
            mode=EvaluationMode.AGGREGATE,
            observable=ObservableSpec(kind="quantum", qubits=qubits, basis=basis),
            expectation=(rel_match.group("relation"),),
        )

    classical_in_match = _CLASSICAL_IN_RE.match(normalized)
    if classical_in_match is not None:
        return PropertySpec(
            id=property_id,
            kind=PropertyKind.ASSERTION,
            scope_kind=ScopeKind.ANCHORS,
            scope_value=(anchor_id,),
            mode=EvaluationMode.POINTWISE_ALL,
            observable=ObservableSpec(
                kind="classical",
                classical_vars=(classical_in_match.group("var"),),
            ),
            expectation=("in", tuple(_parse_scalar(item) for item in _split_top_level(classical_in_match.group("body")))),
        )

    classical_eq_match = _CLASSICAL_EQ_RE.match(normalized)
    if classical_eq_match is not None:
        return PropertySpec(
            id=property_id,
            kind=PropertyKind.ASSERTION,
            scope_kind=ScopeKind.ANCHORS,
            scope_value=(anchor_id,),
            mode=EvaluationMode.POINTWISE_ALL,
            observable=ObservableSpec(
                kind="classical",
                classical_vars=(classical_eq_match.group("var"),),
            ),
            expectation=("eq", _parse_scalar(classical_eq_match.group("value"))),
        )

    raise ValueError(f"Unsupported inline atomic property: {source!r}")


def _parse_qubits_and_basis(args_source: str) -> tuple[tuple[str, ...], str]:
    parts = tuple(part.strip() for part in _split_top_level(args_source) if part.strip())
    if not parts:
        raise ValueError("Quantum observable requires at least one qubit.")
    if parts[-1] in _BASIS_NAMES:
        qubits = parts[:-1]
        basis = parts[-1]
    else:
        qubits = parts
        basis = "Z"
    if not qubits:
        raise ValueError("Quantum observable requires at least one qubit.")
    return qubits, basis


def _parse_distribution(body: str) -> dict[object, float]:
    distribution: dict[object, float] = {}
    for entry in _split_top_level(body):
        if not entry.strip():
            continue
        key_source, value_source = _split_key_value(entry)
        distribution[_parse_outcome(key_source)] = float(value_source.strip())
    return distribution


def _split_key_value(entry: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(entry):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == ":" and depth == 0:
            return entry[:index], entry[index + 1 :]
    raise ValueError(f"Expected key:value distribution entry, got {entry!r}.")


def _parse_outcome(source: str) -> object:
    normalized = source.strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        return tuple(int(part.strip()) for part in _split_top_level(normalized[1:-1]))
    return int(normalized)


def _parse_scalar(source: str) -> object:
    normalized = source.strip()
    if normalized in {"true", "True"}:
        return True
    if normalized in {"false", "False"}:
        return False
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    try:
        return float(normalized)
    except ValueError:
        return normalized


def _split_top_level(source: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(source):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(source[start:index].strip())
            start = index + 1
    parts.append(source[start:].strip())
    return tuple(part for part in parts if part)
