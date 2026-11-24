"""Shared enums for the QAMC IR."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Small compatibility helper for string-valued enums."""


class SummaryKind(StrEnum):
    DENSITY_MATRIX = "density_matrix"
    STATE_VECTOR = "state_vector"
    GROVER_SUBSPACE = "grover_subspace"


class ClassicalType(StrEnum):
    BIT = "bit"
    BOOL = "bool"
    INT = "int"


class BranchKind(StrEnum):
    DETERMINISTIC = "det"
    MEASUREMENT = "meas"
    CLASSICAL_CONTROL = "ctrl"


class AnchorKind(StrEnum):
    NODE = "node"
    FRONTIER = "frontier"
    TERMINAL = "terminal"


class ScopeKind(StrEnum):
    ANCHORS = "anchors"
    INTERVAL = "interval"


class EvaluationMode(StrEnum):
    POINTWISE_ALL = "pointwise_all"
    POINTWISE_ANY = "pointwise_any"
    AGGREGATE = "aggregate"


class PropertyKind(StrEnum):
    ASSERTION = "assertion"
    DISTRIBUTION = "distribution"
    REACHABILITY = "reachability"
    RELATION = "relation"
