"""Load external property bundles from text files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from qamc.external_properties.formula import parse_temporal_formula
from qamc.external_properties.model import ExternalPropertyBundle, TemporalPropertySpec
from qamc.ir.enums import EvaluationMode, PropertyKind, ScopeKind
from qamc.ir.property import ObservableSpec, PropertySpec
from qamc.loader.qam_toml import parse_qam_toml


def load_qprop_file(path: str | Path) -> ExternalPropertyBundle:
    """Load an external property bundle from a .qprop file."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return load_qprop_string(text)


def load_qprop_path(path: str | Path) -> ExternalPropertyBundle:
    """Load a property bundle from one file or a directory of property files."""

    source_path = Path(path)
    if source_path.is_dir():
        files = tuple(
            sorted(
                item
                for item in source_path.iterdir()
                if item.is_file() and item.suffix.lower() in {".qprop", ".ltl"}
            )
        )
        return merge_external_property_bundles(load_qprop_file(item) for item in files)
    return load_qprop_file(source_path)


def merge_external_property_bundles(
    bundles: Iterable[ExternalPropertyBundle],
) -> ExternalPropertyBundle:
    """Merge several external property bundles into one bundle."""

    bundle_tuple = tuple(bundles)
    return ExternalPropertyBundle(
        base_properties=tuple(
            prop for bundle in bundle_tuple for prop in bundle.base_properties
        ),
        temporal_properties=tuple(
            prop for bundle in bundle_tuple for prop in bundle.temporal_properties
        ),
    )


def load_qprop_string(text: str) -> ExternalPropertyBundle:
    """Load an external property bundle from an in-memory string."""

    data = _parse_qprop_text(text)
    return _normalize_external_properties(data)


def _parse_qprop_text(text: str) -> Mapping[str, object]:
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("Empty qprop source")
    if stripped[0] == "{":
        return json.loads(text)
    return parse_qam_toml(text)


def _normalize_external_properties(data: Mapping[str, object]) -> ExternalPropertyBundle:
    root = _as_mapping(data, "root")
    base_properties = tuple(
        _normalize_property(item, "properties[]") for item in _as_sequence(root.get("properties", ()), "properties")
    )
    temporal_properties = tuple(
        _normalize_temporal_property(item, "temporal_properties[]")
        for item in _as_sequence(root.get("temporal_properties", ()), "temporal_properties")
    )
    return ExternalPropertyBundle(
        base_properties=base_properties,
        temporal_properties=temporal_properties,
    )


def _normalize_property(raw: object, location: str) -> PropertySpec:
    mapping = _as_mapping(raw, location)
    observable_raw = _as_mapping(mapping.get("observable", {}), f"{location}.observable")
    return PropertySpec(
        id=str(mapping["id"]),
        kind=PropertyKind(mapping["kind"]),
        scope_kind=ScopeKind(mapping["scope_kind"]),
        scope_value=_normalize_literal(mapping["scope_value"]),
        mode=EvaluationMode(mapping.get("mode", EvaluationMode.AGGREGATE.value)),
        observable=ObservableSpec(
            kind=str(observable_raw.get("kind", "quantum")),
            classical_vars=_tuple_of_str(observable_raw.get("classical_vars", ())),
            qubits=_tuple_of_str(observable_raw.get("qubits", ())),
            basis=_optional_str(observable_raw.get("basis")),
        ),
        expectation=_normalize_expectation(mapping.get("expectation")),
    )


def _normalize_temporal_property(raw: object, location: str) -> TemporalPropertySpec:
    mapping = _as_mapping(raw, location)
    interval_raw = mapping.get("interval")
    interval = None
    if interval_raw is not None:
        interval_parts = _tuple_of_str(interval_raw)
        if len(interval_parts) != 2:
            raise ValueError(f"{location}.interval must contain exactly two anchors.")
        interval = (interval_parts[0], interval_parts[1])
    formula_source = mapping.get("formula")
    formula = None
    if formula_source is not None:
        formula = parse_temporal_formula(str(formula_source))
    operator = str(mapping.get("operator", "formula" if formula is not None else ""))
    if not operator:
        raise ValueError(f"{location} requires either operator or formula.")
    return TemporalPropertySpec(
        id=str(mapping["id"]),
        operator=operator,
        interval=interval,
        operands=_tuple_of_str(mapping.get("operands", ())),
        formula=formula,
    )


def _normalize_expectation(value: object) -> object:
    normalized = _normalize_literal(value)
    if isinstance(normalized, dict):
        return {_coerce_distribution_key(key): inner for key, inner in normalized.items()}
    return normalized


def _coerce_distribution_key(key: object) -> object:
    if isinstance(key, str):
        if key.lstrip("-").isdigit():
            return int(key)
        if "," in key:
            parts = tuple(part.strip() for part in key.split(","))
            if all(part.lstrip("-").isdigit() for part in parts):
                return tuple(int(part) for part in parts)
    return key


def _normalize_literal(value: object) -> object:
    if isinstance(value, list):
        return tuple(_normalize_literal(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_literal(item) for item in value)
    if isinstance(value, Mapping):
        return {_normalize_literal(key): _normalize_literal(inner) for key, inner in value.items()}
    return value


def _tuple_of_str(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_sequence(value, "tuple_of_str"))


def _as_mapping(value: object, location: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at {location}, got {type(value).__name__}.")
    return value


def _as_sequence(value: object, location: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"Expected sequence at {location}, got scalar string.")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected sequence at {location}, got {type(value).__name__}.")
    return tuple(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
