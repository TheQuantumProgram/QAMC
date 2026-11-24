"""Normalization from structured text data to definition-layer QAMC IR."""

from __future__ import annotations

from collections.abc import Mapping
import re

from qamc.ir.enums import (
    AnchorKind,
    BranchKind,
    ClassicalType,
    EvaluationMode,
    PropertyKind,
    ScopeKind,
    SummaryKind,
)
from qamc.ir.model import (
    AbstractUnitSpec,
    AnchorSpec,
    BranchSpec,
    ClassicalProfileSpec,
    ClassicalVarDecl,
    Defaults,
    ExecutionSpec,
    GateDeclaration,
    GateSpec,
    InitialBindingSpec,
    QAMModel,
    StateClassSpec,
    SystemModel,
    TraceSpec,
    TraceStepSpec,
    TransitionClassSpec,
)
from qamc.ir.property import ObservableSpec, PropertySpec


def normalize_qam_data(data: Mapping[str, object]) -> QAMModel:
    """Convert structured mapping data into a typed ``QAMModel``."""

    root = _as_mapping(data, "root")
    system_raw = _as_mapping(root.get("system_model", root.get("system", root)), "system_model")
    qubit_universe = _normalize_qubit_universe(system_raw)

    defaults = _normalize_defaults(system_raw.get("defaults", {}))
    gate_declarations = _normalize_gate_declarations(
        system_raw.get("gate_declarations", system_raw.get("gate_set", system_raw.get("gates", {})))
    )
    transition_classes = _normalize_transition_classes(
        system_raw.get("transition_classes", system_raw.get("transitions", ()))
    )
    _ensure_declared_gates_for_transitions(gate_declarations, transition_classes)
    system_model = SystemModel(
        qubit_universe=qubit_universe,
        program_classical_vars=_normalize_program_classical_vars(system_raw.get("program_classical_vars", {})),
        gate_declarations=gate_declarations,
        defaults=defaults,
        state_classes=_normalize_state_classes(system_raw.get("state_classes", {}), qubit_universe),
        transition_classes=transition_classes,
    )

    traces = _normalize_traces(root.get("traces", root.get("trace", {})))
    executions = _normalize_executions(root.get("executions", root.get("execution", {})), traces)
    properties = _normalize_properties(root.get("properties", ()))

    return QAMModel(
        system_model=system_model,
        traces=traces,
        executions=executions,
        properties=properties,
    )


def _normalize_defaults(raw: object) -> Defaults:
    mapping = _as_mapping(raw, "defaults")
    return Defaults(
        default_summary_kind=_enum(
            SummaryKind,
            mapping.get("default_summary_kind", SummaryKind.DENSITY_MATRIX.value),
            "defaults.default_summary_kind",
        ),
        unit_overlap=_boolish(mapping.get("unit_overlap", True)),
        post_measurement_recoherence=_boolish(mapping.get("post_measurement_recoherence", False)),
        qubit_order_policy=str(mapping.get("qubit_order_policy", "declared_order")),
        default_basis=str(mapping.get("default_basis", "Z")),
        default_tolerance=float(mapping.get("default_tolerance", 1e-9)),
    )


def _normalize_program_classical_vars(raw: object) -> dict[str, ClassicalVarDecl]:
    mapping = _as_mapping(raw, "system_model.program_classical_vars")
    result: dict[str, ClassicalVarDecl] = {}
    for var_name, decl_raw in mapping.items():
        decl = {"type": decl_raw} if not isinstance(decl_raw, Mapping) else _as_mapping(decl_raw, f"program_classical_vars.{var_name}")
        result[var_name] = ClassicalVarDecl(
            name=str(decl.get("name", var_name)),
            type_name=_enum(
                ClassicalType,
                decl.get("type_name", decl.get("type", "bit")),
                f"program_classical_vars.{var_name}.type_name",
            ),
            domain=tuple(_normalize_literal(v) for v in decl.get("domain", ())) or None,
        )
    return result


def _normalize_gate_declarations(raw: object) -> dict[str, GateDeclaration]:
    if not isinstance(raw, Mapping):
        return {
            str(gate_name): GateDeclaration(name=str(gate_name))
            for gate_name in _as_sequence(raw, "system_model.gate_set")
        }
    mapping = _as_mapping(raw, "system_model.gate_declarations")
    result: dict[str, GateDeclaration] = {}
    for gate_name, decl_raw in mapping.items():
        decl = _as_mapping(decl_raw, f"gate_declarations.{gate_name}")
        result[gate_name] = GateDeclaration(
            name=str(decl.get("name", gate_name)),
            arity=_optional_int(decl.get("arity")),
            parameter_count=int(decl.get("parameter_count", 0)),
        )
    return result


def _normalize_state_classes(raw: object, qubit_universe: tuple[str, ...]) -> dict[str, StateClassSpec]:
    mapping = _named_items(raw, "system_model.state_classes")
    result: dict[str, StateClassSpec] = {}
    for state_id, state_raw in mapping.items():
        state = _as_mapping(state_raw, f"state_classes.{state_id}")
        quantum_raw = _as_mapping(state.get("quantum", {}), f"state_classes.{state_id}.quantum")
        profile_raw = _as_mapping(state.get("classical_profile", {}), f"state_classes.{state_id}.classical_profile")
        active_qubits = _state_active_qubits(state, quantum_raw, qubit_universe)
        units_raw = state.get("units", quantum_raw.get("units", ()))
        unit_completion = _normalize_unit_completion(
            state.get("unit_completion", quantum_raw.get("unit_completion", "singletons")),
            f"state_classes.{state_id}.unit_completion",
        )
        units = _normalize_units_for_state(
            units_raw,
            active_qubits,
            f"state_classes.{state_id}.units",
            unit_completion=unit_completion,
            complete_explicit_units=_state_declares_active_qubits(state, quantum_raw),
        )
        active_qubit_count = _state_active_qubit_count(state, quantum_raw, active_qubits, units)
        result[state_id] = StateClassSpec(
            id=str(state.get("id", state_id)),
            classical_profile=_normalize_classical_profile(profile_raw),
            active_qubit_count=active_qubit_count,
            units=units,
            tags=frozenset(_tuple_of_str(state.get("tags", ()))),
        )
    return result


def _normalize_unit(raw: object) -> AbstractUnitSpec:
    if isinstance(raw, str):
        return AbstractUnitSpec(id=f"u_{raw}", qubits=(raw,))
    mapping = _as_mapping(raw, "state_class.unit")
    summary_raw = mapping.get("summary_kind")
    unit_id = mapping.get("id", mapping.get("name"))
    if unit_id is None:
        raise ValueError("State-class unit needs an id or name.")
    return AbstractUnitSpec(
        id=str(unit_id),
        qubits=_tuple_of_str(mapping.get("qubits", ())),
        summary_kind=_enum(SummaryKind, summary_raw, "state_class.unit.summary_kind") if summary_raw is not None else None,
        annotations={str(k): _normalize_literal(v) for k, v in _as_mapping(mapping.get("annotations", {}), "state_class.unit.annotations").items()},
    )


def _normalize_transition_classes(raw: object) -> dict[str, TransitionClassSpec]:
    mapping = _named_items(raw, "system_model.transition_classes")
    result: dict[str, TransitionClassSpec] = {}
    for transition_id, transition_raw in mapping.items():
        transition = _as_mapping(transition_raw, f"transition_classes.{transition_id}")
        gate = _normalize_gate_spec(
            transition.get("gate", {"name": "NOOP"}),
            f"transition_classes.{transition_id}.gate",
        )
        branches_raw = transition.get("branches")
        if branches_raw is None:
            branches = _implicit_branches(transition_id, transition, gate)
        else:
            branches = tuple(
                _normalize_branch(branch_raw, f"transition_classes.{transition_id}.branches", transition_id, index)
                for index, branch_raw in enumerate(_as_sequence(branches_raw, f"transition_classes.{transition_id}.branches"))
            )
        source_class = transition.get("source_class", transition.get("source"))
        if source_class is None:
            raise ValueError(f"Transition {transition_id} needs a source or source_class.")
        result[transition_id] = TransitionClassSpec(
            id=str(transition.get("id", transition_id)),
            source_class=str(source_class),
            gate=gate,
            branches=branches,
        )
    return result


def _normalize_gate_spec(raw: object, location: str) -> GateSpec:
    if isinstance(raw, str):
        return _parse_gate_call(raw, location)
    mapping = _as_mapping(raw, location)
    if "name" not in mapping and len(mapping) == 1:
        gate_name, gate_args = next(iter(mapping.items()))
        arg_mapping = _as_mapping(gate_args, location) if isinstance(gate_args, Mapping) else {"targets": gate_args}
        mapping = {"name": gate_name, **arg_mapping}
    return GateSpec(
        name=str(mapping["name"]),
        targets=_tuple_of_str(mapping.get("targets", ())),
        controls=_tuple_of_str(mapping.get("controls", ())),
        params=tuple(_normalize_literal(v) for v in _as_sequence(mapping.get("params", ()), f"{location}.params")),
        record_as=_optional_str(mapping.get("record_as")),
        extra_qargs=_tuple_of_str(mapping.get("extra_qargs", ())),
    )


def _normalize_branch(raw: object, location: str, transition_id: str, index: int) -> BranchSpec:
    mapping = _as_mapping(raw, location)
    action_gate = mapping.get("action_gate", mapping.get("gate"))
    target_class = mapping.get("target_class", mapping.get("target"))
    if target_class is None:
        raise ValueError(f"Branch at {location} needs a target or target_class.")
    return BranchSpec(
        id=str(mapping.get("id", f"b_{transition_id}_{index}")),
        kind=_branch_kind(mapping, f"{location}.kind"),
        target_class=str(target_class),
        outcome=_normalize_literal(mapping["outcome"]) if "outcome" in mapping else None,
        guard_expr=_normalize_guard_expr(mapping, location),
        action_gate=_normalize_gate_spec(action_gate, f"{location}.action_gate") if action_gate is not None else None,
    )


def _normalize_traces(raw: object) -> dict[str, TraceSpec]:
    if raw is None:
        return {}
    if _is_single_trace(raw):
        trace = _as_mapping(raw, "traces")
        trace_id = str(trace.get("id", "main"))
        raw = {trace_id: trace}
    mapping = _as_mapping(raw, "traces")
    result: dict[str, TraceSpec] = {}
    for trace_id, trace_raw in mapping.items():
        trace = _as_mapping(trace_raw, f"traces.{trace_id}")
        anchors, steps, entry_anchor = _normalize_trace_shape(trace, trace_id)
        result[trace_id] = TraceSpec(
            id=str(trace.get("id", trace_id)),
            anchors=anchors,
            entry_anchor=entry_anchor,
            steps=steps,
        )
    return result


def _normalize_executions(raw: object, traces: dict[str, TraceSpec]) -> dict[str, ExecutionSpec]:
    if raw is None:
        return {}
    if _is_single_execution(raw):
        execution = _as_mapping(raw, "executions")
        execution_id = str(execution.get("id", "run_main"))
        raw = {execution_id: execution}
    mapping = _as_mapping(raw, "executions")
    result: dict[str, ExecutionSpec] = {}
    for execution_id, execution_raw in mapping.items():
        execution = _as_mapping(execution_raw, f"executions.{execution_id}")
        trace_id = str(execution.get("trace_id", execution.get("trace")))
        trace = traces.get(trace_id)
        binding = _as_mapping(execution.get("initial_binding", {}), f"executions.{execution_id}.initial_binding")
        anchor_id = str(binding.get("anchor_id", trace.entry_anchor if trace is not None else ""))
        state_class_id = str(binding.get("state_class_id", _initial_state_class(trace, anchor_id)))
        result[execution_id] = ExecutionSpec(
            id=str(execution.get("id", execution_id)),
            trace_id=trace_id,
            initial_binding=InitialBindingSpec(
                anchor_id=anchor_id,
                state_class_id=state_class_id,
                program_values={str(k): _normalize_literal(v) for k, v in _as_mapping(binding.get("program_values", {}), "initial_binding.program_values").items()},
                measurement_values={str(k): _normalize_literal(v) for k, v in _as_mapping(binding.get("measurement_values", {}), "initial_binding.measurement_values").items()},
                payload_initializer=binding.get("payload_initializer", "zero_density_matrix"),
            ),
            stop_anchor=str(execution.get("stop_anchor", execution.get("stop", _default_stop_anchor(trace)))),
            options={str(k): _normalize_literal(v) for k, v in _as_mapping(execution.get("options", {}), "execution.options").items()},
        )
    return result


def _normalize_properties(raw: object) -> tuple[PropertySpec, ...]:
    properties = _as_sequence(raw, "properties")
    result: list[PropertySpec] = []
    for prop_raw in properties:
        prop = _as_mapping(prop_raw, "properties[]")
        scope_raw = _as_mapping(prop.get("scope", {}), f"properties.{prop.get('id', '<unknown>')}.scope")
        observable_raw = _as_mapping(prop.get("observable", {}), f"properties.{prop.get('id', '<unknown>')}.observable")
        property_kind = _enum(PropertyKind, prop["kind"], f"properties.{prop['id']}.kind")
        expectation = prop.get("expectation", prop.get("expect"))
        if property_kind is PropertyKind.REACHABILITY:
            expectation = _normalize_reachability_expectation(expectation, prop.get("threshold", 0.0))
        scope_kind_raw = prop.get("scope_kind")
        scope_value = prop.get("scope_value")
        if scope_kind_raw is None and scope_raw:
            if "anchors" in scope_raw:
                scope_kind_raw = ScopeKind.ANCHORS.value
                scope_value = scope_raw["anchors"]
            elif "interval" in scope_raw:
                scope_kind_raw = ScopeKind.INTERVAL.value
                scope_value = scope_raw["interval"]
        result.append(
            PropertySpec(
                id=str(prop["id"]),
                kind=property_kind,
                scope_kind=_enum(ScopeKind, scope_kind_raw, f"properties.{prop['id']}.scope_kind"),
                scope_value=_normalize_literal(scope_value),
                mode=_enum(EvaluationMode, prop.get("mode", scope_raw.get("mode", EvaluationMode.AGGREGATE.value)), f"properties.{prop['id']}.mode"),
                observable=ObservableSpec(
                    kind=str(observable_raw.get("kind", "quantum")),
                    classical_vars=_tuple_of_str(observable_raw.get("classical_vars", ())),
                    qubits=_tuple_of_str(observable_raw.get("qubits", ())),
                    basis=_normalize_basis(observable_raw.get("basis")),
                ),
                expectation=_normalize_expectation(expectation),
            )
        )
    return tuple(result)


def _normalize_expectation(value: object) -> object:
    normalized = _normalize_literal(value)
    if isinstance(normalized, dict):
        return {_coerce_distribution_key(k): v for k, v in normalized.items()}
    return normalized


def _normalize_reachability_expectation(expectation: object, threshold: object) -> object:
    outcome = _normalize_literal(expectation)
    if isinstance(outcome, tuple) and len(outcome) == 1:
        outcome = outcome[0]
    return ("probability_gt", outcome, float(_normalize_literal(threshold)))


def _coerce_distribution_key(key: object) -> object:
    if isinstance(key, str):
        if key.lstrip("-").isdigit():
            return int(key)
        if "," in key:
            parts = tuple(part.strip() for part in key.split(","))
            if all(part.lstrip("-").isdigit() for part in parts):
                return tuple(int(part) for part in parts)
    return key


def _normalize_qubit_universe(system_raw: Mapping[str, object]) -> tuple[str, ...]:
    raw = system_raw.get("qubit_universe", system_raw.get("qubits"))
    if raw is not None:
        return _expand_qubit_refs(raw)
    if "n" in system_raw:
        return _qubits_from_count(int(system_raw["n"]))
    return ()


def _qubits_from_count(count: int) -> tuple[str, ...]:
    return tuple(f"q{index}" for index in range(count))


def _expand_qubit_refs(value: object) -> tuple[str, ...]:
    if isinstance(value, int):
        return _qubits_from_count(value)
    if isinstance(value, str):
        stripped = value.strip()
        if ".." in stripped:
            start, end = stripped.split("..", 1)
            start_match = re.fullmatch(r"([A-Za-z_]*)(\d+)", start.strip())
            end_match = re.fullmatch(r"([A-Za-z_]*)(\d+)", end.strip())
            if start_match and end_match:
                start_prefix, start_index = start_match.groups()
                end_prefix, end_index = end_match.groups()
                prefix = end_prefix or start_prefix
                if start_prefix == prefix:
                    begin = int(start_index)
                    finish = int(end_index)
                    step = 1 if finish >= begin else -1
                    return tuple(f"{prefix}{index}" for index in range(begin, finish + step, step))
        return (stripped,)
    expanded: list[str] = []
    for item in _as_sequence(value, "qubit_refs"):
        expanded.extend(_expand_qubit_refs(item))
    return tuple(expanded)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "allowed", "allow"}:
            return True
        if normalized in {"false", "no", "off", "forbidden", "forbid"}:
            return False
    return bool(value)


def _named_items(raw: object, location: str) -> dict[str, object]:
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    result: dict[str, object] = {}
    for index, item in enumerate(_as_sequence(raw, location)):
        mapping = _as_mapping(item, f"{location}[{index}]")
        item_id = mapping.get("id", mapping.get("name"))
        if item_id is None:
            raise ValueError(f"Expected id or name at {location}[{index}].")
        result[str(item_id)] = mapping
    return result


def _state_active_qubits(
    state: Mapping[str, object],
    quantum: Mapping[str, object],
    qubit_universe: tuple[str, ...],
) -> tuple[str, ...]:
    raw = state.get("active", state.get("active_qubits", quantum.get("active", quantum.get("active_qubits"))))
    if raw is not None:
        return _expand_active_qubit_refs(raw, qubit_universe)
    count_raw = quantum.get("n", state.get("active_qubit_count"))
    if count_raw is not None:
        count = int(count_raw)
        if qubit_universe and len(qubit_universe) >= count:
            return qubit_universe[:count]
        return _qubits_from_count(count)
    return ()


def _expand_active_qubit_refs(raw: object, qubit_universe: tuple[str, ...]) -> tuple[str, ...]:
    if _is_active_all(raw):
        if not qubit_universe:
            raise ValueError("state active all requires a declared top-level qubit universe.")
        return qubit_universe

    base: list[str] = []
    excluded: list[str] = []
    saw_exclusion = False
    for item in _active_ref_items(raw):
        if item in {"all", "*"}:
            if not qubit_universe:
                raise ValueError("state active all requires a declared top-level qubit universe.")
            base.extend(qubit_universe)
            continue
        if item.startswith("-"):
            saw_exclusion = True
            excluded.extend(_expand_qubit_refs(item[1:].strip()))
            continue
        base.extend(_expand_qubit_refs(item))

    if saw_exclusion and not base:
        if not qubit_universe:
            raise ValueError("active exclusions require a declared top-level qubit universe.")
        base.extend(qubit_universe)

    excluded_set = set(excluded)
    return tuple(qubit for qubit in dict.fromkeys(base) if qubit not in excluded_set)


def _active_ref_items(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    return tuple(str(item).strip() for item in _as_sequence(raw, "state.active") if str(item).strip())


def _is_active_all(raw: object) -> bool:
    if isinstance(raw, str):
        return raw.strip().lower() in {"all", "*"}
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return _is_active_all(raw[0])
    return False


def _state_declares_active_qubits(
    state: Mapping[str, object],
    quantum: Mapping[str, object],
) -> bool:
    return any(key in state for key in ("active", "active_qubits")) or any(
        key in quantum for key in ("active", "active_qubits")
    )


def _state_active_qubit_count(
    state: Mapping[str, object],
    quantum: Mapping[str, object],
    active_qubits: tuple[str, ...],
    units: tuple[AbstractUnitSpec, ...],
) -> int:
    if "active_qubit_count" in state:
        return int(state["active_qubit_count"])
    if active_qubits:
        return len(tuple(dict.fromkeys(active_qubits)))
    if "n" in quantum:
        return int(quantum["n"])
    return len({qubit for unit in units for qubit in unit.qubits})


def _normalize_units_for_state(
    raw: object,
    active_qubits: tuple[str, ...],
    location: str,
    *,
    unit_completion: str,
    complete_explicit_units: bool,
) -> tuple[AbstractUnitSpec, ...]:
    if isinstance(raw, Mapping):
        units: list[AbstractUnitSpec] = []
        for unit_id, unit_raw in raw.items():
            if isinstance(unit_raw, Mapping):
                unit_data = {"id": unit_id, **_as_mapping(unit_raw, f"{location}.{unit_id}")}
            else:
                unit_data = {"id": unit_id, "qubits": unit_raw}
            units.append(_normalize_unit(unit_data))
        completion_scope = active_qubits if complete_explicit_units else ()
        return _complete_units(tuple(units), completion_scope, unit_completion)

    sequence = _as_sequence(raw, location)
    if sequence:
        completion_scope = active_qubits if complete_explicit_units else ()
        return _complete_units(tuple(_normalize_unit(unit_raw) for unit_raw in sequence), completion_scope, unit_completion)
    if active_qubits:
        return _complete_units((), active_qubits, unit_completion)
    return ()


def _normalize_unit_completion(raw: object, location: str) -> str:
    normalized = str(raw).strip().lower()
    aliases = {
        "single": "singletons",
        "singleton": "singletons",
        "singletons": "singletons",
        "per_qubit": "singletons",
        "per-qubit": "singletons",
        "rest": "remainder",
        "remainder": "remainder",
        "full": "remainder",
        "full_state": "remainder",
        "full-state": "remainder",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported unit completion policy at {location}: {raw!r}.")
    return aliases[normalized]


def _complete_units(
    units: tuple[AbstractUnitSpec, ...],
    active_qubits: tuple[str, ...],
    unit_completion: str,
) -> tuple[AbstractUnitSpec, ...]:
    if not active_qubits:
        return units

    covered_qubits = {qubit for unit in units for qubit in unit.qubits}
    missing_qubits = tuple(qubit for qubit in active_qubits if qubit not in covered_qubits)
    if not missing_qubits:
        return units

    existing_ids = {unit.id for unit in units}
    if unit_completion == "singletons":
        return units + tuple(
            AbstractUnitSpec(id=_unique_unit_id(f"u_{qubit}", existing_ids), qubits=(qubit,))
            for qubit in missing_qubits
        )
    if unit_completion == "remainder":
        base_id = "u_all" if not units else "u_rest"
        return units + (
            AbstractUnitSpec(id=_unique_unit_id(base_id, existing_ids), qubits=missing_qubits),
        )
    raise ValueError(f"Unsupported unit completion policy: {unit_completion!r}.")


def _unique_unit_id(base_id: str, existing_ids: set[str]) -> str:
    if base_id not in existing_ids:
        existing_ids.add(base_id)
        return base_id
    suffix = 2
    while f"{base_id}_{suffix}" in existing_ids:
        suffix += 1
    unit_id = f"{base_id}_{suffix}"
    existing_ids.add(unit_id)
    return unit_id


def _normalize_classical_profile(raw: Mapping[str, object]) -> ClassicalProfileSpec:
    program_raw = raw.get("program", {})
    measurement_raw = raw.get("measurement", {})
    program_vars = raw.get("program_vars")
    measurement_vars = raw.get("measurement_vars")
    return ClassicalProfileSpec(
        program_vars=_tuple_of_str(program_vars) if program_vars is not None else tuple(str(k) for k in _as_mapping(program_raw, "classical_profile.program")),
        measurement_vars=(
            _tuple_of_str(measurement_vars)
            if measurement_vars is not None
            else tuple(str(k) for k in _as_mapping(measurement_raw, "classical_profile.measurement"))
        ),
        type_overrides={
            str(k): str(v)
            for k, v in _as_mapping(raw.get("type_overrides", {}), "type_overrides").items()
        },
    )


def _implicit_branches(
    transition_id: str,
    transition: Mapping[str, object],
    gate: GateSpec,
) -> tuple[BranchSpec, ...]:
    target = transition.get("target_class", transition.get("target"))
    if target is None:
        raise ValueError(f"Transition {transition_id} needs a target or explicit branches.")
    target_class = str(target)
    if gate.is_measurement():
        return (
            BranchSpec(
                id=f"b_{transition_id}_0",
                kind=BranchKind.MEASUREMENT,
                target_class=target_class,
                outcome=0,
            ),
            BranchSpec(
                id=f"b_{transition_id}_1",
                kind=BranchKind.MEASUREMENT,
                target_class=target_class,
                outcome=1,
            ),
        )
    return (
        BranchSpec(
            id=f"b_{transition_id}",
            kind=BranchKind.DETERMINISTIC,
            target_class=target_class,
        ),
    )


def _branch_kind(mapping: Mapping[str, object], location: str) -> BranchKind:
    raw = mapping.get("kind")
    if raw is None:
        if "guard" in mapping or "guard_expr" in mapping:
            raw = BranchKind.CLASSICAL_CONTROL.value
        elif "outcome" in mapping:
            raw = BranchKind.MEASUREMENT.value
        else:
            raw = BranchKind.DETERMINISTIC.value
    aliases = {
        "deterministic": BranchKind.DETERMINISTIC.value,
        "measurement": BranchKind.MEASUREMENT.value,
        "classical_control": BranchKind.CLASSICAL_CONTROL.value,
    }
    raw = aliases.get(str(raw), raw)
    return _enum(BranchKind, raw, location)  # type: ignore[return-value]


def _normalize_guard_expr(mapping: Mapping[str, object], location: str) -> object | None:
    if "guard_expr" in mapping:
        return _normalize_literal(mapping["guard_expr"])
    if "guard" not in mapping:
        return None
    guard = mapping["guard"]
    if isinstance(guard, (list, tuple)):
        return _normalize_literal(guard)
    guard_mapping = _as_mapping(guard, f"{location}.guard")
    if "eq" in guard_mapping:
        var_name, expected = _as_sequence(guard_mapping["eq"], f"{location}.guard.eq")
        return ("eq", str(var_name), _normalize_literal(expected))
    for group in ("measurement", "program"):
        if group in guard_mapping:
            group_mapping = _as_mapping(guard_mapping[group], f"{location}.guard.{group}")
            if len(group_mapping) != 1:
                raise ValueError(f"Expected one guard variable at {location}.guard.{group}.")
            var_name, expected = next(iter(group_mapping.items()))
            return ("eq", str(var_name), _normalize_literal(expected))
    if len(guard_mapping) == 1:
        var_name, expected = next(iter(guard_mapping.items()))
        return ("eq", str(var_name), _normalize_literal(expected))
    raise ValueError(f"Unsupported guard form at {location}: {guard!r}")


def _parse_gate_call(raw: str, location: str) -> GateSpec:
    source = raw.strip()
    measure_match = re.fullmatch(r"measure\s+([A-Za-z_]\w*)\(([^)]*)\)\s+as\s+([A-Za-z_]\w*)", source)
    if measure_match:
        basis, args_source, record_as = measure_match.groups()
        return GateSpec(name=f"M{basis.upper()}", targets=_parse_gate_args(args_source), record_as=record_as)

    match = re.fullmatch(r"([A-Za-z_]\w*)\(([^)]*)\)(?:\s+as\s+([A-Za-z_]\w*))?", source)
    if not match:
        raise ValueError(f"Invalid gate call at {location}: {raw!r}")
    name, args_source, record_as = match.groups()
    args = _parse_gate_args(args_source)
    upper_name = name.upper()
    if upper_name == "P" and len(args) == 2:
        return GateSpec(name=name, targets=(args[0],), params=(_parse_gate_param(args[1]),), record_as=record_as)
    if upper_name in {"CX", "CNOT"} and len(args) == 2:
        return GateSpec(name=name, controls=(args[0],), targets=(args[1],), record_as=record_as)
    if upper_name == "CP" and len(args) == 3:
        return GateSpec(name=name, controls=(args[0],), targets=(args[1],), params=(_parse_gate_param(args[2]),), record_as=record_as)
    if upper_name == "RBS" and len(args) == 3:
        return GateSpec(name=name, targets=args[:2], params=(_parse_gate_param(args[2]),), record_as=record_as)
    if upper_name == "CCX" and len(args) == 3:
        return GateSpec(name=name, controls=args[:2], targets=args[2:], record_as=record_as)
    return GateSpec(name=name, targets=args, record_as=record_as)


def _parse_gate_args(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_gate_param(raw: str) -> object:
    stripped = raw.strip()
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _ensure_declared_gates_for_transitions(
    gate_declarations: dict[str, GateDeclaration],
    transitions: dict[str, TransitionClassSpec],
) -> None:
    for transition in transitions.values():
        _ensure_declared_noop(gate_declarations, transition.gate)
        for branch in transition.branches:
            if branch.action_gate is not None:
                _ensure_declared_noop(gate_declarations, branch.action_gate)


def _ensure_declared_noop(gate_declarations: dict[str, GateDeclaration], gate: GateSpec) -> None:
    if gate.name.upper() == "NOOP" and gate.name not in gate_declarations:
        gate_declarations[gate.name] = GateDeclaration(name=gate.name, arity=0)


def _is_single_trace(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    return any(key in raw for key in ("id", "flow", "anchors", "entry_anchor", "steps"))


def _normalize_trace_shape(
    trace: Mapping[str, object],
    trace_id: str,
) -> tuple[dict[str, AnchorSpec], tuple[TraceStepSpec, ...], str]:
    if "flow" in trace:
        return _normalize_trace_flow(trace, trace_id)

    anchors_raw = _as_mapping(trace.get("anchors", {}), f"traces.{trace_id}.anchors")
    anchors = {
        anchor_id: AnchorSpec(
            id=str(anchor_data.get("id", anchor_id)),
            kind=_enum(AnchorKind, anchor_data["kind"], f"traces.{trace_id}.anchors.{anchor_id}.kind"),
            admissible_classes=_tuple_of_str(anchor_data.get("admissible_classes", ())),
        )
        for anchor_id, anchor_mapping in anchors_raw.items()
        for anchor_data in [_as_mapping(anchor_mapping, f"traces.{trace_id}.anchors.{anchor_id}")]
    }
    steps = tuple(
        TraceStepSpec(
            source_anchor=str(step_data.get("source_anchor", step_data.get("source", step_data.get("from")))),
            transition_class=str(step_data.get("transition_class", step_data.get("transition"))),
            target_anchor=str(step_data.get("target_anchor", step_data.get("target", step_data.get("to")))),
        )
        for step_mapping in _as_sequence(trace.get("steps", ()), f"traces.{trace_id}.steps")
        for step_data in [_as_mapping(step_mapping, f"traces.{trace_id}.steps")]
    )
    return anchors, steps, str(trace["entry_anchor"])


def _normalize_trace_flow(
    trace: Mapping[str, object],
    trace_id: str,
) -> tuple[dict[str, AnchorSpec], tuple[TraceStepSpec, ...], str]:
    anchors: dict[str, AnchorSpec] = {}
    steps: list[TraceStepSpec] = []
    entry_anchor = _optional_str(trace.get("entry_anchor"))

    for index, item_raw in enumerate(_as_sequence(trace.get("flow", ()), f"traces.{trace_id}.flow")):
        item = _as_mapping(item_raw, f"traces.{trace_id}.flow[{index}]")
        if "transition" not in item:
            anchor = _anchor_from_flow_item(item, default_kind=AnchorKind.NODE)
            anchors[anchor.id] = anchor
            if entry_anchor is None:
                entry_anchor = anchor.id
            continue
        if "branches" in item:
            raise ValueError("Branch-specific trace flow is not supported by the current TraceSpec IR.")
        target_anchor = _anchor_from_flow_target(item.get("to"), f"traces.{trace_id}.flow[{index}].to")
        anchors[target_anchor.id] = target_anchor
        steps.append(
            TraceStepSpec(
                source_anchor=str(item["from"]),
                transition_class=str(item["transition"]),
                target_anchor=target_anchor.id,
            )
        )

    if entry_anchor is None:
        raise ValueError(f"Trace {trace_id} needs an entry anchor.")
    return anchors, tuple(steps), entry_anchor


def _anchor_from_flow_item(item: Mapping[str, object], default_kind: AnchorKind) -> AnchorSpec:
    if item.get("terminal") is True:
        anchor_id = str(item.get("frontier", item.get("node", item.get("anchor", item.get("id")))))
        kind = AnchorKind.TERMINAL
    elif "terminal" in item and item.get("terminal") is not True:
        anchor_id = str(item["terminal"])
        kind = AnchorKind.TERMINAL
    elif "frontier" in item and item.get("frontier") is not True:
        anchor_id = str(item["frontier"])
        kind = AnchorKind.FRONTIER
    else:
        anchor_id = str(item.get("node", item.get("anchor", item.get("id"))))
        kind = AnchorKind.TERMINAL if item.get("terminal") is True else default_kind
        if "frontier" in item:
            kind = AnchorKind.FRONTIER
    return AnchorSpec(
        id=anchor_id,
        kind=kind,
        admissible_classes=_flow_admissible_classes(item),
    )


def _anchor_from_flow_target(raw: object, location: str) -> AnchorSpec:
    if isinstance(raw, str):
        return AnchorSpec(id=raw, kind=AnchorKind.NODE, admissible_classes=())
    target = _as_mapping(raw, location)
    default_kind = AnchorKind.FRONTIER if "frontier" in target else AnchorKind.NODE
    return _anchor_from_flow_item(target, default_kind=default_kind)


def _flow_admissible_classes(item: Mapping[str, object]) -> tuple[str, ...]:
    if "admissible_classes" in item:
        return _tuple_of_str(item["admissible_classes"])
    if "class" in item:
        return (str(item["class"]),)
    if "classes" in item:
        return _tuple_of_str(item["classes"])
    return ()


def _is_single_execution(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    return any(key in raw for key in ("id", "trace", "trace_id", "initial_binding", "stop", "stop_anchor"))


def _initial_state_class(trace: TraceSpec | None, anchor_id: str) -> str:
    if trace is None or anchor_id not in trace.anchors:
        return ""
    classes = trace.anchors[anchor_id].admissible_classes
    return classes[0] if classes else ""


def _default_stop_anchor(trace: TraceSpec | None) -> str:
    if trace is None:
        return ""
    for anchor_id, anchor in trace.anchors.items():
        if anchor.kind is AnchorKind.TERMINAL:
            return anchor_id
    order = trace.topological_anchor_order()
    return order[-1] if order else trace.entry_anchor


def _normalize_basis(value: object) -> str | None:
    if value is None:
        return None
    basis = str(value)
    if basis.lower() == "computational":
        return "Z"
    return basis


def _normalize_literal(value: object) -> object:
    if isinstance(value, list):
        return tuple(_normalize_literal(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_literal(item) for item in value)
    if isinstance(value, Mapping):
        return { _normalize_literal(k): _normalize_literal(v) for k, v in value.items() }
    return value


def _enum(enum_cls: type, value: object, location: str) -> object:
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ValueError(f"Invalid enum value at {location}: {value!r}") from exc


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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
