"""Well-formedness checks for definition-layer QAM models."""

from __future__ import annotations

from qamc.ir.enums import AnchorKind, BranchKind, ClassicalType, PropertyKind, ScopeKind
from qamc.ir.model import (
    BranchSpec,
    ExecutionSpec,
    GateSpec,
    QAMModel,
    StateClassSpec,
    SystemModel,
    TraceSpec,
    TransitionClassSpec,
)
from qamc.validate.diagnostics import ValidationReport


def validate_qam_model(model: QAMModel) -> ValidationReport:
    """Validate a definition-layer QAM model."""

    report = ValidationReport()

    _validate_system_model(model.system_model, report)
    for trace in model.traces.values():
        _validate_trace(trace, model.system_model, report)
    for execution in model.executions.values():
        _validate_execution(execution, model, report)
    for prop in model.properties:
        _validate_property(prop, model, report)

    return report


def _validate_system_model(system_model: SystemModel, report: ValidationReport) -> None:
    qubits = system_model.qubit_universe
    qubit_set = set(qubits)
    if not qubits:
        report.error("model.qubits.empty", "The qubit universe must not be empty.", "system_model")
    if len(qubit_set) != len(qubits):
        report.error(
            "model.qubits.duplicate",
            "The qubit universe contains duplicate qubit names.",
            "system_model.qubit_universe",
        )

    for var_name, decl in system_model.program_classical_vars.items():
        if var_name != decl.name:
            report.error(
                "model.classical.key_mismatch",
                f"Classical declaration key {var_name!r} does not match name {decl.name!r}.",
                f"system_model.program_classical_vars[{var_name!r}]",
            )

    for gate_name, decl in system_model.gate_declarations.items():
        if gate_name != decl.name:
            report.error(
                "model.gate.key_mismatch",
                f"Gate declaration key {gate_name!r} does not match name {decl.name!r}.",
                f"system_model.gate_declarations[{gate_name!r}]",
            )

    for state_class_name, state_class in system_model.state_classes.items():
        _validate_state_class(
            state_class_name=state_class_name,
            state_class=state_class,
            system_model=system_model,
            report=report,
        )

    for transition_name, transition in system_model.transition_classes.items():
        _validate_transition_class(
            transition_name=transition_name,
            transition=transition,
            system_model=system_model,
            report=report,
        )


def _validate_state_class(
    *,
    state_class_name: str,
    state_class: StateClassSpec,
    system_model: SystemModel,
    report: ValidationReport,
) -> None:
    location = f"state_class:{state_class_name}"
    if state_class_name != state_class.id:
        report.error(
            "state.key_mismatch",
            f"State class key {state_class_name!r} does not match id {state_class.id!r}.",
            location,
        )

    unit_ids = [unit.id for unit in state_class.units]
    if len(set(unit_ids)) != len(unit_ids):
        report.error(
            "state.unit.duplicate_id",
            f"State class {state_class.id} contains duplicate unit ids.",
            f"{location}.units",
        )

    flattened_qubits: list[str] = []
    for unit in state_class.units:
        if len(set(unit.qubits)) != len(unit.qubits):
            report.error(
                "state.unit.duplicate_qubit",
                f"Unit {unit.id} contains duplicate qubit references.",
                f"{location}.unit:{unit.id}",
            )
        for qubit in unit.qubits:
            flattened_qubits.append(qubit)
            if qubit not in system_model.qubit_universe:
                report.error(
                    "state.unit.unknown_qubit",
                    f"Unit {unit.id} references undeclared qubit {qubit!r}.",
                    f"{location}.unit:{unit.id}",
                )

    distinct_active_qubits = set(flattened_qubits)
    if state_class.active_qubit_count != len(distinct_active_qubits):
        report.error(
            "state.active_qubit_count.mismatch",
            (
                f"State class {state_class.id} declares active_qubit_count="
                f"{state_class.active_qubit_count}, but its units cover "
                f"{len(distinct_active_qubits)} distinct qubits."
            ),
            location,
        )

    if not system_model.defaults.unit_overlap and len(flattened_qubits) != len(distinct_active_qubits):
        report.error(
            "state.unit.overlap_forbidden",
            f"State class {state_class.id} contains overlapping units while overlap is disabled.",
            f"{location}.units",
        )

    declared_program_vars = set(system_model.program_classical_vars)
    for var_name in state_class.classical_profile.program_vars:
        if var_name not in declared_program_vars:
            report.error(
                "state.classical.unknown_program_var",
                f"State class {state_class.id} exposes undeclared program variable {var_name!r}.",
                f"{location}.classical_profile.program_vars",
            )


def _validate_gate_usage(
    *,
    gate: GateSpec,
    allowed_qubits: set[str],
    system_model: SystemModel,
    report: ValidationReport,
    location: str,
) -> None:
    if gate.name not in system_model.gate_declarations:
        report.error(
            "transition.gate.undeclared",
            f"Gate {gate.name!r} is not declared in the model gate set.",
            location,
        )
    for qubit in gate.all_qargs():
        if qubit not in system_model.qubit_universe:
            report.error(
                "transition.gate.unknown_qubit",
                f"Gate {gate.name!r} references undeclared qubit {qubit!r}.",
                location,
            )
        elif qubit not in allowed_qubits:
            report.error(
                "transition.gate.qubit_out_of_scope",
                f"Gate {gate.name!r} references qubit {qubit!r} outside the source state class.",
                location,
            )


def _validate_transition_class(
    *,
    transition_name: str,
    transition: TransitionClassSpec,
    system_model: SystemModel,
    report: ValidationReport,
) -> None:
    location = f"transition:{transition_name}"
    if transition_name != transition.id:
        report.error(
            "transition.key_mismatch",
            f"Transition key {transition_name!r} does not match id {transition.id!r}.",
            location,
        )
    if transition.source_class not in system_model.state_classes:
        report.error(
            "transition.source.unknown_state_class",
            f"Transition {transition.id} references unknown source state class {transition.source_class!r}.",
            location,
        )
        return

    source_state = system_model.state_class(transition.source_class)
    allowed_qubits = set(source_state.active_unit_qubits())
    _validate_gate_usage(
        gate=transition.gate,
        allowed_qubits=allowed_qubits,
        system_model=system_model,
        report=report,
        location=f"{location}.gate",
    )

    branch_ids = transition.branch_ids()
    if len(set(branch_ids)) != len(branch_ids):
        report.error(
            "transition.branch.duplicate_id",
            f"Transition {transition.id} contains duplicate branch ids.",
            f"{location}.branches",
        )

    if transition.gate.is_measurement():
        _validate_measurement_transition(transition, source_state, system_model, report, location)
    else:
        _validate_non_measurement_transition(transition, source_state, system_model, report, location)


def _validate_measurement_transition(
    transition: TransitionClassSpec,
    source_state: StateClassSpec,
    system_model: SystemModel,
    report: ValidationReport,
    location: str,
) -> None:
    if not transition.measurement_branches():
        report.error(
            "transition.measurement.no_branches",
            f"Measurement transition {transition.id} must define measurement branches.",
            location,
        )

    record_as = transition.gate.record_as
    if not record_as:
        report.error(
            "transition.measurement.missing_record_as",
            f"Measurement transition {transition.id} must record its outcome.",
            f"{location}.gate",
        )
        return

    if record_as in system_model.program_classical_vars:
        report.error(
            "transition.measurement.record_as_conflict",
            f"Measurement variable {record_as!r} conflicts with a program classical variable.",
            f"{location}.gate",
        )

    outcomes: list[object] = []
    for branch in transition.branches:
        branch_location = f"{location}.branch:{branch.id}"
        if branch.kind is not BranchKind.MEASUREMENT:
            report.error(
                "transition.measurement.non_measurement_branch",
                f"Measurement transition {transition.id} contains non-measurement branch {branch.id}.",
                branch_location,
            )
        if branch.target_class not in system_model.state_classes:
            report.error(
                "transition.branch.unknown_target_class",
                f"Branch {branch.id} references unknown target state class {branch.target_class!r}.",
                branch_location,
            )
            continue
        target_state = system_model.state_class(branch.target_class)
        if record_as not in target_state.classical_profile.measurement_vars:
            report.error(
                "transition.measurement.target_profile_missing_record",
                (
                    f"Target state class {target_state.id} does not expose measurement variable "
                    f"{record_as!r} required by transition {transition.id}."
                ),
                branch_location,
            )
        outcomes.append(branch.outcome)

    if len(set(outcomes)) != len(outcomes):
        report.error(
            "transition.measurement.duplicate_outcome",
            f"Measurement transition {transition.id} contains duplicate branch outcomes.",
            f"{location}.branches",
        )


def _validate_non_measurement_transition(
    transition: TransitionClassSpec,
    source_state: StateClassSpec,
    system_model: SystemModel,
    report: ValidationReport,
    location: str,
) -> None:
    if not transition.branches:
        report.error(
            "transition.branch.missing",
            f"Transition {transition.id} must declare at least one branch.",
            location,
        )
        return

    for branch in transition.branches:
        branch_location = f"{location}.branch:{branch.id}"
        if branch.target_class not in system_model.state_classes:
            report.error(
                "transition.branch.unknown_target_class",
                f"Branch {branch.id} references unknown target state class {branch.target_class!r}.",
                branch_location,
            )
        if branch.kind is BranchKind.MEASUREMENT:
            report.error(
                "transition.branch.unexpected_measurement_kind",
                f"Non-measurement transition {transition.id} cannot contain measurement branch {branch.id}.",
                branch_location,
            )
        if branch.kind is BranchKind.CLASSICAL_CONTROL:
            if branch.guard_expr is None:
                report.error(
                    "transition.ctrl.missing_guard",
                    f"Control branch {branch.id} is missing its guard expression.",
                    branch_location,
                )
            else:
                _validate_guard(branch, source_state, report, branch_location)
            if branch.action_gate is None:
                report.error(
                    "transition.ctrl.missing_action",
                    f"Control branch {branch.id} is missing its branch-local action gate.",
                    branch_location,
                )
            else:
                _validate_gate_usage(
                    gate=branch.action_gate,
                    allowed_qubits=set(source_state.active_unit_qubits()),
                    system_model=system_model,
                    report=report,
                    location=f"{branch_location}.action_gate",
                )


def _validate_guard(branch: BranchSpec, source_state: StateClassSpec, report: ValidationReport, location: str) -> None:
    guard = branch.guard_expr
    if not isinstance(guard, tuple) or len(guard) != 3 or guard[0] != "eq":
        report.error(
            "transition.ctrl.invalid_guard",
            f"Control branch {branch.id} uses an unsupported guard form {guard!r}.",
            location,
        )
        return
    _, var_name, _expected = guard
    if not source_state.classical_profile.contains(var_name):
        report.error(
            "transition.ctrl.guard_unknown_var",
            f"Control branch {branch.id} references unavailable classical variable {var_name!r}.",
            location,
        )


def _validate_trace(trace: TraceSpec, system_model: SystemModel, report: ValidationReport) -> None:
    location = f"trace:{trace.id}"
    if trace.entry_anchor not in trace.anchors:
        report.error(
            "trace.entry.unknown_anchor",
            f"Trace {trace.id} references unknown entry anchor {trace.entry_anchor!r}.",
            location,
        )

    for anchor_name, anchor in trace.anchors.items():
        if anchor_name != anchor.id:
            report.error(
                "trace.anchor.key_mismatch",
                f"Anchor key {anchor_name!r} does not match id {anchor.id!r}.",
                f"{location}.anchor:{anchor_name}",
            )
        for class_id in anchor.admissible_classes:
            if class_id not in system_model.state_classes:
                report.error(
                    "trace.anchor.unknown_state_class",
                    f"Anchor {anchor.id} admits unknown state class {class_id!r}.",
                    f"{location}.anchor:{anchor_name}",
                )

    for index, step in enumerate(trace.steps):
        step_location = f"{location}.step[{index}]"
        if step.source_anchor not in trace.anchors:
            report.error(
                "trace.step.unknown_source_anchor",
                f"Step {index} references unknown source anchor {step.source_anchor!r}.",
                step_location,
            )
            continue
        if step.target_anchor not in trace.anchors:
            report.error(
                "trace.step.unknown_target_anchor",
                f"Step {index} references unknown target anchor {step.target_anchor!r}.",
                step_location,
            )
            continue
        if step.transition_class not in system_model.transition_classes:
            report.error(
                "trace.step.unknown_transition",
                f"Step {index} references unknown transition class {step.transition_class!r}.",
                step_location,
            )
            continue

        transition = system_model.transition_class(step.transition_class)
        source_anchor = trace.anchors[step.source_anchor]
        target_anchor = trace.anchors[step.target_anchor]

        if source_anchor.admissible_classes and transition.source_class not in source_anchor.admissible_classes:
            report.error(
                "trace.step.source_anchor_incompatible",
                (
                    f"Source anchor {source_anchor.id} does not admit the source state class "
                    f"{transition.source_class!r} of transition {transition.id}."
                ),
                step_location,
            )

        if target_anchor.admissible_classes:
            target_classes = {branch.target_class for branch in transition.branches}
            missing = target_classes.difference(target_anchor.admissible_classes)
            if missing:
                report.error(
                    "trace.step.target_anchor_incompatible",
                    (
                        f"Target anchor {target_anchor.id} does not admit all target state classes of "
                        f"transition {transition.id}: {sorted(missing)!r}."
                    ),
                    step_location,
                )

    try:
        trace.topological_anchor_order()
    except ValueError as exc:
        report.error("trace.cyclic", str(exc), location)
        return

    # A declared step applies to every instance at its source. Its transition
    # branches at runtime; the declared positions themselves form one path.
    position = trace.entry_anchor
    visited = {position}
    for index, step in enumerate(trace.steps):
        if step.source_anchor != position:
            report.error(
                "trace.step.continuation",
                f"Step {index} starts at {step.source_anchor!r}; expected the continuation of {position!r}.",
                f"{location}.step[{index}]",
            )
            return
        position = step.target_anchor
        visited.add(position)

    for anchor in trace.anchors.values():
        if anchor.id not in visited:
            report.error("trace.anchor.unreachable", f"Position {anchor.id!r} is outside the execution path.", location)
        elif anchor.kind is AnchorKind.TERMINAL and anchor.id != position:
            report.error("trace.terminal.has_successor", f"Terminal position {anchor.id!r} has a successor.", location)


def _matches_classical_type(value: object, type_name: ClassicalType) -> bool:
    if type_name is ClassicalType.BOOL:
        return isinstance(value, bool)
    # The parser distinguishes integer literals from Boolean literals and floats.
    if type(value) is not int:
        return False
    return type_name is ClassicalType.INT or (type_name is ClassicalType.BIT and value in (0, 1))


def _validate_execution(execution: ExecutionSpec, model: QAMModel, report: ValidationReport) -> None:
    location = f"execution:{execution.id}"
    if execution.trace_id not in model.traces:
        report.error(
            "execution.unknown_trace",
            f"Execution {execution.id} references unknown trace {execution.trace_id!r}.",
            location,
        )
        return

    trace = model.trace(execution.trace_id)
    binding = execution.initial_binding
    if binding.anchor_id != trace.entry_anchor:
        report.error(
            "execution.initial_binding.entry_mismatch",
            (
                f"Execution {execution.id} binds anchor {binding.anchor_id!r}, "
                f"but trace {trace.id} enters at {trace.entry_anchor!r}."
            ),
            location,
        )

    if binding.anchor_id not in trace.anchors:
        report.error(
            "execution.initial_binding.unknown_anchor",
            f"Execution {execution.id} references unknown initial anchor {binding.anchor_id!r}.",
            location,
        )
    else:
        anchor = trace.anchors[binding.anchor_id]
        if anchor.admissible_classes and binding.state_class_id not in anchor.admissible_classes:
            report.error(
                "execution.initial_binding.anchor_incompatible",
                (
                    f"Initial anchor {anchor.id} does not admit state class {binding.state_class_id!r} "
                    f"used by execution {execution.id}."
                ),
                location,
            )

    if binding.state_class_id not in model.system_model.state_classes:
        report.error(
            "execution.initial_binding.unknown_state_class",
            f"Execution {execution.id} references unknown state class {binding.state_class_id!r}.",
            location,
        )
    else:
        state_class = model.system_model.state_class(binding.state_class_id)
        for var_name, value in binding.program_values.items():
            if var_name not in state_class.classical_profile.program_vars:
                report.error(
                    "execution.initial_binding.unknown_program_value",
                    (
                        f"Execution {execution.id} initializes unavailable program variable "
                        f"{var_name!r} for state class {state_class.id}."
                    ),
                    location,
                )
                continue
            declaration = model.system_model.program_classical_vars.get(var_name)
            if declaration is None:
                continue  # The state class already reports undeclared variables.
            value_location = f"{location}.initial_binding.program_values[{var_name!r}]"
            if not _matches_classical_type(value, declaration.type_name):
                report.error(
                    "execution.initial_binding.program_value_type",
                    f"Initial value {value!r} for {var_name!r} must have type {declaration.type_name.value!r}.",
                    value_location,
                )
            elif declaration.domain is not None and value not in declaration.domain:
                report.error(
                    "execution.initial_binding.program_value_domain",
                    f"Initial value {value!r} for {var_name!r} is outside its declared domain {declaration.domain!r}.",
                    value_location,
                )
        for var_name in binding.measurement_values:
            if var_name not in state_class.classical_profile.measurement_vars:
                report.error(
                    "execution.initial_binding.unknown_measurement_value",
                    (
                        f"Execution {execution.id} initializes unavailable measurement variable "
                        f"{var_name!r} for state class {state_class.id}."
                    ),
                    location,
                )

    if execution.stop_anchor not in trace.anchors:
        report.error(
            "execution.stop_anchor.unknown",
            f"Execution {execution.id} references unknown stop anchor {execution.stop_anchor!r}.",
            location,
        )
    else:
        endpoint = trace.steps[-1].target_anchor if trace.steps else trace.entry_anchor
        if execution.stop_anchor != endpoint:
            report.error(
                "execution.stop_anchor.not_endpoint",
                f"Execution {execution.id} must stop at the final position {endpoint!r}.",
                location,
            )


def _validate_property(prop, model: QAMModel, report: ValidationReport) -> None:
    location = f"property:{prop.id}"
    trace = _infer_property_trace(model)
    if trace is None:
        report.error(
            "property.trace.missing",
            f"Property {prop.id} cannot be validated because no trace is available.",
            location,
        )
        return

    if prop.scope_kind is ScopeKind.ANCHORS:
        anchors = tuple(prop.scope_value)
        if not anchors:
            report.error(
                "property.scope.empty_anchor_set",
                f"Property {prop.id} uses an empty anchor scope.",
                location,
            )
        for anchor_id in anchors:
            if anchor_id not in trace.anchors:
                report.error(
                    "property.scope.unknown_anchor",
                    f"Property {prop.id} references unknown anchor {anchor_id!r}.",
                    location,
                )
    elif prop.scope_kind is ScopeKind.INTERVAL:
        if not isinstance(prop.scope_value, tuple) or len(prop.scope_value) != 2:
            report.error(
                "property.scope.invalid_interval",
                f"Property {prop.id} uses an invalid interval scope {prop.scope_value!r}.",
                location,
            )
        else:
            start_anchor, end_anchor = prop.scope_value
            order = trace.topological_anchor_order()
            if start_anchor not in trace.anchors or end_anchor not in trace.anchors:
                report.error(
                    "property.scope.unknown_interval_anchor",
                    f"Property {prop.id} references unknown interval anchors {prop.scope_value!r}.",
                    location,
                )
            elif order.index(start_anchor) > order.index(end_anchor):
                report.error(
                    "property.scope.invalid_interval_order",
                    f"Property {prop.id} uses a reversed interval {prop.scope_value!r}.",
                    location,
                )

    _validate_classical_observable(prop, model, trace, report)


def _validate_classical_observable(prop, model: QAMModel, trace: TraceSpec, report: ValidationReport) -> None:
    location = f"property:{prop.id}.observable"
    if prop.kind is PropertyKind.ASSERTION and prop.observable.kind != "classical":
        report.error(
            "property.observable.incompatible_kind",
            f"Assertion property {prop.id} requires a classical observation.",
            location,
        )
    if prop.observable.kind != "classical":
        return
    if prop.kind not in {PropertyKind.ASSERTION, PropertyKind.DISTRIBUTION}:
        report.error(
            "property.observable.incompatible_kind",
            f"Property kind {prop.kind.value} does not support classical observations.",
            location,
        )
    if not prop.observable.classical_vars:
        report.error(
            "property.observable.empty_classical_vars",
            f"Property {prop.id} must observe at least one classical variable.",
            location,
        )
        return

    if prop.scope_kind is ScopeKind.ANCHORS:
        selected = tuple(prop.scope_value)
    elif prop.scope_kind is ScopeKind.INTERVAL and isinstance(prop.scope_value, tuple) and len(prop.scope_value) == 2:
        try:
            order = trace.topological_anchor_order()
            start, end = prop.scope_value
            selected = order[order.index(start):order.index(end) + 1]
        except (KeyError, ValueError):
            return  # The scope or trace validator reports the malformed interval.
    else:
        return

    for anchor_id in selected:
        anchor = trace.anchors.get(anchor_id)
        if anchor is None:
            continue
        for class_id in anchor.admissible_classes:
            state_class = model.system_model.state_classes.get(class_id)
            if state_class is None:
                continue
            for name in prop.observable.classical_vars:
                if not state_class.classical_profile.contains(name):
                    report.error(
                        "property.observable.unavailable_classical_var",
                        f"Property {prop.id} observes unavailable classical variable {name!r} "
                        f"at node {anchor_id!r} in state class {class_id!r}.",
                        location,
                    )


def _infer_property_trace(model: QAMModel) -> TraceSpec | None:
    if not model.executions:
        return None
    first_execution = next(iter(model.executions.values()))
    return model.traces.get(first_execution.trace_id)
