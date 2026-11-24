"""Subspace execution for explicitly selected Grover witness models.

Each independent component carries a basis for a set of possible states.  The
initial subspace contains the all-zero input; unitary updates preserve this
inclusion.  Runtime units hold reduced support spaces, with no probabilities
or coefficients for a particular quantum state.
"""

from __future__ import annotations

import numpy as np

from qamc.backend.unit_ops import named_gate_matrix
from qamc.ir.enums import BranchKind, PropertyKind
from qamc.ir.model import GateSpec, QAMModel, StateClassSpec
from qamc.runtime.anchor_store import AnchorStore
from qamc.runtime.execution_context import ExecutionContext
from qamc.runtime.state_node import StateNode, UnitPayload
from qamc.runtime.transition_edge import TransitionEdge


MAX_COMPONENT_QUBITS = 12
SUPPORT_TOLERANCE = 1e-12
SUMMARY_KIND = "grover_subspace"


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _orthonormal_support(columns: np.ndarray) -> np.ndarray:
    left, singular_values, _ = np.linalg.svd(columns, full_matrices=False)
    return _readonly(left[:, singular_values > SUPPORT_TOLERANCE])


def reduced_support_basis(
    basis: np.ndarray,
    qubit_order: tuple[str, ...],
    scope: tuple[str, ...],
) -> np.ndarray:
    """Return an orthonormal basis for the support on an ordered subsystem.

    Slicing each spanning vector along its environment indices gives vectors
    spanning the reduced support.  Their column space also covers every
    superposition and mixture represented by the input subspace.
    """
    if len(set(qubit_order)) != len(qubit_order):
        raise ValueError("Subspace qubit order contains duplicate qubits.")
    if len(set(scope)) != len(scope) or not set(scope).issubset(qubit_order):
        raise ValueError("Subspace observation must use distinct retained qubits.")
    columns = np.asarray(basis, dtype=complex)
    if columns.ndim != 2 or columns.shape[0] != 2 ** len(qubit_order):
        raise ValueError("Subspace basis dimensions do not match its qubit order.")
    selected = tuple(qubit_order.index(qubit) for qubit in scope)
    environment = tuple(index for index in range(len(qubit_order)) if index not in selected)
    tensor = columns.reshape((2,) * len(qubit_order) + (columns.shape[1],))
    slices = tensor.transpose(selected + environment + (len(qubit_order),)).reshape(
        2 ** len(scope), -1,
    )
    return _orthonormal_support(slices)


def _acted_qubits(gate: GateSpec) -> tuple[str, ...]:
    return tuple(dict.fromkeys(gate.controls + gate.targets + gate.extra_qargs))


def _apply_to_basis(
    basis: np.ndarray,
    qubit_order: tuple[str, ...],
    operator: np.ndarray,
    acted_qubits: tuple[str, ...],
) -> np.ndarray:
    positions = tuple(qubit_order.index(qubit) for qubit in acted_qubits)
    rest = tuple(index for index in range(len(qubit_order)) if index not in positions)
    permutation = positions + rest + (len(qubit_order),)
    shape = (2,) * len(qubit_order) + (basis.shape[1],)
    front = basis.reshape(shape).transpose(permutation).reshape(2 ** len(positions), -1)
    updated = (operator @ front).reshape(shape).transpose(np.argsort(permutation))
    return _readonly(updated.reshape(basis.shape))


def _search_groups(model: QAMModel) -> tuple[frozenset[str], ...]:
    universe = set(model.system_model.qubit_universe)
    groups: list[frozenset[str]] = []
    if not model.properties:
        raise ValueError("grover_subspace execution requires a grover_span state property.")
    for prop in model.properties:
        expectation = prop.expectation
        if (
            prop.kind is not PropertyKind.RELATION
            or prop.observable.kind not in {"state", "quantum_state"}
            or not isinstance(expectation, tuple)
            or not expectation
            or expectation[0] != "grover_span"
        ):
            raise ValueError(
                "grover_subspace execution supports only grover_span state relations; "
                f"property {prop.id!r} requests a different observation."
            )
        params = expectation[1] if len(expectation) > 1 else None
        if not isinstance(params, dict):
            raise ValueError(f"Property {prop.id!r} needs grover_span search and phase parameters.")
        search = params.get("search")
        phase = params.get("phase")
        if (
            not isinstance(search, (tuple, list))
            or not search
            or not all(isinstance(qubit, str) for qubit in search)
            or len(set(search)) != len(search)
            or not set(search).issubset(universe)
            or not isinstance(phase, str)
            or phase not in universe
            or phase in search
        ):
            raise ValueError(
                f"Property {prop.id!r} needs distinct declared search qubits "
                "and a declared phase qubit outside that search register."
            )
        group = frozenset(search)
        if group not in groups:
            groups.append(group)
    return tuple(groups)


def run_grover_subspace(
    model: QAMModel, execution_id: str | None = None,
) -> ExecutionContext:
    """Execute a deterministic chain with Grover subspace unit summaries.

    The supported initial abstraction is the span of the zero and uniform
    search-register vectors, with the other component qubits initially zero.
    Components are bounded independently, so allocation depends on component
    size rather than the total number of model qubits.
    """
    system = model.system_model
    if system.defaults.default_summary_kind.value != SUMMARY_KIND:
        raise ValueError("Grover subspace execution requires default_summary_kind grover_subspace.")
    if not model.executions:
        raise ValueError("Grover subspace execution requires an execution declaration.")
    execution = model.execution(execution_id) if execution_id is not None else next(iter(model.executions.values()))
    trace = model.trace(execution.trace_id)
    binding = execution.initial_binding
    initializer = binding.payload_initializer
    if initializer is not None and (
        not isinstance(initializer, str)
        or initializer not in {"zero_density_matrix", "zero_state_vector"}
    ):
        raise ValueError("grover_subspace supports only the all-zero initializer.")
    if binding.measurement_values:
        raise ValueError("grover_subspace does not support initial measurement records.")
    if binding.anchor_id != trace.entry_anchor:
        raise ValueError("Grover subspace execution must start at the trace entry.")

    search_groups = _search_groups(model)
    universe = system.qubit_universe
    if len(set(universe)) != len(universe):
        raise ValueError("Grover subspace qubit universe contains duplicate qubits.")
    parents = {qubit: qubit for qubit in universe}

    def root(qubit: str) -> str:
        while parents[qubit] != qubit:
            parents[qubit] = parents[parents[qubit]]
            qubit = parents[qubit]
        return qubit

    def connect(qubits: tuple[str, ...]) -> None:
        if len(set(qubits)) != len(qubits) or not set(qubits).issubset(parents):
            raise ValueError("Grover subspace supports must contain distinct declared qubits.")
        if qubits:
            leader = root(qubits[0])
            for qubit in qubits[1:]:
                parents[root(qubit)] = leader

    class_ids = {binding.state_class_id}
    gate_operations: list[tuple[tuple[str, ...], np.ndarray | None]] = []
    previous_anchor = trace.entry_anchor
    previous_class = binding.state_class_id
    visited = {previous_anchor}
    for step in trace.steps:
        if step.source_anchor != previous_anchor or step.target_anchor in visited:
            raise ValueError("grover_subspace supports a single ordered chain of trace steps.")
        transition = system.transition_class(step.transition_class)
        if transition.source_class != previous_class:
            raise ValueError("Grover subspace transition source class differs from the preceding state.")
        if len(transition.branches) != 1 or transition.branches[0].kind is not BranchKind.DETERMINISTIC:
            raise ValueError("grover_subspace supports deterministic single-branch transitions only.")
        branch = transition.branches[0]
        if transition.gate.is_measurement() or branch.guard_expr is not None or branch.action_gate is not None:
            raise ValueError("grover_subspace supports unitary transitions without measurement or classical control.")
        acted = _acted_qubits(transition.gate)
        connect(acted)
        operator = None
        if transition.gate.name.upper() != "NOOP":
            operator = named_gate_matrix(transition.gate.name, transition.gate.params)
            if operator.shape != (2 ** len(acted), 2 ** len(acted)) or not np.allclose(
                operator.conj().T @ operator, np.eye(operator.shape[0]), atol=1e-12, rtol=0,
            ):
                raise ValueError(f"Transition {transition.id!r} is not a supported unitary operation.")
        gate_operations.append((acted, operator))
        class_ids.update((transition.source_class, branch.target_class))
        previous_class = branch.target_class
        previous_anchor = step.target_anchor
        visited.add(previous_anchor)
    if visited != set(trace.anchors) or execution.stop_anchor not in visited:
        raise ValueError("grover_subspace requires all trace nodes to belong to its execution chain.")

    for class_id in class_ids:
        for unit in system.state_class(class_id).units:
            if unit.resolved_summary_kind(system.defaults).value != SUMMARY_KIND:
                raise ValueError("grover_subspace execution cannot mix subspace and numerical unit summaries.")
            if not unit.qubits:
                raise ValueError("Grover subspace units must contain at least one qubit.")
            connect(unit.qubits)

    components: dict[str, tuple[str, ...]] = {}
    for qubit in universe:
        key = root(qubit)
        components[key] = components.get(key, ()) + (qubit,)
    qubit_component = {qubit: key for key, qubits in components.items() for qubit in qubits}
    component_search: dict[str, frozenset[str]] = {}
    for key, qubits in components.items():
        if len(qubits) > MAX_COMPONENT_QUBITS:
            raise ValueError(
                f"Grover subspace component has {len(qubits)} qubits; "
                f"the supported maximum is {MAX_COMPONENT_QUBITS}."
            )
        matching_groups = [group for group in search_groups if group.intersection(qubits)]
        if len(matching_groups) > 1:
            raise ValueError("A Grover subspace component crosses distinct search registers.")
        component_search[key] = matching_groups[0] if matching_groups else frozenset()

    # Check every component before constructing any exponentially sized basis.
    workspaces: dict[str, np.ndarray] = {}
    for key, qubits in components.items():
        zero = np.zeros(2 ** len(qubits), dtype=complex)
        zero[0] = 1
        search = component_search[key]
        if not search:
            workspaces[key] = _readonly(zero[:, None])
            continue
        plus = np.asarray([1.0 + 0.0j])
        for qubit in qubits:
            factor = np.asarray([1, 1], dtype=complex) / np.sqrt(2) if qubit in search else np.asarray([1, 0], dtype=complex)
            plus = np.kron(plus, factor)
        workspaces[key] = _orthonormal_support(np.column_stack((zero, plus)))

    def payloads_for(
        state: StateClassSpec,
        previous: dict[str, UnitPayload] | None = None,
        changed_component: str | None = None,
    ) -> dict[str, UnitPayload]:
        payloads: dict[str, UnitPayload] = {}
        for unit in state.units:
            key = qubit_component[unit.qubits[0]]
            existing = previous.get(unit.id) if previous is not None else None
            if existing is not None and existing.qubits == unit.qubits and key != changed_component:
                payloads[unit.id] = existing
            else:
                payloads[unit.id] = UnitPayload(
                    unit_id=unit.id,
                    summary_kind=SUMMARY_KIND,
                    qubits=unit.qubits,
                    data=reduced_support_basis(workspaces[key], components[key], unit.qubits),
                )
        return payloads

    context = ExecutionContext(
        system_model=system,
        trace=trace,
        execution=execution,
        anchors=AnchorStore(anchor_kind={key: anchor.kind.value for key, anchor in trace.anchors.items()}),
    )
    initial_class = system.state_class(binding.state_class_id)
    node = StateNode(
        id=context.new_node_id(),
        state_class_id=initial_class.id,
        active_qubits=tuple(dict.fromkeys(initial_class.active_unit_qubits())),
        program_classical=dict(binding.program_values),
        unit_payloads=payloads_for(initial_class),
        reach_probability=1.0,
    )
    context.add_node(node)
    context.anchors.bind(binding.anchor_id, [node.id])
    for step, (acted, operator) in zip(trace.steps, gate_operations):
        transition = system.transition_class(step.transition_class)
        branch = transition.branches[0]
        changed = None
        if operator is not None:
            changed = qubit_component[acted[0]]
            workspaces[changed] = _apply_to_basis(workspaces[changed], components[changed], operator, acted)
        state = system.state_class(branch.target_class)
        target = StateNode(
            id=context.new_node_id(),
            state_class_id=state.id,
            active_qubits=tuple(dict.fromkeys(state.active_unit_qubits())),
            program_classical=dict(node.program_classical),
            unit_payloads=payloads_for(state, node.unit_payloads, changed),
            reach_probability=1.0,
        )
        context.add_node(target)
        context.add_edge(TransitionEdge(node.id, transition.id, branch.id, target.id, 1.0))
        context.anchors.bind(step.target_anchor, [target.id])
        node = target
    return context
