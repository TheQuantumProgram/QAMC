"""Small, independent full-state reference for modeling-language experiments.

This module shares the parsed QAM IR and its ordered trace steps with the checker.
It does not test parsing independently. Quantum updates, measurement, marginals,
and reduced states use NumPy amplitudes without importing any checker backend,
unit operations, or observable implementation. All declared qubits remain in the
reference vector, including measured qubits omitted from later abstract classes.

Zero-reach records have no normalized physical state. They remain in the trace
for structural comparison, with ``state_vector=None``; measurement probabilities
conditional on such records are undefined rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from qamc.ir.enums import BranchKind, EvaluationMode, ScopeKind
from qamc.ir.model import GateSpec, QAMModel, TraceSpec
from qamc.ir.property import PropertySpec

if TYPE_CHECKING:
    from qamc.runtime.execution_context import ExecutionContext


class ReferenceUnsupported(ValueError):
    """The independent reference does not implement the requested construct."""


class UndefinedReferenceObservation(ReferenceUnsupported):
    """The selected instances have no positive reach for an aggregate."""


@dataclass(slots=True)
class ReferenceNode:
    id: str
    state_class_id: str
    qubit_order: tuple[str, ...]
    program_classical: dict[str, object]
    measurement_classical: dict[str, object]
    reach_probability: float
    state_vector: np.ndarray | None

    def all_classical(self) -> dict[str, object]:
        return {**self.program_classical, **self.measurement_classical}

    def _physical_vector(self) -> np.ndarray:
        if self.state_vector is None:
            raise ReferenceUnsupported(f"Zero-reach record {self.id} has no normalized quantum state.")
        return self.state_vector

    def reduced_density(self, qubits: tuple[str, ...]) -> np.ndarray:
        """Return a reduced density matrix in the requested qubit order."""
        vector = self._physical_vector()
        axes = _requested_axes(self.qubit_order, qubits)
        remainder = tuple(axis for axis in range(len(self.qubit_order)) if axis not in axes)
        amplitudes = vector.reshape((2,) * len(self.qubit_order)).transpose(axes + remainder)
        amplitudes = amplitudes.reshape(2 ** len(qubits), -1)
        return amplitudes @ amplitudes.conj().T

    def distribution(self, qubits: tuple[str, ...], basis: str = "Z") -> dict[object, float]:
        """Return an X- or Z-basis marginal; one-qubit outcomes are integers."""
        vector = self._physical_vector()
        axes = _requested_axes(self.qubit_order, qubits)
        normalized_basis = basis.upper()
        if normalized_basis == "X":
            vector = vector.copy()
            for qubit in qubits:
                vector = _apply_gate(vector, self.qubit_order, GateSpec("H", targets=(qubit,)))
        elif normalized_basis != "Z":
            raise ReferenceUnsupported(f"Reference basis {basis!r} is unsupported.")
        remainder = tuple(axis for axis in range(len(self.qubit_order)) if axis not in axes)
        amplitudes = vector.reshape((2,) * len(self.qubit_order)).transpose(axes + remainder)
        probabilities = np.sum(np.abs(amplitudes.reshape(2 ** len(qubits), -1)) ** 2, axis=1)
        result: dict[object, float] = {}
        for index, probability in enumerate(probabilities):
            bits = tuple((index >> (len(qubits) - axis - 1)) & 1 for axis in range(len(qubits)))
            result[bits[0] if len(bits) == 1 else bits] = float(probability)
        return result


@dataclass(slots=True)
class ReferenceEdge:
    source_node_id: str
    transition_class_id: str
    branch_id: str
    target_node_id: str
    local_probability: float | None


@dataclass(slots=True)
class ReferenceExecution:
    qubit_order: tuple[str, ...]
    trace: TraceSpec
    execution_id: str
    nodes: dict[str, ReferenceNode] = field(default_factory=dict)
    bindings: dict[str, list[str]] = field(default_factory=dict)
    edges: list[ReferenceEdge] = field(default_factory=list)

    def nodes_at(self, anchor_id: str) -> tuple[ReferenceNode, ...]:
        return tuple(self.nodes[node_id] for node_id in self.bindings.get(anchor_id, ()))


def evaluate_reference(
    model: QAMModel,
    execution_id: str | None = None,
    *,
    max_qubits: int = 20,
) -> ReferenceExecution:
    """Execute the selected parsed trace using independent full-state arithmetic.

    This is a correctness reference for small cases, not a performance baseline.
    A vector is saved per positive-reach runtime record, so total storage also
    grows with trace length and measurement branches. Only all-zero initial
    quantum states and equality guards are currently supported.
    """
    execution = model.execution(execution_id) if execution_id else next(iter(model.executions.values()))
    trace = model.trace(execution.trace_id)
    qubits = model.system_model.qubit_universe
    if len(qubits) > max_qubits:
        raise ReferenceUnsupported(f"Reference limit is {max_qubits} qubits; model declares {len(qubits)}.")
    if len(set(qubits)) != len(qubits):
        raise ValueError("Reference qubit names must be unique.")
    binding = execution.initial_binding
    if binding.payload_initializer not in (None, "zero_density_matrix", "zero_state_vector"):
        raise ReferenceUnsupported(f"Reference initializer {binding.payload_initializer!r} is unsupported.")
    vector = np.zeros(2 ** len(qubits), dtype=complex)
    vector[0] = 1.0
    result = ReferenceExecution(qubits, trace, execution.id)
    root = ReferenceNode(
        "r0", binding.state_class_id, qubits, dict(binding.program_values),
        dict(binding.measurement_values), 1.0, vector,
    )
    result.nodes[root.id] = root
    result.bindings[binding.anchor_id] = [root.id]
    incoming_rotations: dict[str, tuple[GateSpec, ReferenceNode]] = {}
    for step in trace.steps:
        transition = model.system_model.transition_class(step.transition_class)
        gate = transition.gate
        rotation = gate.name.upper() in {"P", "CP", "RBS"} and len(gate.params) == 1
        targets: list[str] = []
        for source in result.nodes_at(step.source_anchor):
            is_measurement = transition.gate.is_measurement()
            if is_measurement:
                _validate_measurement(transition.gate, qubits)
                updated = source.state_vector
            else:
                initial = source.state_vector
                effective_gate = gate
                if rotation and source.id in incoming_rotations:
                    previous, parent = incoming_rotations[source.id]
                    previous_args = previous.controls + previous.targets + previous.extra_qargs
                    arguments = gate.controls + gate.targets + gate.extra_qargs
                    angle, last_angle = float(gate.params[0]), float(previous.params[0])
                    if (previous.name.upper() == gate.name.upper() and previous_args == arguments
                            and (angle < 0) != (last_angle < 0)):
                        # Compose opposing rotations on the full vector before the pair.
                        # The intermediate vector remains available at its own trace position.
                        initial = parent.state_vector
                        effective_gate = replace(gate, params=(last_angle + angle,))
                updated = _apply_gate(initial, qubits, effective_gate)
            for branch in transition.branches:
                if is_measurement and branch.kind is not BranchKind.MEASUREMENT:
                    continue
                if branch.kind is BranchKind.CLASSICAL_CONTROL:
                    guard = branch.guard_expr
                    if not isinstance(guard, tuple) or len(guard) != 3 or guard[0] != "eq":
                        raise ReferenceUnsupported(f"Reference guard {guard!r} is unsupported.")
                    if source.all_classical().get(guard[1]) != guard[2]:
                        continue
                records = dict(source.measurement_classical)
                local_probability: float | None = 1.0
                branch_vector = updated
                if is_measurement:
                    local_probability, branch_vector = _measure(
                        updated, qubits, transition.gate.targets[0], branch.outcome,
                    )
                    if transition.gate.record_as:
                        records[transition.gate.record_as] = branch.outcome
                elif branch.kind is BranchKind.CLASSICAL_CONTROL and branch.action_gate is not None:
                    branch_vector = _apply_gate(updated, qubits, branch.action_gate)
                reach = source.reach_probability * (local_probability or 0.0)
                node = ReferenceNode(
                    f"r{len(result.nodes)}", branch.target_class, qubits,
                    dict(source.program_classical), records, reach,
                    None if branch_vector is None else branch_vector.copy(),
                )
                result.nodes[node.id] = node
                if rotation and branch.kind is BranchKind.DETERMINISTIC:
                    incoming_rotations[node.id] = (gate, source)
                result.edges.append(ReferenceEdge(
                    source.id, transition.id, branch.id, node.id, local_probability,
                ))
                targets.append(node.id)
        result.bindings[step.target_anchor] = targets
    return result


def _requested_axes(order: tuple[str, ...], requested: tuple[str, ...]) -> tuple[int, ...]:
    if len(set(requested)) != len(requested):
        raise ValueError("Requested qubits must be distinct.")
    unknown = set(requested).difference(order)
    if unknown:
        raise ValueError(f"Unknown reference qubits: {sorted(unknown)!r}")
    return tuple(order.index(qubit) for qubit in requested)


def _mask(order: tuple[str, ...], qubit: str) -> int:
    return 1 << (len(order) - order.index(qubit) - 1)


def _apply_gate(
    vector: np.ndarray | None, order: tuple[str, ...], gate: GateSpec,
) -> np.ndarray | None:
    """Update amplitudes by bit permutations, paired sums, and phases."""
    name = gate.name.upper()
    arities = {"NOOP": 0, "ID": 1, "I": 1, "H": 1, "X": 1, "Y": 1,
               "Z": 1, "S": 1, "SDG": 1, "T": 1, "TDG": 1, "P": 1,
               "CX": 2, "CNOT": 2, "CZ": 2, "CP": 2, "CCX": 3,
               "TOFFOLI": 3, "SWAP": 2, "RBS": 2}
    if name not in arities:
        raise ReferenceUnsupported(f"Reference gate {gate.name!r} is unsupported.")
    arguments = gate.controls + gate.targets + gate.extra_qargs
    if len(arguments) != arities[name]:
        raise ReferenceUnsupported(f"Reference gate {name} expects {arities[name]} qubits, got {arguments!r}.")
    _requested_axes(order, arguments)
    if len(gate.params) != (1 if name in {"P", "CP", "RBS"} else 0):
        raise ReferenceUnsupported(f"Invalid reference parameters for {name}: {gate.params!r}")
    theta = float(gate.params[0]) if gate.params else 0.0
    if vector is None:
        return None
    result = vector.copy()
    if name in {"NOOP", "ID", "I"}:
        return result
    indices = np.arange(vector.size)
    masks = tuple(_mask(order, argument) for argument in arguments)
    if name in {"X", "CX", "CNOT", "CCX", "TOFFOLI"}:
        condition = np.ones(vector.size, dtype=bool)
        for control in masks[:-1]:
            condition &= (indices & control) != 0
        selected = indices[condition]
        result[selected ^ masks[-1]] = vector[selected]
    elif name in {"H", "Y"}:
        zero = indices[(indices & masks[0]) == 0]
        one = zero | masks[0]
        if name == "H":
            result[zero] = (vector[zero] + vector[one]) / np.sqrt(2)
            result[one] = (vector[zero] - vector[one]) / np.sqrt(2)
        else:
            result[zero] = -1j * vector[one]
            result[one] = 1j * vector[zero]
    elif name in {"SWAP", "RBS"}:
        zero_one = indices[((indices & masks[0]) == 0) & ((indices & masks[1]) != 0)]
        one_zero = zero_one ^ masks[0] ^ masks[1]
        if name == "SWAP":
            result[zero_one], result[one_zero] = vector[one_zero], vector[zero_one]
        else:
            # RBS(theta)|10> = sin(theta)|01> + cos(theta)|10>.
            result[zero_one] = np.cos(theta) * vector[zero_one] + np.sin(theta) * vector[one_zero]
            result[one_zero] = -np.sin(theta) * vector[zero_one] + np.cos(theta) * vector[one_zero]
    else:
        exact_phases = {"Z": -1, "CZ": -1, "S": 1j, "SDG": -1j}
        angle = {"T": np.pi / 4, "TDG": -np.pi / 4}.get(name, theta)
        condition = np.ones(vector.size, dtype=bool)
        for mask in masks:
            condition &= (indices & mask) != 0
        result[condition] *= exact_phases[name] if name in exact_phases else np.exp(1j * angle)
    return result


def _validate_measurement(gate: GateSpec, order: tuple[str, ...]) -> None:
    if (gate.name.upper() != "MZ" or len(gate.targets) != 1
            or gate.controls or gate.extra_qargs or gate.params):
        raise ReferenceUnsupported(f"Reference measurement {gate!r} is unsupported.")
    _requested_axes(order, gate.targets)


def _measure(
    vector: np.ndarray | None, order: tuple[str, ...], qubit: str, outcome: object,
) -> tuple[float | None, np.ndarray | None]:
    if outcome not in (0, 1):
        raise ReferenceUnsupported(f"Reference MZ outcome {outcome!r} is unsupported.")
    if vector is None:
        return None, None
    indices = np.arange(vector.size)
    selected = ((indices & _mask(order, qubit)) != 0) == bool(outcome)
    probability = float(np.sum(np.abs(vector[selected]) ** 2))
    if probability == 0.0:
        return 0.0, None
    collapsed = np.zeros_like(vector)
    collapsed[selected] = vector[selected] / np.sqrt(probability)
    return probability, collapsed


def reference_observation(reference: ReferenceExecution, property_spec: PropertySpec) -> object:
    """Compute an observation over the positive-reach instances at the selected positions."""
    if property_spec.scope_kind is ScopeKind.ANCHORS:
        anchors = tuple(property_spec.scope_value)
    elif property_spec.scope_kind is ScopeKind.INTERVAL:
        order = reference.trace.topological_anchor_order()
        first, last = property_spec.scope_value
        anchors = order[order.index(first):order.index(last) + 1]
    else:
        raise ReferenceUnsupported(f"Reference scope {property_spec.scope_kind!r} is unsupported.")
    nodes = [node for anchor in anchors for node in reference.nodes_at(anchor)
             if node.reach_probability > 0]
    if not nodes and property_spec.mode is EvaluationMode.AGGREGATE:
        raise UndefinedReferenceObservation("An aggregate with zero total reach has no normalized observation.")
    observable = property_spec.observable
    if observable.kind == "classical":
        values = [tuple(node.all_classical().get(name) for name in observable.classical_vars) for node in nodes]
        values = [value[0] if len(value) == 1 else value for value in values]
        if property_spec.kind.value == "assertion":
            return tuple(values)
        observations = [{value: 1.0} for value in values]
    else:
        if observable.kind not in {"quantum", "state", "quantum_state"}:
            raise ReferenceUnsupported(f"Reference observable {observable.kind!r} is unsupported.")
        if observable.kind in {"state", "quantum_state"}:
            observations = [node.reduced_density(observable.qubits) for node in nodes]
        else:
            observations = [node.distribution(observable.qubits, observable.basis or "Z") for node in nodes]
    if property_spec.mode is not EvaluationMode.AGGREGATE:
        return tuple(observations)
    total = sum(node.reach_probability for node in nodes)
    weights = [node.reach_probability / total for node in nodes]
    if observable.kind in {"state", "quantum_state"}:
        return sum(weight * value for weight, value in zip(weights, observations))
    aggregate: dict[object, float] = {}
    for weight, distribution in zip(weights, observations):
        for outcome, probability in distribution.items():
            aggregate[outcome] = aggregate.get(outcome, 0.0) + weight * probability
    return aggregate


@dataclass(slots=True)
class ReferenceComparison:
    structure_matches: bool
    mismatches: tuple[str, ...]
    max_density_error: float
    max_reach_probability_error: float
    max_branch_probability_error: float
    compared_payloads: int
    compared_nodes: int
    compared_edges: int
    skipped_zero_nodes: int
    skipped_undefined_edges: int
    tolerance: float = 1e-9

    @property
    def passed(self) -> bool:
        return self.structure_matches and max(
            self.max_density_error, self.max_reach_probability_error, self.max_branch_probability_error,
        ) <= self.tolerance

    @property
    def matched(self) -> bool:
        return self.passed

    @property
    def supported(self) -> bool:
        # Unsupported arithmetic raises ReferenceUnsupported before comparison.
        return True

    @property
    def status(self) -> str:
        return "matched" if self.matched else "mismatch"


def compare_context(
    reference: ReferenceExecution, context: ExecutionContext, *, tolerance: float = 1e-9,
) -> ReferenceComparison:
    """Compare positional trace bindings, records, probabilities, and local states.

    Runtime identifiers need not agree. Correspondence uses trace node and
    branch occurrence order; differing classical values or classes are reported
    rather than silently matched to another instance. A discrepancy in an
    abstract summary can indicate abstraction loss, not necessarily an engine bug.
    """
    issues: list[str] = []
    actual_to_reference: dict[str, str] = {}
    density_error = reach_error = branch_error = 0.0
    compared_payloads = compared_edges = skipped_zero = skipped_edges = 0
    for anchor in reference.trace.anchors:
        expected_ids = reference.bindings.get(anchor, [])
        actual_ids = context.anchors.nodes_of(anchor)
        if len(expected_ids) != len(actual_ids):
            issues.append(f"{anchor}: expected {len(expected_ids)} instances, got {len(actual_ids)}")
        for index, (expected_id, actual_id) in enumerate(zip(expected_ids, actual_ids)):
            previous = actual_to_reference.setdefault(actual_id, expected_id)
            if previous != expected_id:
                issues.append(f"{anchor}[{index}]: inconsistent instance binding")
    if len(actual_to_reference) != len(context.nodes) or len(set(actual_to_reference.values())) != len(reference.nodes):
        issues.append("The runtime and reference do not have a complete one-to-one instance correspondence")
    for actual_id, expected_id in actual_to_reference.items():
        actual, expected = context.node(actual_id), reference.nodes[expected_id]
        if actual.state_class_id != expected.state_class_id:
            issues.append(f"{actual_id}: state class differs")
        if (actual.program_classical != expected.program_classical
                or actual.measurement_classical != expected.measurement_classical):
            issues.append(f"{actual_id}: classical records differ")
        state_class = context.system_model.state_classes.get(expected.state_class_id)
        if state_class is None:
            issues.append(f"{actual_id}: declared state class is missing")
        elif set(actual.unit_payloads) != set(state_class.unit_ids()):
            issues.append(f"{actual_id}: unit payload identifiers differ from the state class")
        reach_error = max(reach_error, _scalar_error(actual.reach_probability, expected.reach_probability))
        if (actual.reach_probability > 0) != (expected.reach_probability > 0):
            issues.append(f"{actual_id}: positive reach differs from the reference")
        if expected.state_vector is None:
            skipped_zero += 1
            continue
        for unit in actual.unit_payloads.values():
            if unit.summary_kind in {"density_factor", "grover_subspace"}:
                raise ReferenceUnsupported(
                    "The reference vector represents one input. Comparing a Grover input subspace "
                    "requires support containment or a reference mixture over its initial span."
                )
            expected_matrix = expected.reduced_density(unit.qubits)
            matrix = np.asarray(unit.data, dtype=complex)
            if unit.summary_kind == "state_vector":
                vector = matrix.reshape(-1)
                matrix = np.outer(vector, vector.conj())
            elif unit.summary_kind != "density_matrix":
                issues.append(f"{actual_id}/{unit.unit_id}: unsupported summary kind {unit.summary_kind!r}")
                continue
            if matrix.shape != expected_matrix.shape:
                issues.append(f"{actual_id}/{unit.unit_id}: density matrix shape differs")
                continue
            difference = np.abs(matrix - expected_matrix)
            error = float(np.max(difference)) if np.all(np.isfinite(difference)) else float("inf")
            density_error = max(density_error, error)
            compared_payloads += 1
    edge_lookup = {
        (edge.source_node_id, edge.transition_class_id, edge.branch_id, edge.target_node_id): edge
        for edge in reference.edges
    }
    seen_edges: set[tuple[str | None, str, str, str | None]] = set()
    for edge in context.edges:
        key = (actual_to_reference.get(edge.source_node_id), edge.transition_class_id,
               edge.branch_id, actual_to_reference.get(edge.target_node_id))
        expected_edge = edge_lookup.get(key)
        if expected_edge is None or key in seen_edges:
            issues.append(f"Unexpected or duplicate edge: {key!r}")
            continue
        seen_edges.add(key)
        if expected_edge.local_probability is None:
            skipped_edges += 1
            continue
        branch_error = max(branch_error, _scalar_error(edge.local_probability, expected_edge.local_probability))
        if (edge.local_probability > 0) != (expected_edge.local_probability > 0):
            issues.append(f"{key!r}: positive branch probability differs from the reference")
        compared_edges += 1
    if len(seen_edges) != len(reference.edges):
        issues.append("The runtime and reference edge sets differ")
    return ReferenceComparison(
        not issues, tuple(issues), density_error, reach_error, branch_error,
        compared_payloads, len(actual_to_reference), compared_edges, skipped_zero, skipped_edges, tolerance,
    )


def _scalar_error(actual: float, expected: float) -> float:
    difference = abs(float(actual) - float(expected))
    return difference if np.isfinite(difference) else float("inf")
