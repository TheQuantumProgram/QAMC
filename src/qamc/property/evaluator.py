"""Property evaluators for QAMC runtime executions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qamc.backend import DensityMatrixBackend
from qamc.engine.grover_subspace import DENSITY_FACTOR_KIND
from qamc.ir.enums import EvaluationMode, PropertyKind
from qamc.ir.property import ObservableSpec, PropertySpec
from qamc.property.observables import (
    SubspaceInconclusiveError, classical_observation, quantum_density_matrix,
    quantum_distribution, quantum_support_basis,
)
from qamc.property.scopes import ScopeSelector
from qamc.runtime.execution_context import ExecutionContext
from qamc.runtime.state_node import StateNode


@dataclass(slots=True)
class PropertyEvaluationResult:
    """Result of evaluating one property."""

    property_id: str
    passed: bool | None
    selected_anchor_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    observed: object
    diagnostic: str | None = None


class PropertyEvaluator:
    """Evaluate QAM properties on a runtime execution context."""

    def __init__(
        self,
        *,
        backend: DensityMatrixBackend | None = None,
        scope_selector: ScopeSelector | None = None,
    ) -> None:
        self.backend = backend or DensityMatrixBackend()
        self.scope_selector = scope_selector or ScopeSelector()

    def evaluate(self, context: ExecutionContext, property_spec: PropertySpec) -> PropertyEvaluationResult:
        selection = self.scope_selector.select(context, property_spec)
        bound_nodes = (context.node(node_id) for node_id in selection.node_ids)
        selected_nodes = [node for node in bound_nodes if node.reach_probability > 0]

        if not selected_nodes and property_spec.mode is EvaluationMode.AGGREGATE:
            return PropertyEvaluationResult(
                property_id=property_spec.id,
                passed=None,
                selected_anchor_ids=selection.anchor_ids,
                selected_node_ids=(),
                observed=None,
                diagnostic="No selected state instance has positive reach probability.",
            )

        if property_spec.kind is PropertyKind.ASSERTION:
            passed, observed = self._evaluate_assertion(property_spec, selected_nodes)
        elif property_spec.kind is PropertyKind.DISTRIBUTION:
            passed, observed = self._evaluate_distribution(property_spec, selected_nodes)
        elif property_spec.kind is PropertyKind.REACHABILITY:
            passed, observed = self._evaluate_reachability(property_spec, selected_nodes)
        elif property_spec.kind is PropertyKind.RELATION:
            passed, observed = self._evaluate_relation(property_spec, selected_nodes)
        else:
            raise ValueError(f"Unsupported property kind: {property_spec.kind!r}")

        return PropertyEvaluationResult(
            property_id=property_spec.id,
            passed=passed,
            selected_anchor_ids=selection.anchor_ids,
            selected_node_ids=tuple(node.id for node in selected_nodes),
            observed=observed,
        )

    def evaluate_all(
        self,
        context: ExecutionContext,
        properties: tuple[PropertySpec, ...],
    ) -> list[PropertyEvaluationResult]:
        return [self.evaluate(context, property_spec) for property_spec in properties]

    def _evaluate_assertion(
        self,
        property_spec: PropertySpec,
        selected_nodes: list[StateNode],
    ) -> tuple[bool, object]:
        observations = [classical_observation(node, property_spec.observable) for node in selected_nodes]
        matches = [self._assertion_matches(observation, property_spec.expectation) for observation in observations]
        if property_spec.mode is EvaluationMode.POINTWISE_ALL:
            passed = all(matches)
        elif property_spec.mode is EvaluationMode.POINTWISE_ANY:
            passed = any(matches)
        else:
            passed = self._assertion_matches(tuple(observations), property_spec.expectation)
        return passed, tuple(observations)

    def _evaluate_distribution(
        self,
        property_spec: PropertySpec,
        selected_nodes: list[StateNode],
    ) -> tuple[bool, object]:
        if property_spec.observable.kind == "classical":
            distributions = [
                {classical_observation(node, property_spec.observable): 1.0}
                for node in selected_nodes
            ]
        else:
            distributions = [
                quantum_distribution(node, property_spec.observable, self.backend) for node in selected_nodes
            ]

        if property_spec.mode is EvaluationMode.POINTWISE_ALL:
            passed = all(self._distribution_matches(distribution, property_spec.expectation) for distribution in distributions)
            return passed, tuple(distributions)
        if property_spec.mode is EvaluationMode.POINTWISE_ANY:
            passed = any(self._distribution_matches(distribution, property_spec.expectation) for distribution in distributions)
            return passed, tuple(distributions)

        aggregate = self._aggregate_distributions(selected_nodes, distributions)
        passed = self._distribution_matches(aggregate, property_spec.expectation)
        return passed, aggregate

    def _evaluate_relation(
        self,
        property_spec: PropertySpec,
        selected_nodes: list[StateNode],
    ) -> tuple[bool, object]:
        if property_spec.observable.kind in {"state", "quantum_state"}:
            return self._evaluate_state_relation(property_spec, selected_nodes)

        distributions = [
            quantum_distribution(node, property_spec.observable, self.backend) for node in selected_nodes
        ]

        if property_spec.mode is EvaluationMode.POINTWISE_ALL:
            passed = all(self._relation_matches(distribution, property_spec.expectation) for distribution in distributions)
            return passed, tuple(distributions)
        if property_spec.mode is EvaluationMode.POINTWISE_ANY:
            passed = any(self._relation_matches(distribution, property_spec.expectation) for distribution in distributions)
            return passed, tuple(distributions)

        aggregate = self._aggregate_distributions(selected_nodes, distributions)
        passed = self._relation_matches(aggregate, property_spec.expectation)
        return passed, aggregate

    def _evaluate_state_relation(
        self,
        property_spec: PropertySpec,
        selected_nodes: list[StateNode],
    ) -> tuple[bool, object]:
        if any(payload.summary_kind in {"grover_subspace", DENSITY_FACTOR_KIND}
               for node in selected_nodes for payload in node.unit_payloads.values()):
            return self._evaluate_subspace_relation(property_spec, selected_nodes)
        matrices = [
            quantum_density_matrix(node, property_spec.observable, self.backend) for node in selected_nodes
        ]

        if property_spec.mode is EvaluationMode.POINTWISE_ALL:
            passed = all(
                self._state_relation_matches(matrix, property_spec.observable, property_spec.expectation)
                for matrix in matrices
            )
            return passed, tuple(matrices)
        if property_spec.mode is EvaluationMode.POINTWISE_ANY:
            passed = any(
                self._state_relation_matches(matrix, property_spec.observable, property_spec.expectation)
                for matrix in matrices
            )
            return passed, tuple(matrices)

        aggregate = self._aggregate_density_matrices(selected_nodes, matrices)
        passed = self._state_relation_matches(aggregate, property_spec.observable, property_spec.expectation)
        return passed, aggregate

    def _evaluate_subspace_relation(
        self, property_spec: PropertySpec, selected_nodes: list[StateNode],
    ) -> tuple[bool, object]:
        expectation = property_spec.expectation
        if not isinstance(expectation, tuple) or not expectation or expectation[0] != "grover_span":
            raise ValueError("Grover subspace execution supports witness-subspace relations.")
        params = expectation[1] if len(expectation) > 1 and isinstance(expectation[1], dict) else {}
        generators = _grover_span_generators(property_spec.observable.qubits, params)
        bases = [quantum_support_basis(node, property_spec.observable) for node in selected_nodes]
        matches = [_matrix_supported_by_generators(basis @ basis.conj().T, generators, tolerance=1e-8)
                   for basis in bases]
        if property_spec.mode is EvaluationMode.POINTWISE_ANY:
            proved = any(matches)
        else:
            # A mixture of positive-weight states has the span of their supports.
            proved = all(matches)
        if not proved:
            raise SubspaceInconclusiveError(
                f"Property {property_spec.id!r} is not proved by the abstract subspace; "
                "refinement is required to determine a concrete violation."
            )
        if selected_nodes and all(
            payload.summary_kind == DENSITY_FACTOR_KIND
            for node in selected_nodes for payload in node.unit_payloads.values()
        ):
            matrices = tuple(quantum_density_matrix(node, property_spec.observable, self.backend)
                             for node in selected_nodes)
            return True, {"summary_kind": DENSITY_FACTOR_KIND, "representative_density_matrices": matrices}
        return True, {"summary_kind": "grover_subspace", "support_bases": tuple(bases)}

    def _evaluate_reachability(
        self,
        property_spec: PropertySpec,
        selected_nodes: list[StateNode],
    ) -> tuple[bool, object]:
        distributions = [
            quantum_distribution(node, property_spec.observable, self.backend) for node in selected_nodes
        ]

        if property_spec.mode is EvaluationMode.POINTWISE_ALL:
            passed = all(self._reachability_matches(distribution, property_spec.expectation) for distribution in distributions)
            return passed, tuple(distributions)
        if property_spec.mode is EvaluationMode.POINTWISE_ANY:
            passed = any(self._reachability_matches(distribution, property_spec.expectation) for distribution in distributions)
            return passed, tuple(distributions)

        aggregate = self._aggregate_distributions(selected_nodes, distributions)
        passed = self._reachability_matches(aggregate, property_spec.expectation)
        return passed, aggregate

    def _aggregate_distributions(
        self,
        nodes: list[StateNode],
        distributions: list[dict[object, float]],
    ) -> dict[object, float]:
        weighted: dict[object, float] = {}
        total_mass = sum(node.reach_probability for node in nodes)
        for node, distribution in zip(nodes, distributions):
            weight = node.reach_probability / total_mass
            for outcome, probability in distribution.items():
                weighted[outcome] = weighted.get(outcome, 0.0) + weight * probability
        return weighted

    def _aggregate_density_matrices(
        self,
        nodes: list[StateNode],
        matrices: list[np.ndarray],
    ) -> np.ndarray:
        weighted = np.zeros_like(matrices[0], dtype=complex)
        total_mass = sum(node.reach_probability for node in nodes)
        for node, matrix in zip(nodes, matrices):
            weighted += (node.reach_probability / total_mass) * matrix
        return weighted

    def _assertion_matches(self, observation: object, expectation: object) -> bool:
        if callable(expectation):
            return bool(expectation(observation))
        if not isinstance(expectation, tuple) or not expectation:
            return observation == expectation

        op = expectation[0]
        if op == "eq":
            return observation == expectation[1]
        if op == "in":
            return observation in expectation[1]
        if op == "contains":
            return expectation[1] in observation
        raise ValueError(f"Unsupported assertion expectation: {expectation!r}")

    def _distribution_matches(
        self,
        distribution: dict[object, float],
        expectation: object,
        tolerance: float = 1e-9,
    ) -> bool:
        if not isinstance(expectation, dict):
            raise ValueError("Distribution expectation must be a dictionary.")
        outcomes = set(distribution) | set(expectation)
        for outcome in outcomes:
            if abs(distribution.get(outcome, 0.0) - expectation.get(outcome, 0.0)) > tolerance:
                return False
        return True

    def _relation_matches(
        self,
        distribution: dict[object, float],
        expectation: object,
        tolerance: float = 1e-9,
    ) -> bool:
        if not isinstance(expectation, tuple) or not expectation:
            raise ValueError(f"Unsupported relation expectation: {expectation!r}")
        op = expectation[0]
        if op == "equal_bits":
            total_good_mass = 0.0
            for outcome, probability in distribution.items():
                if not isinstance(outcome, tuple):
                    raise ValueError("equal_bits relation expects tuple outcomes.")
                if len(set(outcome)) == 1:
                    total_good_mass += probability
            return abs(total_good_mass - 1.0) <= tolerance
        if op == "support_subset":
            allowed = set(expectation[1])
            return all((outcome in allowed) or abs(probability) <= tolerance for outcome, probability in distribution.items())
        raise ValueError(f"Unsupported relation expectation: {expectation!r}")

    def _state_relation_matches(
        self,
        matrix: np.ndarray,
        observable: ObservableSpec,
        expectation: object,
        tolerance: float = 1e-8,
    ) -> bool:
        if not isinstance(expectation, tuple) or not expectation:
            raise ValueError(f"Unsupported state relation expectation: {expectation!r}")
        op = expectation[0]
        if op == "grover_span":
            params = expectation[1] if len(expectation) > 1 and isinstance(expectation[1], dict) else {}
            generators = _grover_span_generators(observable.qubits, params)
            return _matrix_supported_by_generators(matrix, generators, tolerance=tolerance)
        raise ValueError(f"Unsupported state relation expectation: {expectation!r}")

    def _reachability_matches(
        self,
        distribution: dict[object, float],
        expectation: object,
        tolerance: float = 1e-9,
    ) -> bool:
        if not isinstance(expectation, tuple) or len(expectation) != 3:
            raise ValueError(f"Unsupported reachability expectation: {expectation!r}")
        op, outcome, threshold = expectation
        if op != "probability_gt":
            raise ValueError(f"Unsupported reachability operator: {op!r}")
        return distribution.get(outcome, 0.0) > float(threshold) + tolerance


def _grover_span_generators(
    scope: tuple[str, ...],
    params: dict[object, object],
) -> tuple[np.ndarray, ...]:
    search_qubits = _tuple_param(params, "search")
    phase_qubit = _str_param(params, "phase")
    witnesses = _tuple_param(params, "witnesses", default=("zero", "plus"))
    if not search_qubits:
        raise ValueError("grover_span expects at least one search qubit.")
    if phase_qubit in search_qubits:
        raise ValueError("grover_span phase qubit must be outside the search register.")
    return tuple(
        _grover_witness_density(scope, search_qubits, phase_qubit, witness)
        for witness in witnesses
    )


def _grover_witness_density(
    scope: tuple[str, ...],
    search_qubits: tuple[str, ...],
    phase_qubit: str,
    witness: str,
) -> np.ndarray:
    normalized_witness = witness.lower()
    if normalized_witness not in {"zero", "plus"}:
        raise ValueError(f"Unsupported grover_span witness: {witness!r}")

    vector = np.asarray([1.0 + 0.0j])
    for qubit in scope:
        if qubit == phase_qubit:
            local = _basis_vector("minus")
        elif qubit in search_qubits:
            local = _basis_vector(normalized_witness)
        else:
            local = _basis_vector("zero")
        vector = np.kron(vector, local)
    return np.outer(vector, vector.conj())


def _basis_vector(label: str) -> np.ndarray:
    if label == "zero":
        return np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    if label == "plus":
        return np.asarray([1.0 + 0.0j, 1.0 + 0.0j]) / np.sqrt(2.0)
    if label == "minus":
        return np.asarray([1.0 + 0.0j, -1.0 + 0.0j]) / np.sqrt(2.0)
    raise ValueError(f"Unsupported basis vector: {label!r}")


def _matrix_in_linear_span(
    matrix: np.ndarray,
    generators: tuple[np.ndarray, ...],
    *,
    tolerance: float,
) -> bool:
    if not generators:
        return False
    target = np.asarray(matrix, dtype=complex).reshape(-1)
    basis = np.column_stack([np.asarray(generator, dtype=complex).reshape(-1) for generator in generators])
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    residual = target - basis @ coefficients
    return float(np.linalg.norm(residual)) <= tolerance


def _matrix_supported_by_generators(
    matrix: np.ndarray,
    generators: tuple[np.ndarray, ...],
    *,
    tolerance: float,
) -> bool:
    """Check support containment in the Hilbert span of PSD witness matrices.

    The range of their sum spans the witness vectors. Projecting onto that
    range preserves coherent superpositions as well as mixtures of witnesses.
    """
    if not generators:
        return False
    witness_sum = np.sum(np.asarray(generators, dtype=complex), axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(witness_sum)
    basis = eigenvectors[:, eigenvalues > tolerance]
    if basis.shape[1] == 0:
        return False
    projector = basis @ basis.conj().T
    target = np.asarray(matrix, dtype=complex)
    return float(np.linalg.norm(target - projector @ target @ projector)) <= tolerance


def _tuple_param(
    params: dict[object, object],
    key: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    value = params.get(key, default)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _str_param(params: dict[object, object], key: str) -> str:
    if key not in params:
        raise ValueError(f"grover_span expects {key}=... argument.")
    return str(params[key])
