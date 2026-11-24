"""Observable extraction helpers for property evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from qamc.backend import DensityMatrixBackend
from qamc.engine.grover_subspace import DENSITY_FACTOR_KIND, reduced_density_matrix, reduced_support_basis
from qamc.ir.property import ObservableSpec
from qamc.runtime.state_node import StateNode, UnitPayload


class SubspaceInconclusiveError(ValueError):
    """The abstract support is too broad to establish an assertion.

    This is not a concrete counterexample. It is kept separate from a false
    property result so callers cannot use it as a violated atomic proposition.
    """


def _require_probability_payloads(payloads: list[UnitPayload]) -> None:
    if any(payload.summary_kind in {"grover_subspace", DENSITY_FACTOR_KIND} for payload in payloads):
        raise ValueError("The Grover subspace represents a family of states with unspecified outcome probabilities.")


def quantum_support_basis(node: StateNode, observable: ObservableSpec) -> np.ndarray:
    """Return an orthonormal basis for an abstract observation's support."""
    payload = _payload_covering_qubits(node, observable.qubits)
    if payload is None or payload.summary_kind not in {"grover_subspace", DENSITY_FACTOR_KIND}:
        raise ValueError("A subspace observation requires a covering unit representing that subspace.")
    return reduced_support_basis(payload.data, payload.qubits, observable.qubits)


def classical_observation(node: StateNode, observable: ObservableSpec) -> object:
    values = tuple(node.all_classical().get(var_name) for var_name in observable.classical_vars)
    if len(values) == 1:
        return values[0]
    return values


def quantum_distribution(
    node: StateNode,
    observable: ObservableSpec,
    backend: DensityMatrixBackend,
) -> dict[object, float]:
    payload = _payload_covering_qubits(node, observable.qubits)
    if payload is None:
        payloads = _payloads_intersecting_qubits(node, observable.qubits)
        if not payloads:
            raise ValueError(
                f"Observable qubits {observable.qubits!r} are not covered by any runtime payload."
            )
        return combined_quantum_distribution(payloads, observable.qubits, observable.basis or "Z", backend)
    _require_probability_payloads([payload])
    return backend.observe_distribution(payload, observable.qubits, basis=observable.basis or "Z")


def quantum_density_matrix(
    node: StateNode,
    observable: ObservableSpec,
    backend: DensityMatrixBackend,
) -> np.ndarray:
    payload = _payload_covering_qubits(node, observable.qubits)
    if payload is None:
        raise NotImplementedError(
            "State observables currently require one runtime payload covering "
            f"{observable.qubits!r}."
        )
    if payload.summary_kind == DENSITY_FACTOR_KIND:
        return reduced_density_matrix(payload.data, payload.qubits, observable.qubits)
    if payload.summary_kind == "grover_subspace":
        raise ValueError("An unweighted subspace basis does not specify a density matrix.")
    reduced = backend.partial_trace_payload(payload, observable.qubits)
    if reduced.summary_kind == "state_vector":
        vector = np.asarray(reduced.data, dtype=complex).reshape(-1)
        return np.outer(vector, vector.conj())
    return np.asarray(reduced.data, dtype=complex)


def _payload_covering_qubits(node: StateNode, qubits: tuple[str, ...]) -> UnitPayload | None:
    for payload in node.unit_payloads.values():
        if set(qubits).issubset(set(payload.qubits)):
            return payload
    return None


def _payloads_intersecting_qubits(node: StateNode, qubits: tuple[str, ...]) -> list[UnitPayload]:
    qubit_set = set(qubits)
    payloads = [payload for payload in node.unit_payloads.values() if qubit_set.intersection(payload.qubits)]
    covered = set().union(*(set(payload.qubits) for payload in payloads)) if payloads else set()
    if not qubit_set.issubset(covered):
        missing = tuple(sorted(qubit_set.difference(covered)))
        raise ValueError(f"Observable qubits {missing!r} are missing from the runtime payload family.")
    return payloads


def combined_quantum_distribution(
    payloads: list[UnitPayload],
    target_qubits: tuple[str, ...],
    basis: str,
    backend: DensityMatrixBackend,
) -> dict[object, float]:
    """Combine local payload distributions without mutating runtime payload structure.

    This implements a lightweight factor-combination strategy:

    - if one payload already covers all requested qubits, evaluate directly;
    - if payload scopes are disjoint, multiply local marginals;
    - if payload scopes overlap in a chain/tree-like way, combine them by a
      separator-normalized product, avoiding runtime payload merging.
    """

    _require_probability_payloads(payloads)
    relevant_payloads = []
    for payload in payloads:
        scope = tuple(qubit for qubit in target_qubits if qubit in payload.qubits)
        if scope:
            relevant_payloads.append((scope, backend.observe_distribution(payload, scope, basis=basis)))
    if not relevant_payloads:
        return {}

    if len(relevant_payloads) == 1:
        return relevant_payloads[0][1]

    aggregate_scope, aggregate = relevant_payloads[0]
    used = [False] * len(relevant_payloads)
    used[0] = True

    while not all(used):
        progressed = False
        for index, (scope, distribution) in enumerate(relevant_payloads):
            if used[index]:
                continue
            overlap = tuple(qubit for qubit in aggregate_scope if qubit in scope)
            if overlap:
                aggregate = _combine_distributions(
                    aggregate,
                    aggregate_scope,
                    distribution,
                    scope,
                    overlap,
                )
                aggregate_scope = _ordered_union(aggregate_scope, scope, target_qubits)
                used[index] = True
                progressed = True
            elif set(scope).isdisjoint(set(aggregate_scope)):
                aggregate = _combine_distributions(
                    aggregate,
                    aggregate_scope,
                    distribution,
                    scope,
                    (),
                )
                aggregate_scope = _ordered_union(aggregate_scope, scope, target_qubits)
                used[index] = True
                progressed = True
        if not progressed:
            raise NotImplementedError(
                "Current multi-unit observable combination requires a disjoint or chain-like overlap structure."
            )

    if aggregate_scope != target_qubits:
        aggregate = _reorder_distribution(aggregate, aggregate_scope, target_qubits)
    return aggregate


def _combine_distributions(
    left: dict[object, float],
    left_scope: tuple[str, ...],
    right: dict[object, float],
    right_scope: tuple[str, ...],
    overlap_scope: tuple[str, ...],
    tolerance: float = 1e-12,
) -> dict[object, float]:
    union_scope = _ordered_union(left_scope, right_scope, left_scope + right_scope)
    result: dict[object, float] = defaultdict(float)

    if not overlap_scope:
        for left_key, left_prob in left.items():
            left_map = _assignment_map(left_key, left_scope)
            for right_key, right_prob in right.items():
                right_map = _assignment_map(right_key, right_scope)
                merged_map = dict(left_map)
                merged_map.update(right_map)
                result[_encode_assignment(merged_map, union_scope)] += left_prob * right_prob
        return dict(result)

    left_overlap = _marginalize_distribution(left, left_scope, overlap_scope)
    right_overlap = _marginalize_distribution(right, right_scope, overlap_scope)

    for overlap_key in set(left_overlap) | set(right_overlap):
        if abs(left_overlap.get(overlap_key, 0.0) - right_overlap.get(overlap_key, 0.0)) > 1e-6:
            raise NotImplementedError(
                "Current multi-unit observable combination expects consistent overlap marginals."
            )

    separator = left_overlap
    for left_key, left_prob in left.items():
        left_map = _assignment_map(left_key, left_scope)
        left_overlap_key = _encode_assignment(left_map, overlap_scope)
        for right_key, right_prob in right.items():
            right_map = _assignment_map(right_key, right_scope)
            right_overlap_key = _encode_assignment(right_map, overlap_scope)
            if left_overlap_key != right_overlap_key:
                continue
            denom = separator.get(left_overlap_key, 0.0)
            if abs(denom) <= tolerance:
                continue
            merged_map = dict(left_map)
            merged_map.update(right_map)
            result[_encode_assignment(merged_map, union_scope)] += (left_prob * right_prob) / denom
    return dict(result)


def _marginalize_distribution(
    distribution: dict[object, float],
    full_scope: tuple[str, ...],
    retained_scope: tuple[str, ...],
) -> dict[object, float]:
    marginalized: dict[object, float] = defaultdict(float)
    for outcome, probability in distribution.items():
        assignment = _assignment_map(outcome, full_scope)
        marginalized[_encode_assignment(assignment, retained_scope)] += probability
    return dict(marginalized)


def _assignment_map(outcome: object, scope: tuple[str, ...]) -> dict[str, int]:
    if len(scope) == 1:
        return {scope[0]: int(outcome)}
    return {qubit: int(bit) for qubit, bit in zip(scope, outcome)}


def _encode_assignment(assignment: dict[str, int], scope: tuple[str, ...]) -> object:
    bits = tuple(int(assignment[qubit]) for qubit in scope)
    if len(bits) == 1:
        return bits[0]
    return bits


def _ordered_union(
    left_scope: tuple[str, ...],
    right_scope: tuple[str, ...],
    reference_order: tuple[str, ...],
) -> tuple[str, ...]:
    members = set(left_scope).union(right_scope)
    ordered: list[str] = []
    seen: set[str] = set()
    for qubit in reference_order:
        if qubit in members and qubit not in seen:
            ordered.append(qubit)
            seen.add(qubit)
    return tuple(ordered)


def _reorder_distribution(
    distribution: dict[object, float],
    current_scope: tuple[str, ...],
    target_scope: tuple[str, ...],
) -> dict[object, float]:
    reordered: dict[object, float] = {}
    for outcome, probability in distribution.items():
        assignment = _assignment_map(outcome, current_scope)
        reordered[_encode_assignment(assignment, target_scope)] = probability
    return reordered
