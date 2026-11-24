"""State-vector backend for full-state scalability baselines."""

from __future__ import annotations

import numpy as np

from qamc.backend.density_matrix_backend import DensityMatrixBackend, MeasurementResult
from qamc.backend.unit_ops import (
    apply_local_operator_to_state_vector,
    named_gate_matrix,
    reorder_state_vector,
    state_vector_distribution,
    state_vector_partial_trace,
    state_vector_to_density_matrix,
    zero_state_vector,
)
from qamc.ir.model import GateSpec
from qamc.runtime.state_node import UnitPayload


class StateVectorBackend:
    """Numerical backend for pure full-state payloads.

    The backend keeps full-state payloads as vectors while the model remains
    pure. If a payload is partially traced, the result may be mixed, so it is
    represented as a density matrix and delegated to the density backend from
    then on.
    """

    def __init__(self) -> None:
        self._density = DensityMatrixBackend()

    def initialize_unit_payload(
        self,
        *,
        unit_id: str,
        summary_kind: str,
        qubits: tuple[str, ...],
        initializer: object,
    ) -> UnitPayload:
        if summary_kind == "density_matrix":
            return self._density.initialize_unit_payload(
                unit_id=unit_id,
                summary_kind=summary_kind,
                qubits=qubits,
                initializer=initializer,
            )
        if summary_kind != "state_vector":
            raise ValueError(f"Unsupported summary kind: {summary_kind!r}")
        if initializer not in {"zero_state_vector", "zero_density_matrix"}:
            raise ValueError(f"Unsupported initializer: {initializer!r}")
        return UnitPayload(
            unit_id=unit_id,
            summary_kind="state_vector",
            qubits=qubits,
            data=zero_state_vector(len(qubits)),
        )

    def acted_qubits(self, gate: GateSpec) -> tuple[str, ...]:
        return self._density.acted_qubits(gate)

    def apply_gate_to_payload(self, payload: UnitPayload, gate: GateSpec) -> UnitPayload:
        if payload.summary_kind == "density_matrix":
            return self._density.apply_gate_to_payload(payload, gate)
        if payload.summary_kind != "state_vector":
            raise ValueError(f"Unsupported payload summary kind: {payload.summary_kind!r}")
        if gate.name.upper() == "NOOP":
            return payload.copy()

        acted_qubits = self.acted_qubits(gate)
        for qubit in acted_qubits:
            if qubit not in payload.qubits:
                raise ValueError(
                    f"Gate {gate.name!r} acts on qubit {qubit!r} outside payload {payload.unit_id!r}."
                )
        local_operator = named_gate_matrix(gate.name, gate.params)
        updated = apply_local_operator_to_state_vector(
            np.asarray(payload.data, dtype=complex),
            payload.qubits,
            local_operator,
            acted_qubits,
        )
        return UnitPayload(
            unit_id=payload.unit_id,
            summary_kind="state_vector",
            qubits=payload.qubits,
            data=updated,
        )

    def measure_payload(self, payload: UnitPayload, qubit: str, basis: str = "Z") -> list[MeasurementResult]:
        if payload.summary_kind == "density_matrix":
            return self._density.measure_payload(payload, qubit, basis=basis)
        if payload.summary_kind != "state_vector":
            raise ValueError(f"Unsupported payload summary kind: {payload.summary_kind!r}")
        if qubit not in payload.qubits:
            raise ValueError(f"Qubit {qubit!r} not present in payload {payload.unit_id!r}.")
        if basis.upper() != "Z":
            return self._density.measure_payload(self.to_density_payload(payload), qubit, basis=basis)

        state = np.asarray(payload.data, dtype=complex)
        position = payload.qubits.index(qubit)
        n = len(payload.qubits)
        tensor = state.reshape((2,) * n)
        results: list[MeasurementResult] = []
        for outcome in (0, 1):
            mask = [slice(None)] * n
            mask[position] = outcome
            selected = tensor[tuple(mask)]
            probability = float(np.real_if_close(np.sum(np.abs(selected) ** 2)))
            post_state = np.zeros_like(tensor)
            if probability > 0:
                post_state[tuple(mask)] = selected / np.sqrt(probability)
            results.append(
                MeasurementResult(
                    outcome=outcome,
                    probability=max(probability, 0.0),
                    post_payload=UnitPayload(
                        unit_id=payload.unit_id,
                        summary_kind="state_vector",
                        qubits=payload.qubits,
                        data=post_state.reshape(state.shape),
                    ),
                )
            )
        return results

    def partial_trace_payload(self, payload: UnitPayload, retained_qubits: tuple[str, ...]) -> UnitPayload:
        if payload.summary_kind == "density_matrix":
            return self._density.partial_trace_payload(payload, retained_qubits)
        if payload.summary_kind != "state_vector":
            raise ValueError(f"Unsupported payload summary kind: {payload.summary_kind!r}")
        if retained_qubits == payload.qubits:
            return payload.copy()

        reduced = state_vector_partial_trace(
            np.asarray(payload.data, dtype=complex),
            payload.qubits,
            retained_qubits,
        )
        return UnitPayload(
            unit_id=payload.unit_id,
            summary_kind="density_matrix",
            qubits=retained_qubits,
            data=reduced,
        )

    def merge_unit_payloads(
        self,
        payloads: list[UnitPayload],
        merged_qubits: tuple[str, ...],
        *,
        unit_id: str = "__merged__",
    ) -> UnitPayload:
        if not payloads:
            raise ValueError("At least one payload is required for merging.")
        if any(payload.summary_kind == "density_matrix" for payload in payloads):
            return self._density.merge_unit_payloads(
                [self.to_density_payload(payload) for payload in payloads],
                merged_qubits,
                unit_id=unit_id,
            )

        seen_qubits: set[str] = set()
        for payload in payloads:
            overlap = seen_qubits.intersection(payload.qubits)
            if overlap:
                raise NotImplementedError(
                    "Merging overlapping payloads is not supported in the current backend."
                )
            seen_qubits.update(payload.qubits)
        if seen_qubits != set(merged_qubits):
            raise ValueError("Merged qubit set does not match payload qubit coverage.")

        ordered_payloads = sorted(
            payloads,
            key=lambda payload: min(merged_qubits.index(qubit) for qubit in payload.qubits),
        )
        current_order = ordered_payloads[0].qubits
        merged_state = np.asarray(ordered_payloads[0].data, dtype=complex)
        for payload in ordered_payloads[1:]:
            merged_state = np.kron(merged_state, np.asarray(payload.data, dtype=complex))
            current_order = current_order + payload.qubits
        merged_state = reorder_state_vector(merged_state, current_order, merged_qubits)
        return UnitPayload(
            unit_id=unit_id,
            summary_kind="state_vector",
            qubits=merged_qubits,
            data=merged_state,
        )

    def observe_distribution(
        self,
        payload: UnitPayload,
        qubits: tuple[str, ...],
        basis: str = "Z",
    ) -> dict[object, float]:
        if payload.summary_kind == "density_matrix":
            return self._density.observe_distribution(payload, qubits, basis=basis)
        if payload.summary_kind != "state_vector":
            raise ValueError(f"Unsupported payload summary kind: {payload.summary_kind!r}")

        state = np.asarray(payload.data, dtype=complex)
        basis_name = basis.upper()
        if basis_name == "X":
            h_gate = named_gate_matrix("H")
            for qubit in qubits:
                state = apply_local_operator_to_state_vector(state, payload.qubits, h_gate, (qubit,))
        elif basis_name != "Z":
            raise ValueError(f"Unsupported basis: {basis!r}")
        return state_vector_distribution(state, payload.qubits, qubits)

    def to_density_payload(self, payload: UnitPayload) -> UnitPayload:
        if payload.summary_kind == "density_matrix":
            return payload.copy()
        if payload.summary_kind != "state_vector":
            raise ValueError(f"Unsupported payload summary kind: {payload.summary_kind!r}")
        return UnitPayload(
            unit_id=payload.unit_id,
            summary_kind="density_matrix",
            qubits=payload.qubits,
            data=state_vector_to_density_matrix(np.asarray(payload.data, dtype=complex)),
        )
