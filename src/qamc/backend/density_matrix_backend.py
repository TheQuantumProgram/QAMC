"""Density-matrix backend for the QAMC checker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qamc.backend.unit_ops import (
    apply_local_operator,
    expand_local_operator,
    int_to_bits,
    named_gate_matrix,
    partial_trace,
    reorder_density_matrix,
    single_qubit_projectors,
    zero_density_matrix,
)
from qamc.ir.model import GateSpec
from qamc.runtime.state_node import UnitPayload


@dataclass(slots=True)
class MeasurementResult:
    """One measurement branch result on a density-matrix payload."""

    outcome: object
    probability: float
    post_payload: UnitPayload


class DensityMatrixBackend:
    """Numerical backend based on density matrices."""

    def initialize_unit_payload(
        self,
        *,
        unit_id: str,
        summary_kind: str,
        qubits: tuple[str, ...],
        initializer: object,
    ) -> UnitPayload:
        if summary_kind != "density_matrix":
            raise ValueError(f"Unsupported summary kind: {summary_kind!r}")
        if initializer != "zero_density_matrix":
            raise ValueError(f"Unsupported initializer: {initializer!r}")
        return UnitPayload(
            unit_id=unit_id,
            summary_kind=summary_kind,
            qubits=qubits,
            data=zero_density_matrix(len(qubits)),
        )

    def acted_qubits(self, gate: GateSpec) -> tuple[str, ...]:
        ordered = gate.controls + gate.targets + gate.extra_qargs
        return tuple(dict.fromkeys(ordered))

    def apply_gate_to_payload(self, payload: UnitPayload, gate: GateSpec) -> UnitPayload:
        if gate.name.upper() == "NOOP":
            return payload.copy()
        acted_qubits = self.acted_qubits(gate)
        for qubit in acted_qubits:
            if qubit not in payload.qubits:
                raise ValueError(
                    f"Gate {gate.name!r} acts on qubit {qubit!r} outside payload {payload.unit_id!r}."
                )
        local_operator = named_gate_matrix(gate.name, gate.params)
        rho = np.asarray(payload.data, dtype=complex)
        updated = apply_local_operator(rho, payload.qubits, local_operator, acted_qubits)
        return UnitPayload(
            unit_id=payload.unit_id,
            summary_kind=payload.summary_kind,
            qubits=payload.qubits,
            data=updated,
        )

    def measure_payload(self, payload: UnitPayload, qubit: str, basis: str = "Z") -> list[MeasurementResult]:
        if qubit not in payload.qubits:
            raise ValueError(f"Qubit {qubit!r} not present in payload {payload.unit_id!r}.")

        rho = np.asarray(payload.data, dtype=complex)
        projector0, projector1 = single_qubit_projectors(basis)
        projectors = (projector0, projector1)
        results: list[MeasurementResult] = []
        for outcome, projector in enumerate(projectors):
            full_projector = expand_local_operator(projector, payload.qubits, (qubit,))
            probability = float(np.real_if_close(np.trace(full_projector @ rho)))
            if probability < 0:
                probability = 0.0
            if probability == 0:
                post_rho = np.zeros_like(rho)
            else:
                post_rho = (full_projector @ rho @ full_projector.conj().T) / probability
            results.append(
                MeasurementResult(
                    outcome=outcome,
                    probability=probability,
                    post_payload=UnitPayload(
                        unit_id=payload.unit_id,
                        summary_kind=payload.summary_kind,
                        qubits=payload.qubits,
                        data=post_rho,
                    ),
                )
            )
        return results

    def partial_trace_payload(self, payload: UnitPayload, retained_qubits: tuple[str, ...]) -> UnitPayload:
        reduced = partial_trace(np.asarray(payload.data, dtype=complex), payload.qubits, retained_qubits)
        return UnitPayload(
            unit_id=payload.unit_id,
            summary_kind=payload.summary_kind,
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
        merged_rho = self._density_factor_for_product(ordered_payloads[0])
        for payload in ordered_payloads[1:]:
            merged_rho = np.kron(merged_rho, self._density_factor_for_product(payload))
            current_order = current_order + payload.qubits
        merged_rho = reorder_density_matrix(merged_rho, current_order, merged_qubits)
        return UnitPayload(
            unit_id=unit_id,
            summary_kind="density_matrix",
            qubits=merged_qubits,
            data=merged_rho,
        )

    @staticmethod
    def _density_factor_for_product(payload: UnitPayload) -> np.ndarray:
        """Prevent repeated marginal products from amplifying trace roundoff.

        A split copies the parent's small normalization error into each marginal;
        multiplying those marginals repeatedly would then exponentiate the error.
        Correct only a trace within 1e-12 of one, independently of assertion
        tolerances. Exact zero matrices are impossible-branch placeholders and
        retain zero mass. Other nonunit traces indicate invalid input.
        """
        rho = np.asarray(payload.data, dtype=complex)
        trace = np.trace(rho)
        if trace == 1:
            return rho
        if np.isclose(trace, 1, rtol=0, atol=1e-12):
            return rho / trace.real
        if not np.any(rho):
            return rho
        raise ValueError(
            f"Payload {payload.unit_id!r} has invalid trace {trace!r} for a density-matrix product."
        )

    def observe_distribution(
        self,
        payload: UnitPayload,
        qubits: tuple[str, ...],
        basis: str = "Z",
    ) -> dict[object, float]:
        reduced = self.partial_trace_payload(payload, qubits)
        rho = np.asarray(reduced.data, dtype=complex)
        if basis.upper() == "X":
            h_gate = named_gate_matrix("H")
            for qubit in qubits:
                rho = apply_local_operator(rho, qubits, h_gate, (qubit,))
        elif basis.upper() != "Z":
            raise ValueError(f"Unsupported basis: {basis!r}")

        diagonal = np.real_if_close(np.diag(rho))
        distribution: dict[object, float] = {}
        for index, probability in enumerate(diagonal):
            bits = int_to_bits(index, len(qubits))
            key: object = bits[0] if len(bits) == 1 else bits
            distribution[key] = float(probability)
        return distribution
