"""Runtime state-node objects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(slots=True)
class UnitPayload:
    """Runtime payload attached to one abstract unit.

    Phase 3 keeps payloads structural rather than numerical. The `data` field is
    therefore a placeholder container until the density-matrix backend is added.
    """

    unit_id: str
    summary_kind: str
    qubits: tuple[str, ...]
    data: object

    def copy(self) -> "UnitPayload":
        copied_data = self.data.copy() if hasattr(self.data, "copy") else self.data
        return replace(self, data=copied_data)

    def reindex(self, qubits: tuple[str, ...]) -> "UnitPayload":
        return replace(self, qubits=qubits)


@dataclass(slots=True)
class StateNode:
    """Runtime state-node produced during trace execution."""

    id: str
    state_class_id: str
    active_qubits: tuple[str, ...]
    program_classical: dict[str, object] = field(default_factory=dict)
    measurement_classical: dict[str, object] = field(default_factory=dict)
    unit_payloads: dict[str, UnitPayload] = field(default_factory=dict)
    reach_probability: float = 1.0

    def all_classical(self) -> dict[str, object]:
        merged = dict(self.program_classical)
        merged.update(self.measurement_classical)
        return merged

    def payload_for_unit(self, unit_id: str) -> UnitPayload:
        return self.unit_payloads[unit_id]

    def copy_with(self, **updates: object) -> "StateNode":
        return replace(self, **updates)
