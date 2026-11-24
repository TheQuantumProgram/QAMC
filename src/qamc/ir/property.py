"""Property-layer IR objects for the QAMC checker."""

from __future__ import annotations

from dataclasses import dataclass

from qamc.ir.enums import EvaluationMode, PropertyKind, ScopeKind


@dataclass(slots=True)
class ObservableSpec:
    """Observable declaration attached to a property."""

    kind: str
    classical_vars: tuple[str, ...] = ()
    qubits: tuple[str, ...] = ()
    basis: str | None = None

    def has_classical_part(self) -> bool:
        return bool(self.classical_vars)

    def has_quantum_part(self) -> bool:
        return bool(self.qubits)


@dataclass(slots=True)
class PropertySpec:
    """Definition-layer property specification."""

    id: str
    kind: PropertyKind
    scope_kind: ScopeKind
    scope_value: object
    mode: EvaluationMode
    observable: ObservableSpec
    expectation: object

    def targeted_anchors(self) -> tuple[str, ...]:
        if self.scope_kind is ScopeKind.ANCHORS:
            return tuple(self.scope_value)
        return ()
