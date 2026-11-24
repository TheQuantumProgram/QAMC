"""IR objects for external property bundles."""

from __future__ import annotations

from dataclasses import dataclass

from qamc.external_properties.formula import TemporalFormula
from qamc.ir.property import PropertySpec


@dataclass(slots=True)
class TemporalPropertySpec:
    """Temporal property over base properties and trace anchors."""

    id: str
    operator: str = ""
    interval: tuple[str, str] | None = None
    operands: tuple[str, ...] = ()
    formula: TemporalFormula | None = None


@dataclass(slots=True)
class ExternalPropertyBundle:
    """Bundle of externally supplied base and temporal properties."""

    base_properties: tuple[PropertySpec, ...] = ()
    temporal_properties: tuple[TemporalPropertySpec, ...] = ()


@dataclass(slots=True)
class TemporalPropertyEvaluationResult:
    """Result of evaluating one external temporal property."""

    property_id: str
    operator: str
    passed: bool | None
    selected_anchor_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    observed: object

    @property
    def diagnostic(self) -> str | None:
        if self.passed is None:
            return "An aggregate operand has no selected state instance with positive reach probability."
        return None
