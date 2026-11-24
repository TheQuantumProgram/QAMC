"""Scope selection for property evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from qamc.ir.property import PropertySpec
from qamc.runtime.execution_context import ExecutionContext


@dataclass(slots=True)
class ScopeSelection:
    """Selected nodes for one property scope."""

    anchor_ids: tuple[str, ...]
    node_ids: tuple[str, ...]


class ScopeSelector:
    """Resolve a property scope to concrete runtime nodes."""

    def select(self, context: ExecutionContext, property_spec: PropertySpec) -> ScopeSelection:
        trace_order = context.trace.topological_anchor_order()
        if property_spec.scope_kind.value == "anchors":
            anchor_ids = tuple(property_spec.scope_value)
        elif property_spec.scope_kind.value == "interval":
            start_anchor, end_anchor = property_spec.scope_value
            start_index = trace_order.index(start_anchor)
            end_index = trace_order.index(end_anchor)
            anchor_ids = trace_order[start_index : end_index + 1]
        else:
            raise ValueError(f"Unsupported scope kind: {property_spec.scope_kind!r}")

        node_ids = tuple(
            context.anchors.selected(
                property_spec.scope_kind,
                property_spec.scope_value,
                anchor_order=trace_order,
            )
        )
        return ScopeSelection(anchor_ids=anchor_ids, node_ids=node_ids)
