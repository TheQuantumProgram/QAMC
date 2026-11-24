"""Anchor binding utilities for runtime execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from qamc.ir.enums import ScopeKind


@dataclass(slots=True)
class AnchorStore:
    """Maps trace anchors to runtime node families."""

    anchor_kind: dict[str, str]
    anchor_to_nodes: dict[str, list[str]] = field(default_factory=dict)

    def bind(self, anchor_id: str, node_ids: list[str]) -> None:
        self.anchor_to_nodes[anchor_id] = list(node_ids)

    def nodes_of(self, anchor_id: str) -> list[str]:
        return list(self.anchor_to_nodes.get(anchor_id, []))

    def selected(
        self,
        scope_kind: ScopeKind,
        scope_value: object,
        anchor_order: tuple[str, ...] | None = None,
    ) -> list[str]:
        if scope_kind is ScopeKind.ANCHORS:
            selected: list[str] = []
            for anchor_id in tuple(scope_value):
                selected.extend(self.nodes_of(anchor_id))
            return selected

        if scope_kind is ScopeKind.INTERVAL:
            if anchor_order is None:
                raise ValueError("Interval selection requires anchor order.")
            start_anchor, end_anchor = scope_value
            start_index = anchor_order.index(start_anchor)
            end_index = anchor_order.index(end_anchor)
            selected: list[str] = []
            for anchor_id in anchor_order[start_index : end_index + 1]:
                selected.extend(self.nodes_of(anchor_id))
            return selected

        raise ValueError(f"Unsupported scope kind: {scope_kind!r}")
