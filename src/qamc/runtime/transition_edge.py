"""Runtime transition-edge objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransitionEdge:
    """Runtime edge produced from a transition class and one branch."""

    source_node_id: str
    transition_class_id: str
    branch_id: str
    target_node_id: str
    local_probability: float
