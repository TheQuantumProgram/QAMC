"""Definition-layer IR objects for the QAMC checker."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qamc.ir.enums import (
    AnchorKind,
    BranchKind,
    ClassicalType,
    SummaryKind,
)

if TYPE_CHECKING:
    from qamc.ir.property import PropertySpec


@dataclass(slots=True)
class Defaults:
    """Global default settings for a QAM model."""

    default_summary_kind: SummaryKind = SummaryKind.DENSITY_MATRIX
    unit_overlap: bool = True
    post_measurement_recoherence: bool = False
    qubit_order_policy: str = "declared_order"
    default_basis: str = "Z"
    default_tolerance: float = 1e-9

    def summary_kind_for(self, unit_spec: "AbstractUnitSpec") -> SummaryKind:
        return unit_spec.summary_kind or self.default_summary_kind

    def tolerance_for(self, _property_spec: "PropertySpec | None" = None) -> float:
        return self.default_tolerance


@dataclass(slots=True)
class ClassicalVarDecl:
    """Typed declaration for a program classical variable."""

    name: str
    type_name: ClassicalType
    domain: tuple[object, ...] | None = None


@dataclass(slots=True)
class GateDeclaration:
    """Global declaration of a gate name used in the model."""

    name: str
    arity: int | None = None
    parameter_count: int = 0


@dataclass(slots=True)
class AbstractUnitSpec:
    """Abstract unit declaration inside a state class."""

    id: str
    qubits: tuple[str, ...]
    summary_kind: SummaryKind | None = None
    annotations: dict[str, object] = field(default_factory=dict)

    def resolved_summary_kind(self, defaults: Defaults) -> SummaryKind:
        return defaults.summary_kind_for(self)


@dataclass(slots=True)
class ClassicalProfileSpec:
    """Visible classical interface of a state class."""

    program_vars: tuple[str, ...] = ()
    measurement_vars: tuple[str, ...] = ()
    type_overrides: dict[str, str] = field(default_factory=dict)

    def all_visible_vars(self) -> tuple[str, ...]:
        return self.program_vars + self.measurement_vars

    def contains(self, var_name: str) -> bool:
        return var_name in self.program_vars or var_name in self.measurement_vars


@dataclass(slots=True)
class StateClassSpec:
    """Definition-layer state class."""

    id: str
    classical_profile: ClassicalProfileSpec
    active_qubit_count: int
    units: tuple[AbstractUnitSpec, ...]
    tags: frozenset[str] = frozenset()

    def unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.id for unit in self.units)

    def active_unit_qubits(self) -> tuple[str, ...]:
        return tuple(qubit for unit in self.units for qubit in unit.qubits)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass(slots=True)
class GateSpec:
    """Concrete gate occurrence used by a transition."""

    name: str
    targets: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    params: tuple[object, ...] = ()
    record_as: str | None = None
    extra_qargs: tuple[str, ...] = ()

    def all_qargs(self) -> tuple[str, ...]:
        ordered = self.targets + self.controls + self.extra_qargs
        return tuple(dict.fromkeys(ordered))

    def is_measurement(self) -> bool:
        return self.record_as is not None or self.name.upper().startswith("M")


@dataclass(slots=True)
class BranchSpec:
    """Branch declaration inside a transition class."""

    id: str
    kind: BranchKind
    target_class: str
    outcome: object | None = None
    guard_expr: object | None = None
    action_gate: GateSpec | None = None

    def is_measurement_branch(self) -> bool:
        return self.kind is BranchKind.MEASUREMENT

    def is_control_branch(self) -> bool:
        return self.kind is BranchKind.CLASSICAL_CONTROL


@dataclass(slots=True)
class TransitionClassSpec:
    """Definition-layer transition class."""

    id: str
    source_class: str
    gate: GateSpec
    branches: tuple[BranchSpec, ...]

    def branch_ids(self) -> tuple[str, ...]:
        return tuple(branch.id for branch in self.branches)

    def measurement_branches(self) -> tuple[BranchSpec, ...]:
        return tuple(branch for branch in self.branches if branch.is_measurement_branch())

    def control_branches(self) -> tuple[BranchSpec, ...]:
        return tuple(branch for branch in self.branches if branch.is_control_branch())


@dataclass(slots=True)
class AnchorSpec:
    """Trace anchor declaration."""

    id: str
    kind: AnchorKind
    admissible_classes: tuple[str, ...] = ()


@dataclass(slots=True)
class TraceStepSpec:
    """One definition-layer trace step."""

    source_anchor: str
    transition_class: str
    target_anchor: str


@dataclass(slots=True)
class TraceSpec:
    """Definition-layer trace."""

    id: str
    anchors: dict[str, AnchorSpec]
    entry_anchor: str
    steps: tuple[TraceStepSpec, ...]

    def outgoing(self, anchor_id: str) -> tuple[TraceStepSpec, ...]:
        return tuple(step for step in self.steps if step.source_anchor == anchor_id)

    def incoming(self, anchor_id: str) -> tuple[TraceStepSpec, ...]:
        return tuple(step for step in self.steps if step.target_anchor == anchor_id)

    def topological_anchor_order(self) -> tuple[str, ...]:
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {anchor_id: 0 for anchor_id in self.anchors}
        for step in self.steps:
            adjacency[step.source_anchor].append(step.target_anchor)
            indegree[step.target_anchor] += 1
        queue = deque(anchor_id for anchor_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while queue:
            anchor_id = queue.popleft()
            order.append(anchor_id)
            for target_anchor in adjacency[anchor_id]:
                indegree[target_anchor] -= 1
                if indegree[target_anchor] == 0:
                    queue.append(target_anchor)
        if len(order) != len(self.anchors):
            raise ValueError(f"Trace {self.id} is cyclic or malformed.")
        return tuple(order)


@dataclass(slots=True)
class InitialBindingSpec:
    """Execution entry declaration."""

    anchor_id: str
    state_class_id: str
    program_values: dict[str, object] = field(default_factory=dict)
    measurement_values: dict[str, object] = field(default_factory=dict)
    payload_initializer: object | None = None


@dataclass(slots=True)
class ExecutionSpec:
    """Execution declaration."""

    id: str
    trace_id: str
    initial_binding: InitialBindingSpec
    stop_anchor: str
    options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SystemModel:
    """System-model part of a QAM model."""

    qubit_universe: tuple[str, ...]
    program_classical_vars: dict[str, ClassicalVarDecl]
    gate_declarations: dict[str, GateDeclaration]
    defaults: Defaults
    state_classes: dict[str, StateClassSpec]
    transition_classes: dict[str, TransitionClassSpec]

    def state_class(self, state_class_id: str) -> StateClassSpec:
        return self.state_classes[state_class_id]

    def transition_class(self, transition_class_id: str) -> TransitionClassSpec:
        return self.transition_classes[transition_class_id]


@dataclass(slots=True)
class QAMModel:
    """Complete definition-layer QAM model."""

    system_model: SystemModel
    traces: dict[str, TraceSpec]
    executions: dict[str, ExecutionSpec]
    properties: tuple["PropertySpec", ...]

    def trace(self, trace_id: str) -> TraceSpec:
        return self.traces[trace_id]

    def execution(self, execution_id: str) -> ExecutionSpec:
        return self.executions[execution_id]

    def property_by_id(self, property_id: str) -> "PropertySpec":
        for property_spec in self.properties:
            if property_spec.id == property_id:
                return property_spec
        raise KeyError(property_id)
