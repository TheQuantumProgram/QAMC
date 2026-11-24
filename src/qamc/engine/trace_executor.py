"""Trace-level execution on definition-layer QAM models."""

from __future__ import annotations

from dataclasses import replace

from qamc.backend import DensityMatrixBackend, StateVectorBackend
from qamc.ir.enums import BranchKind, SummaryKind
from qamc.ir.model import GateSpec, QAMModel, SystemModel, TransitionClassSpec
from qamc.runtime.anchor_store import AnchorStore
from qamc.runtime.execution_context import ExecutionContext
from qamc.runtime.state_node import StateNode, UnitPayload
from qamc.engine.transition_executor import TransitionExecutor


def _combined_rotation(
    first: TransitionClassSpec, second: TransitionClassSpec, system: SystemModel,
) -> GateSpec | None:
    """Compose opposing rotations only if the intermediate units retained the operation."""
    gate, following = first.gate, second.gate
    if gate.name.upper() not in {"P", "CP", "RBS"} or following.name.upper() != gate.name.upper():
        return None
    if gate.is_measurement() or following.is_measurement() or len(gate.params) != 1 or len(following.params) != 1:
        return None
    if (gate.targets, gate.controls, gate.extra_qargs) != (following.targets, following.controls, following.extra_qargs):
        return None
    angle, next_angle = float(gate.params[0]), float(following.params[0])
    if (angle < 0) == (next_angle < 0):
        return None
    for transition in (first, second):
        if len(transition.branches) != 1:
            return None
        branch, = transition.branches
        if branch.kind is not BranchKind.DETERMINISTIC or branch.action_gate is not None:
            return None
    if first.branches[0].target_class != second.source_class:
        return None
    units = system.state_class(first.source_class).units
    final_units = system.state_class(second.branches[0].target_class).units
    if units != system.state_class(second.source_class).units or units != final_units:
        return None
    operands = set(gate.all_qargs())
    if not all(operands.isdisjoint(unit.qubits) or operands.issubset(unit.qubits) for unit in units):
        return None
    angle_sum = angle + next_angle
    return replace(gate, params=(angle_sum,)) if angle_sum else GateSpec("NOOP")


class TraceExecutor:
    """Executes one definition-layer trace into a runtime transition model."""

    def __init__(
        self,
        *,
        backend: DensityMatrixBackend | None = None,
        transition_executor: TransitionExecutor | None = None,
    ) -> None:
        self._custom_execution = backend is not None or transition_executor is not None
        self._backend = backend or DensityMatrixBackend()
        self._transition_executor = transition_executor or TransitionExecutor(backend=self._backend)
        self._standard_unitaries = transition_executor is None and type(self._backend) in (DensityMatrixBackend, StateVectorBackend)

    def run(self, model: QAMModel, execution_id: str | None = None) -> ExecutionContext:
        if model.system_model.defaults.default_summary_kind is SummaryKind.GROVER_SUBSPACE:
            if self._custom_execution:
                raise ValueError("Grover subspace execution uses its own backend and transition rules.")
            from qamc.engine.grover_subspace import run_grover_subspace
            return run_grover_subspace(model, execution_id)
        execution = (
            model.execution(execution_id)
            if execution_id is not None
            else next(iter(model.executions.values()))
        )
        trace = model.trace(execution.trace_id)
        context = ExecutionContext(
            system_model=model.system_model,
            trace=trace,
            execution=execution,
            anchors=AnchorStore(
                anchor_kind={anchor_id: anchor.kind.value for anchor_id, anchor in trace.anchors.items()}
            ),
        )

        root_node = self.initialize_root_node(context)
        context.add_node(root_node)
        context.anchors.bind(execution.initial_binding.anchor_id, [root_node.id])

        previous_sources: list[str] = []
        for index, step in enumerate(trace.steps):
            source_node_ids = context.anchors.nodes_of(step.source_anchor)
            if not source_node_ids:
                previous_sources = []
                context.anchors.bind(step.target_anchor, [])
                continue

            transition = model.system_model.transition_class(step.transition_class)
            pair_inputs = None
            if index and self._standard_unitaries and transition.gate.name.upper() in {"P", "CP", "RBS"}:
                previous_step = trace.steps[index - 1]
                if previous_step.target_anchor == step.source_anchor:
                    previous = context.system_model.transition_class(previous_step.transition_class)
                    combined = _combined_rotation(previous, transition, context.system_model)
                    if combined is not None:
                        # Both steps are one-to-one on the ordered source branches.
                        # Evaluate their combined angle from each branch's state before the pair.
                        pair_inputs = previous_sources
                        transition = replace(transition, gate=combined)

            produced_node_ids: list[str] = []
            for branch_index, node_id in enumerate(source_node_ids):
                source_node = context.node(node_id)
                if pair_inputs is not None:
                    before_pair = context.node(pair_inputs[branch_index])
                    source_node = source_node.copy_with(unit_payloads=before_pair.unit_payloads)
                outcomes = self._transition_executor.execute(
                    context=context,
                    source_node=source_node,
                    transition=transition,
                )
                produced_node_ids.extend(outcome.target_node.id for outcome in outcomes)
            previous_sources = source_node_ids
            context.anchors.bind(step.target_anchor, produced_node_ids)

        return context

    def initialize_root_node(self, context: ExecutionContext) -> StateNode:
        binding = context.execution.initial_binding
        state_class = context.system_model.state_class(binding.state_class_id)
        payloads = self._initial_payloads(context, state_class.id)
        return StateNode(
            id=context.new_node_id(),
            state_class_id=state_class.id,
            active_qubits=tuple(dict.fromkeys(state_class.active_unit_qubits())),
            program_classical=dict(binding.program_values),
            measurement_classical=dict(binding.measurement_values),
            unit_payloads=payloads,
            reach_probability=1.0,
        )

    def _initial_payloads(self, context: ExecutionContext, state_class_id: str) -> dict[str, UnitPayload]:
        state_class = context.system_model.state_class(state_class_id)
        payload_initializer = context.execution.initial_binding.payload_initializer
        payloads: dict[str, UnitPayload] = {}
        for unit_spec in state_class.units:
            summary_kind = unit_spec.resolved_summary_kind(context.system_model.defaults).value
            payloads[unit_spec.id] = self._backend.initialize_unit_payload(
                unit_id=unit_spec.id,
                summary_kind=summary_kind,
                qubits=unit_spec.qubits,
                initializer=payload_initializer,
            )
        return payloads
