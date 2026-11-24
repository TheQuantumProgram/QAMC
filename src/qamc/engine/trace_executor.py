"""Trace-level execution on definition-layer QAM models."""

from __future__ import annotations

from qamc.backend import DensityMatrixBackend
from qamc.ir.enums import SummaryKind
from qamc.ir.model import QAMModel
from qamc.runtime.anchor_store import AnchorStore
from qamc.runtime.execution_context import ExecutionContext
from qamc.runtime.state_node import StateNode, UnitPayload
from qamc.engine.transition_executor import TransitionExecutor


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

        for step in trace.steps:
            source_node_ids = context.anchors.nodes_of(step.source_anchor)
            if not source_node_ids:
                context.anchors.bind(step.target_anchor, [])
                continue

            transition = model.system_model.transition_class(step.transition_class)
            produced_node_ids: list[str] = []
            for node_id in source_node_ids:
                source_node = context.node(node_id)
                outcomes = self._transition_executor.execute(
                    context=context,
                    source_node=source_node,
                    transition=transition,
                )
                produced_node_ids.extend(outcome.target_node.id for outcome in outcomes)
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
