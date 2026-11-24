"""Transition execution on runtime state nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qamc.backend import DensityMatrixBackend
from qamc.ir.enums import BranchKind
from qamc.ir.model import BranchSpec, StateClassSpec, SystemModel, TransitionClassSpec
from qamc.runtime.execution_context import ExecutionContext
from qamc.runtime.state_node import StateNode, UnitPayload
from qamc.runtime.transition_edge import TransitionEdge


class JointStateUnavailableError(NotImplementedError):
    """The retained quantum information does not determine a supported joint workspace."""


@dataclass(slots=True)
class BranchOutcome:
    """One runtime branch successor."""

    branch: BranchSpec
    target_node: StateNode
    edge: TransitionEdge


class TransitionExecutor:
    """Expands one transition class on one runtime source node.

    This phase uses the configured numerical backend for unit-local operations.
    """

    def __init__(self, *, backend: DensityMatrixBackend | None = None, measurement_weight_provider=None) -> None:
        self.backend = backend or DensityMatrixBackend()
        self._measurement_weight_provider = measurement_weight_provider

    def execute(
        self,
        *,
        context: ExecutionContext,
        source_node: StateNode,
        transition: TransitionClassSpec,
    ) -> list[BranchOutcome]:
        if transition.gate.is_measurement():
            return self.execute_measurement(
                context=context,
                source_node=source_node,
                transition=transition,
            )

        branches = transition.branches
        if transition.control_branches():
            branches = tuple(
                branch for branch in branches
                if not branch.is_control_branch() or self._guard_holds(source_node, branch)
            )
            if source_node.reach_probability > 0 and len(branches) != 1:
                raise ValueError(
                    f"Control transition {transition.id!r} at state {source_node.id!r} "
                    f"requires exactly one applicable branch; found {len(branches)}."
                )
        branch_outcomes: list[BranchOutcome] = []
        for branch in branches:
            local_probability = 1.0
            target_node = self._build_successor_node(
                context=context,
                source_node=source_node,
                transition=transition,
                branch=branch,
                local_probability=local_probability,
            )
            edge = TransitionEdge(
                source_node_id=source_node.id,
                transition_class_id=transition.id,
                branch_id=branch.id,
                target_node_id=target_node.id,
                local_probability=local_probability,
            )
            context.add_node(target_node)
            context.add_edge(edge)
            branch_outcomes.append(BranchOutcome(branch=branch, target_node=target_node, edge=edge))
        return branch_outcomes

    def execute_measurement(
        self,
        *,
        context: ExecutionContext,
        source_node: StateNode,
        transition: TransitionClassSpec,
    ) -> list[BranchOutcome]:
        results = self._measure_source_units(source_node, transition)
        branch_outcomes: list[BranchOutcome] = []
        for branch in transition.measurement_branches():
            if branch.outcome not in results:
                raise ValueError(f"Measurement branch outcome {branch.outcome!r} is unavailable.")
            local_probability, measured_payloads = results[branch.outcome]
            target_node = self._build_successor_node(
                context=context,
                source_node=source_node,
                transition=transition,
                branch=branch,
                local_probability=local_probability,
                measured_payloads=measured_payloads,
            )
            edge = TransitionEdge(
                source_node_id=source_node.id,
                transition_class_id=transition.id,
                branch_id=branch.id,
                target_node_id=target_node.id,
                local_probability=local_probability,
            )
            context.add_node(target_node)
            context.add_edge(edge)
            branch_outcomes.append(BranchOutcome(branch=branch, target_node=target_node, edge=edge))
        return branch_outcomes

    def _measure_source_units(
        self, source_node: StateNode, transition: TransitionClassSpec,
    ) -> dict[object, tuple[float, dict[str, UnitPayload]]]:
        qubit, = transition.gate.targets
        affected = self._payloads_intersecting_qubits(source_node.unit_payloads, (qubit,))
        if not affected:
            raise ValueError(f"Measurement transition {transition.id} has no payload containing {qubit!r}.")
        if self._measurement_weight_provider is not None:
            return {
                branch.outcome: (float(self._measurement_weight_provider(source_node, transition, branch)), {})
                for branch in transition.measurement_branches()
            }

        if source_node.reach_probability == 0:
            # No conditional probabilities are defined at an unreachable source.
            return {branch.outcome: (0.0, {}) for branch in transition.measurement_branches()}
        self._check_overlap_consistency(affected, context_label=f"Measurement transition {transition.id}")
        results: dict[object, tuple[float, dict[str, UnitPayload]]] = {}
        for payload in sorted(affected, key=lambda item: (len(item.qubits), item.unit_id)):
            for result in self.backend.measure_payload(payload, qubit, basis="Z"):
                probability, updates = results.setdefault(result.outcome, (result.probability, {}))
                if updates:
                    self._check_measurement_probability(
                        probability, result.probability, transition.id, result.outcome,
                    )
                updates[payload.unit_id] = result.post_payload
        return results

    @staticmethod
    def _check_measurement_probability(
        expected: float, actual: float, transition_id: str, outcome: object,
    ) -> None:
        # Agreement to absolute tolerance cannot make zero and positive reach interchangeable.
        if (expected == 0) != (actual == 0) or not np.isclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError(
                f"Measurement transition {transition_id}, outcome {outcome}: "
                f"the retained units disagree on its probability ({expected!r}, {actual!r})."
            )

    def _guard_holds(self, source_node: StateNode, branch: BranchSpec) -> bool:
        if branch.guard_expr is None:
            return False
        if not isinstance(branch.guard_expr, tuple) or len(branch.guard_expr) != 3:
            return False
        op, var_name, expected = branch.guard_expr
        if op != "eq":
            return False
        return source_node.all_classical().get(var_name) == expected

    def _build_successor_node(
        self,
        *,
        context: ExecutionContext,
        source_node: StateNode,
        transition: TransitionClassSpec,
        branch: BranchSpec,
        local_probability: float,
        measured_payloads: dict[str, UnitPayload] | None = None,
    ) -> StateNode:
        target_state = context.system_model.state_class(branch.target_class)
        program_classical = dict(source_node.program_classical)
        measurement_classical = dict(source_node.measurement_classical)

        if branch.kind is BranchKind.MEASUREMENT and transition.gate.record_as is not None:
            measurement_classical[transition.gate.record_as] = branch.outcome

        reach_probability = source_node.reach_probability * local_probability
        if reach_probability == 0:
            target_payloads = {}
            for unit in target_state.units:
                placeholder = self.backend.initialize_unit_payload(
                    unit_id=unit.id, qubits=unit.qubits,
                    summary_kind=unit.resolved_summary_kind(context.system_model.defaults).value,
                    initializer=context.execution.initial_binding.payload_initializer,
                )
                placeholder.data.fill(0)
                target_payloads[unit.id] = placeholder
        elif measured_payloads:
            target_payloads = self._conditioned_targets(
                source_node.unit_payloads, measured_payloads, target_state, transition,
                branch.outcome, local_probability,
            )
        else:
            target_payloads = self._rebuild_payloads(
                source_node=source_node, target_state=target_state,
                system_model=context.system_model, transition=transition, branch=branch,
            )

        return StateNode(
            id=context.new_node_id(),
            state_class_id=target_state.id,
            active_qubits=tuple(dict.fromkeys(target_state.active_unit_qubits())),
            program_classical=program_classical,
            measurement_classical=measurement_classical,
            unit_payloads=target_payloads,
            reach_probability=reach_probability,
        )

    def _conditioned_targets(
        self,
        sources: dict[str, UnitPayload],
        measured: dict[str, UnitPayload],
        target_state: StateClassSpec,
        transition: TransitionClassSpec,
        outcome: object,
        local_probability: float,
    ) -> dict[str, UnitPayload]:
        """Condition retained overlaps before exposing the branch to observations."""
        qubit, = transition.gate.targets
        context_label = f"Measurement transition {transition.id}, outcome {outcome}"
        targets: dict[str, UnitPayload] = {}
        for unit in target_state.units:
            covering = self._payload_covering_qubits(measured, unit.qubits)
            if covering is not None:
                value = self.backend.partial_trace_payload(covering, unit.qubits)
            elif self._measurement_changes_overlap(sources, measured, unit.qubits):
                # Derive this unit from a joint state *before* conditioning. Its
                # old matrix and a new shared marginal alone cannot fix it.
                region = tuple(dict.fromkeys((qubit, *unit.qubits)))
                workspace = self._joint_payload(sources, region, context_label=context_label)
                selected = next(
                    result for result in self.backend.measure_payload(workspace, qubit)
                    if result.outcome == outcome
                )
                self._check_measurement_probability(
                    local_probability, selected.probability, transition.id, outcome,
                )
                value = self.backend.partial_trace_payload(selected.post_payload, unit.qubits)
            else:
                value = self._joint_payload(sources, unit.qubits, context_label=context_label)
            value.unit_id = unit.id
            targets[unit.id] = value

        # An indirectly conditioned unit may also overlap another target. Do
        # not return two conflicting answers for an observation on their overlap.
        try:
            self._check_overlap_consistency(list(targets.values()), context_label=context_label)
        except ValueError as exc:
            raise JointStateUnavailableError(
                f"{exc} Retain a joint source unit for the measurement and the overlapping targets."
            ) from exc
        return targets

    def _measurement_changes_overlap(
        self,
        sources: dict[str, UnitPayload],
        measured: dict[str, UnitPayload],
        qubits: tuple[str, ...],
    ) -> bool:
        for unit_id, after in measured.items():
            shared = tuple(q for q in qubits if q in after.qubits)
            if shared and not self._same_density(
                self.backend.partial_trace_payload(sources[unit_id], shared),
                self.backend.partial_trace_payload(after, shared),
            ):
                return True
        return False

    def _rebuild_payloads(
        self,
        *,
        source_node: StateNode,
        target_state: StateClassSpec,
        system_model: SystemModel,
        transition: TransitionClassSpec,
        branch: BranchSpec,
    ) -> dict[str, UnitPayload]:
        rebuilt: dict[str, UnitPayload] = {}
        base_payloads = {name: payload.copy() for name, payload in source_node.unit_payloads.items()}

        if not transition.gate.is_measurement() and transition.gate.name.upper() != "NOOP":
            self._apply_gate_to_base_payloads(
                base_payloads,
                gate=transition.gate,
                system_model=system_model,
                context_label=f"Transition {transition.id}",
            )

        if branch.kind is BranchKind.CLASSICAL_CONTROL and branch.action_gate is not None:
            acted_qubits = self.backend.acted_qubits(branch.action_gate)
            if branch.action_gate.name.upper() != "NOOP" and acted_qubits:
                self._apply_gate_to_base_payloads(
                    base_payloads,
                    gate=branch.action_gate,
                    system_model=system_model,
                    context_label=f"Branch action {branch.action_gate.name!r}",
                )

        for unit_spec in target_state.units:
            existing = base_payloads.get(unit_spec.id)
            if existing is not None and existing.qubits == unit_spec.qubits:
                rebuilt[unit_spec.id] = existing.copy()
            else:
                reduced = self._joint_payload(
                    base_payloads, unit_spec.qubits,
                    context_label=f"Target unit {unit_spec.id!r} of transition {transition.id}",
                )
                rebuilt[unit_spec.id] = UnitPayload(
                    unit_id=unit_spec.id,
                    summary_kind=reduced.summary_kind,
                    qubits=unit_spec.qubits,
                    data=reduced.data,
                )
        return rebuilt

    def _apply_gate_to_base_payloads(
        self,
        base_payloads: dict[str, UnitPayload],
        *,
        gate,
        system_model: SystemModel,
        context_label: str,
    ) -> None:
        acted_qubits = self.backend.acted_qubits(gate)
        source_payloads = self._payloads_intersecting_qubits(base_payloads, acted_qubits)
        if not source_payloads:
            raise ValueError(f"{context_label} cannot find runtime payloads for qubits {acted_qubits!r}.")

        acted_set = set(acted_qubits)
        covering_payloads = [
            payload for payload in source_payloads if acted_set.issubset(set(payload.qubits))
        ]
        if covering_payloads:
            self._check_overlap_consistency(source_payloads, context_label=context_label)
        if len(covering_payloads) == len(source_payloads):
            for payload in covering_payloads:
                base_payloads[payload.unit_id] = self.backend.apply_gate_to_payload(payload, gate)
            return

        # Every intersecting unit needs its post-operation marginal, including
        # units that contain only some of the gate's qubits.
        workspace_qubits = self._merged_qubit_order(
            system_model=system_model,
            qubits=set(acted_qubits).union(*(payload.qubits for payload in source_payloads)),
        )
        source_payload = self._joint_payload(
            {payload.unit_id: payload for payload in source_payloads},
            workspace_qubits, context_label=context_label,
        )
        updated_payload = self.backend.apply_gate_to_payload(source_payload, gate)
        for payload in source_payloads:
            base_payloads.pop(payload.unit_id, None)
        # A temporary workspace must not overwrite an unrelated declared unit.
        updated_payload.unit_id = "__workspace__"
        while updated_payload.unit_id in base_payloads:
            updated_payload.unit_id += "_"
        base_payloads[updated_payload.unit_id] = updated_payload

    def _joint_payload(
        self,
        payloads: dict[str, UnitPayload],
        qubits: tuple[str, ...],
        *,
        context_label: str,
    ) -> UnitPayload:
        """Obtain a supported joint summary without inventing correlations.

        Disjoint source units retain the model's product assumption. Overlapping
        units can be combined when each product extension has a pure factor
        (the additional subsystem or the state constructed so far). A pure marginal
        is independent of its complement in every compatible quantum state.
        General reconstruction from overlapping mixed marginals is unsupported.
        """
        sources = self._payloads_intersecting_qubits(payloads, qubits)
        required = set(qubits)
        available = set().union(*(payload.qubits for payload in sources))
        missing = required - available
        if missing:
            raise JointStateUnavailableError(
                f"{context_label} has no retained information for qubits {sorted(missing)!r}."
            )
        covering = self._payload_covering_qubits(payloads, qubits)
        if covering is not None:
            return self.backend.partial_trace_payload(covering, qubits)

        overlaps = self._check_overlap_consistency(sources, context_label=context_label)

        candidates = [
            self.backend.partial_trace_payload(
                payload, tuple(q for q in qubits if q in payload.qubits),
            )
            for payload in sources
        ]
        if not overlaps:
            return self.backend.merge_unit_payloads(candidates, qubits)

        for seed in sorted(candidates, key=lambda payload: -len(payload.qubits)):
            factors = [seed]
            covered = set(seed.qubits)
            for candidate in candidates:
                remaining = tuple(q for q in candidate.qubits if q not in covered)
                if not remaining:
                    continue
                marginal = self.backend.partial_trace_payload(candidate, remaining)
                complement = self._normalized_pure_payload(marginal)
                if complement is None:
                    if any(self._normalized_pure_payload(factor) is None for factor in factors):
                        continue
                    # A pure seed is independent of a mixed complement too.
                    complement = marginal
                factors.append(complement)
                covered.update(remaining)
            if covered != required:
                continue
            joint = self.backend.merge_unit_payloads(factors, qubits)
            if all(
                self._same_density(
                    self.backend.partial_trace_payload(joint, candidate.qubits), candidate,
                )
                for candidate in candidates
            ):
                return joint
        raise JointStateUnavailableError(
            f"{context_label} requires a joint summary on {qubits!r}; "
            "retain a unit covering these qubits or supply marginals that determine a product extension."
        )

    def _check_overlap_consistency(
        self, payloads: list[UnitPayload], *, context_label: str,
    ) -> bool:
        overlaps = False
        for index, left in enumerate(payloads):
            for right in payloads[index + 1:]:
                shared = tuple(q for q in left.qubits if q in right.qubits)
                if not shared:
                    continue
                overlaps = True
                left_reduced = self.backend.partial_trace_payload(left, shared)
                right_reduced = self.backend.partial_trace_payload(right, shared)
                if not self._same_density(left_reduced, right_reduced):
                    raise ValueError(
                        f"{context_label}: inconsistent overlapping summaries on {shared!r}."
                    )
        return overlaps

    @staticmethod
    def _density_array(payload: UnitPayload) -> np.ndarray:
        data = np.asarray(payload.data, dtype=complex)
        if payload.summary_kind == "state_vector":
            return np.outer(data, data.conj())
        return data

    def _same_density(self, left: UnitPayload, right: UnitPayload) -> bool:
        left_matrix = self._density_array(left)
        right_matrix = self._density_array(right)
        # Each positive basis weight admits an outcome, however small it is.
        if not np.array_equal(left_matrix.diagonal().real > 0, right_matrix.diagonal().real > 0):
            return False
        return bool(np.allclose(left_matrix, right_matrix, rtol=0, atol=1e-12))

    def _normalized_pure_payload(self, payload: UnitPayload) -> UnitPayload | None:
        """Recognize a pure complement and remove only normalization roundoff."""
        if payload.summary_kind == "state_vector":
            norm = np.vdot(payload.data, payload.data)
            if not np.isclose(norm, 1, rtol=0, atol=1e-12):
                return None
            normalized = payload.copy()
            normalized.data = np.asarray(payload.data, dtype=complex) / np.sqrt(norm.real)
            return normalized
        rho = self._density_array(payload)
        trace = np.trace(rho)
        if not np.isclose(trace, 1, rtol=0, atol=1e-12):
            return None
        rho = rho / trace.real
        if not np.isclose(np.vdot(rho, rho), 1, rtol=0, atol=1e-12):
            return None
        normalized = payload.copy()
        normalized.data = rho
        return normalized

    def _payload_covering_qubits(
        self,
        payloads: dict[str, UnitPayload],
        qubits: tuple[str, ...],
    ) -> UnitPayload | None:
        for payload in payloads.values():
            if set(qubits).issubset(set(payload.qubits)):
                return payload
        return None

    def _payloads_intersecting_qubits(
        self,
        payloads: dict[str, UnitPayload],
        qubits: tuple[str, ...],
    ) -> list[UnitPayload]:
        qubit_set = set(qubits)
        return [payload for payload in payloads.values() if qubit_set.intersection(payload.qubits)]

    def _merged_qubit_order(
        self,
        *,
        system_model: SystemModel,
        qubits: set[str],
    ) -> tuple[str, ...]:
        return tuple(qubit for qubit in system_model.qubit_universe if qubit in qubits)
