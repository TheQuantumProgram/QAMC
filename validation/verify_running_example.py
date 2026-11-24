#!/usr/bin/env python3
"""Compare the six-qubit running example with two independent NumPy calculations.

Run `python validation/verify_running_example.py` from the repository root.
The reference computations compare full-state amplitudes and local density
matrices at every declared position. The checker then executes the supplied
model and its assertions. The JSON report records all three results.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


QUBIT_COUNT = 6
TOLERANCE = 1e-12
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "examples/ghz_ccx_measure_control.qam"
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def bit(basis_index: int, qubit: int) -> int:
    """Use q0 as the most significant bit throughout the reference."""
    return (basis_index >> (QUBIT_COUNT - 1 - qubit)) & 1


def single_qubit_gate(state: np.ndarray, gate: np.ndarray, target: int) -> np.ndarray:
    result = np.zeros_like(state)
    target_mask = 1 << (QUBIT_COUNT - 1 - target)
    for index, amplitude in enumerate(state):
        original_bit = bit(index, target)
        for new_bit in (0, 1):
            output_index = (index & ~target_mask) | (new_bit * target_mask)
            result[output_index] += gate[new_bit, original_bit] * amplitude
    return result


def controlled_x(state: np.ndarray, controls: tuple[int, ...], target: int) -> np.ndarray:
    result = np.zeros_like(state)
    target_mask = 1 << (QUBIT_COUNT - 1 - target)
    for index, amplitude in enumerate(state):
        output_index = index ^ target_mask if all(bit(index, q) for q in controls) else index
        result[output_index] += amplitude
    return result


def measure_z(state: np.ndarray, target: int) -> dict[int, tuple[float, np.ndarray]]:
    outcomes = {}
    for outcome in (0, 1):
        projected = state.copy()
        for index in range(len(state)):
            if bit(index, target) != outcome:
                projected[index] = 0
        probability = float(np.vdot(projected, projected).real)
        if probability > TOLERANCE:
            outcomes[outcome] = (probability, projected / np.sqrt(probability))
    return outcomes


def reduced_density(state: np.ndarray, retained: tuple[int, ...]) -> np.ndarray:
    """Compute a partial trace by assembling amplitudes for each environment."""
    environment = tuple(q for q in range(QUBIT_COUNT) if q not in retained)
    amplitudes = np.zeros((2 ** len(retained), 2 ** len(environment)), dtype=complex)
    for basis_index, amplitude in enumerate(state):
        row = 0
        column = 0
        for qubit in retained:
            row = 2 * row + bit(basis_index, qubit)
        for qubit in environment:
            column = 2 * column + bit(basis_index, qubit)
        amplitudes[row, column] = amplitude
    return amplitudes @ amplitudes.conj().T


def z_probabilities(density: np.ndarray) -> np.ndarray:
    return np.diag(density).real


def distribution_json(probabilities: np.ndarray) -> dict[str, float]:
    width = (len(probabilities) - 1).bit_length()
    return {format(index, f"0{width}b"): float(p) for index, p in enumerate(probabilities)}


def matrix_json(matrix: np.ndarray) -> dict[str, list]:
    return {"real": matrix.real.tolist(), "imaginary": matrix.imag.tolist()}


def close(actual: object, expected: object) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)


def build_reference_positions() -> dict[str, dict[int | None, tuple[float, np.ndarray]]]:
    """Retain the independent full-state reference at every declared position."""
    state = np.zeros(2**QUBIT_COUNT, dtype=complex)
    state[0] = 1
    positions = {"a0": {None: (1.0, state.copy())}}
    state = single_qubit_gate(state, HADAMARD, 0)
    positions["a1"] = {None: (1.0, state.copy())}
    state = controlled_x(state, (0,), 1)
    positions["a2"] = {None: (1.0, state.copy())}
    state = controlled_x(state, (1,), 2)
    positions["a3"] = {None: (1.0, state.copy())}
    state = single_qubit_gate(state, HADAMARD, 3)
    positions["a4"] = {None: (1.0, state.copy())}
    state = controlled_x(state, (2, 3), 4)
    positions["a5"] = {None: (1.0, state.copy())}
    positions["f1"] = measure_z(state, 3)
    positions["f2"] = {}
    positions["f3"] = {}
    for outcome, (probability, branch_state) in positions["f1"].items():
        if outcome == 1:
            branch_state = single_qubit_gate(branch_state, HADAMARD, 5)
        positions["f2"][outcome] = (probability, branch_state.copy())
        branch_state = controlled_x(branch_state, (5,), 4)
        close(np.vdot(branch_state, branch_state).real, 1.0)
        positions["f3"][outcome] = (probability, branch_state)
    return positions


def build_reference() -> tuple[np.ndarray, dict[int, tuple[float, np.ndarray]]]:
    """Keep the original helper interface for other reference computations."""
    positions = build_reference_positions()
    final_branches = {
        outcome: branch for outcome, branch in positions["f3"].items() if outcome is not None
    }
    return positions["a5"][None][1], final_branches


LocalPayload = tuple[tuple[int, ...], np.ndarray]


def _local_bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> shift) & 1 for shift in reversed(range(width)))


def _local_index(bits: tuple[int, ...]) -> int:
    result = 0
    for value in bits:
        result = 2 * result + value
    return result


def local_product(*payloads: LocalPayload) -> LocalPayload:
    """Form an explicitly justified product of disjoint local regions."""
    support: tuple[int, ...] = ()
    density = np.ones((1, 1), dtype=complex)
    for qubits, matrix in payloads:
        assert set(support).isdisjoint(qubits)
        support += qubits
        density = np.kron(density, matrix)
    return support, density


def local_reduce(payload: LocalPayload, retained: tuple[int, ...]) -> LocalPayload:
    """Trace out a local workspace by explicit environment-index matching."""
    support, density = payload
    assert set(retained).issubset(support)
    retained_positions = tuple(support.index(q) for q in retained)
    discarded_positions = tuple(i for i, q in enumerate(support) if q not in retained)
    result = np.zeros((2 ** len(retained), 2 ** len(retained)), dtype=complex)
    basis_bits = [_local_bits(i, len(support)) for i in range(len(density))]
    for i, left in enumerate(basis_bits):
        row = _local_index(tuple(left[k] for k in retained_positions))
        for j, right in enumerate(basis_bits):
            if all(left[k] == right[k] for k in discarded_positions):
                column = _local_index(tuple(right[k] for k in retained_positions))
                result[row, column] += density[i, j]
    return retained, result


def local_apply(payload: LocalPayload, operation: np.ndarray, acted: tuple[int, ...]) -> LocalPayload:
    """Embed an operator in a local workspace without using QAMC's backend."""
    support, density = payload
    positions = tuple(support.index(q) for q in acted)
    embedded = np.zeros_like(density, dtype=complex)
    for source in range(len(density)):
        source_bits = _local_bits(source, len(support))
        local_source = _local_index(tuple(source_bits[i] for i in positions))
        for target in range(len(operation)):
            target_bits = list(source_bits)
            for position, value in zip(positions, _local_bits(target, len(acted))):
                target_bits[position] = value
            embedded[_local_index(tuple(target_bits)), source] = operation[target, local_source]
    return support, embedded @ density @ embedded.conj().T


def local_controlled_x(control_count: int) -> np.ndarray:
    width = control_count + 1
    result = np.zeros((2**width, 2**width), dtype=complex)
    for source in range(len(result)):
        bits = list(_local_bits(source, width))
        if all(bits[:-1]):
            bits[-1] ^= 1
        result[_local_index(tuple(bits)), source] = 1.0
    return result


def build_local_workspace_positions() -> dict:
    """Compute retained summaries solely from preceding local summaries.

    This is an explicit method-level calculation, not execution of the QAM file.
    Product workspaces use the independence established by this program's prefix.
    No full-register reference values enter these calculations.
    """
    zero = np.diag([1.0, 0.0]).astype(complex)
    units = {f"u_q{q}": ((q,), zero.copy()) for q in range(QUBIT_COUNT)}
    positions = {"a0": {None: (1.0, units.copy())}}
    units = dict(units)
    units["u_q0"] = local_apply(units["u_q0"], HADAMARD, (0,))
    positions["a1"] = {None: (1.0, units.copy())}

    # CX01 entangles q0 and q1; q2 is still an independent zero state.
    workspace = local_product(units["u_q0"], units["u_q1"])
    workspace = local_apply(workspace, local_controlled_x(1), (0, 1))
    units = {
        "u01": workspace,
        "u12": local_product(local_reduce(workspace, (1,)), units["u_q2"]),
        "q3": units["u_q3"], "q4": units["u_q4"], "q5": units["u_q5"],
    }
    positions["a2"] = {None: (1.0, units.copy())}

    # The overlapping pairs are not tensor-multiplied. Recover the untouched
    # q2 singleton from u12 and combine it with the Bell-pair summary u01.
    workspace = local_product(units["u01"], local_reduce(units["u12"], (2,)))
    workspace = local_apply(workspace, local_controlled_x(1), (1, 2))
    units = dict(units, u01=local_reduce(workspace, (0, 1)), u12=local_reduce(workspace, (1, 2)))
    positions["a3"] = {None: (1.0, units.copy())}
    units = dict(units, q3=local_apply(units["q3"], HADAMARD, (3,)))
    positions["a4"] = {None: (1.0, units.copy())}

    # q3 and q4 are independent of u12 before CCX. Derive both affected
    # target summaries from the same q1,q2,q3,q4 workspace.
    workspace = local_product(units["u12"], units["q3"], units["q4"])
    workspace = local_apply(workspace, local_controlled_x(2), (2, 3, 4))
    units = {
        "u01": units["u01"], "u12": local_reduce(workspace, (1, 2)),
        "u234": local_reduce(workspace, (2, 3, 4)), "q5": units["q5"],
    }
    positions["a5"] = {None: (1.0, units.copy())}
    for position in ("f1", "f2", "f3"):
        positions[position] = {}
    for outcome in (0, 1):
        projector = np.diag([1.0 - outcome, float(outcome)]).astype(complex)
        support, projected = local_apply(units["u234"], projector, (3,))
        probability = float(np.trace(projected).real)
        assert probability > TOLERANCE
        conditioned = (support, projected / probability)
        branch = {
            "u01": units["u01"], "u12": units["u12"],
            "q4": local_reduce(conditioned, (4,)), "q5": units["q5"],
        }
        positions["f1"][outcome] = (probability, branch.copy())
        if outcome == 1:
            branch = dict(branch, q5=local_apply(branch["q5"], HADAMARD, (5,)))
        positions["f2"][outcome] = (probability, branch.copy())
        workspace = local_product(branch["q4"], branch["q5"])
        workspace = local_apply(workspace, local_controlled_x(1), (5, 4))
        branch = {"u01": branch["u01"], "u12": branch["u12"], "u45": workspace}
        positions["f3"][outcome] = (probability, branch)
    return positions


def local_joint_z_distribution(units: dict) -> np.ndarray:
    """Combine the q01/q12 factors over their q1 separator for this example."""
    pair01 = z_probabilities(units["u01"][1]).reshape(2, 2)
    pair12 = z_probabilities(units["u12"][1]).reshape(2, 2)
    separator = pair01.sum(axis=0)
    close(separator, pair12.sum(axis=1))
    result = np.zeros(8)
    for index in range(8):
        q0, q1, q2 = _local_bits(index, 3)
        if separator[q1] > TOLERANCE:
            result[index] = pair01[q0, q1] * pair12[q1, q2] / separator[q1]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/running-example.json"))
    arguments = parser.parse_args()

    # Compute these paths independently before importing QAMC.
    full_positions = build_reference_positions()
    local_positions = build_local_workspace_positions()
    expected_classes = {
        "a0": "s0", "a1": "s0", "a2": "s1", "a3": "s1", "a4": "s1",
        "a5": "s2", "f1": "s3", "f2": "s3", "f3": "s4",
    }
    expected_groupings = {
        "s0": [(q,) for q in range(QUBIT_COUNT)],
        "s1": [(0, 1), (1, 2), (3,), (4,), (5,)],
        "s2": [(0, 1), (1, 2), (2, 3, 4), (5,)],
        "s3": [(0, 1), (1, 2), (4,), (5,)],
        "s4": [(0, 1), (1, 2), (4, 5)],
    }
    comparisons = {}
    density_count = 0
    largest_error = 0.0
    for position, branches in local_positions.items():
        records = []
        assert set(branches) == set(full_positions[position])
        for outcome, (probability, units) in branches.items():
            full_probability, full_state = full_positions[position][outcome]
            close(probability, full_probability)
            assert [support for support, _ in units.values()] == expected_groupings[expected_classes[position]]
            errors = {}
            for unit_id, (support, density) in units.items():
                expected = reduced_density(full_state, support)
                error = float(np.max(np.abs(density - expected)))
                close(density, expected)
                errors[unit_id] = error
                density_count += 1
                largest_error = max(largest_error, error)
            records.append({"m3": outcome, "weight": probability, "density_max_absolute_errors": errors})
        comparisons[position] = records
    assert density_count == 53

    expected012 = np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
    expected45 = np.array([5 / 8, 1 / 8, 1 / 8, 1 / 8])
    aggregate45 = np.zeros(4)
    full_aggregate45 = np.zeros(4)
    branch_results = {}
    branch_output_distributions = []
    for outcome, (probability, units) in local_positions["f3"].items():
        full_probability, full_state = full_positions["f3"][outcome]
        snapshots = {name: (support, density.copy()) for name, (support, density) in units.items()}
        joint012 = local_joint_z_distribution(units)
        for name, (support, density) in units.items():
            assert support == snapshots[name][0]
            np.testing.assert_array_equal(density, snapshots[name][1])
        full012 = z_probabilities(reduced_density(full_state, (0, 1, 2)))
        output45 = z_probabilities(units["u45"][1])
        full45 = z_probabilities(reduced_density(full_state, (4, 5)))
        close(joint012, expected012)
        close(joint012, full012)
        close(output45, full45)
        aggregate45 += probability * output45
        full_aggregate45 += full_probability * full45
        branch_output_distributions.append(output45)
        branch_results[str(outcome)] = {
            "local_probability": probability, "full_reference_probability": full_probability,
            "local_joint_q012_z": distribution_json(joint012),
            "full_reference_q012_z": distribution_json(full012),
            "local_q45_z": distribution_json(output45),
            "full_reference_q45_z": distribution_json(full45),
            "local_q45_density": matrix_json(units["u45"][1]),
        }
    close(aggregate45, expected45)
    close(full_aggregate45, expected45)
    pointwise_distribution_passed = all(
        np.allclose(distribution, expected45, rtol=0, atol=TOLERANCE)
        for distribution in branch_output_distributions
    )
    uniform_aggregate_passed = bool(np.allclose(aggregate45, np.full(4, 0.25), rtol=0, atol=TOLERANCE))
    assert not pointwise_distribution_passed and not uniform_aggregate_passed

    # Check the actual declaration, then run the unmodified checker separately.
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from qamc.backend import DensityMatrixBackend
    from qamc.checker import run_model
    from qamc.engine import TraceExecutor
    from qamc.engine.transition_executor import TransitionExecutor
    from qamc.ir.enums import EvaluationMode
    from qamc.loader import load_qam_file
    from qamc.property.evaluator import PropertyEvaluator
    from qamc.validate import validate_qam_model

    model = load_qam_file(MODEL_PATH)
    validation = validate_qam_model(model)
    assert validation.ok, validation
    assert set(model.system_model.state_classes) == set(expected_groupings)
    for class_id, grouping in expected_groupings.items():
        state_class = model.system_model.state_class(class_id)
        assert [tuple(int(q[1:]) for q in unit.qubits) for unit in state_class.units] == grouping
    trace = model.trace("main")
    assert {position: anchor.admissible_classes for position, anchor in trace.anchors.items()} == {
        position: (class_id,) for position, class_id in expected_classes.items()
    }
    assert len(model.system_model.transition_classes) == 8 and len(trace.steps) == 8
    expected_steps = [
        ("a0", "t_h0", "a1", "H", ("q0",)),
        ("a1", "t_cx01", "a2", "CX", ("q0", "q1")),
        ("a2", "t_cx12", "a3", "CX", ("q1", "q2")),
        ("a3", "t_h3", "a4", "H", ("q3",)),
        ("a4", "t_ccx", "a5", "CCX", ("q2", "q3", "q4")),
        ("a5", "t_measure_q3", "f1", "MZ", ("q3",)),
        ("f1", "t_cond", "f2", "NOOP", ()),
        ("f2", "t_cx54", "f3", "CX", ("q5", "q4")),
    ]
    for step, (source, name, target, gate_name, gate_qubits) in zip(trace.steps, expected_steps):
        assert (step.source_anchor, step.transition_class, step.target_anchor) == (source, name, target)
        transition = model.system_model.transition_class(name)
        assert transition.source_class == expected_classes[source]
        assert {branch.target_class for branch in transition.branches} == {expected_classes[target]}
        gate = transition.gate
        assert (gate.name, gate.controls + gate.targets + gate.extra_qargs) == (gate_name, gate_qubits)
    measurement = model.system_model.transition_class("t_measure_q3")
    assert measurement.gate.record_as == "m3"
    assert {branch.outcome for branch in measurement.branches} == {0, 1}
    feedback = model.system_model.transition_class("t_cond")
    assert [(branch.guard_expr, branch.action_gate.name, branch.action_gate.targets)
            for branch in feedback.branches] == [
        (("eq", "m3", 0), "ID", ("q5",)), (("eq", "m3", 1), "H", ("q5",)),
    ]
    execution = model.execution("run_main")
    assert execution.stop_anchor == "f3"
    assert execution.initial_binding.anchor_id == "a0"
    assert execution.initial_binding.payload_initializer == "zero_density_matrix"
    properties = {prop.id: prop for prop in model.properties}
    assert set(properties) == {"p_equal_results", "p_output_distribution"}
    assert properties["p_equal_results"].mode is EvaluationMode.POINTWISE_ALL
    assert properties["p_equal_results"].observable.qubits == ("q0", "q1", "q2")
    assert properties["p_equal_results"].expectation == ("equal_bits",)
    assert properties["p_output_distribution"].mode is EvaluationMode.AGGREGATE
    assert properties["p_output_distribution"].observable.qubits == ("q4", "q5")
    assert all(prop.scope_value == ("f3",) for prop in properties.values())
    assert all(prop.observable.basis == "Z" for prop in properties.values())
    close([properties["p_output_distribution"].expectation[_local_bits(i, 2)] for i in range(4)], expected45)

    class RecordingBackend(DensityMatrixBackend):
        def __init__(self) -> None:
            self.initializations = []

        def initialize_unit_payload(self, *, unit_id, summary_kind, qubits, initializer):
            self.initializations.append({"unit_id": unit_id, "qubits": list(qubits), "initializer": initializer})
            return super().initialize_unit_payload(
                unit_id=unit_id, summary_kind=summary_kind, qubits=qubits, initializer=initializer,
            )

    class ObservingExecutor(TransitionExecutor):
        def __init__(self, backend) -> None:
            super().__init__(backend=backend)
            self.last_transition = None
            self.context = None
            self.a2_u12 = None

        def execute(self, *, context, source_node, transition):
            self.last_transition = transition.id
            self.context = context
            if transition.id == "t_cx12":
                self.a2_u12 = np.asarray(source_node.unit_payloads["u12"].data).copy()
            return super().execute(context=context, source_node=source_node, transition=transition)

    backend = RecordingBackend()
    observer = ObservingExecutor(backend)
    executor = TraceExecutor(backend=backend, transition_executor=observer)
    checker = {"declaration_validation": "passed", "completed_execution": False}
    try:
        result = run_model(model, backend=backend, trace_executor=executor)
        assert result.context is not None
        checker_error = 0.0
        checker_weight_error = 0.0
        checker_count = 0
        for position, branches in full_positions.items():
            node_ids = result.context.anchors.nodes_of(position)
            assert len(node_ids) == len(branches)
            assert {result.context.node(node_id).measurement_classical.get("m3")
                    for node_id in node_ids} == set(branches)
            for node_id in node_ids:
                node = result.context.node(node_id)
                full_probability, full_state = branches[node.measurement_classical.get("m3")]
                checker_weight_error = max(checker_weight_error, abs(node.reach_probability - full_probability))
                close(node.reach_probability, full_probability)
                for payload in node.unit_payloads.values():
                    expected = reduced_density(full_state, tuple(int(q[1:]) for q in payload.qubits))
                    checker_error = max(checker_error, float(np.max(np.abs(payload.data - expected))))
                    checker_count += 1
        assert len(result.context.nodes) == 12 and len(result.context.edges) == 11
        evaluator = PropertyEvaluator(backend=backend)
        distribution_property = properties["p_output_distribution"]
        negative_controls = {
            "output_distribution_as_pointwise_all": evaluator.evaluate(
                result.context, replace(distribution_property, mode=EvaluationMode.POINTWISE_ALL),
            ).passed,
            "uniform_aggregate_expectation": evaluator.evaluate(
                result.context, replace(distribution_property, expectation={_local_bits(i, 2): 0.25 for i in range(4)}),
            ).passed,
        }
        assert not any(negative_controls.values())
        checker.update({
            "status": "passed" if result.ok and checker_error <= TOLERANCE and checker_count == 53 else "mismatch",
            "completed_execution": True, "local_density_comparisons": checker_count,
            "maximum_density_error": checker_error,
            "maximum_branch_weight_error": checker_weight_error,
            "property_results": {item.property_id: item.passed for item in result.property_results},
            "negative_controls": {name: {"passed": passed} for name, passed in negative_controls.items()},
        })
    except NotImplementedError as error:
        checker.update({"status": "unsupported", "exception": type(error).__name__, "message": str(error)})
    except Exception as error:
        checker.update({"status": "error", "exception": type(error).__name__, "message": str(error)})
    checker["last_transition"] = observer.last_transition
    checker["initializations_after_root"] = backend.initializations[QUBIT_COUNT:]
    if observer.context is not None:
        checker["runtime_states_constructed"] = len(observer.context.nodes)
        checker["runtime_transitions_constructed"] = len(observer.context.edges)
    if observer.a2_u12 is not None:
        reference12 = reduced_density(full_positions["a2"][None][1], (1, 2))
        checker["a2_u12"] = {
            "actual_density": matrix_json(observer.a2_u12),
            "full_reference_density": matrix_json(reference12),
            "maximum_absolute_error": float(np.max(np.abs(observer.a2_u12 - reference12))),
        }

    report = {
        "status": "passed" if checker["status"] == "passed" else "reference_passed_checker_" + checker["status"],
        "model": str(MODEL_PATH.relative_to(REPOSITORY_ROOT)),
        "tolerance": TOLERANCE,
        "method_reference": {
            "status": "passed",
            "description": "Independent full six-qubit state vector compared with explicit local density-matrix workspaces; neither numerical path imports QAMC's backend.",
            "state_classes": 5, "transition_classes": 8, "trace_positions": 9,
            "reference_instances": 12, "reference_transitions": 11,
            "local_density_comparisons": density_count, "maximum_density_error": largest_error,
            "position_comparisons": comparisons, "branches": branch_results,
            "aggregate_q45_z": distribution_json(aggregate45),
            "assertions": {
                "p_equal_results": {"mode": "pointwise_all", "passed": True},
                "p_output_distribution": {"mode": "aggregate", "passed": True},
            },
            "negative_controls": {
                "output_distribution_as_pointwise_all": {"passed": pointwise_distribution_passed},
                "uniform_aggregate_expectation": {"passed": uniform_aggregate_passed},
            },
            "joint_query": {"units": ["u01", "u12"], "separator": "q1", "payloads_unchanged": True, "u012_stored": False},
            "workspace_conditions": [
                "At CX01, u12 is Tr_q0(rho01) tensor |0><0| on untouched q2.",
                "At CX12, q2 remains independent: rho01 tensor Tr_q1(rho12) supplies the q012 workspace needed to update both overlapping units.",
                "At CCX, q3 and q4 are independent of u12: rho12 tensor rho3 tensor rho4 supplies the q1234 workspace.",
                "The measurement leaves the retained q01 and q12 summaries unchanged for this program, as independently checked against both conditional full states.",
            ],
            "interpretation": "The method-level calculation uses explicit local workspaces; the checker is evaluated separately under current_checker. The full reference is used only for comparison, never to supply the local calculation's target matrices.",
        },
        "current_checker": checker,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REFERENCE PASS: 53 local summaries and both assertions match the full six-qubit calculation; report: {arguments.output}")
    print("REFERENCE PASS: both negative controls fail as expected; joint q012 Z probabilities are 1/2 at 000 and 111.")
    if checker["status"] == "passed":
        print("CHECKER PASS: declaration execution and retained summaries also match the reference.")
        print("CHECKER PASS: both assertions pass and both negative controls fail as expected.")
        return 0
    print(f"CHECKER {checker['status'].upper()}: {checker.get('message', 'Execution differs from the reference.')} (at {observer.last_transition})")
    if "a2_u12" in checker:
        print(f"CHECKER a2 u12 maximum density error: {checker['a2_u12']['maximum_absolute_error']:.6g}.")
    print("The independent method-level reference passes; the current checker has not verified this declaration.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
