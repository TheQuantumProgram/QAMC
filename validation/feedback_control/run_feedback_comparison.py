#!/usr/bin/env python3
"""Compare correct and faulty feedback against an independent full-state reference.

The faulty model replaces the Z correction guarded by m0=1 with ID.
Both models use the same final assertion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(node) -> tuple[int, int]:
    return (int(node.measurement_classical["m0"]), int(node.measurement_classical["m1"]))


def distribution_error(left: dict, right: dict) -> float:
    return max(abs(float(left.get(k, 0)) - float(right.get(k, 0))) for k in set(left) | set(right))


def shown_distribution(value: dict) -> dict[str, float]:
    return {str(k): float(value.get(k, 0)) for k in (0, 1)}


def edge_key(edge, nodes) -> tuple:
    return (
        edge.transition_class_id,
        tuple(sorted(nodes[edge.source_node_id].measurement_classical.items())),
        tuple(sorted(nodes[edge.target_node_id].measurement_classical.items())),
    )


def main() -> None:
    location = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=location.parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project / "src"))
    from qamc.backend import DensityMatrixBackend
    from qamc.checker import run_qam_file
    from qamc.experiments.modeling_reference import compare_context, evaluate_reference, reference_observation
    from qamc.property.observables import quantum_distribution

    manifest = json.loads((location / "source_manifest.json").read_text())
    models = {}
    for name in ("correct", "faulty"):
        path = project / manifest[name]["path"]
        assert sha256(path) == manifest[name]["sha256"], f"Model checksum mismatch: {path}"
        models[name] = path
    correct = models["correct"].read_text()
    faulty = models["faulty"].read_text()
    before, after = manifest["mutation"]["before"], manifest["mutation"]["after"]
    assert correct.count(before) == 1 and correct.replace(before, after, 1) == faulty
    tolerance = 1e-9
    report = {
        "purpose": "Check that the same final X-basis assertion accepts correct feedback and rejects a missing Z correction.",
        "source": manifest,
        "runner_sha256": sha256(Path(__file__)),
        "reference_source": "src/qamc/experiments/modeling_reference.py",
        "reference_sha256": sha256(project / "src/qamc/experiments/modeling_reference.py"),
        "reference_scope": "Shares parsed QAM IR; independent full-state quantum updates, measurement, and observations.",
        "tolerance": tolerance,
        "cases": {},
    }
    property_definitions = []
    for name, expected_pass in [("correct", True), ("faulty", False)]:
        result = run_qam_file(models[name], validate=True)
        assert result.validation.ok, result.validation.diagnostics
        assert len(result.model.properties) == len(result.property_results) == 1
        prop = result.model.properties[0]
        property_definitions.append(prop)
        checked = result.property_results[0]
        reference = evaluate_reference(result.model)
        comparison = compare_context(reference, result.context, tolerance=tolerance)
        assert comparison.passed, comparison
        expected_observation = reference_observation(reference, prop)
        reference_pass = distribution_error(expected_observation, prop.expectation) <= tolerance
        assert checked.passed == reference_pass == expected_pass
        assert distribution_error(checked.observed, expected_observation) <= tolerance

        actual_nodes = {records(result.context.node(node_id)): result.context.node(node_id)
                        for node_id in checked.selected_node_ids}
        reference_nodes = {records(node): node for node in reference.nodes_at("f9")}
        assert set(actual_nodes) == set(reference_nodes) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        branches = []
        backend = DensityMatrixBackend()
        for key in sorted(actual_nodes):
            node, ref_node = actual_nodes[key], reference_nodes[key]
            actual = quantum_distribution(node, prop.observable, backend)
            expected = ref_node.distribution(("q2",), "X")
            error = distribution_error(actual, expected)
            assert error <= tolerance
            assert abs(node.reach_probability - ref_node.reach_probability) <= tolerance
            assert abs(ref_node.reach_probability - 0.25) <= tolerance
            analytic_outcome = 0 if name == "correct" or key[0] == 0 else 1
            analytic = {analytic_outcome: 1.0}
            assert distribution_error(expected, analytic) <= tolerance
            branches.append({
                "measurement_records": {"m0": key[0], "m1": key[1]},
                "checker_reach_probability": node.reach_probability,
                "reference_reach_probability": ref_node.reach_probability,
                "checker_q2_x_distribution": shown_distribution(actual),
                "reference_q2_x_distribution": shown_distribution(expected),
                "observation_max_absolute_error": error,
                "satisfies_same_condition_on_this_branch": distribution_error(expected, prop.expectation) <= tolerance,
            })

        reference_edges = {edge_key(edge, reference.nodes): edge for edge in reference.edges}
        measurement_edges = []
        for edge in result.context.edges:
            if edge.transition_class_id not in {"t_measure_source", "t_measure_pair"}:
                continue
            expected_edge = reference_edges[edge_key(edge, result.context.nodes)]
            assert abs(edge.local_probability - expected_edge.local_probability) <= tolerance
            target = result.context.node(edge.target_node_id)
            measurement_edges.append({
                "transition": edge.transition_class_id,
                "target_measurement_records": target.measurement_classical,
                "checker_local_probability": edge.local_probability,
                "reference_local_probability": expected_edge.local_probability,
            })
        assert len(measurement_edges) == 6
        report["cases"][name] = {
            "model_source": manifest[name]["path"],
            "model_sha256": manifest[name]["sha256"],
            "validation_ok": result.validation.ok,
            "property_id": prop.id,
            "mode": prop.mode.value,
            "selected_position": "f9",
            "observed_qubits": list(prop.observable.qubits),
            "basis": prop.observable.basis,
            "same_expected_distribution": shown_distribution(prop.expectation),
            "expected_judgment": expected_pass,
            "checker_judgment": checked.passed,
            "reference_judgment": reference_pass,
            "checker_aggregate_distribution": shown_distribution(checked.observed),
            "reference_aggregate_distribution": shown_distribution(expected_observation),
            "aggregate_max_absolute_error": distribution_error(checked.observed, expected_observation),
            "terminal_branches": branches,
            "measurement_edges": measurement_edges,
            "full_execution_comparison": asdict(comparison),
        }
    assert property_definitions[0] == property_definitions[1]
    report["same_property_preserved"] = True
    report["comparison_pass"] = True
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
