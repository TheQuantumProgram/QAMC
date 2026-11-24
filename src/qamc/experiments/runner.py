"""Measurements for the circuit and hybrid-program models in Tables 1 and 2."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import time

from qamc.checker import run_qam_file
from qamc.external_properties.loader import load_qprop_file, merge_external_property_bundles
from qamc.external_properties.model import ExternalPropertyBundle
from qamc.ir.model import GateSpec, QAMModel
from qamc.loader import load_qam_file


# Table 2 order. Each program is expected to satisfy its properties.
HYBRID_MODELS = (
    "HybridClassical/Teleportation/teleport_plus.qam",
    "HybridClassical/BitFlipCode/bit_flip_syndrome_q1.qam",
    "HybridClassical/EntanglementSwapping/entanglement_swap_bell.qam",
    "HybridClassical/HybridControlRegister/hybrid_control_register_10.qam",
    "HybridClassical/ProgramMeasurementMux/program_measurement_mux.qam",
    "HybridClassical/MultiwayProgramBranch/multiway_program_branch_10.qam",
)


@dataclass(frozen=True)
class ModelMeasurement:
    question: str
    row: int
    model: str
    qubits: int
    gates: int
    nonempty_source_lines: int
    time_ms: float
    retained_array_mib: float
    satisfied: bool
    expected_satisfied: bool


def measure_paper_models(root: Path) -> list[ModelMeasurement]:
    """Time the checker on the models in the order of the two paper tables."""
    mapping = root.parent.parent / "validation/rq1_property_audit/manifest.json"
    cases = json.loads(mapping.read_text(encoding="utf-8"))["cases"]
    models = [
        ("RQ1", case["row"], Path(case["model"]).relative_to("experiment_data/models"), case["expected_satisfied"])
        for case in cases
    ]
    models.extend(("RQ2", i, Path(name), True) for i, name in enumerate(HYBRID_MODELS, 1))

    rows = []
    for question, index, relative_path, expected in models:
        path = root / relative_path
        model = load_qam_file(path)
        external_properties = load_experiment_properties(path)
        start = time.perf_counter()
        result = run_qam_file(path, external_properties=external_properties)
        time_ms = (time.perf_counter() - start) * 1000.0
        if not result.validation.ok or result.context is None:
            raise ValueError(f"Could not execute {relative_path}: {result.validation.diagnostics}")
        if not (result.property_results or result.temporal_results):
            raise ValueError(f"No properties checked for {relative_path}")
        rows.append(ModelMeasurement(
            question=question,
            row=index,
            model=relative_path.as_posix(),
            qubits=len(model.system_model.qubit_universe),
            gates=_execution_quantum_gate_count(model),
            nonempty_source_lines=_source_line_count(path),
            time_ms=time_ms,
            retained_array_mib=_context_payload_mib(result.context),
            satisfied=result.ok,
            expected_satisfied=expected,
        ))
    return rows


def render_model_measurements(rows: list[ModelMeasurement]) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("question", "row", "model", "qubits", "gates", "nonempty_source_lines",
                     "time_ms", "retained_array_mib", "judgment", "expected_judgment"))
    for row in rows:
        writer.writerow((row.question, row.row, row.model, row.qubits, row.gates,
                         row.nonempty_source_lines, f"{row.time_ms:.6f}", f"{row.retained_array_mib:.6f}",
                         "satisfied" if row.satisfied else "violated",
                         "satisfied" if row.expected_satisfied else "violated"))
    return output.getvalue().rstrip()


def load_experiment_properties(model_path: str | Path) -> ExternalPropertyBundle | None:
    """Return optional external properties checked together with one model."""

    path = Path(model_path)
    candidates = (
        path.with_suffix(".qprop"),
        path.parent / "properties" / f"{path.stem}.qprop",
    )
    bundles = tuple(load_qprop_file(candidate) for candidate in candidates if candidate.exists())
    if not bundles:
        return None
    return merge_external_property_bundles(bundles)


def _execution_quantum_gate_count(model: QAMModel) -> int:
    execution = next(iter(model.executions.values()), None)
    if execution is None:
        return 0
    trace = model.traces[execution.trace_id]  # type: ignore[attr-defined]
    count = 0
    for step in trace.steps:
        transition = model.system_model.transition_class(step.transition_class)
        count += _count_quantum_gate(transition.gate)
        count += sum(_count_quantum_gate(branch.action_gate) for branch in transition.branches)
    return count


def _count_quantum_gate(gate: GateSpec | None) -> int:
    if gate is None:
        return 0
    name = gate.name.upper()
    if name == "NOOP" or gate.is_measurement():
        return 0
    return 1


def _source_line_count(path: str | Path) -> int:
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def _context_payload_mib(context) -> float:
    if context is None:
        return 0.0
    total = 0
    for node in context.nodes.values():
        for payload in node.unit_payloads.values():
            total += int(getattr(payload.data, "nbytes", 0))
    return total / (1024 * 1024)
