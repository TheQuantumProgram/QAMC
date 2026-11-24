"""Check one RQ4 Grover block against two independent full-register trajectories."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qamc.checker import run_model
from qamc.experiments.modeling_reference import ReferenceNode, _apply_gate
from qamc.experiments.rq4 import RQ4_BENCHMARKS
from qamc.loader import load_qam_string
from qamc.property.observables import SubspaceInconclusiveError


def check_block() -> dict:
    source = RQ4_BENCHMARKS["grover"].model_source(10)
    model = load_qam_string(source)
    result = run_model(model)
    assert result.ok and len(result.property_results) == 4
    context = result.context
    order = model.system_model.qubit_universe
    assert order == tuple(f"q{i}" for i in range(10))

    # An orthonormal basis for the input space spanned by |0> and |+> on
    # the first five qubits. The remaining five qubits are initially zero.
    zero = np.zeros(2 ** len(order), dtype=complex)
    zero[0] = 1
    complement = np.zeros_like(zero)
    complement[np.arange(1, 32) << 5] = 1 / np.sqrt(31)
    witnesses = [
        ReferenceNode(str(i), "S_cover", order, {}, {}, 1.0, vector)
        for i, vector in enumerate((zero, complement))
    ]

    locations = [context.trace.entry_anchor] + [step.target_anchor for step in context.trace.steps]
    comparisons = 0
    max_error = 0.0
    for i, location in enumerate(locations):
        if i:
            step = context.trace.steps[i - 1]
            gate = model.system_model.transition_class(step.transition_class).gate
            for witness in witnesses:
                witness.state_vector = _apply_gate(witness.state_vector, order, gate)
        node_id, = context.anchors.nodes_of(location)
        for unit in context.node(node_id).unit_payloads.values():
            expected = (witnesses[0].reduced_density(unit.qubits) + witnesses[1].reduced_density(unit.qubits)) / 2
            actual = unit.data @ unit.data.conj().T
            error = float(np.max(np.abs(actual - expected)))
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
            comparisons += 1
            max_error = max(max_error, error)
    assert (len(locations), comparisons) == (52, 208)

    # Change the marked string by omitting both X(q0) calls around the oracle.
    faulty = source.replace("gates H, X, CCX", "gates H, X, CCX, ID")
    for i in (7, 19):
        transition = f"transition t{i:04d}_x: S_cover -> S_cover {{\n  X(q0)\n}}"
        assert faulty.count(transition) == 1
        faulty = faulty.replace(transition, transition.replace("X(q0)", "ID(q0)"))
    try:
        run_model(load_qam_string(faulty))
    except SubspaceInconclusiveError:
        pass
    else:
        raise AssertionError("The shifted oracle was certified against the original witness space.")
    return {
        "nodes": len(locations),
        "local_density_matrices_compared": comparisons,
        "maximum_absolute_error": max_error,
        "assertions_satisfied": len(result.property_results),
        "shifted_oracle_assertions_proved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/grover-density.json"))
    args = parser.parse_args()
    report = check_block()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['local_density_matrices_compared']} local matrices at "
          f"{report['nodes']} positions and the shifted-oracle control. Report: {args.output}")


if __name__ == "__main__":
    main()
