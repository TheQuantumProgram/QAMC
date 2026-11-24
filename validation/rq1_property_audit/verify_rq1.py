"""Recheck the properties mapped to all 22 manuscript RQ1 rows.

This validates judgments for the mapped properties, using the original program
and abstraction declarations. It does not collect performance measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from qamc.engine import TraceExecutor
from qamc.experiments.modeling_audit import reference_judgment
from qamc.experiments.modeling_reference import evaluate_reference, reference_observation
from qamc.ir.enums import EvaluationMode, PropertyKind, ScopeKind
from qamc.ir.property import ObservableSpec, PropertySpec
from qamc.loader import load_qam_file
from qamc.property import PropertyEvaluator
from qamc.validate import validate_qam_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(bits: str):
    return int(bits) if len(bits) == 1 else tuple(int(bit) for bit in bits)


def property_spec(data: dict) -> PropertySpec:
    kind = PropertyKind(data["kind"])
    if kind is PropertyKind.DISTRIBUTION:
        expectation = {outcome(bits): probability for bits, probability in data["expected_probabilities"].items()}
    elif kind is PropertyKind.REACHABILITY:
        expectation = ("probability_gt", data["outcome"], data["threshold"])
    elif data["relation"] == "equal_bits":
        expectation = ("equal_bits",)
    elif data["relation"] == "support_subset":
        expectation = ("support_subset", frozenset(outcome(bits) for bits in data["allowed_outcomes"]))
    else:
        raise ValueError(f"Unsupported mapped property: {data!r}")
    return PropertySpec(
        data["id"], kind, ScopeKind.ANCHORS, tuple(data["nodes"]),
        EvaluationMode(data["mode"]),
        ObservableSpec("quantum", qubits=tuple(data["qubits"]), basis=data["basis"]),
        expectation,
    )


def json_observation(value):
    if isinstance(value, dict):
        return {str(key): float(probability) for key, probability in value.items() if probability != 0}
    if isinstance(value, (tuple, list)):
        return [json_observation(item) for item in value]
    return value


def check_row(case: dict) -> dict:
    for source in case["source_files"]:
        if sha256(PROJECT / source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed since the mapping was recorded: {source['path']}")
    loaded = load_qam_file(PROJECT / case["model"])
    props = tuple(property_spec(item) for item in case["assertions"])
    if case.get("use_declared_assertions"):
        source_props = {prop.id: prop for prop in loaded.properties}
        for prop in props:
            source_prop = source_props[prop.id]
            if source_prop != prop:
                raise ValueError(f"IQFT mapping differs from its original assertion: {prop.id}")
    model = replace(loaded, properties=props)
    validation = validate_qam_model(model)
    if not validation.ok:
        raise ValueError(f"Invalid mapped model: {validation!r}")
    reference = evaluate_reference(model)
    context = TraceExecutor().run(model)
    evaluator = PropertyEvaluator()
    details = []
    for prop in props:
        actual = evaluator.evaluate(context, prop)
        expected_observation = reference_observation(reference, prop)
        expected_judgment = reference_judgment(prop, expected_observation)
        details.append({
            "property_id": prop.id,
            "selected_positions": list(actual.selected_anchor_ids),
            "checker_satisfied": actual.passed,
            "reference_satisfied": expected_judgment,
            "judgments_agree": actual.passed == expected_judgment,
            "checker_observation": json_observation(actual.observed),
            "reference_observation": json_observation(expected_observation),
        })
    checker = all(item["checker_satisfied"] for item in details)
    reference_judgment_all = all(item["reference_satisfied"] for item in details)
    return {
        "row": case["row"], "model_label": case["model_label"], "model": case["model"],
        "expected_satisfied": case["expected_satisfied"],
        "checker_satisfied": checker, "reference_satisfied": reference_judgment_all,
        "matches_expected": checker == reference_judgment_all == case["expected_satisfied"],
        "all_property_judgments_agree": all(item["judgments_agree"] for item in details),
        "source_files": case["source_files"], "properties": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path; default is stdout.")
    args = parser.parse_args()
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    history = PROJECT / manifest["historical_table_source"]
    if sha256(history) != manifest["historical_table_sha256"]:
        raise ValueError("The historical table transcription changed.")
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Judgments for the 22 mapped RQ1 properties, with the IQFT per-bit thresholds taken from the original model.",
        "historical_performance_rerun": False,
        "manifest_sha256": sha256(manifest_path),
        "code_sha256": {str(path.relative_to(PROJECT)): sha256(path)
                        for path in sorted((PROJECT / "src/qamc").rglob("*.py"))},
        "validation_script_sha256": sha256(Path(__file__)),
        "cases": [],
    }
    for case in manifest["cases"]:
        try:
            row = check_row(case)
        except Exception as exc:
            row = {"row": case["row"], "model_label": case["model_label"], "model": case["model"],
                   "matches_expected": False, "error": f"{type(exc).__name__}: {exc}"}
        result["cases"].append(row)
        print(f"{case['row']:02d} {case['model_label']}: "
              f"{'matched' if row['matches_expected'] else row.get('error', 'judgment mismatch')}", file=sys.stderr)
    result["summary"] = {
        "rows": len(result["cases"]),
        "rows_matching_expected": sum(row["matches_expected"] for row in result["cases"]),
        "expected_satisfactions": sum(case["expected_satisfied"] for case in manifest["cases"]),
        "expected_violations": sum(not case["expected_satisfied"] for case in manifest["cases"]),
        "property_judgments": sum(len(row.get("properties", [])) for row in result["cases"]),
        "all_property_judgments_agree": all(row.get("all_property_judgments_agree", False) for row in result["cases"]),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(json.dumps(result["summary"], sort_keys=True))
    else:
        sys.stdout.write(serialized)
    passed = (result["summary"]["rows_matching_expected"] == len(manifest["cases"])
              and result["summary"]["all_property_judgments_agree"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
