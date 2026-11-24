#!/usr/bin/env python3
"""Check the archived measurements against the paper's table and figure inputs."""

import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiment_data" / "paper_results"
FIGURE_FAMILIES = ("ghz", "bv", "quantum_walk", "iqft_phase")
SCALING_FAMILIES = ("ghz", "bv", "adder", "grover", "simon")
SCALING_SIZES = (50, 100, 200, 300, 400, 500)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with (DATA / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_hashes():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    csv_paths = {path.relative_to(DATA).as_posix() for path in DATA.rglob("*.csv")}
    require(csv_paths == set(manifest["files"]), "The CSV files and the manifest differ")
    for path, entry in manifest["files"].items():
        digest = hashlib.sha256((DATA / path).read_bytes()).hexdigest()
        require(digest == entry["sha256"], f"{path}: SHA-256 mismatch")
        require(len(read_csv(path)) == entry["records"], f"{path}: unexpected number of rows")

    source = ROOT / "src" / "qamc"
    tree = hashlib.sha256()
    for path in sorted(source.rglob("*.py"), key=lambda path: path.relative_to(source).as_posix()):
        tree.update(path.relative_to(source).as_posix().encode("utf-8") + b"\0")
        tree.update(hashlib.sha256(path.read_bytes()).digest())
    expected = manifest["source_revisions"]["main_measurements"]["source_tree_sha256"]
    require(tree.hexdigest() == expected, "The checker source differs from the measured release")


def verify_small_models():
    raw = read_csv("raw/rq1_rq2_measurements.csv")
    cases = json.loads((ROOT / "validation/rq1_property_audit/manifest.json").read_text())["cases"]
    require(len(raw) == 28, "Expected 28 small-model measurements")
    for rq, count in (("RQ1", 22), ("RQ2", 6)):
        table = read_csv(f"{rq.lower()}_table.csv")
        measurements = [r for r in raw if r["question"] == rq]
        require(len(table) == len(measurements) == count, f"{rq}: incomplete table")
        inputs = set()
        for n, (r, row) in enumerate(zip(measurements, table), 1):
            where = f"{rq} row {n}"
            source = ROOT / "experiment_data/models" / r["model"]
            require(source.is_file() and source not in inputs, f"{where}: missing or duplicate input")
            inputs.add(source)
            require(int(r["row"]) == n, f"{where}: row order differs")
            if rq == "RQ1":
                case = cases[n - 1]
                require(int(case["row"]) == n and ROOT / case["model"] == source, f"{where}: model mapping differs")
                expected = "satisfied" if case["expected_satisfied"] else "violated"
            else:
                expected = "satisfied"
            require(r["judgment"] == r["expected_judgment"] == expected, f"{where}: judgment differs")
            require(row["result_latex"] == r"\ensuremath{\surd}", f"{where}: table judgment differs")
            for field in ("qubits", "gates", "nonempty_source_lines"):
                require(row[field] == r[field], f"{where}: {field} differs")
            require(row["time_ms"] == f"{float(r['time_ms']):.1f}", f"{where}: time differs")
            require(row["retained_array_mib"] == f"{float(r['retained_array_mib']):.3f}", f"{where}: memory differs")


def verify_efficiency():
    by_config = defaultdict(list)
    key = lambda r: (r["benchmark_id"], r["style"], int(r["qubits"]))
    for trial in read_csv("raw/rq3_trials.csv"):
        by_config[key(trial)].append(trial)
    expected = {(f, s, n) for f in FIGURE_FAMILIES for s in ("abstract", "full") for n in range(5, 21)}
    summary = read_csv("rq3_scalability.csv")
    require(by_config.keys() == expected, "RQ3: missing or unexpected trial configuration")
    require(len(summary) == len(expected) and {key(r) for r in summary} == expected, "RQ3: summary coverage differs")
    for r in summary:
        cfg = key(r)
        trials = by_config[cfg]
        require(len(trials) == int(r["repeats"]) == 5 and sorted(int(t["trial"]) for t in trials) == list(range(1, 6)),
                f"RQ3 {cfg}: expected five distinct trials")
        for t in trials:
            require(t["status"] == r["status"] == "ok" and t["ok"] == r["ok"] == "1", f"RQ3 {cfg}: judgment differs")
            for field in ("gates", "runtime_nodes", "runtime_edges", "properties"):
                require(t[field] == r[field], f"RQ3 {cfg}: {field} differs")
            require(f"{float(t['payload_mib']):.6f}" == r["payload_mib"], f"RQ3 {cfg}: retained memory differs")
        times = [float(t["wall_time_ms"]) for t in trials]
        expected_stats = {
            "wall_time_ms": statistics.median(times),
            "wall_time_min_ms": min(times),
            "wall_time_max_ms": max(times),
            "wall_time_stdev_ms": statistics.pstdev(times),
            "py_peak_mib": statistics.median(float(t["py_peak_mib"]) for t in trials),
        }
        for field, value in expected_stats.items():
            require(f"{value:.6f}" == r[field], f"RQ3 {cfg}: {field} differs from individual trials")


def verify_scalability():
    measured = {}
    for filename, families in (("rq4_other_families.csv", {"ghz", "bv", "adder", "simon"}),
                               ("rq4_grover.csv", {"grover"})):
        for row in read_csv("raw/" + filename):
            family, n = row["benchmark_id"], int(row["qubits"])
            require(family in families and (family, n) not in measured, f"{filename}: unexpected row {(family, n)}")
            require(row["status"] == "ok" and row["ok"] == "1", f"{family} {n}: property failure")
            # The driver rounds run, check and their sum independently to six decimal places.
            difference = float(row["total_time_ms"]) - float(row["run_time_ms"]) - float(row["check_time_ms"])
            require(abs(difference) <= 0.0000011, f"{family} {n}: inconsistent total time")
            measured[family, n] = row
    expected = [(f, n) for f in SCALING_FAMILIES for n in SCALING_SIZES]
    require(set(measured) == set(expected), "RQ4: missing or unexpected configuration")
    require(read_csv("rq4_large_scale.csv") == [measured[k] for k in expected], "RQ4: assembled table differs")


def verify_reference_results():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    reports = {}
    for key, spec in manifest["validation_outputs"].items():
        path = ROOT / spec["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"], f"{key}: reference report SHA-256 mismatch")
        reports[key] = json.loads(path.read_text(encoding="utf-8"))

    rq1 = reports["rq1"]
    mapping = json.loads((ROOT / "validation/rq1_property_audit/manifest.json").read_text(encoding="utf-8"))
    require(len(rq1["cases"]) == len(mapping["cases"]) == 22, "RQ1: incomplete full-state comparison")
    count = 0
    for row, case in zip(rq1["cases"], mapping["cases"]):
        name = case["model"]
        require(row["model"] == name and row["row"] == case["row"], f"{name}: different full-state input")
        expected = case["expected_satisfied"]
        require(row["checker_satisfied"] == row["reference_satisfied"] == expected, f"{name}: full-state judgment differs")
        properties = row["properties"]
        require([p["property_id"] for p in properties] == [p["id"] for p in case["assertions"]],
                f"{name}: full-state property mapping differs")
        for prop in properties:
            require(prop["checker_satisfied"] == prop["reference_satisfied"], f"{name}: {prop['property_id']} differs")
        count += len(properties)
    require(count == rq1["summary"]["property_judgments"] == 35, "RQ1: expected 35 component judgments")

    feedback = reports["rq2_feedback"]
    require(feedback["same_property_preserved"] and feedback["comparison_pass"], "RQ2: feedback comparison failed")
    for name, expectation, distribution in (("correct", True, (1, 0)), ("faulty", False, (0.5, 0.5))):
        case = feedback["cases"][name]
        require(case["checker_judgment"] == case["reference_judgment"] == expectation,
                f"RQ2 {name}: unexpected feedback judgment")
        branches = case["terminal_branches"]
        outcomes = {(b["measurement_records"]["m0"], b["measurement_records"]["m1"]) for b in branches}
        require(len(branches) == 4 and outcomes == {(0, 0), (0, 1), (1, 0), (1, 1)}, f"RQ2 {name}: missing feedback branches")
        for source in ("checker", "reference"):
            measured = case[source + "_aggregate_distribution"]
            require(all(abs(measured[str(k)] - p) <= 1e-9 for k, p in enumerate(distribution)),
                    f"RQ2 {name}: {source} X-basis distribution differs")
            require(all(abs(b[source + "_reach_probability"] - 0.25) <= 1e-9 for b in branches),
                    f"RQ2 {name}: {source} branch probability differs")


def main():
    try:
        verify_hashes()
        verify_small_models()
        verify_efficiency()
        verify_scalability()
        verify_reference_results()
    except (OSError, KeyError, ValueError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print("Verified checker and CSV hashes, 28 model measurements, 640 efficiency trials, 30 scalability runs, 35 reference judgments and both feedback variants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
