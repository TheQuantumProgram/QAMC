#!/usr/bin/env python3
"""Generate LaTeX for Tables 1–3 from the paper's recorded results."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiment_data/paper_results"
DEFAULT_OUTPUT = ROOT / "runs"
ROW_END = r" \\"
SCALING_FAMILIES = {"ghz": "GHZ", "bv": "BV", "adder": "Adder", "grover": "Grover", "simon": "Simon"}
SCALING_SIZES = (50, 100, 200, 300, 400, 500)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table(caption, label, spacing, layout, headings, rows):
    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:qam-" + label + "}",
        r"\small", rf"\setlength{{\tabcolsep}}{{{spacing}pt}}",
        r"\begin{tabular*}" + layout, r"\hline",
        headings + ROW_END, r"\hline",
    ]
    lines.extend(" & ".join(row) + ROW_END for row in rows)
    lines.extend([r"\hline", r"\end{tabular*}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def model_rows(path, columns, expected_count):
    records = read_csv(path)
    if len(records) != expected_count or len({r["model"] for r in records}) != expected_count:
        raise ValueError(f"Expected {expected_count} distinct models in {path}")
    for record in records:
        yield [record[column] for column in columns]


def scaling_rows(path):
    records = read_csv(path)
    indexed = {(r["benchmark_id"], int(r["qubits"])): r for r in records}
    expected = {(family, size) for family in SCALING_FAMILIES for size in SCALING_SIZES}
    if len(indexed) != len(records) or indexed.keys() != expected:
        raise ValueError(f"Expected one row per RQ4 family and register size in {path}")
    for family, label in SCALING_FAMILIES.items():
        for size in SCALING_SIZES:
            row = indexed[family, size]
            if row["status"] != "ok" or row["ok"] != "1":
                raise ValueError(f"Unsuccessful measurement for {family}, {size} qubits")
            yield [
                label if size == SCALING_SIZES[0] else "", str(size), row["gates"], row["lines"],
                f"{float(row['run_time_ms']):.1f}", f"{float(row['check_time_ms']):.1f}",
                f"{float(row['total_time_ms']):.1f}", f"{float(row['payload_mib']):.3f}",
                r"\ensuremath{\surd}",
            ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model = ("model", "qubits", "gates", "nonempty_source_lines", "time_ms", "retained_array_mib")
    result = ("property_latex", "result_latex")
    try:
        correctness = table(
            "Correctness-checking results. Ln. counts non-empty model source lines, "
            "T is the recorded end-to-end checking time in ms, and Mem is stored quantum-array "
            "memory in MiB. A check mark denotes agreement with the expected judgment, "
            "including expected violations for faulty and counterexample models.",
            "correctness", 2,
            r"{0.98\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.18\textwidth}"
            r"rrrrr>{\raggedright\arraybackslash}p{0.40\textwidth}c@{}}",
            r"Model & $n$ & G & Ln. & T & Mem & Property & Res.",
            model_rows(args.input_dir / "rq1_table_transcription.csv", model + result, 22),
        )
        hybrid = table(
            "Hybrid-program support results. Class. lists the recorded measurements and initialized "
            "program variables used by the model. The size, time, memory, and result columns "
            r"have the meanings given in Table~\ref{tab:qam-correctness}. Final quantum distributions "
            "are evaluated in aggregate.",
            "hybrid", 2,
            r"{0.94\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.14\textwidth}"
            r"rrrrr>{\raggedright\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}"
            r"p{0.36\textwidth}c@{}}",
            r"Model & $n$ & G & Ln. & T & Mem & Class. & Property & Res.",
            model_rows(args.input_dir / "rq2_table_transcription.csv", model + ("classical_latex",) + result, 6),
        )
        scaling = table(
            "Scalability results for the five benchmark families.", "scaling", 3,
            r"{0.94\textwidth}{@{\extracolsep{\fill}}lrrrrrrrc@{}}",
            "Family & Size & Gates & Lines & Run (ms) & Check (ms) & Total (ms) & Mem (MiB) & Result",
            scaling_rows(args.input_dir / "rq4_large_scale.csv"),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for number, content in enumerate((correctness, hybrid, scaling), 1):
            output = args.output_dir / f"table{number}.tex"
            output.write_text(content, encoding="utf-8")
            print(f"Wrote {output}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
