"""CLI for running all QAMC experiment models under a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from qamc.experiments.rq3 import (
    RQ3_DEFAULT_FULL_PAYLOAD_CAP_MIB,
    RQ3_DEFAULT_FULL_QUBIT_LIMIT,
    RQ3_DEFAULT_REPEATS,
    RQ3_DEFAULT_SIZES,
    render_rq3_markdown_summary,
    run_rq3_scalability,
    write_rq3_outputs,
)
from qamc.experiments.rq4 import (
    RQ4_DEFAULT_SIZES,
    render_rq4_latex_table,
    render_rq4_markdown_summary,
    run_rq4_large_scale,
    write_rq4_outputs,
)
from qamc.experiments.runner import measure_paper_models, render_model_measurements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qam-evaluate",
        description="Run the QAMC paper experiments.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("experiment_data/models"),
        help="Model directory in the paper artifact (RQ1 and RQ2).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--measure-models",
        action="store_true",
        help="Measure the RQ1 and RQ2 models and emit CSV (default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the model measurement CSV or experiment summary to this path.",
    )
    mode.add_argument(
        "--rq3-scalability",
        action="store_true",
        help="Run the RQ3 full-vs-abstract scalability experiment.",
    )
    parser.add_argument(
        "--rq3-results-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for RQ3 CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--rq3-figure",
        type=Path,
        help="Optional path for the generated RQ3 figure (.pdf/.png/.svg via matplotlib, or .tex via pgfplots).",
    )
    parser.add_argument(
        "--rq3-full-qubit-limit",
        type=int,
        default=RQ3_DEFAULT_FULL_QUBIT_LIMIT,
        help="Largest qubit count attempted for full-state RQ3 models.",
    )
    parser.add_argument(
        "--rq3-sizes",
        default=f"{RQ3_DEFAULT_SIZES[0]}-{RQ3_DEFAULT_SIZES[-1]}",
        help="RQ3 qubit sizes as a comma list or range, e.g. 5-20 or 10,15,20.",
    )
    parser.add_argument(
        "--rq3-repeats",
        type=int,
        default=RQ3_DEFAULT_REPEATS,
        help="Number of repeated runs per RQ3 row; reported time is the median.",
    )
    parser.add_argument(
        "--rq3-full-payload-cap-mib",
        type=float,
        default=RQ3_DEFAULT_FULL_PAYLOAD_CAP_MIB,
        help="Skip full-state rows whose estimated runtime payload exceeds this MiB cap; use 0 to disable.",
    )
    mode.add_argument(
        "--rq4-large-scale",
        action="store_true",
        help="Run the RQ4 large-scale abstraction-oriented experiment.",
    )
    parser.add_argument(
        "--rq4-results-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for RQ4 CSV/Markdown/LaTeX outputs.",
    )
    parser.add_argument(
        "--rq4-sizes",
        default=",".join(str(size) for size in RQ4_DEFAULT_SIZES),
        help="RQ4 qubit sizes as a comma list or range, e.g. 50,100,200,300,400,500.",
    )
    parser.add_argument(
        "--rq4-table",
        choices=("markdown", "latex"),
        default="markdown",
        help="Table format emitted to stdout for RQ4 runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.rq3_scalability:
        full_payload_cap_mib = (
            None if args.rq3_full_payload_cap_mib <= 0 else args.rq3_full_payload_cap_mib
        )
        rq3_report = run_rq3_scalability(
            sizes=_parse_size_spec(args.rq3_sizes),
            full_qubit_limit=args.rq3_full_qubit_limit,
            repeats=args.rq3_repeats,
            full_payload_cap_mib=full_payload_cap_mib,
        )
        write_rq3_outputs(
            rq3_report,
            results_dir=args.rq3_results_dir,
            figure_path=args.rq3_figure,
        )
        text = render_rq3_markdown_summary(rq3_report)
        _emit(text, args.output)
        return 0 if all(row.status == "skipped" or (row.status == "ok" and row.ok) for row in rq3_report.rows) else 2

    if args.rq4_large_scale:
        rq4_report = run_rq4_large_scale(sizes=_parse_size_spec(args.rq4_sizes))
        write_rq4_outputs(rq4_report, results_dir=args.rq4_results_dir)
        text = (
            render_rq4_latex_table(rq4_report)
            if args.rq4_table == "latex"
            else render_rq4_markdown_summary(rq4_report)
        )
        _emit(text, args.output)
        return 0 if rq4_report.all_ok else 2

    rows = measure_paper_models(args.root.resolve())
    _emit(render_model_measurements(rows), args.output)
    return 0 if all(row.satisfied == row.expected_satisfied for row in rows) else 2


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


def _parse_size_spec(spec: str) -> tuple[int, ...]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = (int(part.strip()) for part in spec.split("-", 1))
        if end < start:
            raise SystemExit("Size range end must be >= start")
        return tuple(range(start, end + 1))
    sizes = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not sizes:
        raise SystemExit("Size list must not be empty")
    return sizes


if __name__ == "__main__":
    raise SystemExit(main())
