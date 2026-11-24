"""CLI for running the QAM checker on ``.qam`` files."""

from __future__ import annotations

import argparse
from pathlib import Path

from qamc.checker import run_qam_file
from qamc.external_properties.loader import load_qprop_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qam-check",
        description="Run the QAM checker on a .qam model file.",
    )
    parser.add_argument("path", type=Path, help="Path to the .qam model file.")
    parser.add_argument("--execution", dest="execution_id", help="Execution id to run.")
    parser.add_argument(
        "--properties",
        dest="properties_path",
        type=Path,
        help="Optional path to an external .qprop/.ltl property file or properties directory.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip definition-layer well-formedness checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    external_properties = None
    if args.properties_path is not None:
        external_properties = load_qprop_path(args.properties_path)
    result = run_qam_file(
        args.path,
        execution_id=args.execution_id,
        validate=not args.no_validate,
        external_properties=external_properties,
    )

    print(f"Model: {args.path}")
    if args.properties_path is not None:
        print(f"External properties: {args.properties_path}")
    if result.execution_id is not None:
        print(f"Execution: {result.execution_id}")

    if result.validation.diagnostics:
        print("Diagnostics:")
        for diagnostic in result.validation.diagnostics:
            print(
                f"  [{diagnostic.severity}] {diagnostic.code} "
                f"@ {diagnostic.location}: {diagnostic.message}"
            )

    if not result.validation.ok:
        print("Validation failed.")
        return 1

    if result.property_results:
        print("Properties:")
        for prop_result in result.property_results:
            status = "PASS" if prop_result.passed else "FAIL"
            print(
                f"  [{status}] {prop_result.property_id} "
                f"anchors={prop_result.selected_anchor_ids} "
                f"nodes={prop_result.selected_node_ids} "
                f"observed={prop_result.observed!r}"
            )
    else:
        print("No properties declared.")

    if result.temporal_results:
        print("Temporal properties:")
        for temporal_result in result.temporal_results:
            status = "PASS" if temporal_result.passed else "FAIL"
            print(
                f"  [{status}] {temporal_result.property_id} "
                f"operator={temporal_result.operator} "
                f"anchors={temporal_result.selected_anchor_ids} "
                f"nodes={temporal_result.selected_node_ids} "
                f"observed={temporal_result.observed!r}"
            )

    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
