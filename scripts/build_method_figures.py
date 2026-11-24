#!/usr/bin/env python3
"""Export Figures 1–4 from their editable draw.io and LaTeX sources."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "figures"
DIAGRAMS = {
    1: "qam_model_structure.drawio",
    3: "qam_modeling_example_models.drawio",
    4: "qam_framework.drawio",
}


def build_circuit(target: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for source in ("qam_modeling_example_circuit.tex", "figure2.tex"):
            shutil.copyfile(SOURCES / source, work / source)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source],
                cwd=work, check=True,
            )
        shutil.copyfile(work / "figure2.pdf", target)


def export_diagram(drawio: str, source: Path, target: Path) -> None:
    # WSL can run a Windows draw.io installation. Its CLI takes Windows paths.
    paths = [str(target), str(source)]
    if os.name != "nt" and drawio.lower().endswith(".exe") and shutil.which("wslpath"):
        paths = [subprocess.check_output(["wslpath", "-w", path], text=True).strip()
                 for path in paths]
    subprocess.run(
        [drawio, "--export", "--format", "pdf", "--crop", "--output", *paths],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=range(1, 5),
                        default=range(1, 5), help="Figure numbers to build (default: 1–4).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--drawio", help="Path to the draw.io Desktop executable.")
    args = parser.parse_args()
    drawio = args.drawio or shutil.which("drawio") or shutil.which("draw.io")
    if any(i in DIAGRAMS for i in args.figures) and not drawio:
        parser.error("Specify --drawio or put the draw.io Desktop executable on PATH.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for number in args.figures:
        output = output_dir / f"figure{number}.pdf"
        # Keep the previous output intact if a tool exits before writing a PDF.
        with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
            target = Path(temporary) / output.name
            if number == 2:
                build_circuit(target)
            else:
                export_diagram(drawio, SOURCES / DIAGRAMS[number], target)
            if not target.is_file() or target.stat().st_size == 0:
                parser.exit(1, f"Figure {number} export did not produce a PDF.\n")
            target.replace(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
