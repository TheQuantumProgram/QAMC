#!/usr/bin/env python3
"""Render Figure 5 from the 128 full and abstract model measurements for RQ3."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiment_data/paper_results/rq3_scalability.csv"
DEFAULT_OUTPUT = ROOT / "runs/figure5.pdf"
SIZES = tuple(range(5, 21))
FAMILIES = (
    ("ghz", "GHZ", "#0072B2"),
    ("bv", "BV", "#D55E00"),
    ("quantum_walk", "Quantum walk", "#009E73"),
    ("iqft_phase", "IQFT phase", "#CC79A7"),
)


def read_measurements(path: Path) -> list[dict[str, str]]:
    """Validate the complete archived series without modifying its values."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(family, style, n) for family, _, _ in FAMILIES
                for style in ("abstract", "full") for n in SIZES}
    seen = set()
    for row in rows:
        key = (row["benchmark_id"], row["style"], int(row["qubits"]))
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate measurement: {key}")
        seen.add(key)
        if row["status"] != "ok" or row["ok"] != "1" or row["repeats"] != "5":
            raise ValueError(f"Expected a successful five-repeat measurement: {key}")
        for metric in ("wall_time_ms", "payload_mib"):
            value = float(row[metric])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {metric} for {key}: {row[metric]}")
    if seen != expected:
        raise ValueError(f"Missing measurements: {sorted(expected - seen)}")
    return rows


def format_log_tick(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def render(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    figure.patch.set_facecolor("white")
    panels = (
        ("wall_time_ms", "Time (ms)", "Wall-clock time", (1.0, 10000.0),
         (1.0, 10.0, 100.0, 1000.0, 10000.0)),
        ("payload_mib", "Memory (MiB)", "Retained quantum-array memory", (0.001, 1000.0),
         (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)),
    )
    for axis, (metric, ylabel, title, limits, ticks) in zip(axes, panels):
        axis.set_facecolor("#FAFAF7")
        axis.set_title(title, pad=7)
        axis.set_xlabel("Qubits (n)")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.set_ylim(*limits)
        axis.set_yticks(ticks)
        axis.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
        axis.set_xlim(min(SIZES) - 0.45, max(SIZES) + 0.45)
        axis.set_xticks(SIZES)
        axis.set_axisbelow(True)
        axis.grid(True, which="major", color="#D7D9DD", linewidth=0.55, alpha=0.7)
        axis.grid(True, which="minor", axis="y", color="#ECEDEF", linewidth=0.35, alpha=0.8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#6F7580")
        for family, _, color in FAMILIES:
            for style in ("abstract", "full"):
                points = sorted((row for row in rows
                                 if row["benchmark_id"] == family and row["style"] == style),
                                key=lambda row: int(row["qubits"]))
                abstract = style == "abstract"
                axis.plot(
                    [int(row["qubits"]) for row in points],
                    [float(row[metric]) for row in points],
                    color=color, linestyle="-" if abstract else (0, (4, 2)),
                    linewidth=1.55, marker="o" if abstract else "s", markersize=3.9,
                    markerfacecolor=color if abstract else "white",
                    markeredgecolor=color, markeredgewidth=0.85,
                    alpha=0.96 if abstract else 0.88,
                )
    family_handles = [Line2D([0], [0], color=color, linewidth=2.0, label=label)
                      for _, label, color in FAMILIES]
    style_handles = [
        Line2D([0], [0], color="#2F3440", marker="o", linewidth=1.7, label="abstract"),
        Line2D([0], [0], color="#2F3440", marker="s", markerfacecolor="white",
               linestyle=(0, (4, 2)), linewidth=1.7, label="full"),
    ]
    figure.legend(handles=family_handles + style_handles, loc="lower center", ncol=6,
                  frameon=False, bbox_to_anchor=(0.5, 0.01), handlelength=1.8,
                  columnspacing=1.25)
    figure.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.23, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Figure filename (default: runs/figure5.pdf)")
    args = parser.parse_args(argv)
    try:
        rows = read_measurements(args.input_csv)
        render(rows, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output} using {len(rows)} measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
