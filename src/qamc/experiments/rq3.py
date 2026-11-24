"""RQ3 scalability experiments for full vs. abstraction-oriented models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import csv
from dataclasses import dataclass
import gc
from io import StringIO
import os
from pathlib import Path
from statistics import median, pstdev
import time
import tracemalloc

from qamc.backend import StateVectorBackend
from qamc.checker import run_model
from qamc.experiments.runner import _execution_quantum_gate_count
from qamc.loader import load_qam_string


RQ3_DEFAULT_SIZES = tuple(range(5, 21))
RQ3_DEFAULT_FULL_QUBIT_LIMIT = 20
RQ3_DEFAULT_REPEATS = 5
RQ3_DEFAULT_FULL_PAYLOAD_CAP_MIB = 1000.0
_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class RQ3BenchmarkSpec:
    """One scalable benchmark family used by the RQ3 comparison."""

    id: str
    label: str
    model_source: Callable[[int, str], str]


@dataclass(frozen=True, slots=True)
class RQ3ScalabilityRow:
    """One measured RQ3 full/abstract run."""

    benchmark_id: str
    benchmark_label: str
    style: str
    qubit_count: int
    gate_count: int
    status: str
    ok: bool
    wall_time_ms: float
    payload_mib: float
    py_peak_mib: float
    runtime_node_count: int
    runtime_edge_count: int
    property_count: int
    repeat_count: int = 1
    wall_time_min_ms: float = 0.0
    wall_time_max_ms: float = 0.0
    wall_time_stdev_ms: float = 0.0
    note: str = ""


@dataclass(frozen=True, slots=True)
class RQ3ScalabilityReport:
    """Measured rows for the RQ3 scalability comparison."""

    rows: tuple[RQ3ScalabilityRow, ...]
    sizes: tuple[int, ...]
    full_qubit_limit: int
    repeat_count: int
    full_payload_cap_mib: float | None
    trials: tuple[tuple[int, RQ3ScalabilityRow], ...] = ()

    @property
    def ok_rows(self) -> tuple[RQ3ScalabilityRow, ...]:
        return tuple(row for row in self.rows if row.status == "ok")


def run_rq3_scalability(
    *,
    benchmark_names: Sequence[str] | None = None,
    sizes: Sequence[int] = RQ3_DEFAULT_SIZES,
    full_qubit_limit: int = RQ3_DEFAULT_FULL_QUBIT_LIMIT,
    repeats: int = RQ3_DEFAULT_REPEATS,
    full_payload_cap_mib: float | None = RQ3_DEFAULT_FULL_PAYLOAD_CAP_MIB,
) -> RQ3ScalabilityReport:
    """Run paired abstraction/full scalability models and collect costs."""

    selected = _selected_benchmarks(benchmark_names)
    rows: list[RQ3ScalabilityRow] = []
    trials: list[tuple[int, RQ3ScalabilityRow]] = []
    for benchmark in selected:
        for qubit_count in sizes:
            for style in ("abstract", "full"):
                if style == "full" and qubit_count > full_qubit_limit:
                    rows.append(
                        _skipped_row(
                            benchmark=benchmark,
                            style=style,
                            qubit_count=qubit_count,
                            note=(
                                "full-state vector skipped; "
                                f"estimated base payload {_full_state_vector_payload_mib(qubit_count):.1f} MiB"
                            ),
                        )
                    )
                    continue
                if style == "full" and full_payload_cap_mib is not None:
                    estimated_payload = _estimated_full_context_payload_mib(benchmark, qubit_count)
                    if estimated_payload > full_payload_cap_mib:
                        rows.append(
                            _skipped_row(
                                benchmark=benchmark,
                                style=style,
                                qubit_count=qubit_count,
                                note=(
                                    "full-state vector skipped; "
                                    f"estimated runtime memory {estimated_payload:.1f} MiB "
                                    f"exceeds cap {full_payload_cap_mib:.1f} MiB"
                                ),
                            )
                        )
                        continue
                measurements = []
                for trial in range(1, max(1, repeats) + 1):
                    measurement = _measure_model_once(benchmark, qubit_count, style)
                    measurements.append(measurement)
                    trials.append((trial, measurement))
                    gc.collect()
                rows.append(_summarize_trials(measurements))
    return RQ3ScalabilityReport(
        rows=tuple(rows),
        sizes=tuple(int(size) for size in sizes),
        full_qubit_limit=full_qubit_limit,
        repeat_count=max(1, repeats),
        full_payload_cap_mib=full_payload_cap_mib,
        trials=tuple(trials),
    )


def render_rq3_csv(report: RQ3ScalabilityReport) -> str:
    """Render RQ3 rows as CSV for archival experiment data."""

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        (
            "benchmark_id",
            "benchmark",
            "style",
            "qubits",
            "gates",
            "status",
            "ok",
            "wall_time_ms",
            "payload_mib",
            "py_peak_mib",
            "runtime_nodes",
            "runtime_edges",
            "properties",
            "repeats",
            "wall_time_min_ms",
            "wall_time_max_ms",
            "wall_time_stdev_ms",
            "note",
        )
    )
    for row in report.rows:
        writer.writerow(
            (
                row.benchmark_id,
                row.benchmark_label,
                row.style,
                row.qubit_count,
                row.gate_count,
                row.status,
                int(row.ok),
                f"{row.wall_time_ms:.6f}",
                f"{row.payload_mib:.6f}",
                f"{row.py_peak_mib:.6f}",
                row.runtime_node_count,
                row.runtime_edge_count,
                row.property_count,
                row.repeat_count,
                f"{row.wall_time_min_ms:.6f}",
                f"{row.wall_time_max_ms:.6f}",
                f"{row.wall_time_stdev_ms:.6f}",
                row.note,
            )
        )
    return output.getvalue().rstrip()


def render_rq3_markdown_summary(report: RQ3ScalabilityReport) -> str:
    """Render a compact text summary for command-line runs."""

    lines = [
        "# QAMC RQ3 Scalability",
        "",
        f"sizes: {', '.join(str(size) for size in report.sizes)}",
        f"full qubit limit: {report.full_qubit_limit}",
        f"repeats: {report.repeat_count}",
        (
            "full memory cap: disabled"
            if report.full_payload_cap_mib is None
            else f"full memory cap: {report.full_payload_cap_mib:.1f} MiB"
        ),
        "",
        "| benchmark | style | n | gates | status | median ms | min--max ms | memory MiB | nodes | edges |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.benchmark_label} | {row.style} | {row.qubit_count} | {row.gate_count} | "
            f"{row.status} | {row.wall_time_ms:.3f} | "
            f"{row.wall_time_min_ms:.3f}--{row.wall_time_max_ms:.3f} | {row.payload_mib:.3f} | "
            f"{row.runtime_node_count} | {row.runtime_edge_count} |"
        )
    return "\n".join(lines)


def render_rq3_pgfplots(report: RQ3ScalabilityReport) -> str:
    """Render a LaTeX/pgfplots two-panel figure for the RQ3 trends."""

    return "\n".join(
        [
            "% QAMC RQ3 scalability figure. Generated from the RQ3 measurements.",
            "\\begin{figure*}[t]",
            "\\centering",
            _render_axis(
                report,
                metric="wall_time_ms",
                title="Wall-clock time",
                ylabel="Time (ms)",
                include_legend=True,
            ),
            "\\hfill",
            _render_axis(
                report,
                metric="payload_mib",
                title="Runtime memory",
                ylabel="Memory (MiB)",
                include_legend=False,
            ),
            "\\par\\vspace{0.4ex}",
            "\\ref{rq3-scalability-legend}",
            "\\caption{RQ3 full-state versus abstraction-oriented checking cost. Both y axes use logarithmic scales. Missing full-state points are skipped when the configured payload safety limit is exceeded.}",
            "\\Description{Two line charts comparing QAMC checking time and runtime memory for abstract and full-state models across benchmark sizes.}",
            "\\label{fig:rq3-scalability}",
            "\\end{figure*}",
        ]
    )


def render_rq3_matplotlib_figure(report: RQ3ScalabilityReport, output_path: str | Path) -> None:
    """Render a publication-style RQ3 figure with matplotlib."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qamc-matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "ghz": "#0072B2",
        "bv": "#D55E00",
        "quantum_walk": "#009E73",
        "iqft_phase": "#CC79A7",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.patch.set_facecolor("white")
    _plot_rq3_matplotlib_panel(
        axes[0],
        report,
        metric="wall_time_ms",
        ylabel="Time (ms)",
        title="Wall-clock time",
        ylim=(2.0, 10000.0),
        yticks=(10.0, 100.0, 1000.0, 10000.0),
        colors=colors,
        formatter=FuncFormatter(_format_log_tick),
    )
    _plot_rq3_matplotlib_panel(
        axes[1],
        report,
        metric="payload_mib",
        ylabel="Memory (MiB)",
        title="Runtime memory",
        ylim=(0.001, 1000.0),
        yticks=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        colors=colors,
        formatter=FuncFormatter(_format_log_tick),
    )

    benchmark_handles = [
        Line2D([0], [0], color=colors[benchmark.id], linewidth=2.0, label=benchmark.label)
        for benchmark in RQ3_BENCHMARKS.values()
    ]
    style_handles = [
        Line2D([0], [0], color="#2F3440", marker="o", linewidth=1.7, label="abstract"),
        Line2D(
            [0],
            [0],
            color="#2F3440",
            marker="s",
            markerfacecolor="white",
            linestyle=(0, (4, 2)),
            linewidth=1.7,
            label="full",
        ),
    ]
    fig.legend(
        handles=benchmark_handles + style_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.8,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.23, wspace=0.28)
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_rq3_outputs(
    report: RQ3ScalabilityReport,
    *,
    results_dir: str | Path,
    figure_path: str | Path | None = None,
) -> None:
    """Write CSV, Markdown, and optional figure outputs."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rq3_scalability.csv").write_text(render_rq3_csv(report) + "\n", encoding="utf-8")
    with (output_dir / "rq3_trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("benchmark_id", "style", "qubits", "trial", "gates", "status", "ok",
                         "wall_time_ms", "payload_mib", "py_peak_mib", "runtime_nodes",
                         "runtime_edges", "properties", "note"))
        for trial, row in report.trials:
            writer.writerow((row.benchmark_id, row.style, row.qubit_count, trial, row.gate_count,
                             row.status, int(row.ok), row.wall_time_ms, row.payload_mib,
                             row.py_peak_mib, row.runtime_node_count, row.runtime_edge_count,
                             row.property_count, row.note))
    (output_dir / "rq3_scalability.md").write_text(
        render_rq3_markdown_summary(report) + "\n",
        encoding="utf-8",
    )
    if figure_path is not None:
        path = Path(figure_path)
        if path.suffix == ".tex":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_rq3_pgfplots(report) + "\n", encoding="utf-8")
        else:
            render_rq3_matplotlib_figure(report, path)


def _plot_rq3_matplotlib_panel(
    ax,
    report: RQ3ScalabilityReport,
    *,
    metric: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    yticks: tuple[float, ...],
    colors: dict[str, str],
    formatter,
) -> None:
    ax.set_facecolor("#FAFAF7")
    ax.set_title(title, pad=7)
    ax.set_xlabel("Qubits (n)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlim(min(report.sizes) - 0.45, max(report.sizes) + 0.45)
    ax.set_xticks(list(report.sizes))
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#D7D9DD", linewidth=0.55, alpha=0.7)
    ax.grid(True, which="minor", axis="y", color="#ECEDEF", linewidth=0.35, alpha=0.8)
    first_skipped_full_size = _first_skipped_full_size(report)
    if first_skipped_full_size is not None:
        ax.axvspan(first_skipped_full_size - 0.45, max(report.sizes) + 0.45, color="#EEF0F3", zorder=0)
        ax.text(
            0.98,
            0.055,
            "full rows capped",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#69707D",
            fontsize=6.3,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#6F7580")

    for benchmark in RQ3_BENCHMARKS.values():
        for style in ("abstract", "full"):
            points = sorted(
                (
                    row
                    for row in report.rows
                    if row.benchmark_id == benchmark.id and row.style == style and row.status == "ok"
                ),
                key=lambda item: item.qubit_count,
            )
            if not points:
                continue
            x_values = [row.qubit_count for row in points]
            y_values = [max(float(getattr(row, metric)), 1e-9) for row in points]
            if style == "abstract":
                linestyle = "-"
                marker = "o"
                marker_face = colors[benchmark.id]
                alpha = 0.96
            else:
                linestyle = (0, (4, 2))
                marker = "s"
                marker_face = "white"
                alpha = 0.88
            ax.plot(
                x_values,
                y_values,
                color=colors[benchmark.id],
                linestyle=linestyle,
                linewidth=1.55,
                marker=marker,
                markersize=3.9,
                markerfacecolor=marker_face,
                markeredgecolor=colors[benchmark.id],
                markeredgewidth=0.85,
                alpha=alpha,
            )


def _format_log_tick(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 1:
        return f"{value:g}"
    if value >= 0.01:
        return f"{value:g}"
    return f"{value:.3f}"


def _first_skipped_full_size(report: RQ3ScalabilityReport) -> int | None:
    skipped_sizes = [row.qubit_count for row in report.rows if row.style == "full" and row.status == "skipped"]
    return min(skipped_sizes) if skipped_sizes else None


def _selected_benchmarks(benchmark_names: Sequence[str] | None) -> tuple[RQ3BenchmarkSpec, ...]:
    if benchmark_names is None:
        return tuple(RQ3_BENCHMARKS.values())
    return tuple(RQ3_BENCHMARKS[name] for name in benchmark_names)


def _summarize_trials(measurements: Sequence[RQ3ScalabilityRow]) -> RQ3ScalabilityRow:
    first_error = next((row for row in measurements if row.status != "ok"), None)
    if first_error is not None:
        return first_error

    times = [row.wall_time_ms for row in measurements]
    peaks = [row.py_peak_mib for row in measurements]
    representative = measurements[-1]
    return RQ3ScalabilityRow(
        benchmark_id=representative.benchmark_id,
        benchmark_label=representative.benchmark_label,
        style=representative.style,
        qubit_count=representative.qubit_count,
        gate_count=representative.gate_count,
        status=representative.status,
        ok=representative.ok,
        wall_time_ms=median(times),
        payload_mib=representative.payload_mib,
        py_peak_mib=median(peaks),
        runtime_node_count=representative.runtime_node_count,
        runtime_edge_count=representative.runtime_edge_count,
        property_count=representative.property_count,
        repeat_count=len(measurements),
        wall_time_min_ms=min(times),
        wall_time_max_ms=max(times),
        wall_time_stdev_ms=pstdev(times) if len(times) > 1 else 0.0,
        note=representative.note,
    )


def _measure_model_once(
    benchmark: RQ3BenchmarkSpec,
    qubit_count: int,
    style: str,
) -> RQ3ScalabilityRow:
    source = benchmark.model_source(qubit_count, style)
    start = time.perf_counter()
    tracemalloc.start()
    try:
        model = load_qam_string(source)
        result = run_model(model, backend=_backend_for_style(style))
        current, peak = tracemalloc.get_traced_memory()
        del current
    except Exception as exc:  # pragma: no cover - defensive path for batch experiments.
        current, peak = tracemalloc.get_traced_memory()
        del current
        tracemalloc.stop()
        return RQ3ScalabilityRow(
            benchmark_id=benchmark.id,
            benchmark_label=benchmark.label,
            style=style,
            qubit_count=qubit_count,
            gate_count=0,
            status="error",
            ok=False,
            wall_time_ms=(time.perf_counter() - start) * 1000.0,
            payload_mib=0.0,
            py_peak_mib=peak / _MIB,
            runtime_node_count=0,
            runtime_edge_count=0,
            property_count=0,
            note=str(exc),
        )
    tracemalloc.stop()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    context = result.context
    status = "ok" if result.ok else "failed"
    note = "" if result.ok else "; ".join(diag.message for diag in result.validation.diagnostics)
    if result.has_undefined:
        status = "undefined"
        note = "; ".join(r.diagnostic for r in (*result.property_results, *result.temporal_results) if r.diagnostic)
    return RQ3ScalabilityRow(
        benchmark_id=benchmark.id,
        benchmark_label=benchmark.label,
        style=style,
        qubit_count=qubit_count,
        gate_count=_execution_quantum_gate_count(model),
        status=status,
        ok=result.ok,
        wall_time_ms=elapsed_ms,
        payload_mib=_context_payload_mib(context),
        py_peak_mib=peak / _MIB,
        runtime_node_count=len(context.nodes) if context is not None else 0,
        runtime_edge_count=len(context.edges) if context is not None else 0,
        property_count=len(result.property_results) + len(result.temporal_results),
        repeat_count=1,
        wall_time_min_ms=elapsed_ms,
        wall_time_max_ms=elapsed_ms,
        note=note,
    )


def _skipped_row(
    *,
    benchmark: RQ3BenchmarkSpec,
    style: str,
    qubit_count: int,
    note: str,
) -> RQ3ScalabilityRow:
    return RQ3ScalabilityRow(
        benchmark_id=benchmark.id,
        benchmark_label=benchmark.label,
        style=style,
        qubit_count=qubit_count,
        gate_count=0,
        status="skipped",
        ok=False,
        wall_time_ms=0.0,
        payload_mib=0.0,
        py_peak_mib=0.0,
        runtime_node_count=0,
        runtime_edge_count=0,
        property_count=0,
        note=note,
    )


def _context_payload_mib(context) -> float:
    if context is None:
        return 0.0
    total = 0
    for node in context.nodes.values():
        for payload in node.unit_payloads.values():
            total += int(getattr(payload.data, "nbytes", 0))
    return total / _MIB


def _full_state_vector_payload_mib(qubit_count: int) -> float:
    return (16 * (2**qubit_count)) / _MIB


def _estimated_full_context_payload_mib(benchmark: RQ3BenchmarkSpec, qubit_count: int) -> float:
    try:
        model = load_qam_string(benchmark.model_source(qubit_count, "full"))
    except Exception:
        return float("inf")
    runtime_nodes = _execution_quantum_gate_count(model) + 1
    return runtime_nodes * _full_state_vector_payload_mib(qubit_count)


def _backend_for_style(style: str):
    if style == "full":
        return StateVectorBackend()
    return None


def _render_axis(
    report: RQ3ScalabilityReport,
    *,
    metric: str,
    title: str,
    ylabel: str,
    include_legend: bool,
) -> str:
    colors = {
        "ghz": "blue",
        "bv": "red",
        "quantum_walk": "teal",
        "iqft_phase": "orange",
    }
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "width=0.48\\textwidth,",
        "height=0.34\\textwidth,",
        "ymode=log,",
        "grid=both,",
        "xlabel={Qubits},",
        f"ylabel={{{ylabel}}},",
        f"title={{{title}}},",
        "tick label style={font=\\scriptsize},",
        "label style={font=\\scriptsize},",
        "title style={font=\\small},",
    ]
    if include_legend:
        lines.extend(
            [
                "legend to name=rq3-scalability-legend,",
                "legend style={font=\\scriptsize, legend columns=4},",
            ]
        )
    lines.append("]")
    for benchmark in RQ3_BENCHMARKS.values():
        for style in ("abstract", "full"):
            points = [
                row
                for row in report.rows
                if row.benchmark_id == benchmark.id and row.style == style and row.status == "ok"
            ]
            if not points:
                continue
            coordinates = " ".join(
                f"({row.qubit_count},{max(getattr(row, metric), 1e-6):.6f})"
                for row in sorted(points, key=lambda item: item.qubit_count)
            )
            dash = "solid" if style == "abstract" else "dashed"
            mark = "*" if style == "abstract" else "square*"
            options = f"color={colors[benchmark.id]}, {dash}, mark={mark}"
            if not include_legend:
                options = f"{options}, forget plot"
            lines.append(f"\\addplot+[{options}] coordinates {{{coordinates}}};")
            if include_legend:
                lines.append(f"\\addlegendentry{{{benchmark.label} {style}}}")
    lines.extend(["\\end{axis}", "\\end{tikzpicture}"])
    return "\n".join(lines)


def _unit_block(style: str, extra_units: Iterable[tuple[str, Sequence[str]]] = ()) -> str:
    completion = "remainder" if style == "full" else "singletons"
    lines = ["  units {", f"    unit_completion {completion}"]
    if style != "full":
        for unit_id, qubits in extra_units:
            lines.append(f"    {unit_id}: {', '.join(qubits)}")
    lines.append("  }")
    return "\n".join(lines)


def _state_block(
    state_id: str,
    style: str,
    *,
    initial: bool = False,
    extra_units: Iterable[tuple[str, Sequence[str]]] = (),
) -> str:
    tag = " initial" if initial else ""
    return "\n".join(
        [
            f"state {state_id}{tag} {{",
            "  active all",
            _unit_block(style, extra_units),
            "}",
        ]
    )


def _header(model_id: str, qubit_count: int, gates: str, style: str) -> str:
    lines = [f"model {model_id}", "", f"qubits q0..q{qubit_count - 1}", f"gates {gates}"]
    if style == "full":
        lines.append("default_summary_kind state_vector")
    lines.append("")
    return "\n".join(lines)


def _trace_lines(steps: Sequence[tuple[str, str, str]]) -> str:
    lines = ["trace main {", "  a0: S0 initial"]
    source_anchor = "a0"
    for index, (transition_id, target_anchor, target_state) in enumerate(steps, start=1):
        kind = "terminal " if index == len(steps) else ""
        lines.append(f"  {source_anchor} --{transition_id}--> {target_anchor}: {kind}{target_state}")
        source_anchor = target_anchor
    lines.append("}")
    return "\n".join(lines)


def _ghz_source(qubit_count: int, style: str) -> str:
    states = [_state_block("S0", style, initial=True), _state_block("S1", style)]
    for state_index in range(2, qubit_count + 1):
        pairs = (
            (f"u{pair}_{pair + 1}", (f"q{pair}", f"q{pair + 1}"))
            for pair in range(state_index - 1)
        )
        states.append(_state_block(f"S{state_index}", style, extra_units=pairs))
    transitions = [
        "transition t_h0: S0 -> S1 {\n  H(q0)\n}",
    ]
    steps = [("t_h0", "a1", "S1")]
    for target in range(1, qubit_count):
        transition_id = f"t_cx_{target - 1}_{target}"
        transitions.append(
            f"transition {transition_id}: S{target} -> S{target + 1} {{\n  CX(q{target - 1}, q{target})\n}}"
        )
        anchor = f"f{target + 1}" if target == qubit_count - 1 else f"a{target + 1}"
        steps.append((transition_id, anchor, f"S{target + 1}"))
    return "\n\n".join(
        [
            _header(f"rq3_ghz_{style}_{qubit_count}", qubit_count, "H, CX", style),
            *states,
            *transitions,
            _trace_lines(steps),
            "execution run_main {\n  trace main\n}",
            (
                "property p_local_ghz_pair distribution {\n"
                f"  anchors f{qubit_count}\n"
                "  observe dist(q0, q1, Z)\n"
                "  expect {(0,0): 0.5, (1,1): 0.5}\n"
                "}"
            ),
        ]
    )


def _bv_source(qubit_count: int, style: str) -> str:
    hidden = tuple(index for index in range(qubit_count - 1) if index % 3 == 0)
    state_count = 2 * (qubit_count - 1) + 2 + len(hidden)
    states = [_state_block(f"S{index}", style, initial=(index == 0)) for index in range(state_count + 1)]
    transitions: list[str] = []
    steps: list[tuple[str, str, str]] = []
    current = 0

    def add_transition(gate: str) -> None:
        nonlocal current
        transition_id = f"t{current}_{gate.split('(', 1)[0].lower()}"
        transitions.append(f"transition {transition_id}: S{current} -> S{current + 1} {{\n  {gate}\n}}")
        anchor = f"f{current + 1}" if current + 1 == state_count else f"a{current + 1}"
        steps.append((transition_id, anchor, f"S{current + 1}"))
        current += 1

    ancilla = qubit_count - 1
    add_transition(f"X(q{ancilla})")
    add_transition(f"H(q{ancilla})")
    for index in range(qubit_count - 1):
        add_transition(f"H(q{index})")
    for index in hidden:
        add_transition(f"CX(q{index}, q{ancilla})")
    for index in range(qubit_count - 1):
        add_transition(f"H(q{index})")

    return "\n\n".join(
        [
            _header(f"rq3_bv_{style}_{qubit_count}", qubit_count, "H, X, CX", style),
            *states,
            *transitions,
            _trace_lines(steps),
            "execution run_main {\n  trace main\n}",
            (
                "property p_hidden_first_bit distribution {\n"
                f"  anchors f{state_count}\n"
                "  observe dist(q0, Z)\n"
                "  expect {1: 1}\n"
                "}"
            ),
        ]
    )


def _quantum_walk_source(qubit_count: int, style: str) -> str:
    walk_unit = (("walk", ("q0", "q1", "q2")),)
    states = [
        _state_block("S0", style, initial=True),
        _state_block("S1", style),
        *(_state_block(f"S{index}", style, extra_units=walk_unit) for index in range(2, 8)),
    ]
    transitions = [
        "transition t_coin_0: S0 -> S1 {\n  H(q0)\n}",
        "transition t_shift_0_carry: S1 -> S2 {\n  CCX(q0, q1, q2)\n}",
        "transition t_shift_0_low: S2 -> S3 {\n  CX(q0, q1)\n}",
        "transition t_coin_1: S3 -> S4 {\n  H(q0)\n}",
        "transition t_shift_1_carry: S4 -> S5 {\n  CCX(q0, q1, q2)\n}",
        "transition t_shift_1_low: S5 -> S6 {\n  CX(q0, q1)\n}",
        "transition t_release_walk: S6 -> S7 {\n  ID(q0)\n}",
    ]
    steps = [
        ("t_coin_0", "a1", "S1"),
        ("t_shift_0_carry", "a2", "S2"),
        ("t_shift_0_low", "a3", "S3"),
        ("t_coin_1", "a4", "S4"),
        ("t_shift_1_carry", "a5", "S5"),
        ("t_shift_1_low", "a6", "S6"),
        ("t_release_walk", "f7", "S7"),
    ]
    return "\n\n".join(
        [
            _header(f"rq3_quantum_walk_{style}_{qubit_count}", qubit_count, "H, CX, CCX, ID", style),
            *states,
            *transitions,
            _trace_lines(steps),
            "execution run_main {\n  trace main\n}",
            (
                "property p_position_after_two_steps distribution {\n"
                "  anchors f7\n"
                "  observe dist(q1, q2, Z)\n"
                "  expect {(0,0): 0.25, (1,0): 0.5, (0,1): 0.25}\n"
                "}"
            ),
        ]
    )


def _iqft_phase_source(qubit_count: int, style: str) -> str:
    states = [_state_block("S0", style, initial=True), _state_block("S1", style)]
    for state_index in range(2, qubit_count + 1):
        pair = (("phase", ("q0", f"q{state_index - 1}")),)
        states.append(_state_block(f"S{state_index}", style, extra_units=pair))
    transitions = ["transition t_h0: S0 -> S1 {\n  H(q0)\n}"]
    steps = [("t_h0", "a1", "S1")]
    for target in range(1, qubit_count):
        transition_id = f"t_cp_0_{target}"
        angle = -3.141592653589793 / (2**target)
        transitions.append(
            f"transition {transition_id}: S{target} -> S{target + 1} {{\n  CP(q0, q{target}, {angle})\n}}"
        )
        anchor = f"f{target + 1}" if target == qubit_count - 1 else f"a{target + 1}"
        steps.append((transition_id, anchor, f"S{target + 1}"))
    return "\n\n".join(
        [
            _header(f"rq3_iqft_phase_{style}_{qubit_count}", qubit_count, "H, CP", style),
            *states,
            *transitions,
            _trace_lines(steps),
            "execution run_main {\n  trace main\n}",
            (
                "property p_phase_reference distribution {\n"
                f"  anchors f{qubit_count}\n"
                "  observe dist(q0, X)\n"
                "  expect {0: 1}\n"
                "}"
            ),
        ]
    )


RQ3_BENCHMARKS: dict[str, RQ3BenchmarkSpec] = {
    "ghz": RQ3BenchmarkSpec("ghz", "GHZ", _ghz_source),
    "bv": RQ3BenchmarkSpec("bv", "BV", _bv_source),
    "quantum_walk": RQ3BenchmarkSpec("quantum_walk", "Quantum walk", _quantum_walk_source),
    "iqft_phase": RQ3BenchmarkSpec("iqft_phase", "IQFT phase", _iqft_phase_source),
}
