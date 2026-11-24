"""RQ4 large-scale abstraction-oriented experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import time

from qamc.engine import TraceExecutor
from qamc.experiments.runner import _execution_quantum_gate_count
from qamc.loader import load_qam_string
from qamc.property import PropertyEvaluator
from qamc.validate import validate_qam_model


RQ4_DEFAULT_SIZES = (50, 100, 200, 300, 400, 500)
_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class RQ4BenchmarkSpec:
    """One large-scale abstraction-oriented benchmark family."""

    id: str
    label: str
    model_source: Callable[[int], str]


@dataclass(frozen=True, slots=True)
class RQ4LargeScaleRow:
    """One measured RQ4 large-scale run."""

    benchmark_id: str
    benchmark_label: str
    qubit_count: int
    gate_count: int
    line_count: int
    status: str
    ok: bool
    run_time_ms: float
    check_time_ms: float
    payload_mib: float
    note: str = ""

    @property
    def total_time_ms(self) -> float:
        return self.run_time_ms + self.check_time_ms


@dataclass(frozen=True, slots=True)
class RQ4LargeScaleReport:
    """Measured rows for the RQ4 large-scale abstraction experiment."""

    rows: tuple[RQ4LargeScaleRow, ...]
    sizes: tuple[int, ...]

    @property
    def all_ok(self) -> bool:
        return all(row.status == "ok" and row.ok for row in self.rows)


def run_rq4_large_scale(
    *,
    benchmark_names: Sequence[str] | None = None,
    sizes: Sequence[int] = RQ4_DEFAULT_SIZES,
) -> RQ4LargeScaleReport:
    """Run abstraction-oriented large-scale models and collect costs."""

    rows: list[RQ4LargeScaleRow] = []
    for benchmark in _selected_benchmarks(benchmark_names):
        for qubit_count in sizes:
            rows.append(_measure_model_once(benchmark, int(qubit_count)))
    return RQ4LargeScaleReport(rows=tuple(rows), sizes=tuple(int(size) for size in sizes))


def render_rq4_csv(report: RQ4LargeScaleReport) -> str:
    """Render RQ4 rows as CSV for archival experiment data."""

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        (
            "benchmark_id",
            "benchmark",
            "qubits",
            "gates",
            "lines",
            "status",
            "ok",
            "run_time_ms",
            "check_time_ms",
            "total_time_ms",
            "payload_mib",
            "note",
        )
    )
    for row in report.rows:
        writer.writerow(
            (
                row.benchmark_id,
                row.benchmark_label,
                row.qubit_count,
                row.gate_count,
                row.line_count,
                row.status,
                int(row.ok),
                f"{row.run_time_ms:.6f}",
                f"{row.check_time_ms:.6f}",
                f"{row.total_time_ms:.6f}",
                f"{row.payload_mib:.6f}",
                row.note,
            )
        )
    return output.getvalue().rstrip()


def render_rq4_markdown_summary(report: RQ4LargeScaleReport) -> str:
    """Render a compact RQ4 summary for command-line runs."""

    lines = [
        "# QAMC RQ4 Large-Scale Abstraction",
        "",
        f"sizes: {', '.join(str(size) for size in report.sizes)}",
        "",
        "| family | size | gates | lines | status | run ms | check ms | total ms | memory MiB | result |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.benchmark_label} | {row.qubit_count} | {row.gate_count} | "
            f"{row.line_count} | {row.status} | "
            f"{row.run_time_ms:.3f} | {row.check_time_ms:.3f} | "
            f"{row.total_time_ms:.3f} | "
            f"{row.payload_mib:.3f} | {'yes' if row.ok else 'no'} |"
        )
    return "\n".join(lines)


def render_rq4_latex_table(report: RQ4LargeScaleReport) -> str:
    """Render the RQ4 large-scale results as a paper table."""

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{RQ4 large-scale abstraction-oriented checking results.}",
        "\\label{tab:rq4-large-scale}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrrrrc}",
        "\\hline",
        "Family & Size & Gates & Lines & Run (ms) & Check (ms) & Total (ms) & Mem (MiB) & Result \\\\",
        "\\hline",
    ]
    previous_family = ""
    for row in report.rows:
        family = row.benchmark_label if row.benchmark_label != previous_family else ""
        previous_family = row.benchmark_label
        result = "\\ensuremath{\\surd}" if row.ok and row.status == "ok" else "--"
        lines.append(
            f"{family} & {row.qubit_count} & {row.gate_count} & {row.line_count} & "
            f"{row.run_time_ms:.1f} & {row.check_time_ms:.1f} & "
            f"{row.total_time_ms:.1f} & "
            f"{row.payload_mib:.3f} & {result} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def write_rq4_outputs(
    report: RQ4LargeScaleReport,
    *,
    results_dir: str | Path,
) -> None:
    """Write RQ4 CSV, Markdown, and LaTeX table outputs."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rq4_large_scale.csv").write_text(render_rq4_csv(report) + "\n", encoding="utf-8")
    (output_dir / "rq4_large_scale.md").write_text(
        render_rq4_markdown_summary(report) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rq4_large_scale.tex").write_text(
        render_rq4_latex_table(report) + "\n",
        encoding="utf-8",
    )


def _selected_benchmarks(benchmark_names: Sequence[str] | None) -> tuple[RQ4BenchmarkSpec, ...]:
    if benchmark_names is None:
        return tuple(RQ4_BENCHMARKS.values())
    return tuple(RQ4_BENCHMARKS[name] for name in benchmark_names)


def _measure_model_once(benchmark: RQ4BenchmarkSpec, qubit_count: int) -> RQ4LargeScaleRow:
    source = benchmark.model_source(qubit_count)
    line_count = _source_line_count(source)
    run_start = time.perf_counter()
    try:
        model = load_qam_string(source)
        gate_count = _execution_quantum_gate_count(model)
        validation = validate_qam_model(model)
        context = None
        if validation.ok:
            selected_execution_id = next(iter(model.executions), None)
            context = TraceExecutor().run(model, execution_id=selected_execution_id)
    except Exception as exc:  # pragma: no cover - defensive path for batch experiments.
        return RQ4LargeScaleRow(
            benchmark_id=benchmark.id,
            benchmark_label=benchmark.label,
            qubit_count=qubit_count,
            gate_count=0,
            line_count=line_count,
            status="error",
            ok=False,
            run_time_ms=(time.perf_counter() - run_start) * 1000.0,
            check_time_ms=0.0,
            payload_mib=0.0,
            note=str(exc),
        )
    run_time_ms = (time.perf_counter() - run_start) * 1000.0
    if not validation.ok or context is None:
        return RQ4LargeScaleRow(
            benchmark_id=benchmark.id,
            benchmark_label=benchmark.label,
            qubit_count=qubit_count,
            gate_count=gate_count,
            line_count=line_count,
            status="invalid",
            ok=False,
            run_time_ms=run_time_ms,
            check_time_ms=0.0,
            payload_mib=0.0,
            note="; ".join(diagnostic.message for diagnostic in validation.diagnostics),
        )

    check_start = time.perf_counter()
    try:
        property_results = PropertyEvaluator().evaluate_all(context, model.properties)
    except Exception as exc:  # pragma: no cover - defensive path for batch experiments.
        return RQ4LargeScaleRow(
            benchmark_id=benchmark.id,
            benchmark_label=benchmark.label,
            qubit_count=qubit_count,
            gate_count=gate_count,
            line_count=line_count,
            status="error",
            ok=False,
            run_time_ms=run_time_ms,
            check_time_ms=(time.perf_counter() - check_start) * 1000.0,
            payload_mib=_context_payload_mib(context),
            note=str(exc),
        )
    check_time_ms = (time.perf_counter() - check_start) * 1000.0
    ok = all(result.passed for result in property_results)
    return RQ4LargeScaleRow(
        benchmark_id=benchmark.id,
        benchmark_label=benchmark.label,
        qubit_count=qubit_count,
        gate_count=gate_count,
        line_count=line_count,
        status="ok" if ok else "failed",
        ok=ok,
        run_time_ms=run_time_ms,
        check_time_ms=check_time_ms,
        payload_mib=_context_payload_mib(context),
        note="" if ok else _failed_property_note(property_results),
    )


def _context_payload_mib(context) -> float:
    if context is None:
        return 0.0
    total = 0
    for node in context.nodes.values():
        for payload in node.unit_payloads.values():
            total += int(getattr(payload.data, "nbytes", 0))
    return total / _MIB


def _source_line_count(source: str) -> int:
    return sum(1 for line in source.splitlines() if line.strip())


def _failed_property_note(property_results) -> str:
    failed_ids = [result.property_id for result in property_results if not result.passed]
    if not failed_ids:
        return ""
    if len(failed_ids) <= 5:
        return f"failed properties: {', '.join(failed_ids)}"
    return f"failed properties: {', '.join(failed_ids[:5])}, ..."


def _header(model_id: str, qubit_count: int, gates: str) -> str:
    return "\n".join([f"model {model_id}", "", f"qubits q0..q{qubit_count - 1}", f"gates {gates}", ""])


def _single_state(state_id: str = "S", *, extra_units: Sequence[tuple[str, Sequence[str]]] = ()) -> str:
    lines = [
        f"state {state_id} initial {{",
        "  active all",
        "  units {",
        "    unit_completion singletons",
    ]
    for unit_id, qubits in extra_units:
        lines.append(f"    {unit_id}: {', '.join(qubits)}")
    lines.extend(["  }", "}"])
    return "\n".join(lines)


def _transition(transition_id: str, gate: str, state_id: str = "S") -> str:
    return f"transition {transition_id}: {state_id} -> {state_id} {{\n  {gate}\n}}"


def _trace(steps: Sequence[str], state_id: str = "S") -> str:
    lines = ["trace main {", f"  a0: {state_id} initial"]
    source_anchor = "a0"
    for index, transition_id in enumerate(steps, start=1):
        target_anchor = f"f{index}" if index == len(steps) else f"a{index}"
        kind = "terminal " if index == len(steps) else ""
        lines.append(f"  {source_anchor} --{transition_id}--> {target_anchor}: {kind}{state_id}")
        source_anchor = target_anchor
    lines.append("}")
    return "\n".join(lines)


def _execution() -> str:
    return "execution run_main {\n  trace main\n}"


def _distribution_property(
    property_id: str,
    anchor_id: str,
    qubit: str,
    expected: int,
) -> str:
    return "\n".join(
        [
            f"property {property_id} distribution {{",
            f"  anchors {anchor_id}",
            f"  observe dist({qubit}, Z)",
            f"  expect {{{expected}: 1}}",
            "}",
        ]
    )


def _ghz_source(qubit_count: int) -> str:
    units = [
        (f"adj_{index}_{index + 1}", (f"q{index}", f"q{index + 1}"))
        for index in range(qubit_count - 1)
    ]
    transitions = ["transition t_h0: S -> S {\n  H(q0)\n}"]
    steps = ["t_h0"]
    for target in range(1, qubit_count):
        transition_id = f"t_cx_{target - 1}_{target}"
        transitions.append(
            f"transition {transition_id}: S -> S {{\n  CX(q{target - 1}, q{target})\n}}"
        )
        steps.append(transition_id)

    properties = [
        "\n".join(
            [
                f"property p_adjacent_{index}_{index + 1} relation {{",
                f"  anchors f{qubit_count}",
                f"  observe dist(q{index}, q{index + 1}, Z)",
                "  expect equal_bits",
                "}",
            ]
        )
        for index in range(qubit_count - 1)
    ]

    return "\n\n".join(
        [
            _header(f"rq4_ghz_{qubit_count}", qubit_count, "H, CX"),
            _single_state(extra_units=units),
            *transitions,
            _trace(steps),
            _execution(),
            *properties,
        ]
    )


def _bv_source(qubit_count: int) -> str:
    ancilla = qubit_count - 1
    hidden = tuple(index for index in range(qubit_count - 1) if index % 7 == 0)
    transitions: list[str] = []
    steps: list[str] = []

    def add(gate: str) -> None:
        transition_id = f"t{len(steps):04d}_{gate.split('(', 1)[0].lower()}"
        transitions.append(_transition(transition_id, gate))
        steps.append(transition_id)

    add(f"X(q{ancilla})")
    add(f"H(q{ancilla})")
    for index in range(qubit_count - 1):
        add(f"H(q{index})")
    for index in hidden:
        add(f"CX(q{index}, q{ancilla})")
    for index in range(qubit_count - 1):
        add(f"H(q{index})")

    properties = [
        _distribution_property(
            f"p_hidden_{index}",
            f"f{len(steps)}",
            f"q{index}",
            1 if index in hidden else 0,
        )
        for index in range(qubit_count - 1)
    ]
    return "\n\n".join(
        [
            _header(f"rq4_bv_{qubit_count}", qubit_count, "H, X, CX"),
            _single_state(),
            *transitions,
            _trace(steps),
            _execution(),
            *properties,
        ]
    )


def _adder_source(qubit_count: int) -> str:
    if qubit_count % 2 != 0:
        raise ValueError("RQ4 adder expects an even qubit count.")
    half = qubit_count // 2
    transitions: list[str] = []
    steps: list[str] = []
    bits = [0] * qubit_count

    def add(gate: str) -> None:
        transition_id = f"t{len(steps):04d}_{gate.split('(', 1)[0].lower()}"
        transitions.append(_transition(transition_id, gate))
        steps.append(transition_id)
        _apply_classical_gate(bits, gate)

    for index in range(1, half, 2):
        add(f"X(q{index})")
    for index in range(half, min(qubit_count, half + 2)):
        add(f"X(q{index})")

    for index in range(half - 1):
        add(f"CX(q{index + 1}, q{half + index})")
        add(f"CX(q{index + 1}, q{index})")
        add(f"CCX(q{index}, q{half + index}, q{index + 1})")
    add(f"CX(q{half - 1}, q{qubit_count - 1})")
    for index in range(half - 2, -1, -1):
        add(f"CCX(q{index}, q{half + index}, q{index + 1})")
        add(f"CX(q{index + 1}, q{index})")
        add(f"CX(q{index}, q{half + index})")

    properties = [
        _distribution_property(f"p_output_q{index}", f"f{len(steps)}", f"q{index}", expected)
        for index, expected in enumerate(bits)
    ]
    return "\n\n".join(
        [
            _header(f"rq4_adder_{qubit_count}", qubit_count, "X, CX, CCX"),
            _single_state(),
            *transitions,
            _trace(steps),
            _execution(),
            *properties,
        ]
    )


def _apply_classical_gate(bits: list[int], gate: str) -> None:
    name, args = gate.split("(", 1)
    qubits = [int(item.strip().removeprefix("q")) for item in args.rstrip(")").split(",")]
    if name == "X":
        bits[qubits[0]] ^= 1
    elif name == "CX":
        if bits[qubits[0]]:
            bits[qubits[1]] ^= 1
    elif name == "CCX":
        if bits[qubits[0]] and bits[qubits[1]]:
            bits[qubits[2]] ^= 1
    else:  # pragma: no cover - generator-local defensive guard.
        raise ValueError(f"Unsupported classical gate simulation: {gate!r}")


def _grover_source(qubit_count: int) -> str:
    if qubit_count % 10 != 0:
        raise ValueError("RQ4 Grover expects a qubit count divisible by 10.")
    blocks = qubit_count // 10
    units: list[tuple[str, tuple[str, ...]]] = []
    transitions: list[str] = []
    steps: list[str] = []

    def add(gate: str) -> None:
        transition_id = f"t{len(steps):04d}_{gate.split('(', 1)[0].lower()}"
        transitions.append(_transition(transition_id, gate, state_id="S_cover"))
        steps.append(transition_id)

    for block in range(blocks):
        base = block * 10
        units.extend(_grover_units(block, base))
        for gate in _grover_block_gates(base):
            add(gate)

    final_anchor = f"f{len(steps)}"
    properties = [
        prop
        for block in range(blocks)
        for prop in _grover_properties(block, block * 10, final_anchor)
    ]

    return "\n\n".join(
        [
            _header(f"rq4_grover_{qubit_count}", qubit_count, "H, X, CCX"),
            "default_summary_kind grover_subspace",
            _single_state("S_cover", extra_units=units),
            *transitions,
            _trace(steps, state_id="S_cover"),
            _execution(),
            *properties,
        ]
    )


def _grover_units(block: int, base: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (f"b{block}_w0", tuple(f"q{base + offset}" for offset in (0, 1, 2, 5, 6))),
        (f"b{block}_w1", tuple(f"q{base + offset}" for offset in (2, 3, 5, 6, 7))),
        (f"b{block}_w2", tuple(f"q{base + offset}" for offset in (3, 4, 6, 7, 9))),
        (f"b{block}_w3", (f"q{base + 8}",)),
    )


def _grover_block_gates(base: int) -> tuple[str, ...]:
    q = lambda offset: f"q{base + offset}"
    gates: list[str] = []
    gates.extend(f"H({q(offset)})" for offset in range(5))
    gates.extend((f"X({q(9)})", f"H({q(9)})"))
    gates.extend(f"X({q(offset)})" for offset in range(5))
    gates.extend(
        (
            f"CCX({q(0)}, {q(1)}, {q(5)})",
            f"CCX({q(5)}, {q(2)}, {q(6)})",
            f"CCX({q(6)}, {q(3)}, {q(7)})",
            f"CCX({q(7)}, {q(4)}, {q(9)})",
            f"CCX({q(6)}, {q(3)}, {q(7)})",
            f"CCX({q(5)}, {q(2)}, {q(6)})",
            f"CCX({q(0)}, {q(1)}, {q(5)})",
        )
    )
    gates.extend(f"X({q(offset)})" for offset in range(5))
    gates.extend(f"H({q(offset)})" for offset in range(5))
    gates.extend(f"X({q(offset)})" for offset in range(5))
    gates.extend(
        (
            f"CCX({q(0)}, {q(1)}, {q(5)})",
            f"CCX({q(5)}, {q(2)}, {q(6)})",
            f"CCX({q(6)}, {q(3)}, {q(7)})",
            f"CCX({q(7)}, {q(4)}, {q(9)})",
            f"CCX({q(6)}, {q(3)}, {q(7)})",
            f"CCX({q(5)}, {q(2)}, {q(6)})",
            f"CCX({q(0)}, {q(1)}, {q(5)})",
        )
    )
    gates.extend(f"X({q(offset)})" for offset in range(5))
    gates.extend(f"H({q(offset)})" for offset in range(5))
    return tuple(gates)


def _grover_properties(block: int, base: int, anchor_id: str) -> tuple[str, ...]:
    q = lambda offset: f"q{base + offset}"
    search = ", ".join(q(offset) for offset in range(5))
    phase = q(9)
    scopes = (
        (0, (0, 1, 2, 5, 6)),
        (1, (2, 3, 5, 6, 7)),
        (2, (3, 4, 6, 7, 9)),
        (3, (8,)),
    )
    return tuple(
        "\n".join(
            [
                f"property p_grover_b{block}_w{window} relation {{",
                f"  anchors {anchor_id}",
                f"  observe state({', '.join(q(offset) for offset in scope)})",
                f"  expect grover_span(search={search}; phase={phase}; witnesses=zero, plus)",
                "}",
            ]
        )
        for window, scope in scopes
    )


def _simon_source(qubit_count: int) -> str:
    if qubit_count % 2 != 0:
        raise ValueError("RQ4 Simon expects an even qubit count.")
    half = qubit_count // 2
    units = [(f"pair_{index}", (f"q{index}", f"q{half + index}")) for index in range(1, half)]
    transitions: list[str] = []
    steps: list[str] = []

    def add(gate: str) -> None:
        transition_id = f"t{len(steps):04d}_{gate.split('(', 1)[0].lower()}"
        transitions.append(_transition(transition_id, gate))
        steps.append(transition_id)

    for index in range(half):
        add(f"H(q{index})")
    for index in range(1, half):
        add(f"CX(q{index}, q{half + index})")
    for index in range(half):
        add(f"H(q{index})")

    properties = [
        _distribution_property("p_hidden_period_bit", f"f{len(steps)}", "q0", 0),
    ]
    properties.extend(
        "\n".join(
            [
                f"property p_output_marginal_{index} distribution {{",
                f"  anchors f{len(steps)}",
                f"  observe dist(q{half + index}, Z)",
                "  expect {0: 0.5, 1: 0.5}",
                "}",
            ]
        )
        for index in range(1, half)
    )
    return "\n\n".join(
        [
            _header(f"rq4_simon_{qubit_count}", qubit_count, "H, CX"),
            _single_state(extra_units=units),
            *transitions,
            _trace(steps),
            _execution(),
            *properties,
        ]
    )


RQ4_BENCHMARKS: dict[str, RQ4BenchmarkSpec] = {
    "ghz": RQ4BenchmarkSpec("ghz", "GHZ", _ghz_source),
    "bv": RQ4BenchmarkSpec("bv", "BV", _bv_source),
    "adder": RQ4BenchmarkSpec("adder", "Adder", _adder_source),
    "grover": RQ4BenchmarkSpec("grover", "Grover", _grover_source),
    "simon": RQ4BenchmarkSpec("simon", "Simon", _simon_source),
}
