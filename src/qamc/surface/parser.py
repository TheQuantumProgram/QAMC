"""Minimal block-style Surface QAM parser.

This parser is intentionally small. It covers the first paper-facing surface
syntax while lowering to the structured data accepted by ``normalize_qam_data``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import re


class SurfaceQAMParseError(ValueError):
    """Raised when text is not valid block-style Surface QAM."""


def parse_surface_qam(text: str) -> Mapping[str, object]:
    """Parse a compact block-style Surface QAM source into structured data."""

    lines = _strip_lines(text)
    if not lines or not lines[0].startswith("model "):
        raise SurfaceQAMParseError("Surface QAM source must start with a model declaration.")

    data: dict[str, object] = {
        "state_classes": [],
        "transitions": [],
        "properties": [],
    }
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("model "):
            data["model_id"] = line.split(None, 1)[1]
            index += 1
            continue
        if line.startswith("qubits "):
            data["qubits"] = _parse_csv_refs(line.removeprefix("qubits "))
            index += 1
            continue
        if line.startswith("gates "):
            data["gate_set"] = _parse_csv_refs(line.removeprefix("gates "))
            index += 1
            continue
        if line.startswith("default_summary_kind"):
            data.setdefault("defaults", {})["default_summary_kind"] = _parse_default_summary_kind(line)
            index += 1
            continue
        if line.startswith("classical "):
            data["program_classical_vars"] = _parse_program_classical_vars(line.removeprefix("classical "))
            index += 1
            continue

        header, body, index = _read_block(lines, index)
        if header.startswith("state "):
            data["state_classes"].append(_parse_state(header, body))
        elif header.startswith("transition "):
            data["transitions"].append(_parse_transition(header, body))
        elif header.startswith("trace "):
            data["trace"] = _parse_trace(header, body)
        elif header.startswith("execution "):
            data["execution"] = _parse_execution(header, body)
        elif header.startswith("property "):
            data["properties"].append(_parse_property(header, body))
        else:
            raise SurfaceQAMParseError(f"Unsupported top-level declaration: {header!r}")

    return data


def _strip_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if line:
            lines.append(line)
    return lines


def _unquoted_delimiters(line: str, delimiters: str) -> Iterator[tuple[int, str]]:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in {"'", '"'}:
            quote = char
        elif char in delimiters:
            yield index, char


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    for index, _ in _unquoted_delimiters(line, "#"):
        return line[:index]
    return line


def _read_block(lines: list[str], index: int) -> tuple[str, list[str], int]:
    header = lines[index]
    if header.startswith(("state ", "transition ")):
        return _read_declaration_block(lines, index)
    if not header.endswith("{"):
        raise SurfaceQAMParseError(f"Expected block declaration at line: {header!r}")
    header = header[:-1].strip()
    body: list[str] = []
    nested_depth = 0
    index += 1
    while index < len(lines):
        line = lines[index]
        if line == "}" and nested_depth == 0:
            return header, body, index + 1
        if line == "}":
            nested_depth -= 1
            body.append(line)
            index += 1
            continue
        if line.endswith("{"):
            nested_depth += 1
        body.append(line)
        index += 1
    raise SurfaceQAMParseError(f"Unclosed block: {header!r}")


def _read_declaration_block(lines: list[str], index: int) -> tuple[str, list[str], int]:
    """Read state or transition clauses separated by newlines or semicolons."""
    header, opening, first_body_line = lines[index].partition("{")
    if not opening:
        raise SurfaceQAMParseError(f"Expected block declaration at line: {header!r}")
    header = header.strip()
    body: list[str] = []
    in_units = False
    closing = {"(": ")", "[": "]", "{": "}"}
    for line_index in range(index, len(lines)):
        line = first_body_line if line_index == index else lines[line_index]
        if not any(delimiter in line for delimiter in "{};"):
            if line.strip():
                body.append(line.strip())
            continue
        start = 0
        expected: list[str] = []
        for offset, char in _unquoted_delimiters(line, "()[]{};"):
            if char in closing:
                if (char == "{" and not expected and header.startswith("state ")
                        and line[start:offset].strip() == "units"):
                    if in_units:
                        raise SurfaceQAMParseError("Nested units blocks are not supported.")
                    body.append("units {")
                    in_units = True
                    start = offset + 1
                else:
                    expected.append(closing[char])
                continue
            if expected:
                if char != ";" and char != expected.pop():
                    raise SurfaceQAMParseError(f"Mismatched delimiter in declaration: {line!r}")
                continue
            if char in ")]":
                raise SurfaceQAMParseError(f"Unexpected delimiter in declaration: {line!r}")
            clause = line[start:offset].strip()
            if clause:
                body.append(clause)
            start = offset + 1
            if char == "}":
                if in_units:
                    body.append("}")
                    in_units = False
                elif line[start:].strip():
                    raise SurfaceQAMParseError(f"Unexpected text after {header!r}.")
                else:
                    return header, body, line_index + 1
        if expected:
            raise SurfaceQAMParseError(f"Unclosed delimiter in declaration: {line!r}")
        tail = line[start:].strip()
        if tail:
            body.append(tail)
    raise SurfaceQAMParseError(f"Unclosed block: {header!r}")


def _parse_state(header: str, body: list[str]) -> dict[str, object]:
    match = re.fullmatch(r"state\s+([A-Za-z_]\w*)(?:\s+(.+))?", header)
    if not match:
        raise SurfaceQAMParseError(f"Invalid state header: {header!r}")
    state_id, tag_source = match.groups()
    state: dict[str, object] = {"id": state_id}
    tags = tuple(tag_source.split()) if tag_source else ()
    if tags:
        state["tags"] = list(tags)

    profile: dict[str, object] = {}
    index = 0
    while index < len(body):
        line = body[index]
        if line.startswith("active "):
            state["active"] = _parse_active_refs(line.removeprefix("active "))
        elif re.fullmatch(r"units\s*\{", line):
            _header, unit_body, index = _read_block(body, index)
            units, unit_completion = _parse_units_block(unit_body)
            state["units"] = units
            if unit_completion is not None:
                state["unit_completion"] = unit_completion
            continue
        elif line.startswith("units "):
            state["units"] = _parse_units(line.removeprefix("units "))
        elif line.startswith("measurement "):
            profile["measurement"] = {
                name: "measured_bit"
                for name in _parse_csv_refs(line.removeprefix("measurement "))
            }
        elif line.startswith("program "):
            profile["program"] = {
                item.split(":", 1)[0].strip(): item.split(":", 1)[1].strip() if ":" in item else "defined"
                for item in _split_top_level(line.removeprefix("program "), ",")
            }
        else:
            raise SurfaceQAMParseError(f"Unsupported state line: {line!r}")
        index += 1
    if profile:
        state["classical_profile"] = profile
    return state


def _parse_transition(header: str, body: list[str]) -> dict[str, object]:
    match = re.fullmatch(r"transition\s+([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)(?:\s*->\s*([A-Za-z_]\w*))?", header)
    if not match:
        raise SurfaceQAMParseError(f"Invalid transition header: {header!r}")
    transition_id, source, target = match.groups()
    transition: dict[str, object] = {"id": transition_id, "source": source}
    if target is not None:
        transition["target"] = target

    branches: list[dict[str, object]] = []
    gate_line: str | None = None
    for line in body:
        if line.startswith("when "):
            branches.append(_parse_control_branch(line))
        else:
            if gate_line is not None:
                raise SurfaceQAMParseError(f"Transition {transition_id} has more than one gate line.")
            gate_line = line

    if gate_line is not None:
        transition["gate"] = gate_line
    if branches:
        transition["branches"] = branches
    return transition


def _parse_control_branch(line: str) -> dict[str, object]:
    match = re.fullmatch(r"when\s+([A-Za-z_]\w*)\s*==\s*(.+?)\s*->\s*([A-Za-z_]\w*)\s*:\s*(.+)", line)
    if not match:
        raise SurfaceQAMParseError(f"Invalid control branch: {line!r}")
    var_name, expected, target, gate = match.groups()
    return {
        "kind": "classical_control",
        "target": target,
        "guard": {var_name: _parse_scalar(expected)},
        "gate": gate.strip(),
    }


def _parse_trace(header: str, body: list[str]) -> dict[str, object]:
    match = re.fullmatch(r"trace\s+([A-Za-z_]\w*)", header)
    if not match:
        raise SurfaceQAMParseError(f"Invalid trace header: {header!r}")
    flow: list[dict[str, object]] = []
    for line in body:
        step_match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*--([A-Za-z_]\w*)-->\s*([A-Za-z_]\w*)\s*:\s*(?:(frontier|terminal)\s+)?([A-Za-z_]\w*)",
            line,
        )
        if step_match:
            source, transition, target, kind, class_id = step_match.groups()
            target_data: dict[str, object]
            if kind == "terminal":
                target_data = {"frontier": target, "class": class_id, "terminal": True}
            elif kind == "frontier":
                target_data = {"frontier": target, "class": class_id}
            else:
                target_data = {"node": target, "class": class_id}
            flow.append({"transition": transition, "from": source, "to": target_data})
            continue

        anchor_match = re.fullmatch(r"([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)(?:\s+initial)?", line)
        if anchor_match:
            anchor_id, class_id = anchor_match.groups()
            flow.append({"node": anchor_id, "class": class_id})
            continue
        raise SurfaceQAMParseError(f"Invalid trace line: {line!r}")
    return {"id": match.group(1), "flow": flow}


def _parse_execution(header: str, body: list[str]) -> dict[str, object]:
    match = re.fullmatch(r"execution\s+([A-Za-z_]\w*)", header)
    if not match:
        raise SurfaceQAMParseError(f"Invalid execution header: {header!r}")
    execution: dict[str, object] = {"id": match.group(1)}
    initial_binding: dict[str, object] = {}
    for line in body:
        key, value = _split_keyword_value(line)
        if key == "trace":
            execution["trace"] = value
        elif key in {"stop", "stop_anchor"}:
            execution["stop"] = value
        elif key == "program":
            initial_binding.setdefault("program_values", {}).update(_parse_assignments(value))
        else:
            raise SurfaceQAMParseError(f"Unsupported execution line: {line!r}")
    if initial_binding:
        execution["initial_binding"] = initial_binding
    return execution


def _parse_property(header: str, body: list[str]) -> dict[str, object]:
    match = re.fullmatch(r"property\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)", header)
    if not match:
        raise SurfaceQAMParseError(f"Invalid property header: {header!r}")
    property_id, kind = match.groups()
    prop: dict[str, object] = {"id": property_id, "kind": kind}
    expectation_source: str | None = None
    for line in body:
        key, value = _split_keyword_value(line)
        if key in {"anchors", "nodes"}:
            prop["scope"] = {"anchors": _parse_csv_refs(value)}
        elif key == "mode":
            prop["mode"] = value
        elif key == "observe":
            prop["observable"] = _parse_observable(value)
        elif key == "expect":
            expectation_source = value
        elif key == "threshold":
            prop["threshold"] = _parse_scalar(value)
        else:
            raise SurfaceQAMParseError(f"Unsupported property line: {line!r}")
    if expectation_source is not None:
        if kind == "assertion":
            prop["expect"] = _parse_classical_expectation(expectation_source)
        elif kind == "distribution" and prop.get("observable", {}).get("kind") == "classical":
            prop["expect"] = _parse_classical_distribution(expectation_source)
        else:
            prop["expect"] = _parse_expectation(expectation_source)
    if kind == "assertion" and prop.get("mode") not in {"pointwise_all", "pointwise_any"}:
        raise SurfaceQAMParseError("A classical assertion requires mode pointwise_all or pointwise_any.")
    return prop


def _parse_observable(source: str) -> dict[str, object]:
    stripped = source.strip()
    match = re.fullmatch(r"classical\(([^)]*)\)", stripped)
    if match:
        names = [name.strip() for name in match.group(1).split(",")]
        if any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in names):
            raise SurfaceQAMParseError("A classical observation needs a nonempty list of variable names.")
        if len(set(names)) != len(names):
            raise SurfaceQAMParseError("A classical observation must use distinct variable names.")
        return {"kind": "classical", "classical_vars": names}

    match = re.fullmatch(r"dist\(([^)]*)\)", stripped)
    if match:
        args = _parse_csv_refs(match.group(1))
        if len(args) >= 2 and args[-1].upper() in {"X", "Y", "Z", "COMPUTATIONAL"}:
            return {"qubits": args[:-1], "basis": args[-1]}
        return {"qubits": args}

    match = re.fullmatch(r"state\(([^)]*)\)", stripped)
    if match:
        return {"kind": "state", "qubits": _parse_csv_refs(match.group(1))}

    raise SurfaceQAMParseError(f"Unsupported observable: {source!r}")


def _parse_classical_expectation(source: str) -> object:
    stripped = source.strip()
    if stripped.startswith("=="):
        return ["eq", _parse_classical_literal(stripped[2:])]
    membership = re.fullmatch(r"in\s*\{(.*)\}", stripped)
    if membership:
        items = _split_top_level(membership.group(1), ",")
        values = [] if not membership.group(1).strip() else [_parse_classical_literal(item) for item in items]
        return ["in", values]
    raise SurfaceQAMParseError("A classical assertion expects '== <value>' or 'in {<values>}'.")


def _parse_classical_literal(source: str) -> object:
    stripped = source.strip()
    if not stripped:
        raise SurfaceQAMParseError("A classical assertion needs an expected value.")
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip()
        return [] if not inner else [_parse_classical_literal(item) for item in _split_top_level(inner, ",")]
    value = _parse_scalar(stripped)
    if isinstance(value, str) and not (
        len(stripped) >= 2 and stripped[0] in {"'", '"'} and stripped[-1] == stripped[0]
    ):
        raise SurfaceQAMParseError(f"Expected a literal or tuple in classical assertion: {source!r}")
    return value


def _parse_classical_distribution(source: str) -> dict[object, object]:
    stripped = source.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise SurfaceQAMParseError("A classical distribution expects a probability map.")
    result: dict[object, object] = {}
    for entry in _split_top_level(stripped[1:-1], ","):
        if not entry:
            continue
        fields = _split_top_level(entry, ":")
        if len(fields) != 2:
            raise SurfaceQAMParseError(f"Invalid classical distribution entry: {entry!r}")
        key = _parse_classical_literal(fields[0])
        if isinstance(key, list):
            if any(isinstance(item, list) for item in key):
                raise SurfaceQAMParseError("A classical outcome is a scalar or a tuple of scalars.")
            key = tuple(key)
        result[key] = _parse_scalar(fields[1])
    return result


def _parse_expectation(source: str) -> object:
    stripped = source.strip()
    grover_match = re.fullmatch(r"grover_span\((.*)\)", stripped)
    if grover_match:
        return ["grover_span", _parse_grover_span_args(grover_match.group(1))]
    if stripped.startswith("{") and stripped.endswith("}"):
        entries = _split_top_level(stripped[1:-1], ",")
        result: dict[str, object] = {}
        for entry in entries:
            if not entry:
                continue
            if ":" not in entry:
                raise SurfaceQAMParseError(f"Invalid expectation entry: {entry!r}")
            key, value = entry.split(":", 1)
            result[key.strip().strip('"').strip("'").strip("()")] = _parse_scalar(value)
        return result
    if stripped.startswith("(") and stripped.endswith(")"):
        return [_parse_scalar(item) for item in _split_top_level(stripped[1:-1], ",")]
    return [_parse_scalar(stripped)]


def _parse_grover_span_args(source: str) -> dict[str, object]:
    args: dict[str, object] = {}
    for entry in _split_top_level(source, ";"):
        if not entry:
            continue
        if "=" not in entry:
            raise SurfaceQAMParseError(f"Invalid grover_span argument: {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if key in {"search", "witnesses"}:
            args[key] = _parse_csv_refs(value)
        elif key == "phase":
            args[key] = str(_parse_scalar(value))
        else:
            raise SurfaceQAMParseError(f"Unsupported grover_span argument: {key!r}")

    if "search" not in args or "phase" not in args:
        raise SurfaceQAMParseError("grover_span expects search=... and phase=... arguments.")
    args.setdefault("witnesses", ["zero", "plus"])
    return args


def _parse_units(source: str) -> dict[str, object]:
    units: dict[str, object] = {}
    for item in _split_top_level(source, ","):
        unit_id, qubits = _parse_unit_declaration(item)
        units[unit_id] = qubits
    return units


def _parse_active_refs(source: str) -> object:
    stripped = source.strip()
    if stripped in {"all", "*"}:
        return stripped
    return _parse_csv_refs(source)


def _parse_default_summary_kind(line: str) -> str:
    if "=" in line:
        value = line.split("=", 1)[1].strip()
    else:
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SurfaceQAMParseError(f"Expected default_summary_kind value: {line!r}")
        value = parts[1].strip()
    return str(_parse_scalar(value))


def _parse_units_block(body: list[str]) -> tuple[dict[str, object], str | None]:
    units: dict[str, object] = {}
    unit_completion: str | None = None
    for line in body:
        if line.startswith("unit_completion "):
            unit_completion = line.removeprefix("unit_completion ").strip()
            continue
        if line.startswith("unit_completion:"):
            unit_completion = line.split(":", 1)[1].strip()
            continue
        unit_id, qubits = _parse_unit_declaration(line)
        units[unit_id] = qubits
    return units, unit_completion


def _parse_unit_declaration(source: str) -> tuple[str, list[str]]:
    stripped = source.strip()
    match = re.fullmatch(r"([A-Za-z_]\w*)\(([^)]*)\)", stripped)
    if match:
        unit_id, qubits = match.groups()
        return unit_id, _parse_csv_refs(qubits)
    if ":" in stripped:
        unit_id, qubits = stripped.split(":", 1)
        unit_id = unit_id.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", unit_id):
            raise SurfaceQAMParseError(f"Invalid unit id: {unit_id!r}")
        return unit_id, _parse_csv_refs(qubits)
    if not stripped:
        raise SurfaceQAMParseError("Empty unit declaration.")
    return stripped, [stripped]


def _parse_csv_refs(source: str) -> list[str]:
    return [item.strip() for item in _split_top_level(source, ",") if item.strip()]


def _parse_program_classical_vars(source: str) -> dict[str, dict[str, object]]:
    declarations: dict[str, dict[str, object]] = {}
    for item in _split_top_level(source, ","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" in stripped:
            name, type_name = stripped.split(":", 1)
            name = name.strip()
            type_name = type_name.strip()
        else:
            name = stripped
            type_name = "bit"
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise SurfaceQAMParseError(f"Invalid classical variable name: {name!r}")
        declarations[name] = {"type": type_name}
    return declarations


def _parse_assignments(source: str) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for item in _split_top_level(source, ","):
        stripped = item.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise SurfaceQAMParseError(f"Expected assignment: {stripped!r}")
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise SurfaceQAMParseError(f"Invalid assignment name: {name!r}")
        assignments[name] = _parse_scalar(value)
    return assignments


def _split_keyword_value(line: str) -> tuple[str, str]:
    parts = line.split(None, 1)
    if len(parts) != 2:
        raise SurfaceQAMParseError(f"Expected keyword and value: {line!r}")
    return parts[0], parts[1].strip()


def _parse_scalar(source: str) -> object:
    stripped = source.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        return stripped.strip('"').strip("'")


def _split_top_level(source: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    escaped = False
    for char in source:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "({[":
            depth += 1
            current.append(char)
            continue
        if char in ")}]":
            depth -= 1
            current.append(char)
            continue
        if char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return parts
