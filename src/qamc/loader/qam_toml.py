"""A small TOML-subset parser for QAM concrete syntax."""

from __future__ import annotations

import ast
from collections.abc import Mapping


class QAMTomlParseError(ValueError):
    """Raised when the restricted TOML front end cannot parse the input."""


def parse_qam_toml(text: str) -> Mapping[str, object]:
    """Parse a restricted TOML syntax into nested Python data."""

    root: dict[str, object] = {}
    current: dict[str, object] = root

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        if line.startswith("[[") and line.endswith("]]"):
            path = _parse_header_path(line[2:-2].strip(), line_no)
            current = _ensure_array_table(root, path, line_no)
            continue

        if line.startswith("[") and line.endswith("]"):
            path = _parse_header_path(line[1:-1].strip(), line_no)
            current = _ensure_table(root, path, line_no)
            continue

        if "=" not in line:
            raise QAMTomlParseError(f"Line {line_no}: expected key = value assignment")

        key, raw_value = line.split("=", 1)
        key = _normalize_key(key.strip())
        if not key:
            raise QAMTomlParseError(f"Line {line_no}: empty key")
        current[key] = _parse_value(raw_value.strip(), line_no)

    return root


def _strip_comment(line: str) -> str:
    in_string = False
    escaped = False
    quote = ""
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
        elif char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _parse_header_path(raw: str, line_no: int) -> list[str]:
    path = [_normalize_key(part.strip()) for part in raw.split(".") if part.strip()]
    if not path:
        raise QAMTomlParseError(f"Line {line_no}: empty table header")
    return path


def _ensure_table(root: dict[str, object], path: list[str], line_no: int) -> dict[str, object]:
    current = root
    for segment in path:
        if segment in current:
            value = current[segment]
            if isinstance(value, list):
                if not value:
                    raise QAMTomlParseError(
                        f"Line {line_no}: table segment '{segment}' refers to an empty array-of-tables"
                    )
                last = value[-1]
                if not isinstance(last, dict):
                    raise QAMTomlParseError(f"Line {line_no}: invalid array-of-tables entry for '{segment}'")
                current = last
            elif isinstance(value, dict):
                current = value
            else:
                raise QAMTomlParseError(f"Line {line_no}: segment '{segment}' is not a table")
        else:
            current[segment] = {}
            current = current[segment]
    return current


def _ensure_array_table(root: dict[str, object], path: list[str], line_no: int) -> dict[str, object]:
    if not path:
        raise QAMTomlParseError(f"Line {line_no}: empty array-of-tables header")

    current = root
    for segment in path[:-1]:
        if segment in current:
            value = current[segment]
            if isinstance(value, list):
                if not value:
                    raise QAMTomlParseError(
                        f"Line {line_no}: parent segment '{segment}' refers to an empty array-of-tables"
                    )
                last = value[-1]
                if not isinstance(last, dict):
                    raise QAMTomlParseError(f"Line {line_no}: invalid array-of-tables entry for '{segment}'")
                current = last
            elif isinstance(value, dict):
                current = value
            else:
                raise QAMTomlParseError(f"Line {line_no}: segment '{segment}' is not a table")
        else:
            current[segment] = {}
            current = current[segment]

    final = path[-1]
    existing = current.get(final)
    if existing is None:
        current[final] = []
        existing = current[final]
    if not isinstance(existing, list):
        raise QAMTomlParseError(f"Line {line_no}: segment '{final}' is not an array-of-tables")

    item: dict[str, object] = {}
    existing.append(item)
    return item


def _parse_value(raw: str, line_no: int) -> object:
    if raw.startswith('"') or raw.startswith("'"):
        return ast.literal_eval(raw)

    if raw.startswith("[") and raw.endswith("]"):
        return _parse_array(raw[1:-1], line_no)

    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if raw.startswith("{") and raw.endswith("}"):
        return _parse_inline_table(raw[1:-1], line_no)

    try:
        if any(char in raw for char in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        # Bare identifiers are accepted as strings in this restricted syntax.
        return raw


def _normalize_key(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return ast.literal_eval(raw)
    return raw


def _parse_array(raw: str, line_no: int) -> list[object]:
    items = _split_top_level(raw, ",")
    if not items or (len(items) == 1 and not items[0].strip()):
        return []
    return [_parse_value(item.strip(), line_no) for item in items]


def _parse_inline_table(raw: str, line_no: int) -> dict[str, object]:
    result: dict[str, object] = {}
    items = _split_top_level(raw, ",")
    for item in items:
        if not item.strip():
            continue
        if "=" not in item:
            raise QAMTomlParseError(f"Line {line_no}: invalid inline-table entry '{item.strip()}'")
        key, value = item.split("=", 1)
        result[key.strip().strip('"').strip("'")] = _parse_value(value.strip(), line_no)
    return result


def _split_top_level(raw: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    escaped = False
    depth_square = 0
    depth_curly = 0

    for char in raw:
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

        if char == "[":
            depth_square += 1
            current.append(char)
            continue
        if char == "]":
            depth_square -= 1
            current.append(char)
            continue
        if char == "{":
            depth_curly += 1
            current.append(char)
            continue
        if char == "}":
            depth_curly -= 1
            current.append(char)
            continue

        if char == delimiter and depth_square == 0 and depth_curly == 0:
            parts.append("".join(current))
            current = []
            continue

        current.append(char)

    parts.append("".join(current))
    return parts
