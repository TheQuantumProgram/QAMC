"""File and string loaders for QAM models."""

from __future__ import annotations

import json
from pathlib import Path
import re

from qamc.ir.model import QAMModel
from qamc.loader.normalize import normalize_qam_data
from qamc.loader.qam_toml import QAMTomlParseError, parse_qam_toml
from qamc.surface import SurfaceQAMParseError, parse_surface_qam


_LEGACY_EXPECTED_JUDGMENT_RE = re.compile(
    r"(?m)^\s*expected_judgment\s+[A-Za-z_][A-Za-z0-9_]*\s*\n?"
)


def _parse_qam_text(text: str) -> dict[str, object]:
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("Empty QAM source")

    if stripped[0] == "{":
        return json.loads(text)

    try:
        return parse_qam_toml(text)  # type: ignore[return-value]
    except QAMTomlParseError:
        try:
            return parse_surface_qam(_strip_legacy_experiment_metadata(text))  # type: ignore[return-value]
        except SurfaceQAMParseError:
            # Preserve JSON as a compatibility fallback even if the input does
            # not start with a TOML table header.
            return json.loads(text)


def _strip_legacy_experiment_metadata(text: str) -> str:
    """Keep old experiment files loadable without making metadata Surface QAM."""

    return _LEGACY_EXPECTED_JUDGMENT_RE.sub("", text)


def load_qam_file(path: str | Path) -> QAMModel:
    """Load a QAM model from a TOML- or JSON-backed ``.qam`` file."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    data = _parse_qam_text(text)
    return normalize_qam_data(data)


def load_qam_string(text: str) -> QAMModel:
    """Load a QAM model from an in-memory TOML or JSON string."""

    data = _parse_qam_text(text)
    return normalize_qam_data(data)
