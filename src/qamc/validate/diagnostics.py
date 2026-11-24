"""Diagnostic structures for QAMC validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class Diagnostic:
    """One validation diagnostic item."""

    code: str
    message: str
    location: str
    severity: str = "error"


@dataclass(slots=True)
class ValidationReport:
    """Aggregated validation result."""

    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "warning")

    def add(self, severity: str, code: str, message: str, location: str) -> None:
        self.diagnostics.append(
            Diagnostic(code=code, message=message, location=location, severity=severity)
        )

    def error(self, code: str, message: str, location: str) -> None:
        self.add("error", code, message, location)

    def warning(self, code: str, message: str, location: str) -> None:
        self.add("warning", code, message, location)

    def has_code(self, code: str) -> bool:
        return any(d.code == code for d in self.diagnostics)
