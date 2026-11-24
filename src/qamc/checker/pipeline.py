"""End-to-end checker pipeline from model loading to property evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qamc.engine import TraceExecutor
from qamc.external_properties.model import ExternalPropertyBundle, TemporalPropertyEvaluationResult
from qamc.external_properties.temporal import TemporalPropertyEvaluator
from qamc.ir.model import QAMModel
from qamc.loader import load_qam_file
from qamc.property import PropertyEvaluationResult, PropertyEvaluator
from qamc.runtime.execution_context import ExecutionContext
from qamc.validate import validate_qam_model
from qamc.validate.diagnostics import ValidationReport


@dataclass(slots=True)
class CheckerRunResult:
    """Result bundle for one checker run."""

    model: QAMModel
    validation: ValidationReport
    execution_id: str | None
    context: ExecutionContext | None
    property_results: list[PropertyEvaluationResult]
    temporal_results: list[TemporalPropertyEvaluationResult]

    @property
    def ok(self) -> bool:
        return (
            self.validation.ok
            and all(result.passed for result in self.property_results)
            and all(result.passed for result in self.temporal_results)
        )


def run_model(
    model: QAMModel,
    *,
    execution_id: str | None = None,
    validate: bool = True,
    backend=None,
    external_properties: ExternalPropertyBundle | None = None,
    trace_executor: TraceExecutor | None = None,
    property_evaluator: PropertyEvaluator | None = None,
) -> CheckerRunResult:
    """Run validation, execution, and property evaluation on a typed model."""

    validation = validate_qam_model(model) if validate else ValidationReport()
    selected_execution_id = execution_id or next(iter(model.executions), None)
    if not validation.ok:
        return CheckerRunResult(
            model=model,
            validation=validation,
            execution_id=selected_execution_id,
            context=None,
            property_results=[],
            temporal_results=[],
        )

    executor = trace_executor or TraceExecutor(backend=backend)
    evaluator = property_evaluator or PropertyEvaluator(backend=backend)
    context = executor.run(model, execution_id=selected_execution_id)
    properties = model.properties
    temporal_results: list[TemporalPropertyEvaluationResult] = []
    if external_properties is not None:
        properties = properties + external_properties.base_properties
    property_results = evaluator.evaluate_all(context, properties)
    if external_properties is not None and external_properties.temporal_properties:
        temporal_results = TemporalPropertyEvaluator(property_evaluator=evaluator).evaluate_all(
            context,
            external_properties,
            available_properties=properties,
        )
    return CheckerRunResult(
        model=model,
        validation=validation,
        execution_id=selected_execution_id,
        context=context,
        property_results=property_results,
        temporal_results=temporal_results,
    )


def run_qam_file(
    path: str | Path,
    *,
    execution_id: str | None = None,
    validate: bool = True,
    backend=None,
    external_properties: ExternalPropertyBundle | None = None,
    trace_executor: TraceExecutor | None = None,
    property_evaluator: PropertyEvaluator | None = None,
) -> CheckerRunResult:
    """Load a TOML-backed ``.qam`` file and run the checker pipeline."""

    model = load_qam_file(path)
    return run_model(
        model,
        execution_id=execution_id,
        validate=validate,
        backend=backend,
        external_properties=external_properties,
        trace_executor=trace_executor,
        property_evaluator=property_evaluator,
    )
