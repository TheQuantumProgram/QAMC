"""Evaluation of external temporal properties over one execution context."""

from __future__ import annotations

from dataclasses import replace

from qamc.external_properties.formula import (
    AnchorFormula,
    AlwaysFormula,
    AndFormula,
    AtomFormula,
    EventuallyFormula,
    InlineAtomFormula,
    NextFormula,
    NotFormula,
    TemporalFormula,
    TrueFormula,
    UntilFormula,
)
from qamc.external_properties.inline_atoms import inline_atom_to_property
from qamc.external_properties.model import (
    ExternalPropertyBundle,
    TemporalPropertyEvaluationResult,
    TemporalPropertySpec,
)
from qamc.ir.enums import ScopeKind
from qamc.ir.property import PropertySpec
from qamc.property import PropertyEvaluationResult, PropertyEvaluator
from qamc.runtime.execution_context import ExecutionContext


class TemporalPropertyEvaluator:
    """Evaluate temporal formulas over existing base property semantics."""

    def __init__(self, *, property_evaluator: PropertyEvaluator | None = None) -> None:
        self.property_evaluator = property_evaluator or PropertyEvaluator()

    def evaluate_all(
        self,
        context: ExecutionContext,
        bundle: ExternalPropertyBundle,
        *,
        available_properties: tuple[PropertySpec, ...],
    ) -> list[TemporalPropertyEvaluationResult]:
        property_map = {prop.id: prop for prop in available_properties}
        return [
            self.evaluate(context, temporal_property, property_map)
            for temporal_property in bundle.temporal_properties
        ]

    def evaluate(
        self,
        context: ExecutionContext,
        temporal_property: TemporalPropertySpec,
        property_map: dict[str, PropertySpec],
    ) -> TemporalPropertyEvaluationResult:
        if temporal_property.formula is not None:
            return self._evaluate_formula_property(context, temporal_property, property_map)
        operator = temporal_property.operator
        if operator == "always":
            return self._evaluate_always(context, temporal_property, property_map)
        if operator == "eventually":
            return self._evaluate_eventually(context, temporal_property, property_map)
        if operator == "until":
            return self._evaluate_until(context, temporal_property, property_map)
        raise ValueError(f"Unsupported temporal operator: {operator!r}")

    def _evaluate_formula_property(
        self,
        context: ExecutionContext,
        temporal_property: TemporalPropertySpec,
        property_map: dict[str, PropertySpec],
    ) -> TemporalPropertyEvaluationResult:
        if temporal_property.formula is None:
            raise ValueError(f"Temporal property {temporal_property.id!r} has no formula.")
        anchor_ids = self._interval_anchor_ids(context, temporal_property)
        cache: dict[tuple[str, str], PropertyEvaluationResult] = {}
        passed = self._evaluate_formula(
            context,
            temporal_property.formula,
            anchor_ids,
            0,
            property_map,
            cache,
        )
        truth_by_anchor = tuple(
            (
                anchor_id,
                self._evaluate_formula(
                    context,
                    temporal_property.formula,
                    anchor_ids,
                    index,
                    property_map,
                    cache,
                ),
            )
            for index, anchor_id in enumerate(anchor_ids)
        )
        return TemporalPropertyEvaluationResult(
            property_id=temporal_property.id,
            operator=temporal_property.operator,
            passed=passed,
            selected_anchor_ids=anchor_ids,
            selected_node_ids=self._selected_node_ids(tuple(cache.values())),
            observed={
                "formula": str(temporal_property.formula),
                "truth_by_anchor": truth_by_anchor,
            },
        )

    def _evaluate_formula(
        self,
        context: ExecutionContext,
        formula: TemporalFormula,
        anchor_ids: tuple[str, ...],
        index: int,
        property_map: dict[str, PropertySpec],
        cache: dict[tuple[str, str], PropertyEvaluationResult],
    ) -> bool:
        if isinstance(formula, TrueFormula):
            return True
        if isinstance(formula, AtomFormula):
            return self._evaluate_atom(context, formula.property_id, anchor_ids[index], property_map, cache).passed
        if isinstance(formula, InlineAtomFormula):
            return self._evaluate_inline_atom(context, formula.source, anchor_ids[index], cache).passed
        if isinstance(formula, AnchorFormula):
            target_anchor_id = self._resolve_formula_anchor(context, formula.anchor_id)
            return self._evaluate_formula(
                context,
                formula.operand,
                (target_anchor_id,),
                0,
                property_map,
                cache,
            )
        if isinstance(formula, NotFormula):
            return not self._evaluate_formula(context, formula.operand, anchor_ids, index, property_map, cache)
        if isinstance(formula, AndFormula):
            return self._evaluate_formula(context, formula.left, anchor_ids, index, property_map, cache) and self._evaluate_formula(
                context,
                formula.right,
                anchor_ids,
                index,
                property_map,
                cache,
            )
        if isinstance(formula, NextFormula):
            return index + 1 < len(anchor_ids) and self._evaluate_formula(
                context,
                formula.operand,
                anchor_ids,
                index + 1,
                property_map,
                cache,
            )
        if isinstance(formula, EventuallyFormula):
            return any(
                self._evaluate_formula(context, formula.operand, anchor_ids, candidate, property_map, cache)
                for candidate in range(index, len(anchor_ids))
            )
        if isinstance(formula, AlwaysFormula):
            return all(
                self._evaluate_formula(context, formula.operand, anchor_ids, candidate, property_map, cache)
                for candidate in range(index, len(anchor_ids))
            )
        if isinstance(formula, UntilFormula):
            for witness in range(index, len(anchor_ids)):
                if self._evaluate_formula(context, formula.right, anchor_ids, witness, property_map, cache) and all(
                    self._evaluate_formula(context, formula.left, anchor_ids, earlier, property_map, cache)
                    for earlier in range(index, witness)
                ):
                    return True
            return False
        raise TypeError(f"Unsupported temporal formula: {formula!r}")

    def _resolve_formula_anchor(self, context: ExecutionContext, anchor_id: str) -> str:
        if anchor_id in {"final", "terminal"}:
            return context.execution.stop_anchor
        return anchor_id

    def _evaluate_atom(
        self,
        context: ExecutionContext,
        property_id: str,
        anchor_id: str,
        property_map: dict[str, PropertySpec],
        cache: dict[tuple[str, str], PropertyEvaluationResult],
    ) -> PropertyEvaluationResult:
        key = (property_id, anchor_id)
        if key not in cache:
            property_spec = self._resolve_property_id(property_id, property_map)
            cache[key] = self.property_evaluator.evaluate(
                context,
                self._localized_property(property_spec, anchor_id),
            )
        return cache[key]

    def _evaluate_inline_atom(
        self,
        context: ExecutionContext,
        source: str,
        anchor_id: str,
        cache: dict[tuple[str, str], PropertyEvaluationResult],
    ) -> PropertyEvaluationResult:
        key = (f"inline:{source}", anchor_id)
        if key not in cache:
            property_spec = inline_atom_to_property(
                source,
                property_id=f"inline:{source}",
                anchor_id=anchor_id,
            )
            cache[key] = self.property_evaluator.evaluate(context, property_spec)
        return cache[key]

    def _evaluate_always(
        self,
        context: ExecutionContext,
        temporal_property: TemporalPropertySpec,
        property_map: dict[str, PropertySpec],
    ) -> TemporalPropertyEvaluationResult:
        anchor_ids = self._interval_anchor_ids(context, temporal_property)
        operand = self._resolve_unary_operand(temporal_property, property_map)
        local_results = tuple(
            self.property_evaluator.evaluate(context, self._localized_property(operand, anchor_id))
            for anchor_id in anchor_ids
        )
        return self._wrap_result(
            temporal_property,
            anchor_ids,
            local_results,
            all(result.passed for result in local_results),
        )

    def _evaluate_eventually(
        self,
        context: ExecutionContext,
        temporal_property: TemporalPropertySpec,
        property_map: dict[str, PropertySpec],
    ) -> TemporalPropertyEvaluationResult:
        anchor_ids = self._interval_anchor_ids(context, temporal_property)
        operand = self._resolve_unary_operand(temporal_property, property_map)
        local_results = tuple(
            self.property_evaluator.evaluate(context, self._localized_property(operand, anchor_id))
            for anchor_id in anchor_ids
        )
        return self._wrap_result(
            temporal_property,
            anchor_ids,
            local_results,
            any(result.passed for result in local_results),
        )

    def _evaluate_until(
        self,
        context: ExecutionContext,
        temporal_property: TemporalPropertySpec,
        property_map: dict[str, PropertySpec],
    ) -> TemporalPropertyEvaluationResult:
        anchor_ids = self._interval_anchor_ids(context, temporal_property)
        if len(temporal_property.operands) != 2:
            raise ValueError("until requires exactly two operand property ids.")
        lhs = self._resolve_property_id(temporal_property.operands[0], property_map)
        rhs = self._resolve_property_id(temporal_property.operands[1], property_map)
        lhs_results = tuple(
            self.property_evaluator.evaluate(context, self._localized_property(lhs, anchor_id))
            for anchor_id in anchor_ids
        )
        rhs_results = tuple(
            self.property_evaluator.evaluate(context, self._localized_property(rhs, anchor_id))
            for anchor_id in anchor_ids
        )
        witness_indices = [
            index for index, rhs_result in enumerate(rhs_results) if rhs_result.passed
        ]
        passed = any(all(lhs_results[j].passed for j in range(index)) for index in witness_indices)
        observed = {
            "lhs": tuple((anchor_ids[i], lhs_results[i].passed) for i in range(len(anchor_ids))),
            "rhs": tuple((anchor_ids[i], rhs_results[i].passed) for i in range(len(anchor_ids))),
            "witnesses": tuple(anchor_ids[i] for i in witness_indices),
        }
        node_ids = self._selected_node_ids(lhs_results)
        return TemporalPropertyEvaluationResult(
            property_id=temporal_property.id,
            operator=temporal_property.operator,
            passed=passed,
            selected_anchor_ids=anchor_ids,
            selected_node_ids=node_ids,
            observed=observed,
        )

    def _interval_anchor_ids(
        self,
        context: ExecutionContext,
        temporal_property: TemporalPropertySpec,
    ) -> tuple[str, ...]:
        trace_order = context.trace.topological_anchor_order()
        if temporal_property.interval is None:
            return trace_order
        start_anchor, end_anchor = temporal_property.interval
        start_index = trace_order.index(start_anchor)
        end_index = trace_order.index(end_anchor)
        return trace_order[start_index : end_index + 1]

    def _resolve_unary_operand(
        self,
        temporal_property: TemporalPropertySpec,
        property_map: dict[str, PropertySpec],
    ) -> PropertySpec:
        if len(temporal_property.operands) != 1:
            raise ValueError(
                f"Temporal property {temporal_property.id!r} requires exactly one operand."
            )
        return self._resolve_property_id(temporal_property.operands[0], property_map)

    def _resolve_property_id(
        self,
        property_id: str,
        property_map: dict[str, PropertySpec],
    ) -> PropertySpec:
        try:
            return property_map[property_id]
        except KeyError as exc:
            raise ValueError(f"Unknown operand property id: {property_id!r}") from exc

    def _localized_property(self, property_spec: PropertySpec, anchor_id: str) -> PropertySpec:
        return replace(
            property_spec,
            scope_kind=ScopeKind.ANCHORS,
            scope_value=(anchor_id,),
        )

    def _wrap_result(
        self,
        temporal_property: TemporalPropertySpec,
        anchor_ids: tuple[str, ...],
        local_results: tuple[PropertyEvaluationResult, ...],
        passed: bool,
    ) -> TemporalPropertyEvaluationResult:
        return TemporalPropertyEvaluationResult(
            property_id=temporal_property.id,
            operator=temporal_property.operator,
            passed=passed,
            selected_anchor_ids=anchor_ids,
            selected_node_ids=self._selected_node_ids(local_results),
            observed=tuple(
                (anchor_id, result.passed, result.observed)
                for anchor_id, result in zip(anchor_ids, local_results)
            ),
        )

    def _selected_node_ids(
        self,
        local_results: tuple[PropertyEvaluationResult, ...],
    ) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for result in local_results:
            for node_id in result.selected_node_ids:
                if node_id not in seen:
                    ordered.append(node_id)
                    seen.add(node_id)
        return tuple(ordered)
