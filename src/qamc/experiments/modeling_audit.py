"""Independent predicate evaluation for the full-state reference calculations."""

from __future__ import annotations

import numpy as np

from qamc.experiments.modeling_reference import ReferenceUnsupported
from qamc.ir.enums import EvaluationMode, PropertyKind
from qamc.ir.property import PropertySpec


def observation_error(actual: object, expected: object) -> float:
    """Maximum entrywise discrepancy, with absent outcomes interpreted as zero."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = [abs(actual.get(key, 0) - expected.get(key, 0)) for key in actual.keys() | expected.keys()]
        difference = max(differences, default=0.0)
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return float("inf")
        difference = max((observation_error(a, b) for a, b in zip(actual, expected)), default=0.0)
    elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            return float("inf")
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    else:
        difference = 0.0 if actual == expected else float("inf")
    return float(difference) if np.isfinite(difference) else float("inf")


def reference_judgment(property_spec: PropertySpec, observed: object, tolerance: float = 1e-9) -> bool:
    """Evaluate the candidate assertion predicates without the checker evaluator."""
    expectation = property_spec.expectation

    def matches(value: object) -> bool:
        if property_spec.kind is PropertyKind.DISTRIBUTION:
            return observation_error(value, expectation) <= tolerance
        if property_spec.kind is PropertyKind.RELATION:
            if expectation[0] == "equal_bits":
                return abs(sum(probability for outcome, probability in value.items()
                               if len(set(outcome)) == 1) - 1.0) <= tolerance
            if expectation[0] == "support_subset":
                return all(probability <= tolerance or outcome in expectation[1]
                           for outcome, probability in value.items())
        if property_spec.kind is PropertyKind.REACHABILITY:
            operation, outcome, threshold = expectation
            if operation == "probability_gt":
                return value.get(outcome, 0.0) > threshold + tolerance
        if property_spec.kind is PropertyKind.ASSERTION:
            if not isinstance(expectation, tuple):
                return value == expectation
            if expectation[0] == "eq":
                return value == expectation[1]
            if expectation[0] == "in":
                return value in expectation[1]
            if expectation[0] == "contains":
                return expectation[1] in value
        raise ReferenceUnsupported(f"Independent audit predicate {expectation!r} is unsupported.")

    if property_spec.mode is EvaluationMode.POINTWISE_ALL:
        return all(matches(value) for value in observed)
    if property_spec.mode is EvaluationMode.POINTWISE_ANY:
        return any(matches(value) for value in observed)
    return matches(observed)
