"""Turn counterfactual option outcomes into training targets (pure functions, no simulator).

Each skill option at a decision is tried by replaying the episode up to that decision, taking the
option with its oracle parameters and letting the rule selector finish the episode. Its utility is

- ``1 - STEP_COST * (decisions_to_end - 1)`` when the task completes (fewer decisions is better),
- ``0`` when it does not, and ``-VIOLATION_COST`` more if the run broke an invariant;

abstaining is worth ``ABSTAIN_UTILITY``: better than any failing option, worse than any success.
The hard choice target is the best option (ties go to the rule teacher's option, then to the
earlier offered option) and the soft target is ``softmax(utility / SOFT_TEMPERATURE)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mushin.schema import ABSTAIN

LABEL_VERSION = "cf-v1"
STEP_COST = 0.05
VIOLATION_COST = 0.5
ABSTAIN_UTILITY = 0.1
SOFT_TEMPERATURE = 0.1
NUMBER_BINS = 16


def utility(outcome: Mapping[str, Any]) -> float:
    if outcome.get("success") is None:
        return ABSTAIN_UTILITY
    if outcome["success"]:
        steps = max(1, int(outcome.get("decisions_to_end") or 1))
        value = 1.0 - STEP_COST * (steps - 1)
    else:
        value = 0.0
    if outcome.get("violations"):
        value -= VIOLATION_COST
    return round(value, 6)


def choice_targets(
    options: Sequence[str], outcomes: Mapping[str, Mapping[str, Any]], teacher: str
) -> tuple[dict[str, float], str, dict[str, float]]:
    """Utilities, hard target and soft target over ``options`` (abstain included)."""
    values = {o: (ABSTAIN_UTILITY if o == ABSTAIN else utility(outcomes[o])) for o in options}
    best = max(values.values())
    tied = [o for o in options if values[o] == best]
    hard = teacher if teacher in tied else tied[0]
    peak = max(v / SOFT_TEMPERATURE for v in values.values())
    weights = {o: math.exp(v / SOFT_TEMPERATURE - peak) for o, v in values.items()}
    total = sum(weights.values())
    soft = {o: round(w / total, 6) for o, w in weights.items()}
    drift = round(1.0 - sum(soft.values()), 6)
    soft[hard] = round(soft[hard] + drift, 6)
    return values, hard, soft


def number_edges(spec: Mapping[str, Any], bins: int = NUMBER_BINS) -> list[float]:
    low = float(spec.get("minimum", 0.0))
    high = float(spec.get("maximum", 1.0))
    return [round(low + (high - low) * i / bins, 6) for i in range(bins + 1)]


def bin_of(value: float, edges: Sequence[float]) -> int:
    """Index of the bin holding ``value`` (values outside the range go to the end bins)."""
    for index in range(len(edges) - 1):
        if value < edges[index + 1]:
            return index
    return len(edges) - 2


def bin_center(index: int, edges: Sequence[float]) -> float:
    return (edges[index] + edges[index + 1]) / 2


def parameter_labels(
    schema: Mapping[str, Any],
    oracle: Mapping[str, Any],
    enum_success: Mapping[str, Sequence[bool | None]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Typed targets for each parameter in a skill's input schema, from the oracle parameters.

    Numbers become value bins over the schema's range; enums become an index into the allowed
    values, with the counterfactual success of each value when it was tried."""
    out: dict[str, dict[str, Any]] = {}
    for name, spec in (schema.get("properties") or {}).items():
        if name not in oracle:
            continue
        if "enum" in spec:
            values = [str(v) for v in spec["enum"]]
            label: dict[str, Any] = {
                "kind": "enum",
                "values": values,
                "index": values.index(str(oracle[name])),
            }
            if enum_success and name in enum_success:
                label["success"] = list(enum_success[name])
            out[name] = label
        elif spec.get("type") in ("number", "integer"):
            edges = number_edges(spec)
            value = float(oracle[name])
            out[name] = {
                "kind": "bins", "edges": edges, "bin": bin_of(value, edges), "value": value
            }
    return out
