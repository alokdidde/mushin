"""Exploration episodes: the rule selector with epsilon-random choices, logging every decision.

At each decision the explorer renders the frames, asks the protocol's rule selector (the teacher)
and, with probability ``epsilon``, replaces its choice by a uniformly random offered option
(an executable eligible skill or abstain). Explored skills get the oracle parameters half the
time and random in-range parameters otherwise, so parameter mistakes are seen too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from mushin.data.bodies import oracle_parameters
from mushin.schema import ABSTAIN, option_id


def random_parameters(schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in (schema.get("properties") or {}).items():
        if "enum" in spec:
            out[key] = rng.choice(spec["enum"])
        elif spec.get("type") == "number":
            out[key] = round(rng.uniform(spec.get("minimum", 0.0), spec.get("maximum", 1.0)), 4)
        elif spec.get("type") == "integer":
            out[key] = rng.randint(spec.get("minimum", 0), spec.get("maximum", 3))
    return out


def offered(request: dict[str, Any], executable: frozenset[str]) -> list[dict[str, Any]]:
    """Eligible skills that have an implementation to run, in the request's order."""
    return [
        e for e in request["eligible"]
        if option_id(e["skill_id"], e["skill_version"]) in executable
    ]


def as_choice(decision: dict[str, Any]) -> dict[str, Any]:
    if decision.get("action") != "select":
        return {"option_id": ABSTAIN, "parameters": None}
    return {
        "option_id": option_id(decision["skill_id"], decision["skill_version"]),
        "parameters": dict(decision.get("parameters") or {}),
    }


def as_decision(
    entry: dict[str, Any] | None, parameters: dict[str, Any], reason: str | None
) -> dict[str, Any]:
    """A selector decision; without ``reason`` it has the rule selectors' exact shape (the
    reason is recorded in the evidence bundle, so replays must match it)."""
    if entry is None:
        return {"action": "abstain", "reason": reason or "no_eligible_skill"}
    decision = {
        "action": "select",
        "skill_id": entry["skill_id"],
        "skill_version": entry["skill_version"],
        "parameters": parameters,
    }
    if reason is not None:
        decision["reason"] = reason
    return decision


@dataclass
class Step:
    decision_id: str
    request: dict[str, Any]
    teacher: dict[str, Any]
    decision: dict[str, Any]
    source: str
    frames: dict[str, bytes | None] = field(default_factory=dict)


class Explorer:
    """A ``DecisionSource`` that explores around the rule teacher and logs each step."""

    def __init__(
        self,
        rule: Any,
        inputs: dict[str, dict[str, Any]],
        executable: frozenset[str],
        epsilon: float,
        rng: random.Random,
        frames: Any = None,
    ) -> None:
        self.rule, self.inputs, self.executable = rule, inputs, executable
        self.epsilon, self.rng, self.frames = epsilon, rng, frames
        self.steps: list[Step] = []

    def decide(self, decision_id: str, role: str, request: dict[str, Any]) -> dict[str, Any]:
        images = self.frames.render() if self.frames is not None else {}
        teacher = self.rule.decide(decision_id, role, request)
        decision, source = teacher, "rule"
        if self.rng.random() < self.epsilon:
            options: list[dict[str, Any] | None] = [*offered(request, self.executable), None]
            pick = self.rng.choice(options)
            source = "explore"
            parameters: dict[str, Any] = {}
            if pick is not None:
                if self.rng.random() < 0.5:
                    parameters = oracle_parameters(pick["skill_id"], request["observation"])
                else:
                    parameters = random_parameters(self.inputs.get(pick["skill_id"], {}), self.rng)
            decision = as_decision(pick, parameters, "exploration")
        self.steps.append(Step(decision_id, request, teacher, decision, source, images))
        return decision


class Replay:
    """A ``DecisionSource`` that repeats logged decisions, forces one at ``at``, then follows
    the rule selector; it keeps every request so the replayed prefix can be checked."""

    def __init__(self, prefix: list[dict[str, Any]], forced: dict[str, Any], rule: Any) -> None:
        self.prefix, self.forced, self.rule = prefix, forced, rule
        self.requests: list[dict[str, Any]] = []

    def decide(self, decision_id: str, role: str, request: dict[str, Any]) -> dict[str, Any]:
        index = len(self.requests)
        self.requests.append(request)
        if index < len(self.prefix):
            return self.prefix[index]
        if index == len(self.prefix):
            return self.forced
        return self.rule.decide(decision_id, role, request)
