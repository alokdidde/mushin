"""Counterfactual labels: replay an episode up to a decision, try each option there, and let the
rule selector finish.

The MuJoCo worlds are deterministic (a case replayed with the same decisions gives a byte-identical
evidence bundle), so the replay reaches the same state as the logged episode; ``reproduced``
records that the replayed requests matched the logged ones. Each tried option gets its outcome,
the number of decisions it took to finish, and the head hash of its evidence bundle so the label
can be re-derived (``mushin.data.verify``). Enum parameters are tried value by value.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mushin.data.bodies import oracle_parameters
from mushin.data.explore import Replay, Step, as_decision, offered
from mushin.schema import option_id

#: ``run(case, selector) -> (CaseRun, world)``.
Runner = Callable[[Any, Any], tuple[Any, Any]]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bundle_head(bundle: dict[str, Any]) -> str | None:
    events = bundle.get("events") or []
    return events[-1].get("sha256") if events else None


def outcome_of(run: Any, world: Any = None, parameters: Any = None, start: int = 0,
               decisions: int | None = None) -> dict[str, Any]:
    verdict = run.verdict or {}
    return {
        "success": verdict.get("primary") == "completed",
        "primary": verdict.get("primary"),
        "status": run.report.status,
        "violations": verdict.get("violations"),
        "decisions_to_end": None if decisions is None else decisions - start,
        "parameters": parameters,
        "bundle_head": bundle_head(run.bundle),
        "final": world.final_state() if world is not None else None,
    }


def try_option(
    runner: Runner,
    case: Any,
    steps: list[Step],
    at: int,
    entry: dict[str, Any],
    parameters: dict[str, Any],
    rule_factory: Callable[[], Any],
) -> dict[str, Any]:
    prefix = [s.decision for s in steps[:at]]
    forced = as_decision(entry, parameters, "counterfactual")
    replay = Replay(prefix, forced, rule_factory())
    run, world = runner(case, replay)
    seen = [canonical(r) for r in replay.requests[: at + 1]]
    logged = [canonical(s.request) for s in steps[: at + 1]]
    outcome = outcome_of(run, world, parameters, at, len(replay.requests))
    outcome["reproduced"] = seen == logged
    return outcome


def label_step(
    runner: Runner,
    case: Any,
    steps: list[Step],
    at: int,
    executable: frozenset[str],
    inputs: dict[str, dict[str, Any]],
    rule_factory: Callable[[], Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[bool | None]]]]:
    """Outcome per offered skill option (oracle parameters), and per-value success of each enum
    parameter of each option."""
    request = steps[at].request
    outcomes: dict[str, dict[str, Any]] = {}
    enums: dict[str, dict[str, list[bool | None]]] = {}
    for entry in offered(request, executable):
        oid = option_id(entry["skill_id"], entry["skill_version"])
        oracle = oracle_parameters(entry["skill_id"], request["observation"])
        outcomes[oid] = try_option(runner, case, steps, at, entry, oracle, rule_factory)
        for name, spec in (inputs.get(entry["skill_id"], {}).get("properties") or {}).items():
            if "enum" not in spec or name not in oracle:
                continue
            results: list[bool | None] = []
            for value in spec["enum"]:
                if value == oracle[name]:
                    results.append(outcomes[oid]["success"])
                    continue
                variant = dict(oracle, **{name: value})
                tried = try_option(runner, case, steps, at, entry, variant, rule_factory)
                results.append(tried["success"])
            enums.setdefault(oid, {})[name] = results
    return outcomes, enums
