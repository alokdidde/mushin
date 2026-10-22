"""``mushin.decision/1.0``: one planner decision from an uzumaki episode with its labels.

A record holds what the planner saw (task goal, observation, recent history, camera frames and the
offered options), what the rule teacher would have done, what was actually taken, and labels for
the three typed questions Mushin answers:

- choice: which option to take (a hard target and a soft distribution over the options),
- yes/no: whether each skill option leads to task success (``option_outcomes[o].success``),
- value bins / enums: the right value for each parameter of each skill option (``params``).

Skill cards (instructions, applicability, input schema) are stored once per dataset in
``cards.json`` and referenced by ``card_sha256`` (uzumaki's ``content_sha256``).
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

import jsonschema

SCHEMA_ID = "mushin.decision/1.0"
ABSTAIN = "abstain"


class RecordError(ValueError):
    """A record that does not satisfy ``mushin.decision/1.0`` or its cross-field rules."""


def option_id(skill_id: str, skill_version: int) -> str:
    return f"{skill_id}@{skill_version}"


@cache
def _validator() -> jsonschema.Draft202012Validator:
    text = resources.files("mushin.schema").joinpath("decision-1.0.schema.json").read_text()
    return jsonschema.Draft202012Validator(json.loads(text))


def validate_record(record: Any) -> None:
    """Raise ``RecordError`` unless ``record`` is a valid ``mushin.decision/1.0`` record."""
    errors = sorted(_validator().iter_errors(record), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.path) or "<root>"
        raise RecordError(f"{where}: {first.message}")
    ids = [o["option_id"] for o in record["options"]]
    if len(set(ids)) != len(ids):
        raise RecordError("options: duplicate option_id")
    if ids[-1] != ABSTAIN or ids.count(ABSTAIN) != 1:
        raise RecordError("options: abstain must be offered exactly once, last")
    for option in record["options"][:-1]:
        if option["kind"] != "skill" or option["option_id"] != option_id(
            option["skill_id"], option["skill_version"]
        ):
            raise RecordError(f"options: malformed skill option {option['option_id']}")
    labels = record["labels"]
    known = set(ids)
    for name in ("option_outcomes", "utility", "choice_soft"):
        if set(labels[name]) != known:
            raise RecordError(f"labels/{name}: must cover exactly the offered options")
    if labels["choice"] not in known:
        raise RecordError("labels/choice: not an offered option")
    if abs(sum(labels["choice_soft"].values()) - 1.0) > 1e-6:
        raise RecordError("labels/choice_soft: does not sum to 1")
    if not set(labels["params"]) <= known - {ABSTAIN}:
        raise RecordError("labels/params: unknown option")
    for choice in ("teacher", "taken"):
        if record[choice]["option_id"] not in known:
            raise RecordError(f"{choice}: not an offered option")
    if record["step"] != int(record["record_id"].rsplit("/", 1)[1]) or not record[
        "record_id"
    ].startswith(record["episode_id"] + "/"):
        raise RecordError("record_id: must be <episode_id>/<step>")
