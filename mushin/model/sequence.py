"""Render a decision record as one token sequence with marked read-out positions.

The sequence holds the task card, the observation, recent history, the camera frames and then
every option (skills with their cards, and abstain), each ending at a marked token; a skill's
parameters follow it as marked slots (one per number, one per enum value). A final ``Decide``
token is the decision read-out. Options are shuffled during training so the head cannot learn
their order.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mushin.schema import ABSTAIN

ABSTAIN_TEXT = "abstain: do not run any skill now and hand the decision back to the harness."


@dataclass
class Segment:
    kind: str  # "text" or "image"
    value: Any  # text, or a PIL image
    #: Read-out tag of the segment's last token: ("option", oid), ("number", oid, name),
    #: ("enum", oid, name, index) or ("decide",).
    tag: tuple[Any, ...] | None = None


@dataclass
class Layout:
    """Read-out positions after tokenization."""

    decide: int = -1
    options: list[int] = field(default_factory=list)
    option_ids: list[str] = field(default_factory=list)
    #: (option index, parameter name, position, number of bins)
    numbers: list[tuple[int, str, int]] = field(default_factory=list)
    #: (option index, parameter name, [positions of each enum value])
    enums: list[tuple[int, str, list[int]]] = field(default_factory=list)


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(", ", ": "))


def _inputs_text(schema: Mapping[str, Any]) -> str:
    parts = []
    for name, spec in (schema.get("properties") or {}).items():
        if "enum" in spec:
            parts.append(f"{name} (one of {', '.join(map(str, spec['enum']))})")
        else:
            low, high = spec.get("minimum"), spec.get("maximum")
            parts.append(f"{name} ({spec.get('type', 'value')} {low} to {high})")
    return "; ".join(parts) or "none"


def _history_text(history: Sequence[Mapping[str, Any]], with_results: bool) -> str:
    if not history:
        return "none"
    lines = []
    for entry in history:
        params = _compact(entry["parameters"]) if entry["parameters"] else ""
        line = f"- step {entry['step']}: {entry['option_id']} {params}".rstrip()
        if with_results:
            results = ", ".join(
                f"{r['tool']} {r['status']}" + (f" ({r['reason']})" if r.get("reason") else "")
                for r in entry["tool_results"]
            ) or "no tool calls"
            line += f" -> {results}"
        lines.append(line)
    return "\n".join(lines)


def segments(
    record: Mapping[str, Any],
    cards: Mapping[str, Any],
    images: Mapping[str, Any] | None = None,
    rng: random.Random | None = None,
    with_results: bool = False,
) -> list[Segment]:
    """The record as segments; ``images`` maps ``scene``/``wrist`` to loaded frames.

    Tool results of earlier steps are left out by default: uzumaki's decision requests do not
    carry them yet, so a deployed planner could not see them."""
    task = cards["tasks"][record["protocol"]]
    out = [Segment("text", (
        f"{task['intro']}\nGoal: {record['task']['goal']}\n"
        f"Observation: {_compact(record['observation'])}\n"
        f"Recent steps:\n{_history_text(record['history'], with_results)}\n"
    ))]
    for view in ("scene", "wrist"):
        if images and images.get(view) is not None:
            out.append(Segment("text", f"{view.capitalize()} camera: "))
            out.append(Segment("image", images[view]))
            out.append(Segment("text", "\n"))
    out.append(Segment("text", "Options:\n"))
    options = list(record["options"])
    if rng is not None:
        rng.shuffle(options)
    for option in options:
        oid = option["option_id"]
        if oid == ABSTAIN:
            out.append(Segment("text", f"* {ABSTAIN_TEXT}", ("option", oid)))
            out.append(Segment("text", "\n"))
            continue
        card = cards["skills"][option["card_sha256"]]
        out.append(Segment("text", (
            f"* {oid}: {card['instructions']} Applies when: {card['applies_when']} "
            f"Inputs: {_inputs_text(card['input_schema'])}."
        ), ("option", oid)))
        for name, spec in (card["input_schema"].get("properties") or {}).items():
            if "enum" in spec:
                for index, value in enumerate(spec["enum"]):
                    out.append(Segment("text", f"\n  {name} = {value}", ("enum", oid, name, index)))
            elif spec.get("type") in ("number", "integer"):
                out.append(Segment("text", f"\n  {name} value", ("number", oid, name)))
        out.append(Segment("text", "\n"))
    out.append(Segment("text", "Decide:", ("decide",)))
    return out


def layout_from_tags(tags: Sequence[tuple[tuple[Any, ...], int]]) -> Layout:
    """Group tagged positions (in sequence order) into a ``Layout``."""
    layout = Layout()
    index_of: dict[str, int] = {}
    enum_slots: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for tag, position in tags:
        if tag[0] == "option":
            index_of[tag[1]] = len(layout.options)
            layout.options.append(position)
            layout.option_ids.append(tag[1])
        elif tag[0] == "number":
            layout.numbers.append((index_of[tag[1]], tag[2], position))
        elif tag[0] == "enum":
            enum_slots.setdefault((tag[1], tag[2]), []).append((tag[3], position))
        elif tag[0] == "decide":
            layout.decide = position
    for (oid, name), slots in enum_slots.items():
        layout.enums.append((index_of[oid], name, [p for _, p in sorted(slots)]))
    return layout
