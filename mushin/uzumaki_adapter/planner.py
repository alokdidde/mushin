"""``MushinPlanner``: an uzumaki ``DecisionSource`` backed by a Mushin checkpoint.

For each decision it builds the same view the training records hold (task card, observation
fields, the last four steps with their tool results, scene and wrist frames, the offered skills
and abstain), runs one pass and returns a decision of the rule selector's shape. It abstains when
abstain is the top option or the top option's probability is below ``min_confidence``; numbers
are set to the centre of the most likely bin and enums to the most likely value. The driver still
validates every decision and the independent evaluator decides success.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mushin.data.explore import offered
from mushin.data.labels import bin_center, number_edges
from mushin.model.head import MushinModel
from mushin.model.sequence import layout_from_tags, segments
from mushin.schema import ABSTAIN, option_id


def number_value(logits: torch.Tensor, spec: Mapping[str, Any]) -> float:
    """Expected value over the most likely bin and its neighbours (finer than one bin)."""
    edges = number_edges(spec)
    probs = torch.softmax(logits.float(), -1)
    top = int(probs.argmax())
    window = range(max(0, top - 1), min(len(probs), top + 2))
    weights = [float(probs[i]) for i in window]
    total = sum(w * bin_center(i, edges) for w, i in zip(weights, window, strict=True))
    return total / sum(weights)


class MushinPlanner:
    def __init__(self, model: MushinModel, snapshot: Any, card: Any, protocol: str,
                 executable: frozenset[str], frames: Any = None,
                 min_confidence: float = 0.0) -> None:
        self.model, self.snapshot, self.card = model.eval(), snapshot, card
        self.protocol, self.executable, self.frames = protocol, executable, frames
        self.min_confidence = min_confidence
        self.history: list[dict[str, Any]] = []
        self.last: dict[str, Any] | None = None

    def _cards(self, entries: list[Mapping[str, Any]]) -> dict[str, Any]:
        skills = {}
        for e in entries:
            skill = self.snapshot.get(e["skill_id"], e["skill_version"])
            skills[e["content_sha256"]] = {
                "instructions": skill.instructions,
                "applies_when": skill.applicability["summary"],
                "input_schema": skill.inputs,
            }
        task = {"intro": self.card.intro, "goal": self.card.goal}
        return {"skills": skills, "tasks": {self.protocol: task}}

    def view(self, request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        entries = offered(dict(request), self.executable)
        fields = ("observation_id", *self.card.observation_fields)
        record = {
            "protocol": self.protocol,
            "task": {"protocol": self.protocol, "goal": self.card.goal},
            "observation": {k: request["observation"].get(k) for k in fields},
            "history": self.history[-4:],
            "options": [
                {"option_id": option_id(e["skill_id"], e["skill_version"]), "kind": "skill",
                 "skill_id": e["skill_id"], "skill_version": e["skill_version"],
                 "card_sha256": e["content_sha256"]} for e in entries
            ] + [{"option_id": ABSTAIN, "kind": "abstain"}],
        }
        return record, self._cards(entries)

    def note_results(self, tool_results: list[dict[str, Any]]) -> None:
        """Tool results of the last decision, for the history the next decision sees."""
        if self.history:
            self.history[-1]["tool_results"] = tool_results

    @torch.inference_mode()
    def decide(self, decision_id: str, role: str, request: dict[str, Any]) -> dict[str, Any]:
        record, cards = self.view(request)
        images = self.frames.render() if self.frames is not None else None
        if images:
            from io import BytesIO

            from PIL import Image

            images = {k: Image.open(BytesIO(v)) for k, v in images.items() if v is not None}
        encoded = self.model.backbone.encode(segments(record, cards, images))
        layout = layout_from_tags(encoded.tags)
        prediction = self.model(self.model.backbone.collate([encoded]), [layout])[0]
        probs = torch.softmax(prediction.choice.float(), -1)
        best = int(probs.argmax())
        chosen = layout.option_ids[best]
        confidence = float(probs[best])
        success = torch.sigmoid(prediction.success.float())
        info = {"model": "mushin", "confidence": round(confidence, 4),
                "p_success": {o: round(float(s), 4)
                              for o, s in zip(layout.option_ids, success, strict=True)
                              if o != ABSTAIN}}
        if chosen == ABSTAIN or confidence < self.min_confidence:
            self.history.append({"step": len(self.history), "option_id": ABSTAIN,
                                 "parameters": None, "tool_results": []})
            return {"action": "abstain", "reason": "mushin: abstain or low confidence",
                    "mushin": info}
        option = next(o for o in record["options"] if o["option_id"] == chosen)
        schema = cards["skills"][option["card_sha256"]]["input_schema"]
        parameters: dict[str, Any] = {}
        for index, name, logits in prediction.numbers:
            if index == best:
                parameters[name] = round(number_value(logits, schema["properties"][name]), 6)
        for index, name, logits in prediction.enums:
            if index == best:
                parameters[name] = schema["properties"][name]["enum"][int(logits.argmax())]
        self.history.append({"step": len(self.history), "option_id": chosen,
                             "parameters": parameters, "tool_results": []})
        return {"action": "select", "skill_id": option["skill_id"],
                "skill_version": option["skill_version"], "parameters": parameters,
                "reason": f"mushin p={confidence:.2f}", "mushin": info}
