"""Decision records as model inputs and aligned targets."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from mushin.data.splits import load as load_splits
from mushin.data.splits import read_records
from mushin.model.backbone import Backbone, Encoded
from mushin.model.sequence import Layout, layout_from_tags, segments

IGNORE = -100


@dataclass
class Targets:
    soft: torch.Tensor  # [n_options]
    hard: int
    success: torch.Tensor  # [n_options], nan where unlabeled (abstain)
    numbers: list[int]  # bin per number slot, IGNORE if unlabeled
    enums: list[int]  # value index per enum slot, IGNORE if unlabeled


@dataclass
class Example:
    record_id: str
    encoded: Encoded
    layout: Layout
    targets: Targets


def load_dataset(data: Path, splits: Sequence[str] | None = None,
                 limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cards = json.loads((data / "cards.json").read_text())
    owner = load_splits(data) if splits is not None else {}
    records = []
    for record in read_records(data / "records.jsonl"):
        if splits is not None and owner.get(record["episode_id"]) not in splits:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records, cards


def targets_for(record: dict[str, Any], layout: Layout) -> Targets:
    labels = record["labels"]
    ids = layout.option_ids
    soft = torch.tensor([labels["choice_soft"][o] for o in ids], dtype=torch.float32)
    success = torch.tensor(
        [math.nan if labels["option_outcomes"][o]["success"] is None
         else float(labels["option_outcomes"][o]["success"]) for o in ids],
        dtype=torch.float32,
    )
    params = labels["params"]
    numbers = [params.get(ids[o], {}).get(name, {}).get("bin", IGNORE)
               for o, name, _ in layout.numbers]
    enums = [params.get(ids[o], {}).get(name, {}).get("index", IGNORE)
             for o, name, _ in layout.enums]
    return Targets(soft, ids.index(labels["choice"]), success, numbers, enums)


def make_example(record: dict[str, Any], cards: dict[str, Any], backbone: Backbone,
                 data: Path | None, rng: random.Random | None = None,
                 frames: bool = True) -> Example:
    images = None
    if frames and data is not None:
        images = {v: Image.open(data / p) for v, p in record["frames"].items() if p is not None}
    encoded = backbone.encode(segments(record, cards, images, rng))
    layout = layout_from_tags(encoded.tags)
    return Example(record["record_id"], encoded, layout, targets_for(record, layout))


class DecisionDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]], cards: dict[str, Any], backbone: Backbone,
                 data: Path | None, shuffle_options: bool, frames: bool = True,
                 seed: int = 0) -> None:
        self.records, self.cards, self.backbone = records, cards, backbone
        self.data, self.shuffle, self.frames = data, shuffle_options, frames
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Example:
        rng = random.Random(self.rng.random()) if self.shuffle else None
        return make_example(self.records[index], self.cards, self.backbone, self.data, rng,
                            self.frames)


def collate(backbone: Backbone):
    def run(examples: list[Example]) -> tuple[dict[str, Any], list[Example]]:
        return backbone.collate([e.encoded for e in examples]), examples

    return run
