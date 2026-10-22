"""Typed decision heads read at the tagged positions of one shared-state pass.

- choice: dot product between the projected decision token and each option's last token
  (abstain included), divided by a fitted temperature;
- yes/no: P(success) per skill option from its last token;
- value bins: a distribution over ``NUMBER_BINS`` for each number parameter slot;
- enums: dot product between a parameter query and each value's slot.

The scores do not depend on the order options are listed in except through the backbone's
context, and options are shuffled in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mushin.data.labels import NUMBER_BINS
from mushin.model.backbone import Backbone, scaled_dot
from mushin.model.sequence import Layout


@dataclass
class Prediction:
    option_ids: list[str]
    choice: torch.Tensor  # [n_options] logits
    success: torch.Tensor  # [n_options] logits (the abstain entry is unused)
    numbers: list[tuple[int, str, torch.Tensor]]  # (option index, name, [bins] logits)
    enums: list[tuple[int, str, torch.Tensor]]  # (option index, name, [values] logits)


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.decide = nn.Linear(hidden_size, width)
        self.option = nn.Linear(hidden_size, width)
        self.success = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.Linear(width, 1))
        self.number = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(),
                                    nn.Linear(width, NUMBER_BINS))
        self.enum_query = nn.Linear(hidden_size, width)
        self.enum_key = nn.Linear(hidden_size, width)
        # Fitted on the calibration split (mushin.train.calibrate); 1.0 until then.
        self.register_buffer("choice_temperature", torch.ones(()))
        self.register_buffer("success_temperature", torch.ones(()))

    def forward(self, hidden: torch.Tensor, layout: Layout) -> Prediction:
        h = self.norm(hidden.float())
        options = h[layout.options]
        choice = scaled_dot(self.decide(h[layout.decide]), self.option(options))
        success = self.success(options).squeeze(-1)
        numbers = [(o, name, self.number(h[p])) for o, name, p in layout.numbers]
        enums = [
            (o, name, scaled_dot(self.enum_query(h[layout.options[o]]), self.enum_key(h[ps])))
            for o, name, ps in layout.enums
        ]
        return Prediction(layout.option_ids, choice / self.choice_temperature,
                          success / self.success_temperature, numbers, enums)


class MushinModel(nn.Module):
    def __init__(self, backbone: Backbone, head_width: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = DecisionHead(backbone.hidden_size, head_width)

    def forward(self, batch: dict[str, Any], layouts: list[Layout]) -> list[Prediction]:
        hidden = self.backbone(batch)
        return [self.head(hidden[row], layout) for row, layout in enumerate(layouts)]

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Only what training changes (head, LoRA, connector), for small checkpoints."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        names |= {n for n, _ in self.named_buffers() if n.startswith("head.")}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}
