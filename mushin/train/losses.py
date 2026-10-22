"""SFT objective: soft cross-entropy on the choice plus a Brier term, BCE on P(success) per skill,
and cross-entropy on parameter bins and enum values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from mushin.model.head import Prediction
from mushin.train.data import IGNORE, Targets

DEFAULT_WEIGHTS = {"choice": 1.0, "brier": 0.1, "success": 1.0, "params": 0.5}


def decision_loss(predictions: Sequence[Prediction], targets: Sequence[Targets],
                  weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> tuple[torch.Tensor, dict]:
    parts = {k: [] for k in ("choice", "brier", "success", "params")}
    for p, t in zip(predictions, targets, strict=True):
        soft = t.soft.to(p.choice.device)
        log_probs = F.log_softmax(p.choice, -1)
        parts["choice"].append(-(soft * log_probs).sum())
        parts["brier"].append(((log_probs.exp() - soft) ** 2).sum())
        labelled = ~torch.isnan(t.success)
        if labelled.any():
            parts["success"].append(F.binary_cross_entropy_with_logits(
                p.success[labelled.to(p.success.device)], t.success[labelled].to(p.success.device)))
        slots = [*zip(p.numbers, t.numbers, strict=True), *zip(p.enums, t.enums, strict=True)]
        for (_, _, logits), target in slots:
            if target != IGNORE:
                wanted = torch.tensor([target], device=logits.device)
                parts["params"].append(F.cross_entropy(logits.unsqueeze(0), wanted))
    device = predictions[0].choice.device
    means = {k: torch.stack(v).mean() if v else torch.zeros((), device=device)
             for k, v in parts.items()}
    total = sum(weights.get(k, 0.0) * v for k, v in means.items())
    return total, {k: float(v.detach()) for k, v in means.items()}
