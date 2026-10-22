"""Fit the choice and success temperatures on the calibration split.

    python -m mushin.train.calibrate --checkpoint runs/x/checkpoint.pt --data DIR [--split calib]

Temperatures are fitted by minimising NLL (LBFGS on log T) with the model frozen; they are stored
in the checkpoint's head buffers, and the metrics before and after go to ``calibration.json``
next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mushin.train import checkpoint
from mushin.train.data import DecisionDataset, collate, load_dataset
from mushin.train.sft import autocast, evaluate


def fit_temperature(logits: list[torch.Tensor], loss_fn) -> float:
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = sum(loss_fn(z / log_t.exp(), i) for i, z in enumerate(logits)) / len(logits)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.train.calibrate", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="calib")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    options = parser.parse_args(argv)
    saved = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = checkpoint.build_model(config)
    checkpoint.restore(model, options.checkpoint)
    model.to(options.device).eval()
    records, cards = load_dataset(options.data, [options.split])
    loader = torch.utils.data.DataLoader(
        DecisionDataset(records, cards, model.backbone, options.data, False,
                        config["data"].get("frames", True)),
        batch_size=config["train"].get("batch_size", 8), collate_fn=collate(model.backbone))
    model.head.choice_temperature.fill_(1.0)
    model.head.success_temperature.fill_(1.0)
    before = evaluate(model, loader, options.device)
    choice, success, hard, labels = [], [], [], []
    with torch.no_grad():
        for batch, examples in loader:
            with autocast(options.device):
                predictions = model(batch, [e.layout for e in examples])
            for p, e in zip(predictions, examples, strict=True):
                choice.append(p.choice.float().cpu())
                hard.append(e.targets.hard)
                mask = ~torch.isnan(e.targets.success)
                if mask.any():
                    success.append(p.success.float().cpu()[mask])
                    labels.append(e.targets.success[mask])
    t_choice = fit_temperature(choice, lambda z, i: F.cross_entropy(
        z.unsqueeze(0), torch.tensor([hard[i]])))
    t_success = fit_temperature(success, lambda z, i: F.binary_cross_entropy_with_logits(
        z, labels[i])) if success else 1.0
    model.head.choice_temperature.fill_(t_choice)
    model.head.success_temperature.fill_(t_success)
    after = evaluate(model, loader, options.device)
    checkpoint.save(model, options.checkpoint, config, {"step": saved.get("step", 0)})
    report = {"split": options.split, "records": len(records), "choice_temperature": t_choice,
              "success_temperature": t_success, "before": before, "after": after}
    (options.checkpoint.parent / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("choice_temperature", "success_temperature")}))
    print("choice ECE", round(before["choice"]["ece"], 4), "->", round(after["choice"]["ece"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
