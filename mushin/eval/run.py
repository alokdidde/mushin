"""Evaluate a checkpoint on dataset splits.

    python -m mushin.eval.run --checkpoint runs/x/checkpoint.pt --data DIR
        [--splits val test test_body] [--out results.json]

Reports choice, P(success) and parameter metrics per split, plus per body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mushin.train import checkpoint
from mushin.train.data import DecisionDataset, collate, load_dataset
from mushin.train.sft import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.eval.run", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test", "test_body"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    options = parser.parse_args(argv)
    model = checkpoint.load_for_inference(options.checkpoint, options.device)
    config = torch.load(options.checkpoint, map_location="cpu", weights_only=False)["config"]
    frames = config["data"].get("frames", True)
    results = {}
    for split in options.splits:
        records, cards = load_dataset(options.data, [split])
        if not records:
            continue
        groups = {"all": records}
        for record in records:
            groups.setdefault(f"body:{record['body']}", []).append(record)
        results[split] = {}
        for name, group in groups.items():
            loader = torch.utils.data.DataLoader(
                DecisionDataset(group, cards, model.backbone, options.data, False, frames),
                batch_size=config["train"].get("batch_size", 8),
                collate_fn=collate(model.backbone))
            results[split][name] = evaluate(model, loader, options.device)
    text = json.dumps(results, indent=1)
    if options.out:
        options.out.write_text(text)
    print(json.dumps({s: {"accuracy": r["all"]["choice"]["accuracy"],
                          "ece": r["all"]["choice"]["ece"]} for s, r in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
