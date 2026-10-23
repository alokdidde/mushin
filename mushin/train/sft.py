"""Supervised fine-tuning of the decision heads (``stage: warmup``, backbone frozen) and then of
LoRA adapters plus the vision connector (``stage: lora``).

    python -m mushin.train.sft --config configs/sft_tiny.yaml [--set train.max_steps=50 ...]

Writes ``<out>/checkpoint.pt`` (trainable weights + config), ``<out>/log.jsonl`` and
``<out>/summary.json`` with validation metrics. A checkpoint is also written every
``train.checkpoint_minutes`` and training resumes from it when restarted with the same ``out``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from mushin.eval.metrics import choice_metrics, success_metrics
from mushin.model.head import MushinModel
from mushin.train import checkpoint
from mushin.train.data import IGNORE, DecisionDataset, collate, load_dataset
from mushin.train.losses import DEFAULT_WEIGHTS, decision_loss


def load_config(path: Path, overrides: list[str] = ()) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        node = config
        *parents, leaf = key.split(".")
        for name in parents:
            node = node.setdefault(name, {})
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):  # YAML 1.1 reads "2e-3" as a string
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[leaf] = parsed
    return config


def device_of(config: dict[str, Any]) -> str:
    wanted = config.get("train", {}).get("device", "auto")
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return wanted


@torch.no_grad()
def evaluate(model: MushinModel, loader: Any, device: str) -> dict[str, Any]:
    model.eval()
    probs, hard, soft, s_probs, s_labels = [], [], [], [], []
    params_right, params_total = 0, 0
    for batch, examples in loader:
        with autocast(device):
            predictions = model(batch, [e.layout for e in examples])
        for p, e in zip(predictions, examples, strict=True):
            t = e.targets
            probs.append(torch.softmax(p.choice.float(), -1).tolist())
            hard.append(t.hard)
            soft.append(t.soft.tolist())
            labelled = ~torch.isnan(t.success)
            s_probs += torch.sigmoid(p.success.float().cpu()[labelled]).tolist()
            s_labels += t.success[labelled].tolist()
            for (_, _, logits), target in [*zip(p.numbers, t.numbers, strict=True),
                                           *zip(p.enums, t.enums, strict=True)]:
                if target != IGNORE:
                    params_total += 1
                    params_right += int(int(logits.argmax()) == target)
    model.train()
    return {
        "choice": choice_metrics(probs, hard, soft),
        "success": success_metrics(s_probs, s_labels),
        "params_accuracy": params_right / params_total if params_total else math.nan,
    }


def autocast(device: str) -> Any:
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def loaders(config: dict[str, Any], model: MushinModel) -> tuple[Any, Any]:
    data_cfg = config["data"]
    data = Path(data_cfg["dir"])
    frames = data_cfg.get("frames", True)
    train_records, cards = load_dataset(data, data_cfg.get("train_splits"), data_cfg.get("limit"))
    eval_splits = data_cfg.get("eval_splits")
    if eval_splits == "train":
        eval_records = train_records
    else:
        eval_records, _ = load_dataset(data, eval_splits, data_cfg.get("eval_limit"))
    batch = config["train"].get("batch_size", 8)
    workers = config["train"].get("workers", 0)
    make = collate(model.backbone)
    train = torch.utils.data.DataLoader(
        DecisionDataset(train_records, cards, model.backbone, data, True, frames,
                        config["train"].get("seed", 0)),
        batch_size=batch, shuffle=True, collate_fn=make, num_workers=workers,
        generator=torch.Generator().manual_seed(config["train"].get("seed", 0)),
    )
    evaluation = torch.utils.data.DataLoader(
        DecisionDataset(eval_records, cards, model.backbone, data, False, frames),
        batch_size=batch, shuffle=False, collate_fn=make, num_workers=workers,
    )
    return train, evaluation


def train(config: dict[str, Any]) -> dict[str, Any]:
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    device = device_of(config)
    model = checkpoint.build_model(config)
    if config.get("init"):
        checkpoint.restore(model, Path(config["init"]))
    model.to(device).train()
    train_loader, eval_loader = loaders(config, model)
    t = config["train"]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": head, "lr": t.get("head_lr", t.get("lr", 1e-3))}]
    if rest:
        groups.append({"params": rest, "lr": t.get("lr", 1e-4)})
    optimizer = torch.optim.AdamW(groups, weight_decay=t.get("weight_decay", 0.0))
    accumulate = t.get("grad_accum", 1)
    max_steps = t.get("max_steps") or t.get("epochs", 1) * math.ceil(len(train_loader) / accumulate)
    warmup = t.get("warmup_steps", 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
        1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max_steps))))
    weights = {**DEFAULT_WEIGHTS, **config.get("loss", {})}
    state_path = out / "checkpoint.pt"
    step = 0
    if state_path.exists():
        saved = checkpoint.restore(model, state_path)
        if "optimizer" in saved and saved.get("step", 0) < max_steps:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
        step = saved.get("step", 0)
    log = (out / "log.jsonl").open("a")
    started = last_save = time.monotonic()
    every = t.get("checkpoint_minutes", 15) * 60
    eval_every = t.get("eval_every", 0)
    micro = 0
    while step < max_steps:
        for batch, examples in train_loader:
            with autocast(device):
                predictions = model(batch, [e.layout for e in examples])
            loss, parts = decision_loss(predictions, [e.targets for e in examples], weights)
            (loss / accumulate).backward()
            micro += 1
            if micro % accumulate:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],
                                           t.get("clip", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % t.get("log_every", 10) == 0 or step == 1:
                entry = {"step": step, "loss": float(loss), **parts,
                         "lr": scheduler.get_last_lr()[0],
                         "elapsed_s": round(time.monotonic() - started, 1)}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
            if eval_every and step % eval_every == 0:
                metrics = evaluate(model, eval_loader, device)
                log.write(json.dumps({"step": step, "eval": metrics}) + "\n")
                log.flush()
            if time.monotonic() - last_save > every:
                checkpoint.save(model, state_path, config, {
                    "step": step, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()})
                last_save = time.monotonic()
            if step >= max_steps:
                break
    metrics = evaluate(model, eval_loader, device)
    checkpoint.save(model, state_path, config, {"step": step})
    summary = {"steps": step, "train_s": round(time.monotonic() - started, 1),
               "train_records": len(train_loader.dataset),
               "eval_records": len(eval_loader.dataset), "eval": metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["eval"], indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.train.sft", description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=1e-4")
    options = parser.parse_args(argv)
    train(load_config(options.config, options.set))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
