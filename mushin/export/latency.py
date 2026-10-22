"""Per-decision latency of one shared-state pass.

    python -m mushin.export.latency --data DIR (--checkpoint runs/x/checkpoint.pt |
        --backbone '{"kind": "vlm", "name": "Qwen/Qwen3.5-0.8B", "dtype": "float16"}')
        [--records 20] [--repeat 5] [--device cuda] [--out latency.json]

Times preprocessing (tokenization and image processing) and the model forward separately for
real records with their frames, at batch size 1, after warm-up, and reports median and p90 with
the sequence length, device, dtype and peak memory, as the evaluation plan asks.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from mushin.model.backbone import build_backbone
from mushin.model.head import MushinModel
from mushin.train import checkpoint
from mushin.train.data import load_dataset, make_example


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.export.latency", description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", type=json.loads)
    parser.add_argument("--records", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    options = parser.parse_args(argv)
    if options.checkpoint:
        model = checkpoint.load_for_inference(options.checkpoint, options.device)
    elif options.backbone:
        model = MushinModel(build_backbone(options.backbone)).to(options.device).eval()
    else:
        parser.error("give --checkpoint or --backbone")
    records, cards = load_dataset(options.data, None, options.records)
    prep, forward, lengths = [], [], []
    if options.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, record in enumerate(records):
            for attempt in range(options.repeat + (options.warmup if index == 0 else 0)):
                started = time.perf_counter()
                example = make_example(record, cards, model.backbone, options.data)
                batch = model.backbone.collate([example.encoded])
                prepared = time.perf_counter()
                model(batch, [example.layout])
                sync(options.device)
                done = time.perf_counter()
                if index == 0 and attempt < options.warmup:
                    continue
                prep.append((prepared - started) * 1000)
                forward.append((done - prepared) * 1000)
                lengths.append(len(example.encoded.input_ids))
    report = {
        "device": options.device,
        "device_name": torch.cuda.get_device_name() if options.device.startswith("cuda")
        else platform.processor() or platform.machine(),
        "dtype": str(next(model.parameters()).dtype),
        "records": len(records),
        "tokens_median": statistics.median(lengths),
        "image_tokens_median": statistics.median(
            sum(make_example(r, cards, model.backbone, options.data).encoded.image_mask)
            for r in records[:5]),
        "forward_ms": {"median": statistics.median(forward), "p90": percentile(forward, 0.9)},
        "preprocess_ms": {"median": statistics.median(prep), "p90": percentile(prep, 0.9)},
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20
        if options.device.startswith("cuda") else None,
    }
    text = json.dumps(report, indent=1)
    if options.out:
        options.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
