"""Turn a training checkpoint into a release folder for the Hugging Face Hub.

    python -m mushin.export.release runs/qwen35_08b_lora/checkpoint.pt release/

Writes ``model.safetensors`` (the trained weights only: LoRA, vision connector, heads and the
calibrated temperatures) and ``mushin_config.json`` (the stage config plus the training step), the
two files ``mushin.train.checkpoint.load_pretrained`` reads. Safetensors cannot run code on load,
unlike the pickled ``checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mushin.train.checkpoint import RELEASE_CONFIG, RELEASE_WEIGHTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.export.release", description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    options = parser.parse_args(argv)
    from safetensors.torch import save_file

    saved = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    options.out.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in saved["state"].items()}
    save_file(state, str(options.out / RELEASE_WEIGHTS), metadata={"format": "pt"})
    config = {**saved["config"], "step": saved.get("step", 0)}
    (options.out / RELEASE_CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"tensors": len(state), "parameters": sum(v.numel() for v in state.values()),
                      "step": config["step"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
