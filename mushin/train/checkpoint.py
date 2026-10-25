"""Build a model from a stage config and save or restore its trained weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mushin.model.backbone import VisionLanguageBackbone, build_backbone
from mushin.model.head import MushinModel


def build_model(config: dict[str, Any]) -> MushinModel:
    torch.manual_seed(config.get("train", {}).get("seed", 0))
    backbone = build_backbone(dict(config["backbone"]))
    model = MushinModel(backbone, config.get("head_width", 256))
    if config.get("stage") == "lora":
        lora = config.get("lora", {})
        if isinstance(backbone, VisionLanguageBackbone):
            backbone.lora(lora.get("rank", 16), lora.get("alpha", 32), lora.get("dropout", 0.05),
                          lora.get("targets", DEFAULT_LORA_TARGETS))
        # The tiny backbone has no adapters: it is trained in full in this stage.
    else:
        backbone.head_parameters_only()
    return model


#: Attention, gated-delta and MLP projections of Qwen3.5's language model.
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj", "in_proj_qkv", "in_proj_z", "out_proj")


def save(model: MushinModel, path: Path, config: dict[str, Any], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"config": config, "state": model.trainable_state(), **(extra or {})}, tmp)
    tmp.replace(path)


def restore(model: MushinModel, path: Path) -> dict[str, Any]:
    """Load weights saved by ``save`` (from this stage or an earlier one) into ``model``."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    missing = set(saved["state"]) - set(model.state_dict())
    if missing:
        raise KeyError(f"checkpoint has weights the model lacks: {sorted(missing)[:5]}")
    model.load_state_dict(saved["state"], strict=False)
    return saved


def load_for_inference(path: Path, device: str = "cpu") -> MushinModel:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["config"])
    model.load_state_dict(saved["state"], strict=False)
    return model.to(device).eval()


#: Files of a released model (``python -m mushin.export.release``).
RELEASE_CONFIG, RELEASE_WEIGHTS = "mushin_config.json", "model.safetensors"


def load_pretrained(source: str | Path, device: str = "cpu",
                    revision: str | None = None) -> MushinModel:
    """Load a released model from a local directory or a Hugging Face repo id.

    The release holds the stage config and the trained weights only (LoRA, vision connector,
    heads and calibrated temperatures); the base backbone named in the config is downloaded by
    transformers as usual.
    """
    from safetensors.torch import load_file

    folder = Path(source)
    if not folder.is_dir():
        from huggingface_hub import snapshot_download

        folder = Path(snapshot_download(str(source), revision=revision,
                                        allow_patterns=[RELEASE_CONFIG, RELEASE_WEIGHTS]))
    config = json.loads((folder / RELEASE_CONFIG).read_text())
    model = build_model(config)
    state = load_file(str(folder / RELEASE_WEIGHTS))
    missing = set(state) - set(model.state_dict())
    if missing:
        raise KeyError(f"release has weights the model lacks: {sorted(missing)[:5]}")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()
