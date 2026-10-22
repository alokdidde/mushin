"""Backbones: a pretrained vision-language model with its LM head removed, or a tiny transformer
for CPU tests.

Both turn ``Segment`` lists (``mushin.model.sequence``) into one token sequence per decision and
return the last hidden states; the heads read them at the tagged positions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mushin.model.sequence import Segment


@dataclass
class Encoded:
    input_ids: list[int]
    image_mask: list[int]
    images: list[Any] = field(default_factory=list)
    tags: list[tuple[tuple[Any, ...], int]] = field(default_factory=list)


class Backbone(nn.Module):
    hidden_size: int

    def encode_text(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_image(self, image: Any) -> tuple[list[int], Any]:
        """Placeholder token IDs for one image and its preprocessed payload."""
        raise NotImplementedError

    def encode(self, segments: Sequence[Segment]) -> Encoded:
        out = Encoded([], [])
        for segment in segments:
            if segment.kind == "image":
                ids, payload = self.encode_image(segment.value)
                out.images.append(payload)
                out.image_mask += [int(i == self.image_token_id) for i in ids]
            else:
                ids = self.encode_text(segment.value)
                out.image_mask += [0] * len(ids)
            out.input_ids += ids
            if segment.tag is not None:
                out.tags.append((segment.tag, len(out.input_ids) - 1))
        return out

    image_token_id: int = -1

    def collate(self, batch: Sequence[Encoded]) -> dict[str, Any]:
        width = max(len(e.input_ids) for e in batch)
        ids = torch.full((len(batch), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        image_mask = torch.zeros((len(batch), width), dtype=torch.int)
        for row, e in enumerate(batch):
            ids[row, : len(e.input_ids)] = torch.tensor(e.input_ids)
            mask[row, : len(e.input_ids)] = 1
            image_mask[row, : len(e.input_ids)] = torch.tensor(e.image_mask, dtype=torch.int)
        return {"input_ids": ids, "attention_mask": mask, "image_mask": image_mask,
                "images": [p for e in batch for p in e.images]}

    pad_token_id: int = 0

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def head_parameters_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class TinyBackbone(Backbone):
    """Byte tokens, a small causal transformer and a conv image encoder (4 tokens per image).
    For tests and CPU smoke runs; it knows no language, so it only fits what it is trained on."""

    PAD, IMAGE, OFFSET = 0, 1, 4
    IMAGE_TOKENS = 4

    def __init__(self, hidden_size: int = 64, layers: int = 2, heads: int = 4,
                 max_len: int = 4096) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.image_token_id = self.IMAGE
        self.pad_token_id = self.PAD
        self.embed = nn.Embedding(256 + self.OFFSET, hidden_size)
        self.position = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, heads, hidden_size * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=4), nn.GELU(), nn.Conv2d(16, hidden_size, 4, stride=4),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

    def encode_text(self, text: str) -> list[int]:
        return [b + self.OFFSET for b in text.encode()]

    def encode_image(self, image: Any) -> tuple[list[int], Any]:
        import numpy as np

        pixels = np.asarray(image.convert("RGB").resize((64, 64)), dtype="float32") / 255.0
        return [self.IMAGE] * self.IMAGE_TOKENS, torch.from_numpy(pixels).permute(2, 0, 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        ids = batch["input_ids"]
        hidden = self.embed(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        if batch["images"]:
            pixels = torch.stack(batch["images"]).to(hidden.device)
            features = self.vision(pixels).flatten(2).transpose(1, 2).reshape(-1, self.hidden_size)
            hidden = hidden.masked_scatter(
                batch["image_mask"].bool().unsqueeze(-1).to(hidden.device),
                features.to(hidden.dtype),
            )
        causal = nn.Transformer.generate_square_subsequent_mask(ids.shape[1], device=ids.device)
        padding = (batch["attention_mask"] == 0).to(ids.device)
        causal = causal.to(hidden.dtype)
        padding = torch.zeros_like(padding, dtype=hidden.dtype).masked_fill(padding, float("-inf"))
        return self.encoder(hidden, mask=causal, src_key_padding_mask=padding, is_causal=True)


class VisionLanguageBackbone(Backbone):
    """A Qwen3.5-style vision-language model (``Qwen3_5Model``: vision tower, merger and language
    model) without its LM head. ``name`` is a Hugging Face model ID or local path; ``config``
    builds a randomly initialised model instead (tests)."""

    def __init__(self, name: str | None = None, *, config: Any = None, tokenizer: Any = None,
                 image_processor: Any = None, dtype: torch.dtype = torch.float32,
                 attn_implementation: str | None = None) -> None:
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        if name is not None:
            processor = AutoProcessor.from_pretrained(name)
            tokenizer = tokenizer or processor.tokenizer
            image_processor = image_processor or processor.image_processor
            kwargs = {"dtype": dtype}
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation
            full = AutoModel.from_pretrained(name, **kwargs)
            config = full.config
        else:
            full = AutoModel.from_config(config)
        # AutoModel gives the base model (no LM head); a generation wrapper is unwrapped.
        self.model = getattr(full, "model", full) if hasattr(full, "lm_head") else full
        self.config = config
        text = getattr(config, "text_config", config)
        self.hidden_size = text.hidden_size
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = config.image_token_id
        self.vision_start = config.vision_start_token_id
        self.vision_end = config.vision_end_token_id
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = pad if pad is not None else 0
        vision = config.vision_config
        self.merge = vision.spatial_merge_size

    def encode_text(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode_plain"):
            return self.tokenizer.encode_plain(text)
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_image(self, image: Any) -> tuple[list[int], Any]:
        out = self.image_processor(images=[image.convert("RGB")], return_tensors="pt")
        grid = out["image_grid_thw"][0]
        count = int(grid.prod()) // (self.merge**2)
        ids = [self.vision_start] + [self.image_token_id] * count + [self.vision_end]
        return ids, (out["pixel_values"], grid)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        kwargs: dict[str, Any] = {}
        if batch["images"]:
            dtype = next(self.model.visual.parameters()).dtype
            kwargs["pixel_values"] = torch.cat([p for p, _ in batch["images"]]).to(device, dtype)
            kwargs["image_grid_thw"] = torch.stack([g for _, g in batch["images"]]).to(device)
            kwargs["mm_token_type_ids"] = batch["image_mask"].to(device)
        out = self.model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
            **kwargs,
        )
        return out.last_hidden_state

    def lora(self, rank: int, alpha: int, dropout: float, targets: Sequence[str]) -> None:
        """LoRA on the language model, and the vision merger (connector) trained in full."""
        from peft import LoraConfig, inject_adapter_in_model

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                            target_modules=list(targets))
        inject_adapter_in_model(config, self.model.language_model)
        merger = getattr(self.model.visual, "merger", None)
        if merger is not None:
            for parameter in merger.parameters():
                parameter.requires_grad_(True)


class ByteTokenizer:
    """Stand-in tokenizer for randomly initialised test models (bytes 0..255 shifted by 4)."""

    pad_token_id = 0

    def encode_plain(self, text: str) -> list[int]:
        return [b + 4 for b in text.encode()]


def tiny_qwen35_config() -> Any:
    """A few-layer Qwen3.5 with the real layer types (linear + full attention) and vision tower,
    small enough to train on a CPU."""
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config={
            "vocab_size": 512, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "max_position_embeddings": 8192,
        },
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "num_position_embeddings": 256,
        },
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
    )


def build_backbone(spec: dict[str, Any]) -> Backbone:
    kind = spec.get("kind", "vlm")
    if kind == "tiny":
        return TinyBackbone(**{k: v for k, v in spec.items() if k != "kind"})
    if kind == "tiny_qwen35":
        from transformers import Qwen2VLImageProcessor

        processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2,
                                          min_pixels=32 * 32, max_pixels=64 * 64)
        return VisionLanguageBackbone(config=tiny_qwen35_config(), tokenizer=ByteTokenizer(),
                                      image_processor=processor)
    dtype = getattr(torch, spec.get("dtype", "bfloat16"))
    return VisionLanguageBackbone(spec["name"], dtype=dtype,
                                  attn_implementation=spec.get("attn_implementation"))


def scaled_dot(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return (keys @ query.unsqueeze(-1)).squeeze(-1) / math.sqrt(query.shape[-1])
