"""Model, losses and calibration on CPU with tiny backbones (no downloads)."""

import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mushin.model.backbone import build_backbone  # noqa: E402
from mushin.model.head import MushinModel  # noqa: E402
from mushin.model.sequence import layout_from_tags, segments  # noqa: E402
from mushin.schema import ABSTAIN  # noqa: E402
from mushin.train import calibrate, sft  # noqa: E402
from mushin.train.data import make_example  # noqa: E402
from mushin.train.losses import decision_loss  # noqa: E402


def with_frames(tmp_path, small_dir, small_records):
    """A copy of the fixture with a synthetic scene frame on every record."""
    from PIL import Image

    data = tmp_path / "data"
    shutil.copytree(small_dir, data)
    (data / "images").mkdir()
    Image.new("RGB", (224, 224), (40, 120, 60)).save(data / "images" / "f.jpg")
    with (data / "records.jsonl").open("w") as handle:
        for record in small_records:
            record = dict(record, frames={"scene": "images/f.jpg", "wrist": None})
            handle.write(json.dumps(record) + "\n")
    return data


def test_layout_marks_every_option_and_parameter(small_records):
    cards = json.loads(
        (__import__("pathlib").Path(__file__).parent / "fixtures/m0_small/cards.json").read_text()
    )
    record = small_records[0]
    segs = segments(record, cards)
    tags = [(s.tag, i) for i, s in enumerate(segs) if s.tag]
    layout = layout_from_tags(tags)
    assert sorted(layout.option_ids) == sorted(o["option_id"] for o in record["options"])
    assert ABSTAIN in layout.option_ids and layout.decide == len(segs) - 1


@pytest.mark.parametrize("kind", ["tiny", "tiny_qwen35"])
def test_forward_backward_with_images(tmp_path, small_dir, small_records, kind):
    data = with_frames(tmp_path, small_dir, small_records)
    cards = json.loads((data / "cards.json").read_text())
    backbone = build_backbone({"kind": kind})
    model = MushinModel(backbone, 32)
    records = [json.loads(line) for line in (data / "records.jsonl").read_text().splitlines()]
    examples = [make_example(r, cards, backbone, data) for r in records[:3]]
    assert all(sum(e.encoded.image_mask) > 0 for e in examples)
    batch = backbone.collate([e.encoded for e in examples])
    predictions = model(batch, [e.layout for e in examples])
    loss, parts = decision_loss(predictions, [e.targets for e in examples])
    loss.backward()
    assert torch.isfinite(loss) and parts["choice"] > 0
    assert all(p.choice.shape[0] == len(e.layout.option_ids)
               for p, e in zip(predictions, examples, strict=True))


def test_sft_learns_and_calibrates(tmp_path, small_dir, small_records):
    data = with_frames(tmp_path, small_dir, small_records)
    from mushin.data.splits import build

    config = {"held_out_bodies": [], "held_out_protocols": [],
              "fractions": {"train": 0.0, "val": 0.0, "calib": 1.0, "test": 0.0}}
    (data / "splits.json").write_text(json.dumps(build(data, config)))
    out = tmp_path / "run"
    config = {
        "stage": "lora", "backbone": {"kind": "tiny", "hidden_size": 32, "layers": 1},
        "head_width": 32,
        "data": {"dir": str(data), "train_splits": None, "eval_splits": "train"},
        "train": {"device": "cpu", "max_steps": 60, "batch_size": 8, "lr": 3e-3,
                  "head_lr": 3e-3, "log_every": 1000, "seed": 0},
        "out": str(out),
    }
    first = sft.evaluate(*_model_and_loader(config))
    summary = sft.train(config)
    assert summary["eval"]["choice"]["soft_nll"] < first["choice"]["soft_nll"]
    assert calibrate.main(["--checkpoint", str(out / "checkpoint.pt"), "--data", str(data),
                           "--device", "cpu"]) == 0
    report = json.loads((out / "calibration.json").read_text())
    assert report["after"]["choice"]["nll"] <= report["before"]["choice"]["nll"] + 1e-6


def _model_and_loader(config):
    from mushin.train import checkpoint

    model = checkpoint.build_model(config)
    _, loader = sft.loaders(config, model)
    return model, loader, "cpu"
