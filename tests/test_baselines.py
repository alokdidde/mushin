import json
import shutil

from mushin.data.splits import build
from mushin.eval import baselines


def test_baselines_score_every_held_out_split(tmp_path, small_dir):
    data = tmp_path / "d"
    shutil.copytree(small_dir, data)
    config = {"held_out_bodies": ["go1"], "held_out_protocols": [],
              "fractions": {"train": 0.6, "val": 0.4}}
    (data / "splits.json").write_text(json.dumps(build(data, config)))
    results = baselines.run(data)
    assert set(results["baselines"]) == {"rule", "frequency", "linear"}
    for splits in results["baselines"].values():
        assert "train" not in splits and "test_body" in splits
        for metrics in splits.values():
            assert 0.0 <= metrics["choice"]["accuracy"] <= 1.0


def test_config_overrides_parse_numbers(tmp_path):
    from mushin.train.sft import load_config

    path = tmp_path / "c.yaml"
    path.write_text("train: {lr: 1.0e-3}\n")
    config = load_config(path, ["train.lr=2e-3", "train.epochs=3", "out=runs/x"])
    assert config["train"] == {"lr": 0.002, "epochs": 3} and config["out"] == "runs/x"
