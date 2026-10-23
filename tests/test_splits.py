import json
import shutil

import pytest

from mushin.data.splits import build, load, split_of
from mushin.data.verify import check_records

CONFIG = {"held_out_bodies": ["go1"], "held_out_protocols": ["handover-v1"],
          "fractions": {"train": 0.85, "val": 0.05, "calib": 0.05, "test": 0.05}}


def test_held_out_bodies_and_protocols():
    assert split_of("go1", "pick-place-v1", "mushin-train-x", CONFIG) == "test_body"
    assert split_of("aloha", "handover-v1", "mushin-train-x", CONFIG) == "test_protocol"


def test_hash_split_is_deterministic_and_roughly_proportional():
    seeds = [f"mushin-train-m0-so101-{i:06d}" for i in range(4000)]
    first = [split_of("so101", "pick-place-v1", s, CONFIG) for s in seeds]
    assert first == [split_of("so101", "pick-place-v1", s, CONFIG) for s in seeds]
    assert 0.82 < first.count("train") / len(first) < 0.88
    assert set(first) == {"train", "val", "calib", "test"}


def test_manifest_keeps_episodes_whole_and_pins_records(tmp_path, small_dir):
    shutil.copytree(small_dir, tmp_path / "d")
    manifest = build(tmp_path / "d", CONFIG)
    owners = {}
    for name, ids in manifest["episodes"].items():
        for episode in ids:
            assert owners.setdefault(episode, name) == name
    (tmp_path / "d" / "splits.json").write_text(json.dumps(manifest))
    assert set(load(tmp_path / "d").values()) <= set(manifest["episodes"])
    with (tmp_path / "d" / "records.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError):
        load(tmp_path / "d")


def test_verify_accepts_the_fixture(small_dir):
    records, problems = check_records(small_dir)
    assert records and problems == []


def test_finalize_rebuilds_a_cut_short_run(tmp_path, small_dir):
    from mushin.data.finalize import finalize
    from mushin.data.splits import read_records

    shutil.copytree(small_dir, tmp_path / "d")
    data = tmp_path / "d"
    lines = (data / "records.jsonl").read_text().splitlines(keepends=True)
    (data / "records.jsonl").unlink()
    (data / "records.partial.jsonl").write_text("".join(reversed(lines)) + '{"torn')
    assert finalize(data) == len(lines)
    keys = [(r["episode_id"], r["step"]) for r in read_records(data / "records.jsonl")]
    assert keys == sorted(keys)
    assert json.loads((data / "stats.json").read_text())["complete"] is False
    build(data, CONFIG)
