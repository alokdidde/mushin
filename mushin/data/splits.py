"""Frozen splits: whole held-out bodies and task packs, then a hash split of the remaining
episodes by seed. Every decision of an episode lands in the same split.

    python -m mushin.data.splits DATA_DIR --config configs/data_m0.yaml

writes ``DATA_DIR/splits.json`` with the config, the hash of ``records.jsonl`` and the episode IDs
of each split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "val", "calib", "test")


def split_of(body: str, protocol: str, seed: str, config: Mapping[str, Any]) -> str:
    if body in config.get("held_out_bodies", []):
        return "test_body"
    if protocol in config.get("held_out_protocols", []):
        return "test_protocol"
    fractions = config["fractions"]
    point = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big") / 2**64
    edge = 0.0
    for name in SPLITS:
        edge += fractions.get(name, 0.0)
        if point < edge:
            return name
    return SPLITS[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build(data: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    episodes: dict[str, set[str]] = {}
    decisions: Counter[str] = Counter()
    for record in read_records(data / "records.jsonl"):
        name = split_of(record["body"], record["protocol"], record["seed"], config)
        episodes.setdefault(name, set()).add(record["episode_id"])
        decisions[name] += 1
    return {
        "format": "mushin.splits/1.0",
        "config": dict(config),
        "records_sha256": file_sha256(data / "records.jsonl"),
        "decisions": dict(sorted(decisions.items())),
        "episodes": {name: sorted(ids) for name, ids in sorted(episodes.items())},
    }


def load(data: Path) -> dict[str, str]:
    """Episode ID -> split, checked against the records file it was built from."""
    manifest = json.loads((data / "splits.json").read_text())
    if manifest["records_sha256"] != file_sha256(data / "records.jsonl"):
        raise ValueError("splits.json does not match records.jsonl; rebuild the splits")
    return {e: name for name, ids in manifest["episodes"].items() for e in ids}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.data.splits", description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    options = parser.parse_args(argv)
    config = yaml.safe_load(options.config.read_text())["splits"]
    manifest = build(options.data, config)
    (options.data / "splits.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["decisions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
