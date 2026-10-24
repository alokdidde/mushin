"""Turn the output of a generator run, cut short or downloaded from the Hub, into a normal
data directory.

    python -m mushin.data.finalize DATA_DIR --config configs/data_m0.yaml

A run stopped by its deadline leaves only ``records.partial.jsonl`` (episodes in finishing order,
maybe a torn last line); a run whose pod died may have left only the ``shards/`` it uploaded.
This unpacks the shards' frames into ``images/`` when they are missing, writes ``records.jsonl``
sorted by episode and step from the best source there is, then the splits. ``records.jsonl``
that already exists is kept.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import yaml

from mushin.data import splits


def write_sorted(source: Path, target: Path) -> tuple[int, str]:
    """Copy records from ``source`` to ``target`` sorted by episode and step, dropping a torn
    last line; returns the count and the SHA-256 of what was written."""
    keyed = []
    with source.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break  # torn last line from a kill mid-write
            keyed.append(((record["episode_id"], record["step"]), line))
    keyed.sort(key=lambda kv: kv[0])
    digest = hashlib.sha256()
    with target.open("w") as handle:
        for _, line in keyed:
            digest.update(line.encode())
            handle.write(line)
    return len(keyed), digest.hexdigest()


def write_shard(out: Path, index: int, records: list[dict[str, Any]]) -> None:
    """``shards/sNNNNN/``: the records and a tar of their frames, then a ``DONE`` marker that
    tells ``scripts/pod_job.sh`` the shard is ready to upload. Shards keep the Hub's per-folder
    file count small (it caps a folder at 10k files)."""
    folder = out / "shards" / f"s{index:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with tarfile.open(folder / "images.tar", "w") as tar:
        for record in records:
            for path in record["frames"].values():
                if path:
                    tar.add(out / path, arcname=path)
    (folder / "DONE").touch()


def _extract(path: Path, data: Path) -> None:
    with tarfile.open(path) as tar:
        tar.extractall(data, filter="data")


def unpack_shards(data: Path, workers: int = 8) -> int:
    """Extract every ``shards/*/images.tar`` into ``data``, several tars at a time, printing
    progress; returns the number of shards."""
    tars = sorted(data.glob("shards/*/images.tar"))
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max(1, workers)) as pool:
        jobs = [pool.submit(_extract, path, data) for path in tars]
        for done, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            job.result()
            if done % 10 == 0 or done == len(tars):
                print(f"unpacked {done}/{len(tars)} shards, {time.monotonic() - started:.0f} s",
                      flush=True)
    return len(tars)


def finalize(data: Path) -> int:
    """Write ``records.jsonl`` from ``records.partial.jsonl`` or, failing that, the shards'
    records; returns the decision count."""
    source = data / "records.partial.jsonl"
    if not source.exists():
        with source.open("w") as handle:
            for path in sorted(data.glob("shards/*/records.jsonl")):
                handle.write(path.read_text())
    count, digest = write_sorted(source, data / "records.jsonl")
    episodes = set()
    for record in splits.read_records(data / "records.jsonl"):
        episodes.add(record["episode_id"])
    stats_path = data / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats.update(complete=False, episodes=len(episodes), decisions=count,
                 records_sha256=digest)
    stats_path.write_text(json.dumps(stats, indent=2))
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.data.finalize", description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    options = parser.parse_args(argv)
    if not (options.data / "images").is_dir():
        print(f"unpacked {unpack_shards(options.data)} shards")
    if (options.data / "records.jsonl").exists():
        print("records.jsonl already present")
        return 0
    count = finalize(options.data)
    config = yaml.safe_load(options.config.read_text())["splits"]
    (options.data / "splits.json").write_text(
        json.dumps(splits.build(options.data, config), indent=1))
    print(f"wrote {count} decisions and their splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
