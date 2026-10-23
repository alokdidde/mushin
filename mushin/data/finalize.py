"""Turn the records of a generator run that was cut short into a normal data directory.

    python -m mushin.data.finalize DATA_DIR --config configs/data_m0.yaml

A run stopped by its deadline leaves only ``records.partial.jsonl`` (episodes in finishing order,
maybe a torn last line). This writes ``records.jsonl`` sorted by episode and step, then the
splits. A directory that already has ``records.jsonl`` is left as it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from mushin.data import splits


def finalize(data: Path) -> int:
    """Write ``records.jsonl`` from ``records.partial.jsonl``; returns the decision count."""
    records = []
    with (data / "records.partial.jsonl").open() as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # torn last line from the kill at the deadline
    records.sort(key=lambda r: (r["episode_id"], r["step"]))
    digest = hashlib.sha256()
    with (data / "records.jsonl").open("w") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True) + "\n"
            digest.update(line.encode())
            handle.write(line)
    stats_path = data / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats.update(complete=False, episodes=len({r["episode_id"] for r in records}),
                 decisions=len(records), records_sha256=digest.hexdigest())
    stats_path.write_text(json.dumps(stats, indent=2))
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.data.finalize", description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    options = parser.parse_args(argv)
    if (options.data / "records.jsonl").exists():
        print("records.jsonl already present; nothing to do")
        return 0
    count = finalize(options.data)
    config = yaml.safe_load(options.config.read_text())["splits"]
    (options.data / "splits.json").write_text(
        json.dumps(splits.build(options.data, config), indent=1))
    print(f"wrote {count} decisions and their splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
