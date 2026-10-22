"""Check a generated dataset (the M0 gate).

    python -m mushin.data.verify DATA_DIR [--reproduce N] [--uzumaki DIR]

1. Every record is a valid ``mushin.decision/1.0`` record and every referenced card and frame
   exists.
2. Every seed is a training seed (``mushin-train-*``), so no uzumaki development or held-out case
   is in the data.
3. ``splits.json``, if present, was built from this ``records.jsonl``.
4. With ``--reproduce N``: N random option labels are re-derived in the simulator and must give
   the same evidence-bundle head hash and outcome.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from mushin.data import bodies as B
from mushin.data.splits import load, read_records
from mushin.schema import ABSTAIN, RecordError, validate_record


def check_records(data: Path) -> tuple[list[dict[str, Any]], list[str]]:
    cards = json.loads((data / "cards.json").read_text())
    problems: list[str] = []
    records = []
    for record in read_records(data / "records.jsonl"):
        rid = record.get("record_id", "?")
        try:
            validate_record(record)
        except RecordError as error:
            problems.append(f"{rid}: {error}")
            continue
        if not record["seed"].startswith(B.TRAIN_SEED_PREFIX):
            problems.append(f"{rid}: non-training seed {record['seed']}")
        for option in record["options"]:
            if option["kind"] == "skill" and option["card_sha256"] not in cards["skills"]:
                problems.append(f"{rid}: missing card for {option['option_id']}")
        if record["protocol"] not in cards["tasks"]:
            problems.append(f"{rid}: missing task card {record['protocol']}")
        for path in record["frames"].values():
            if path is not None and not (data / path).is_file():
                problems.append(f"{rid}: missing frame {path}")
        records.append(record)
    return records, problems


def reproduce(records: list[dict[str, Any]], count: int, root: Path) -> list[str]:
    """Re-run the episode prefix and one labelled option for ``count`` random records."""
    B.use_uzumaki(root)
    from uzumaki.baseline import BASELINE_BUDGETS, RULE_SELECTORS
    from uzumaki.sim.run import run_sim_case

    from mushin.data.counterfactual import try_option
    from mushin.data.explore import Step, as_decision

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_episode.setdefault(record["episode_id"], []).append(record)
    rng = random.Random(0)
    problems = []
    for record in rng.sample(records, min(count, len(records))):
        body = B.BODIES[record["body"]]
        rig = body.rig()
        case, _ = B.sample_case(body, record["seed"], random.Random(record["seed"]))
        snapshot = B.snapshot(body.name, str(root))
        episode = sorted(by_episode[record["episode_id"]], key=lambda r: r["step"])

        def logged(r: dict[str, Any]) -> dict[str, Any]:
            taken = r["taken"]
            reason = "exploration" if taken["source"] == "explore" else None
            if taken["option_id"] == ABSTAIN:
                return as_decision(None, {}, reason)
            skill, version = taken["option_id"].rsplit("@", 1)
            entry = {"skill_id": skill, "skill_version": int(version)}
            return as_decision(entry, taken["parameters"] or {}, reason)

        # The logged requests are not stored, so the reproduction check compares outcomes and
        # bundle heads only.
        steps = [Step(f"d{r['step'] + 1}", {}, {}, logged(r), "replay") for r in episode]
        choices = [o for o in record["options"] if o["kind"] == "skill"]
        option = rng.choice(choices)
        expected = record["labels"]["option_outcomes"][option["option_id"]]
        entry = {"skill_id": option["skill_id"], "skill_version": option["skill_version"]}

        def runner(
            case: Any, selector: Any, snapshot: Any = snapshot, rig: Any = rig,
            protocol: str = body.protocol,
        ) -> tuple[Any, Any]:
            return run_sim_case(case, snapshot, selector=selector, planner_id="mushin-explorer",
                                budgets=BASELINE_BUDGETS, rig=rig, protocol=protocol)

        got = try_option(runner, case, steps, record["step"], entry, expected["parameters"],
                         RULE_SELECTORS[body.protocol][1])
        for key in ("bundle_head", "primary", "success", "decisions_to_end"):
            if got[key] != expected[key]:
                problems.append(
                    f"{record['record_id']} {option['option_id']}: {key} {got[key]!r} != "
                    f"{expected[key]!r}"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.data.verify", description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--reproduce", type=int, default=0)
    parser.add_argument("--uzumaki", type=Path, default=None)
    options = parser.parse_args(argv)
    records, problems = check_records(options.data)
    if (options.data / "splits.json").exists():
        try:
            load(options.data)
        except ValueError as error:
            problems.append(str(error))
    if options.reproduce:
        problems += reproduce(records, options.reproduce, options.uzumaki or B.uzumaki_root())
    for problem in problems[:50]:
        print(problem, file=sys.stderr)
    print(json.dumps({"records": len(records), "problems": len(problems),
                      "reproduced": options.reproduce}))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
