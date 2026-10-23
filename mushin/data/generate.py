"""Generate ``mushin.decision/1.0`` records from uzumaki's MuJoCo scenes.

    python -m mushin.data.generate --out DIR --episodes N [--bodies so101,panda,...]
        [--family m0] [--epsilon 0.3] [--workers 4] [--no-frames] [--uzumaki DIR]

Episodes are spread round-robin over the bodies. Each one runs on a fresh training seed
``mushin-train-<family>-<body>-<index>`` with random faults, explores around the rule teacher,
renders scene and wrist frames at each decision and then labels every decision counterfactually.
Writes ``records.jsonl`` (sorted by episode and step), ``images/``, ``cards.json`` (skill cards
by content hash, and task cards by protocol) and ``stats.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mushin
from mushin.data import bodies as B
from mushin.data.counterfactual import label_step, outcome_of
from mushin.data.explore import Explorer, as_choice, offered
from mushin.data.finalize import write_shard, write_sorted
from mushin.data.labels import LABEL_VERSION, choice_targets, parameter_labels
from mushin.schema import ABSTAIN, SCHEMA_ID, option_id, validate_record

GENERATOR = "explore-v1"


def episode_seed(family: str, body: str, index: int) -> str:
    return f"{B.TRAIN_SEED_PREFIX}{family}-{body}-{index:06d}"


def history_results(bundle: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Non-observe tool results that followed each decision, from the evidence bundle."""
    tools: dict[str, str] = {}
    out: list[list[dict[str, Any]]] = []
    for event in bundle.get("events") or []:
        data = event.get("data") or {}
        if event["kind"] == "decision":
            out.append([])
        elif event["kind"] == "tool_call":
            tools[data.get("call_id")] = data.get("tool")
        elif event["kind"] == "tool_result" and out:
            tool = tools.get(data.get("call_id"), "?")
            if tool in ("observe", "monitor"):
                continue
            result = data.get("result") or {}
            reason = result.get("reason") if data.get("status") != "ok" else None
            out[-1].append({"tool": tool, "status": data.get("status"), "reason": reason})
    return out


def skill_cards(body: B.Body, root: str) -> dict[str, dict[str, Any]]:
    from uzumaki.planner import _skill_card

    snapshot = B.snapshot(body.name, root)
    cards = {}
    for skill in snapshot.eligible(body.protocol):
        label = option_id(skill.skill_id, skill.skill_version)
        if label in B.executable_skills():
            cards[skill.sha256] = _skill_card(snapshot, skill.skill_id, skill.skill_version)
    return cards


def run_episode(job: tuple[str, str, int, float, str, bool, str]) -> dict[str, Any]:
    body_name, family, index, epsilon, out, frames_on, root = job
    B.use_uzumaki(Path(root))
    from uzumaki.baseline import BASELINE_BUDGETS, RULE_SELECTORS
    from uzumaki.sim.run import run_sim_case

    body = B.BODIES[body_name]
    rig = body.rig()
    seed = episode_seed(family, body_name, index)
    episode_id = seed.removeprefix(B.TRAIN_SEED_PREFIX)
    rng = random.Random(seed)
    case, faults = B.sample_case(body, seed, rng)
    snapshot = B.snapshot(body_name, root)
    inputs = {s.skill_id: s.inputs for s in snapshot.eligible(body.protocol)}
    executable = B.executable_skills()
    rule_factory = RULE_SELECTORS[body.protocol][1]
    card = B.task_card(body)

    def runner(case: Any, selector: Any, frame_hook: Any = None) -> tuple[Any, Any]:
        return run_sim_case(
            case, snapshot, frame_hook=frame_hook, selector=selector,
            planner_id="mushin-explorer", budgets=BASELINE_BUDGETS, rig=rig,
            protocol=body.protocol,
        )

    started = time.monotonic()
    frames = None
    if frames_on:
        from mushin.data.render import Frames

        frames = Frames(rig.camera, body.wrist_camera, getattr(rig, "site", None),
                        random.Random(seed + ":render"))
    explorer = Explorer(rule_factory(), inputs, executable, epsilon, rng, frames)
    run, world = runner(case, explorer, frames)
    if frames is not None:
        frames.close()
    episode = outcome_of(run, world, None, 0, len(explorer.steps))
    results = history_results(run.bundle)
    sim_started = time.monotonic()
    images: dict[str, bytes] = {}
    records = []
    history: list[dict[str, Any]] = []
    shown_fields = ("observation_id", *card.observation_fields)
    provenance = {
        "uzumaki_commit": B.PINNED_COMMIT,
        "mushin_version": mushin.__version__,
        "generator": GENERATOR,
        "label_version": LABEL_VERSION,
    }
    for at, step in enumerate(explorer.steps):
        request = step.request
        entries = offered(request, executable)
        options = [
            {"option_id": option_id(e["skill_id"], e["skill_version"]), "kind": "skill",
             "skill_id": e["skill_id"], "skill_version": e["skill_version"],
             "card_sha256": e["content_sha256"]}
            for e in entries
        ] + [{"option_id": ABSTAIN, "kind": "abstain"}]
        outcomes, enums = label_step(lambda c, s: runner(c, s), case, explorer.steps, at,
                                     executable, inputs, rule_factory)
        outcomes[ABSTAIN] = {"success": None, "primary": "abstained"}
        teacher = as_choice(step.teacher)
        if teacher["option_id"] not in {o["option_id"] for o in options}:
            teacher = {"option_id": ABSTAIN, "parameters": None}
        utilities, hard, soft = choice_targets([o["option_id"] for o in options], outcomes,
                                               teacher["option_id"])
        params = {}
        for e in entries:
            oid = option_id(e["skill_id"], e["skill_version"])
            labels = parameter_labels(inputs.get(e["skill_id"], {}), outcomes[oid]["parameters"],
                                      enums.get(oid))
            if labels:
                params[oid] = labels
        frame_paths: dict[str, str | None] = {"scene": None, "wrist": None}
        for view, data in (step.frames or {}).items():
            if data is not None:
                path = f"images/{episode_id}-{at}-{view}.jpg"
                images[path] = data
                frame_paths[view] = path
        taken = dict(as_choice(step.decision), source=step.source)
        record = {
            "schema": SCHEMA_ID,
            "record_id": f"{episode_id}/{at}",
            "episode_id": episode_id,
            "step": at,
            "body": body_name,
            "protocol": body.protocol,
            "seed": seed,
            "faults": faults,
            "task": {"protocol": body.protocol, "goal": card.goal},
            "observation": {k: request["observation"].get(k) for k in shown_fields},
            "history": [dict(h) for h in history],
            "frames": frame_paths,
            "options": options,
            "teacher": teacher,
            "taken": taken,
            "labels": {
                "option_outcomes": outcomes,
                "utility": utilities,
                "choice": hard,
                "choice_soft": soft,
                "params": params,
            },
            "episode": episode,
            "provenance": provenance,
        }
        validate_record(record)
        records.append(record)
        history.append({"step": at, "option_id": taken["option_id"],
                        "parameters": taken["parameters"],
                        "tool_results": results[at] if at < len(results) else []})
        history = history[-4:]
    for path, data in images.items():
        (Path(out) / path).write_bytes(data)
    return {
        "body": body_name,
        "index": index,
        "records": records,
        "primary": episode["primary"],
        "render_s": frames.render_s if frames else 0.0,
        "episode_s": sim_started - started,
        "label_s": time.monotonic() - sim_started,
        "image_bytes": sum(len(d) for d in images.values()),
    }


class RecordTally:
    """Record-level stats, counted as records stream past instead of held in memory."""

    def __init__(self) -> None:
        self.count = 0
        self.choice_targets: Counter[str] = Counter()
        self.taken_source: Counter[str] = Counter()
        self.successes = self.labelled = self.unreproduced = self.teacher_matches = 0

    def add(self, r: dict[str, Any]) -> None:
        self.count += 1
        choice = r["labels"]["choice"]
        self.choice_targets["abstain" if choice == ABSTAIN else choice.split("@")[0]] += 1
        self.taken_source[r["taken"]["source"]] += 1
        for o in r["labels"]["option_outcomes"].values():
            if o["success"] is not None:
                self.successes += o["success"]
                self.labelled += 1
            self.unreproduced += o.get("reproduced") is False
        self.teacher_matches += r["teacher"]["option_id"] == choice

    def stats(self) -> dict[str, Any]:
        return {
            "choice_targets": dict(self.choice_targets),
            "taken_source": dict(self.taken_source),
            "option_success_rate": round(self.successes / max(1, self.labelled), 3),
            "unreproduced": self.unreproduced,
            "teacher_matches_target": round(self.teacher_matches / max(1, self.count), 3),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.data.generate", description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--start", type=int, default=0, help="first episode index")
    parser.add_argument("--bodies", default=",".join(B.BODIES))
    parser.add_argument("--family", default="m0")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count()))
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--uzumaki", type=Path, default=None)
    parser.add_argument("--shard-episodes", type=int, default=0,
                        help="also write shards/ of this many episodes each (0: off)")
    parser.add_argument("--stall-s", type=float, default=1800.0,
                        help="stop early when no episode finishes for this long")
    options = parser.parse_args(argv)
    root = str(B.use_uzumaki(options.uzumaki))
    names = [n for n in options.bodies.split(",") if n]
    unknown = set(names) - set(B.BODIES)
    if unknown:
        parser.error(f"unknown bodies: {sorted(unknown)}")
    out: Path = options.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    cards: dict[str, Any] = {"skills": {}, "tasks": {}}
    from uzumaki import planner

    for name in names:
        body = B.BODIES[name]
        cards["skills"].update(skill_cards(body, root))
        card = getattr(planner, body.card)
        cards["tasks"][body.protocol] = {"intro": card.intro, "goal": card.goal,
                                         "eligibility_rule": card.eligibility_rule,
                                         "observation_fields": list(card.observation_fields)}
    (out / "cards.json").write_text(json.dumps(cards, indent=1, sort_keys=True))
    # Resume: episodes already in shards/ (downloaded by pod_job.sh after a pod died) are skipped.
    done_ids = {json.loads(line)["episode_id"]
                for path in out.glob("shards/*/records.jsonl")
                for line in path.read_text().splitlines() if line.strip()}
    jobs = [
        (names[i % len(names)], options.family, i, options.epsilon, str(out),
         not options.no_frames, root)
        for i in range(options.start, options.start + options.episodes)
        if episode_seed(options.family, names[i % len(names)], i).removeprefix(
            B.TRAIN_SEED_PREFIX) not in done_ids
    ]
    if done_ids:
        print(f"resuming: {len(done_ids)} episodes already in shards/", flush=True)
    started = time.monotonic()
    results = []
    tally = RecordTally()
    shard: list[dict[str, Any]] = []
    shards = len(list(out.glob("shards/s*")))
    context = multiprocessing.get_context("spawn")
    # Records are appended as each episode finishes, so a run cut off by its deadline (or a
    # hung worker) still leaves every finished episode on disk. Only slim per-episode results
    # stay in memory: holding every record OOM-killed a 32-worker run.
    with context.Pool(options.workers, maxtasksperchild=20) as pool, \
            (out / "records.partial.jsonl").open("w") as partial:
        pending = pool.imap_unordered(run_episode, jobs)
        for done in range(1, len(jobs) + 1):
            try:
                result = pending.next(timeout=options.stall_s)
            except multiprocessing.TimeoutError:
                print(f"no episode finished in {options.stall_s:.0f} s; stopping early", flush=True)
                break
            records = result.pop("records")
            result["decisions"] = len(records)
            results.append(result)
            for record in records:
                partial.write(json.dumps(record, sort_keys=True) + "\n")
                tally.add(record)
            partial.flush()
            shard.extend(records)
            if options.shard_episodes and done % options.shard_episodes == 0:
                write_shard(out, shards, shard)
                shards, shard = shards + 1, []
            if done % 25 == 0 or done == len(jobs):
                elapsed = time.monotonic() - started
                print(f"{done}/{len(jobs)} episodes, {elapsed:.0f} s", flush=True)
                (out / "progress.json").write_text(json.dumps({
                    "episodes_done": done, "episodes": len(jobs), "decisions": tally.count,
                    "shards": shards, "elapsed_s": round(elapsed)}))
        pool.terminate()
    if options.shard_episodes and shard:
        write_shard(out, shards, shard)
    if options.shard_episodes:
        # The shards hold every episode, including those of earlier pods when resuming.
        with (out / "records.partial.jsonl").open("w") as handle:
            for path in sorted(out.glob("shards/*/records.jsonl")):
                handle.write(path.read_text())
    count, digest = write_sorted(out / "records.partial.jsonl", out / "records.jsonl")
    stats = {
        "episodes": len(results) + len(done_ids),
        "complete": len(results) == len(jobs),
        "decisions": count,
        "records_sha256": digest,
        "wall_s": round(time.monotonic() - started, 1),
        "workers": options.workers,
        "cpu_s": {
            "episode": round(sum(r["episode_s"] for r in results), 1),
            "render": round(sum(r["render_s"] for r in results), 1),
            "label": round(sum(r["label_s"] for r in results), 1),
        },
        "image_bytes": sum(r["image_bytes"] for r in results),
        "episodes_by_body": dict(Counter(r["body"] for r in results)),
        "episode_outcomes": dict(Counter(str(r["primary"]) for r in results)),
        **tally.stats(),
        "uzumaki_commit": B.PINNED_COMMIT,
        "family": options.family,
        "epsilon": options.epsilon,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "records.partial.jsonl").unlink()
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
