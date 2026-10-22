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
import hashlib
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
    jobs = [
        (names[i % len(names)], options.family, i, options.epsilon, str(out),
         not options.no_frames, root)
        for i in range(options.start, options.start + options.episodes)
    ]
    started = time.monotonic()
    results = []
    context = multiprocessing.get_context("spawn")
    # Records are appended as each episode finishes, so a run cut off by its deadline (or a
    # hung worker) still leaves every finished episode on disk for the upload.
    with context.Pool(options.workers, maxtasksperchild=50) as pool, \
            (out / "records.partial.jsonl").open("w") as partial:
        pending = pool.imap_unordered(run_episode, jobs)
        for done in range(1, len(jobs) + 1):
            try:
                result = pending.next(timeout=options.stall_s)
            except multiprocessing.TimeoutError:
                print(f"no episode finished in {options.stall_s:.0f} s; stopping early", flush=True)
                break
            results.append(result)
            for record in result["records"]:
                partial.write(json.dumps(record, sort_keys=True) + "\n")
            partial.flush()
            if done % 25 == 0 or done == len(jobs):
                elapsed = time.monotonic() - started
                print(f"{done}/{len(jobs)} episodes, {elapsed:.0f} s", flush=True)
                (out / "progress.json").write_text(json.dumps({
                    "episodes_done": done, "episodes": len(jobs),
                    "decisions": sum(len(r["records"]) for r in results),
                    "elapsed_s": round(elapsed)}))
        pool.terminate()
    results.sort(key=lambda r: r["index"])
    digest = hashlib.sha256()
    count = 0
    with (out / "records.jsonl").open("w") as handle:
        for result in results:
            for record in result["records"]:
                line = json.dumps(record, sort_keys=True) + "\n"
                digest.update(line.encode())
                handle.write(line)
                count += 1
    records = [r for result in results for r in result["records"]]
    stats = {
        "episodes": len(results),
        "complete": len(results) == len(jobs),
        "decisions": count,
        "records_sha256": digest.hexdigest(),
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
        "choice_targets": dict(Counter(
            "abstain" if r["labels"]["choice"] == ABSTAIN else r["labels"]["choice"].split("@")[0]
            for r in records)),
        "taken_source": dict(Counter(r["taken"]["source"] for r in records)),
        "option_success_rate": round(
            sum(o["success"] for r in records for o in r["labels"]["option_outcomes"].values()
                if o["success"] is not None)
            / max(1, sum(1 for r in records for o in r["labels"]["option_outcomes"].values()
                         if o["success"] is not None)), 3),
        "unreproduced": sum(
            1 for r in records for o in r["labels"]["option_outcomes"].values()
            if o.get("reproduced") is False),
        "teacher_matches_target": round(
            sum(r["teacher"]["option_id"] == r["labels"]["choice"] for r in records)
            / max(1, count), 3),
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
