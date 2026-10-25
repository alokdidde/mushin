"""Closed-loop benchmark: whole uzumaki episodes decided by a Mushin checkpoint vs the rule
selector.

    python -m mushin.eval.closed_loop --checkpoint runs/x/checkpoint.pt --out DIR
        [--episodes 50] [--bodies so101,panda,...] [--family bench] [--min-confidence 0]

Each episode runs on a fresh training-only seed ``mushin-train-<family>-<body>-<index>`` with the
same random faults the data generator samples, once with the rule selector (uzumaki's teacher) and
once with ``MushinPlanner``; uzumaki's independent evaluator decides success. The model sees the
tool results of its previous decision, as the training records do. One JSON line per episode goes
to ``episodes.jsonl`` as it finishes, and ``summary.json`` has success rate, verdicts, decisions,
abstains and decision latency per body and planner.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mushin.data import bodies as B
from mushin.data.generate import history_results


class Timed:
    """A decision source that times each call of the one it wraps."""

    def __init__(self, source: Any) -> None:
        self.source, self.ms = source, []

    def decide(self, decision_id: str, role: str, request: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        output = self.source.decide(decision_id, role, request)
        self.ms.append((time.perf_counter() - start) * 1000)
        return output

    def note_results(self, results: list[dict[str, Any]]) -> None:
        if hasattr(self.source, "note_results"):
            self.source.note_results(results)


def feed_tool_results() -> None:
    """Before each decision, hand the source the tool results of its previous decision."""
    from uzumaki.evidence import EventRecorder

    if getattr(EventRecorder.decide, "_mushin", False):
        return
    original = EventRecorder.decide

    def decide(self: Any, source: Any, decision_id: str, role: str,
               request: dict[str, Any]) -> dict[str, Any]:
        if hasattr(source, "note_results"):
            results = history_results({"events": self._events})
            if results:
                source.note_results(results[-1])
        return original(self, source, decision_id, role, request)

    decide._mushin = True  # type: ignore[attr-defined]
    EventRecorder.decide = decide  # type: ignore[method-assign]


def run_one(body: B.Body, case: Any, selector: Any, root: str, frames: Any) -> dict[str, Any]:
    from uzumaki.baseline import BASELINE_BUDGETS
    from uzumaki.sim.run import run_sim_case

    timed = Timed(selector)
    run, _ = run_sim_case(case, B.snapshot(body.name, root), frame_hook=frames, selector=timed,
                          planner_id="mushin-bench", budgets=BASELINE_BUDGETS, rig=body.rig(),
                          protocol=body.protocol)
    verdict = run.verdict or {}
    outputs = [e["data"]["output"] for e in run.bundle.get("events") or []
               if e["kind"] == "decision"]
    return {
        "success": verdict.get("primary") == "completed",
        "primary": verdict.get("primary"),
        "status": run.report.status,
        "decisions": len(outputs),
        "abstains": sum(o.get("action") == "abstain" for o in outputs),
        "decision_ms": [round(ms, 2) for ms in timed.ms],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for planner in ("rule", "mushin"):
        groups: dict[str, list[dict[str, Any]]] = {"all": []}
        for row in rows:
            groups["all"].append(row[planner])
            groups.setdefault(f"body:{row['body']}", []).append(row[planner])
        out[planner] = {}
        for name, runs in groups.items():
            ms = [m for r in runs for m in r["decision_ms"]]
            decisions = sum(r["decisions"] for r in runs)
            out[planner][name] = {
                "episodes": len(runs),
                "success_rate": sum(r["success"] for r in runs) / len(runs),
                "verdicts": dict(Counter(str(r["primary"]) for r in runs)),
                "decisions_mean": statistics.fmean(r["decisions"] for r in runs),
                "abstain_rate": sum(r["abstains"] for r in runs) / max(1, decisions),
                "decision_ms_median": statistics.median(ms) if ms else None,
                "decision_ms_p90": sorted(ms)[int(0.9 * (len(ms) - 1))] if ms else None,
            }
    both = rows
    out["paired"] = {
        "mushin_only": sum(r["mushin"]["success"] and not r["rule"]["success"] for r in both),
        "rule_only": sum(r["rule"]["success"] and not r["mushin"]["success"] for r in both),
        "both": sum(r["rule"]["success"] and r["mushin"]["success"] for r in both),
        "neither": sum(not r["rule"]["success"] and not r["mushin"]["success"] for r in both),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.eval.closed_loop", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50, help="per body")
    parser.add_argument("--bodies", default=",".join(B.BODIES))
    parser.add_argument("--family", default="bench")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--uzumaki", type=Path, default=None)
    options = parser.parse_args(argv)
    root = str(B.use_uzumaki(options.uzumaki))
    from uzumaki.baseline import RULE_SELECTORS

    from mushin.data.render import Frames
    from mushin.train import checkpoint
    from mushin.uzumaki_adapter import MushinPlanner

    feed_tool_results()
    model = checkpoint.load_for_inference(options.checkpoint, options.device)
    options.out.mkdir(parents=True, exist_ok=True)
    log = options.out / "episodes.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    done = {row["seed"] for row in rows}
    bodies = [B.BODIES[name] for name in options.bodies.split(",")]
    for index in range(options.episodes):  # round-robin, so a cut-short run covers every body
        for body in bodies:
            seed = f"{B.TRAIN_SEED_PREFIX}{options.family}-{body.name}-{index:06d}"
            if seed in done:
                continue
            case, faults = B.sample_case(body, seed, random.Random(seed))
            rig = body.rig()
            rule = run_one(body, case, RULE_SELECTORS[body.protocol][1](), root, None)
            frames = Frames(rig.camera, body.wrist_camera, getattr(rig, "site", None), None)
            planner = MushinPlanner(model, B.snapshot(body.name, root), B.task_card(body),
                                    body.protocol, B.executable_skills(), frames,
                                    options.min_confidence)
            mushin = run_one(body, case, planner, root, frames)
            frames.close()
            row = {"seed": seed, "body": body.name, "faults": faults, "rule": rule,
                   "mushin": mushin}
            rows.append(row)
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{seed}: rule {rule['primary']}, mushin {mushin['primary']}", flush=True)
        (options.out / "summary.json").write_text(json.dumps(summarize(rows), indent=1))
    summary = summarize(rows)
    (options.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({p: summary[p]["all"] for p in ("rule", "mushin")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
