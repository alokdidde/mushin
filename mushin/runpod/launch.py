"""Run one training stage on RunPod: choose the machine, start it under a cost cap, watch it,
and stop it.

    python -m mushin.runpod.launch configs/stages/sft_warmup.yaml --dry-run [--offers FILE]
    python -m mushin.runpod.launch configs/stages/sft_warmup.yaml --approve-cap 12

``--dry-run`` prints the ranked machines and the chosen one's estimated hours, cost and cap and
starts nothing (``--offers`` reads a saved offer list instead of the live API). A real start
requires ``--approve-cap`` at or above the computed cap: that number is the one Alok approved.
While the pod runs the launcher polls it and stops it when the job ends, when the running cost
reaches the cap, or at ``max_hours``; the pod also stops itself (``scripts/pod_job.sh``). Every
run writes ``runs/launch/<stage>-<time>.json`` with the machine, hours and cost.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from mushin.runpod.catalog import Offer, Ranked, rank
from mushin.runpod.client import RunPodClient

REPO = Path(__file__).resolve().parents[2]
POLL_S = 60
#: Time past max_hours before the launcher stops a pod, so its own upload can finish.
UPLOAD_GRACE_H = 0.5
JOB_SCRIPT = "scripts/pod_job.sh"


def load_stage(path: Path) -> dict[str, Any]:
    stage = yaml.safe_load(path.read_text())
    for key in ("name", "resources", "work_units", "command", "max_hours", "image"):
        if key not in stage:
            raise ValueError(f"{path}: missing {key}")
    return stage


def read_offers(path: Path) -> list[Offer]:
    return [Offer(**row) for row in json.loads(path.read_text())]


def choose(stage: dict[str, Any], offers: list[Offer]) -> list[Ranked]:
    ranked = rank(offers, stage["resources"], stage["work_units"],
                  stage.get("measured_speed"), stage.get("allow_spot", True))
    budget = stage.get("budget_usd")
    if budget is not None:
        ranked = [r for r in ranked if r.cap <= budget]
    # A machine that would hit max_hours first never finishes the job.
    ranked = [r for r in ranked if r.est_hours <= stage["max_hours"]]
    return ranked


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def pod_deadline_h(stage: dict[str, Any], pick: Ranked) -> float:
    """Hours the pod's own job may run, so the pod stays under its cap even if the launcher dies."""
    return min(stage["max_hours"], pick.cap / pick.offer.price_hr - UPLOAD_GRACE_H)


def pod_spec(stage: dict[str, Any], pick: Ranked, commit: str) -> dict[str, Any]:
    env = {
        "MUSHIN_COMMIT": commit,
        "MUSHIN_COMMAND": stage["command"],
        "MUSHIN_MAX_SECONDS": str(int(pod_deadline_h(stage, pick) * 3600)),
        "MUSHIN_EXTRAS": stage.get("extras", "train"),
        "MUSHIN_OUT_DIR": stage.get("out_dir", "runs"),
        **{k: str(v) for k, v in (stage.get("env") or {}).items()},
    }
    # Secrets are passed by reference to RunPod secrets, never inlined.
    for name in stage.get("secrets", []):
        env[name] = "{{ RUNPOD_SECRET_" + name + " }}"
    spec: dict[str, Any] = {
        "name": f"mushin-{stage['name']}",
        "imageName": stage["image"],
        "cloudType": pick.offer.cloud,
        "interruptible": pick.offer.spot,
        "containerDiskInGb": stage.get("disk_gb", 60),
        "env": env,
        "dockerStartCmd": ["bash", "-c",
                           f"curl -fsSL https://raw.githubusercontent.com/alokdidde/mushin/"
                           f"{commit}/{JOB_SCRIPT} | bash"],
    }
    if pick.offer.gpu == "cpu":
        spec.update(computeType="CPU", vcpuCount=pick.offer.vcpus,
                    cpuFlavorIds=[pick.offer.flavor or "cpu3c"])
    else:
        spec.update(computeType="GPU", gpuTypeIds=[pick.offer.gpu], gpuCount=1)
    if stage.get("network_volume_id"):
        spec["networkVolumeId"] = stage["network_volume_id"]
    return spec


def watch(client: RunPodClient, pod_id: str, price_hr: float, cap: float, max_hours: float,
          report: dict[str, Any], report_path: Path) -> None:
    started = time.monotonic()
    while True:
        hours = (time.monotonic() - started) / 3600
        cost = hours * price_hr
        pod = client.get_pod(pod_id) or {}
        status = pod.get("desiredStatus") or pod.get("status")
        report.update(hours=round(hours, 3), cost=round(cost, 2), status=status)
        report_path.write_text(json.dumps(report, indent=2))
        if status in ("EXITED", "TERMINATED", "STOPPED"):
            report["ended"] = "job finished"
            break
        if cost >= cap or hours >= max_hours + UPLOAD_GRACE_H:
            report["ended"] = "cost cap" if cost >= cap else "max hours"
            client.stop_pod(pod_id)
            break
        time.sleep(POLL_S)
    client.terminate_pod(pod_id)
    report["terminated"] = True
    report_path.write_text(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mushin.runpod.launch", description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offers", type=Path, help="saved offer list (JSON) instead of the API")
    parser.add_argument("--save-offers", type=Path, help="write the live offer list here")
    parser.add_argument("--approve-cap", type=float, help="the cost cap Alok approved, in USD")
    parser.add_argument("--top", type=int, default=8)
    options = parser.parse_args(argv)
    stage = load_stage(options.stage)
    client = None
    if options.offers:
        offers = read_offers(options.offers)
    else:
        client = RunPodClient()
        offers = client.offers(stage.get("vcpus_per_gpu", 8))
        if options.save_offers:
            options.save_offers.write_text(json.dumps([asdict(o) for o in offers], indent=1))
    ranked = choose(stage, offers)
    if not ranked:
        print("no available machine fits this stage within its budget", file=sys.stderr)
        return 2
    print(f"stage {stage['name']}: {stage['work_units']} work units")
    for r in ranked[: options.top]:
        o = r.offer
        name = o.gpu if o.gpu != "cpu" else f"cpu {o.flavor} {o.vcpus} vCPU"
        print(f"  {name:32} {o.cloud:9} {'spot' if o.spot else 'on-demand':9} "
              f"${o.price_hr:.2f}/h  ~{r.est_hours:.1f} h  ~${r.est_cost:.2f}  cap ${r.cap:.0f}")
    pick = ranked[0]
    print(f"chosen: {pick.offer.gpu} ({pick.offer.cloud}, "
          f"{'spot' if pick.offer.spot else 'on-demand'}), estimated {pick.est_hours:.1f} h "
          f"and ${pick.est_cost:.2f}; cost cap ${pick.cap:.0f}")
    if options.dry_run:
        print("dry run: nothing started")
        return 0
    if options.approve_cap is None or options.approve_cap < pick.cap:
        print(f"refusing to start: --approve-cap must be at least ${pick.cap:.0f}",
              file=sys.stderr)
        return 3
    client = client or RunPodClient()
    commit = git_commit()
    pod = client.create_pod(pod_spec(stage, pick, commit))
    pod_id = pod["id"]
    report_path = REPO / "runs" / "launch" / f"{stage['name']}-{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"stage": stage["name"], "pod_id": pod_id, "commit": commit,
              "machine": pick.as_dict(), "approved_cap": options.approve_cap}
    print(f"started pod {pod_id}; watching (report: {report_path})")
    try:
        watch(client, pod_id, pick.offer.price_hr, options.approve_cap,
              stage["max_hours"], report, report_path)
    except BaseException:
        client.stop_pod(pod_id)
        raise
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
