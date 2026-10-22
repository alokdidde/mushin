"""The eight uzumaki bodies Mushin trains on: task pack, rig, skill set, task card and geometry.

uzumaki is imported from a checkout pinned in ``UZUMAKI_COMMIT`` (``scripts/fetch_uzumaki.sh``);
``use_uzumaki`` puts its ``src`` on ``sys.path`` and resolves its fixture skill sets.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PINNED_COMMIT = (REPO / "UZUMAKI_COMMIT").read_text().strip()
TRAIN_SEED_PREFIX = "mushin-train-"


def uzumaki_root() -> Path:
    return Path(os.environ.get("UZUMAKI_ROOT", REPO / "third_party" / "uzumaki")).resolve()


def use_uzumaki(root: Path | None = None) -> Path:
    """Make the pinned uzumaki checkout importable and return its root."""
    root = (root or uzumaki_root()).resolve()
    if not (root / "src" / "uzumaki").is_dir():
        raise FileNotFoundError(f"no uzumaki checkout at {root} (run scripts/fetch_uzumaki.sh)")
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("MUJOCO_GL", "egl")
    return root


@dataclass(frozen=True)
class Body:
    name: str
    protocol: str
    #: ``module:attribute`` of the rig, and of the geometry generator for training seeds.
    rig_ref: str
    geometry_ref: str
    skills_file: str
    card: str
    #: Named model camera used as the wrist view; ``None`` follows the rig's tool site instead.
    wrist_camera: str | None
    #: Whether the unreachable-target fault applies (tabletop arms only).
    unreachable: bool = False

    def rig(self) -> Any:
        return _attr(self.rig_ref)

    def geometry(self, seed: str) -> dict[str, Any]:
        return _attr(self.geometry_ref)(seed)


ARM_GEOMETRY = "uzumaki.sim.cases:generate_geometry"
PICK_PLACE_SKILLS = "tests/fixtures/skills_pick_place_v1.json"
BODIES: Mapping[str, Body] = {
    b.name: b
    for b in (
        Body("so101", "pick-place-v1", "uzumaki.sim.so101:RIG", ARM_GEOMETRY,
             PICK_PLACE_SKILLS, "PICK_PLACE_CARD", "wrist_cam", unreachable=True),
        Body("panda", "pick-place-v1", "uzumaki.sim.panda:RIG", ARM_GEOMETRY,
             PICK_PLACE_SKILLS, "PICK_PLACE_CARD", None, unreachable=True),
        Body("spot", "pick-place-v1", "uzumaki.sim.spot:RIG", ARM_GEOMETRY,
             PICK_PLACE_SKILLS, "PICK_PLACE_CARD", None, unreachable=True),
        Body("g1", "pick-place-v1", "uzumaki.sim.g1:RIG", ARM_GEOMETRY,
             PICK_PLACE_SKILLS, "PICK_PLACE_CARD", None, unreachable=True),
        Body("go1", "pick-place-v1", "uzumaki.sim.go1:RIG", ARM_GEOMETRY,
             PICK_PLACE_SKILLS, "PICK_PLACE_CARD", "jaw/gripper_gopro", unreachable=True),
        Body("aloha", "handover-v1", "uzumaki.sim.aloha:GIVER",
             "uzumaki.sim.aloha:generate_geometry",
             "tests/fixtures/skills_handover_v1.json", "HANDOVER_CARD", "wrist_cam_left"),
        Body("tidybot", "fetch-v1", "uzumaki.sim.tidybot:RIG",
             "uzumaki.sim.tidybot:generate_geometry", "tests/fixtures/skills_fetch_v1.json",
             "FETCH_CARD", "wrist"),
        Body("go1_walk", "fetch-v1", "uzumaki.sim.go1_walk:RIG",
             "uzumaki.sim.tidybot:generate_geometry", "tests/fixtures/skills_fetch_v1.json",
             "FETCH_CARD", "jaw/gripper_gopro"),
    )
}


def _attr(ref: str) -> Any:
    module, _, attribute = ref.partition(":")
    return getattr(importlib.import_module(module), attribute)


def task_card(body: Body) -> Any:
    return _attr(f"uzumaki.planner:{body.card}")


@cache
def snapshot(body_name: str, root: str) -> Any:
    """The sealed registry snapshot of a body's skill set (cached per process)."""
    from uzumaki.broker import PICK_PLACE_V1_TOOL_VERSION
    from uzumaki.fixture_env import PROTOCOL_TOOLS
    from uzumaki.sim.run import load_snapshot

    body = BODIES[body_name]
    tools = {tool: PICK_PLACE_V1_TOOL_VERSION for tool in PROTOCOL_TOOLS[body.protocol]}
    return load_snapshot(Path(root) / body.skills_file, tools)


def executable_skills() -> frozenset[str]:
    from uzumaki.baseline import BASELINE_SKILL_BODIES

    return frozenset(BASELINE_SKILL_BODIES)


def oracle_parameters(skill_id: str, observation: Mapping[str, Any]) -> dict[str, Any]:
    """The parameters the rule selectors would pass to ``skill_id`` in this situation."""
    target = observation.get("target_pose") or {}
    if skill_id in ("pick-box", "pick-and-place"):
        return {"grasp_height_m": float(target.get("z_m", 0.0))}
    if skill_id == "place-box":
        return {"approach": "vertical"}
    if skill_id == "go-to":
        return {"to": "target" if observation.get("gripper_state") == "holding" else "object"}
    return {}


def sample_case(body: Body, seed: str, rng: random.Random) -> Any:
    """A fresh case on a training-only seed with random injected faults."""
    if not seed.startswith(TRAIN_SEED_PREFIX):
        raise ValueError(f"training seeds must start with {TRAIN_SEED_PREFIX!r}")
    template = next(c for c in body.rig().cases() if c.case_id == "nominal")
    geometry = body.geometry(seed)
    target = tuple(geometry["target_xy_m"])
    faults: dict[str, Any] = {"unreachable": False}
    if body.unreachable and rng.random() < 0.08:
        from uzumaki.sim.cases import BASE_XY

        target = tuple(round(b + 3.0 * (t - b), 6) for t, b in zip(target, BASE_XY, strict=True))
        faults["unreachable"] = True
    changes: dict[str, Any] = {
        "misgrasp_picks": rng.choice([0, 0, 0, 0, 0, 0, 1, 1, 2, 3]),
        "stale_observes": (rng.choice([1, 2]),) if rng.random() < 0.1 else (),
        "crash_on_call": rng.randint(2, 6) if rng.random() < 0.03 else None,
        "withdraw_goal_after_call": rng.randint(2, 6) if rng.random() < 0.03 else None,
    }
    if body.protocol == "handover-v1":
        changes["dropped_handovers"] = rng.choice([1, 2]) if rng.random() < 0.15 else 0
    if body.protocol == "fetch-v1":
        changes["stopped_short_drives"] = rng.choice([1, 2]) if rng.random() < 0.15 else 0
    faults.update({k: list(v) if isinstance(v, tuple) else v for k, v in changes.items()})
    case = dataclasses.replace(
        template,
        case_id=seed.removeprefix(TRAIN_SEED_PREFIX),
        geometry=seed,
        box_size_m=geometry["box_size_m"],
        initial_xy_m=tuple(geometry["initial_xy_m"]),
        initial_yaw_deg=geometry["initial_yaw_deg"],
        target_xy_m=target,
        **changes,
    )
    return case, faults


