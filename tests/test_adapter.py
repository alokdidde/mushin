"""MushinPlanner drives a simulated uzumaki episode (skipped without uzumaki and MuJoCo)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
ROOT = Path(os.environ.get("UZUMAKI_ROOT", Path(__file__).parents[1] / "third_party" / "uzumaki"))
pytestmark = pytest.mark.skipif(not (ROOT / "src" / "uzumaki").is_dir(), reason="no uzumaki")


@pytest.mark.parametrize("body_name", ["so101", "tidybot"])
def test_planner_runs_an_episode(body_name):
    from mushin.data import bodies as B

    B.use_uzumaki(ROOT)
    from uzumaki.sim.run import run_sim_case

    from mushin.data.render import Frames
    from mushin.model.backbone import TinyBackbone
    from mushin.model.head import MushinModel
    from mushin.uzumaki_adapter import MushinPlanner

    body = B.BODIES[body_name]
    rig = body.rig()
    snapshot = B.snapshot(body_name, str(ROOT))
    case = next(c for c in rig.cases() if c.case_id == "nominal")
    frames = Frames(rig.camera, body.wrist_camera, getattr(rig, "site", None), None, size=64)
    torch.manual_seed(0)
    planner = MushinPlanner(MushinModel(TinyBackbone(32, 1), 32), snapshot, B.task_card(body),
                            body.protocol, B.executable_skills(), frames)
    run, _ = run_sim_case(case, snapshot, frame_hook=frames, selector=planner,
                          planner_id="mushin", rig=rig, protocol=body.protocol)
    frames.close()
    assert run.verdict is not None and planner.history
    assert run.report.status != "failed" or run.verdict.get("primary") != "invalid_decision"
