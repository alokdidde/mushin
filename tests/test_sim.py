"""End-to-end generation in MuJoCo; skipped unless an uzumaki checkout and MuJoCo are present."""

import json
import os
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
ROOT = Path(os.environ.get("UZUMAKI_ROOT", Path(__file__).parents[1] / "third_party" / "uzumaki"))
pytestmark = pytest.mark.skipif(not (ROOT / "src" / "uzumaki").is_dir(), reason="no uzumaki")


def test_generate_and_reproduce(tmp_path):
    from mushin.data import generate, verify

    out = tmp_path / "data"
    assert generate.main(["--out", str(out), "--episodes", "2", "--bodies", "so101,tidybot",
                          "--workers", "1", "--no-frames", "--uzumaki", str(ROOT)]) == 0
    stats = json.loads((out / "stats.json").read_text())
    assert stats["decisions"] >= 2 and stats["unreproduced"] == 0
    assert verify.main([str(out), "--reproduce", "3", "--uzumaki", str(ROOT)]) == 0
