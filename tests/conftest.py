import json
import os
from pathlib import Path

import pytest

# Headless MuJoCo rendering; must be set before mujoco is first imported.
os.environ.setdefault("MUJOCO_GL", "egl")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_dir() -> Path:
    return FIXTURES / "m0_small"


@pytest.fixture
def small_records(small_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (small_dir / "records.jsonl").read_text().splitlines()]
