import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_dir() -> Path:
    return FIXTURES / "m0_small"


@pytest.fixture
def small_records(small_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (small_dir / "records.jsonl").read_text().splitlines()]
