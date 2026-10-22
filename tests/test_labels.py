import pytest

from mushin.data.labels import (
    ABSTAIN_UTILITY,
    bin_of,
    choice_targets,
    number_edges,
    parameter_labels,
    utility,
)
from mushin.schema import ABSTAIN


def ok(steps=1, violations=0):
    return {"success": True, "primary": "completed", "decisions_to_end": steps,
            "violations": violations}


FAIL = {"success": False, "primary": "misgrasp", "decisions_to_end": 3, "violations": 0}


def test_utility_orders_success_abstain_failure():
    assert utility(ok(1)) > utility(ok(3)) > ABSTAIN_UTILITY > utility(FAIL)
    assert utility(ok(1, violations=1)) < utility(ok(1))


def test_choice_prefers_fewer_decisions_and_breaks_ties_to_teacher():
    options = ["pick-box@1", "pick-and-place@1", ABSTAIN]
    outcomes = {"pick-box@1": ok(2), "pick-and-place@1": ok(1)}
    _, hard, soft = choice_targets(options, outcomes, "pick-box@1")
    assert hard == "pick-and-place@1"
    assert soft["pick-and-place@1"] > soft["pick-box@1"] > soft[ABSTAIN]
    assert sum(soft.values()) == pytest.approx(1.0, abs=1e-9)
    outcomes = {"pick-box@1": ok(1), "pick-and-place@1": ok(1)}
    assert choice_targets(options, outcomes, "pick-box@1")[1] == "pick-box@1"


def test_abstain_wins_when_every_skill_fails():
    options = ["pick-box@1", ABSTAIN]
    _, hard, _ = choice_targets(options, {"pick-box@1": FAIL}, "pick-box@1")
    assert hard == ABSTAIN


def test_number_bins_cover_the_schema_range():
    edges = number_edges({"minimum": 0, "maximum": 0.2}, bins=4)
    assert edges == [0.0, 0.05, 0.1, 0.15, 0.2]
    assert [bin_of(v, edges) for v in (0.0, 0.049, 0.05, 0.2, 9.0, -1.0)] == [0, 0, 1, 3, 3, 0]


def test_parameter_labels_types():
    schema = {"properties": {
        "grasp_height_m": {"type": "number", "minimum": 0, "maximum": 0.2},
        "approach": {"type": "string", "enum": ["vertical", "angled"]},
    }}
    labels = parameter_labels(schema, {"grasp_height_m": 0.02, "approach": "vertical"},
                              {"approach": [True, False]})
    assert labels["grasp_height_m"]["kind"] == "bins" and labels["grasp_height_m"]["bin"] == 1
    assert labels["approach"] == {"kind": "enum", "values": ["vertical", "angled"], "index": 0,
                                  "success": [True, False]}
