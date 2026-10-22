import copy

import pytest

from mushin.schema import ABSTAIN, RecordError, validate_record


def test_fixture_records_validate(small_records):
    assert len(small_records) >= 8
    for record in small_records:
        validate_record(record)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.update(seed="so101-dev-001"), "seed"),
        (lambda r: r["options"].pop(), "abstain"),
        (lambda r: r["labels"].update(choice="nope"), "choice"),
        (lambda r: r["labels"]["choice_soft"].update({ABSTAIN: 0.9}), "sum"),
        (lambda r: r["labels"]["utility"].pop(ABSTAIN), "utility"),
        (lambda r: r.update(record_id=r["episode_id"] + "/99"), "record_id"),
        (lambda r: r["taken"].update(source="oracle"), "source"),
    ],
)
def test_invalid_records_are_rejected(small_records, mutate, message):
    record = copy.deepcopy(small_records[0])
    mutate(record)
    with pytest.raises(RecordError, match=message):
        validate_record(record)
