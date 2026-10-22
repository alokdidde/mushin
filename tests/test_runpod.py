"""Machine choice and launch guards, offline (no API calls)."""

import json
from pathlib import Path

import pytest

from mushin.runpod import launch
from mushin.runpod.catalog import OVERHEAD_HOURS, Offer, rank

ROOT = Path(__file__).parents[1]
OFFERS = ROOT / "configs" / "stages" / "offers_example.json"


def offer(gpu, price, memory=24, spot=False, vcpus=8, available=True):
    return Offer(gpu, memory, vcpus, price, "COMMUNITY", spot, available)


def test_rank_by_total_cost_not_hourly_price():
    resources = {"min_gpu_gb": 24, "min_vcpus": 8, "est_hours_per_unit": 1.0}
    slow_cheap = offer("NVIDIA GeForce RTX 3090", 0.22)  # 0.55x
    fast = offer("NVIDIA H100 80GB HBM3", 0.50, 80)  # 2.4x
    ranked = rank([slow_cheap, fast], resources, work_units=10)
    assert ranked[0].offer is fast
    assert ranked[0].est_hours == pytest.approx(10 / 2.4 + OVERHEAD_HOURS)


def test_rank_filters_memory_availability_spot_and_unknown_gpus():
    resources = {"min_gpu_gb": 24, "min_vcpus": 8, "est_hours_per_unit": 1.0}
    offers = [offer("NVIDIA RTX A4000", 0.1, 16), offer("NVIDIA L40S", 0.5, 48, available=False),
              offer("Mystery GPU", 0.01, 80), offer("NVIDIA GeForce RTX 4090", 0.2, spot=True),
              offer("NVIDIA GeForce RTX 4090", 0.3)]
    assert [r.offer.price_hr for r in rank(offers, resources, 1)] == [0.2, 0.3]
    assert [r.offer.price_hr for r in rank(offers, resources, 1, allow_spot=False)] == [0.3]


def test_measured_speed_overrides_the_table():
    resources = {"min_gpu_gb": 24, "est_hours_per_unit": 1.0}
    offers = [offer("NVIDIA GeForce RTX 4090", 0.34), offer("NVIDIA GeForce RTX 3090", 0.22)]
    assert rank(offers, resources, 10)[0].offer.gpu.endswith("4090")
    measured = {"NVIDIA GeForce RTX 3090": 0.9}
    assert rank(offers, resources, 10, measured)[0].offer.gpu.endswith("3090")


def test_cpu_stage_uses_cpu_offers_only():
    resources = {"min_gpu_gb": 0, "min_vcpus": 32, "core_hours_per_unit": 1.6}
    offers = [offer("cpu", 0.64, 0, vcpus=32), offer("cpu", 0.2, 0, vcpus=16),
              offer("NVIDIA GeForce RTX 4090", 0.2)]
    ranked = rank(offers, resources, 200)
    assert [r.offer.vcpus for r in ranked] == [32]


@pytest.mark.parametrize("stage", sorted((ROOT / "configs" / "stages").glob("m*.yaml")))
def test_dry_run_starts_nothing(stage, capsys, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("dry run must not call the API")

    monkeypatch.setattr(launch, "RunPodClient", fail)
    assert launch.main([str(stage), "--dry-run", "--offers", str(OFFERS)]) == 0
    out = capsys.readouterr().out
    assert "chosen:" in out and "cost cap" in out and "nothing started" in out


def test_start_requires_an_approved_cap(monkeypatch, capsys):
    monkeypatch.setattr(launch, "RunPodClient", lambda: pytest.fail("must not start"))
    stage = ROOT / "configs" / "stages" / "m5_lora.yaml"
    assert launch.main([str(stage), "--offers", str(OFFERS)]) == 3
    assert launch.main([str(stage), "--offers", str(OFFERS), "--approve-cap", "0.5"]) == 3


def test_pod_spec_keeps_secrets_out_and_pins_the_commit():
    stage = launch.load_stage(ROOT / "configs" / "stages" / "m5_warmup.yaml")
    pick = launch.choose(stage, launch.read_offers(OFFERS))[0]
    spec = launch.pod_spec(stage, pick, "abc123")
    assert spec["env"]["MUSHIN_COMMIT"] == "abc123"
    assert spec["env"]["HF_TOKEN"] == "{{ RUNPOD_SECRET_HF_TOKEN }}"
    assert "abc123/scripts/pod_job.sh" in " ".join(spec["dockerStartCmd"])
    assert spec["gpuTypeIds"] == [pick.offer.gpu]
    json.dumps(spec)
