"""Machine choice: fit, then rank by estimated total cost (price per hour x estimated hours).

A stage states its minimum GPU memory and vCPUs and its estimated hours on the reference GPU
(an RTX 4090) for its workload. Other GPUs scale by ``RELATIVE_SPEED`` (rough bf16 training
throughput relative to the 4090, from public specs; replaced by a measurement once the 5-minute
benchmark has run on one machine). A faster GPU wins when it finishes enough sooner to cost less.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: Throughput relative to an RTX 4090 for small-model bf16 fine-tuning (estimates).
RELATIVE_SPEED: Mapping[str, float] = {
    "NVIDIA GeForce RTX 3090": 0.55,
    "NVIDIA GeForce RTX 4090": 1.0,
    "NVIDIA GeForce RTX 5090": 1.4,
    "NVIDIA RTX A4500": 0.45,
    "NVIDIA RTX A5000": 0.5,
    "NVIDIA RTX A6000": 0.65,
    "NVIDIA RTX 4000 Ada Generation": 0.4,
    "NVIDIA RTX 6000 Ada Generation": 1.05,
    "NVIDIA L4": 0.35,
    "NVIDIA A40": 0.65,
    "NVIDIA L40": 1.0,
    "NVIDIA L40S": 1.1,
    "NVIDIA A100 80GB PCIe": 1.3,
    "NVIDIA A100-SXM4-80GB": 1.45,
    "NVIDIA H100 PCIe": 1.9,
    "NVIDIA H100 80GB HBM3": 2.4,
    "NVIDIA H100 NVL": 2.2,
    "NVIDIA H200": 2.6,
}
#: Hours added to every run for pod start, image pull, model download and upload.
OVERHEAD_HOURS = 0.25
#: The cost cap is the estimate times this margin, rounded up to the next dollar.
CAP_MARGIN = 1.5


@dataclass(frozen=True)
class Offer:
    gpu: str  # RunPod GPU type ID (its display name for NVIDIA cards) or "cpu"
    memory_gb: int
    vcpus: int
    price_hr: float
    cloud: str  # "COMMUNITY" or "SECURE"
    spot: bool
    available: bool = True


@dataclass(frozen=True)
class Ranked:
    offer: Offer
    est_hours: float
    est_cost: float
    cap: float

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self.offer), "est_hours": round(self.est_hours, 2),
                "est_cost": round(self.est_cost, 2), "cap": self.cap}


def speed(gpu: str, measured: Mapping[str, float] | None = None) -> float | None:
    if measured and gpu in measured:
        return measured[gpu]
    return RELATIVE_SPEED.get(gpu)


def rank(offers: Iterable[Offer], resources: Mapping[str, Any], work_units: float,
         measured: Mapping[str, float] | None = None, allow_spot: bool = True) -> list[Ranked]:
    """Offers that fit, cheapest estimated total cost first.

    ``resources``: ``min_gpu_gb`` (0 for CPU jobs), ``min_vcpus``, ``est_hours_per_unit`` on the
    reference machine, and for CPU jobs ``core_hours_per_unit``. ``work_units`` is the job size
    in those units (for example 100k records)."""
    out = []
    cpu_job = not resources.get("min_gpu_gb")
    for offer in offers:
        if not offer.available or (offer.spot and not allow_spot):
            continue
        if offer.vcpus < resources.get("min_vcpus", 0):
            continue
        if cpu_job:
            if offer.gpu != "cpu":
                continue
            hours = resources["core_hours_per_unit"] * work_units / offer.vcpus
        else:
            if offer.gpu == "cpu" or offer.memory_gb < resources["min_gpu_gb"]:
                continue
            factor = speed(offer.gpu, measured)
            if factor is None:
                continue
            hours = resources["est_hours_per_unit"] * work_units / factor
        hours += OVERHEAD_HOURS
        cost = hours * offer.price_hr
        out.append(Ranked(offer, hours, cost, float(math.ceil(cost * CAP_MARGIN))))
    out.sort(key=lambda r: (r.est_cost, r.est_hours))
    return out


def measured_speed(gpu: str, units_per_hour: float, resources: Mapping[str, Any]) -> float:
    """Relative speed of ``gpu`` from a benchmark: reference hours per unit over measured."""
    return resources["est_hours_per_unit"] * units_per_hour
