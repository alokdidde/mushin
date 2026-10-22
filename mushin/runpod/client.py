"""Minimal RunPod API client (standard library only).

Prices and availability come from the GraphQL API (``gpuTypes`` and ``cpuFlavors``); pods are
created, read, stopped and terminated through the REST API (``/v1/pods``). The key is read from
``RUNPOD_API_KEY`` and never logged. Field names follow RunPod's public API docs; the training
thread checks them with a read-only ``offers`` call before the first paid start.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from mushin.runpod.catalog import Offer

GRAPHQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"

GPU_QUERY = """
query GpuTypes {
  gpuTypes {
    id displayName memoryInGb secureCloud communityCloud
    securePrice communityPrice secureSpotPrice communitySpotPrice
    lowestPrice(input: {gpuCount: 1}) { stockStatus maxUnreservedGpuCount }
  }
}
"""

CPU_FLAVORS = ("cpu3c", "cpu3g", "cpu5c", "cpu5g")
#: vCPU counts offered for CPU pods; RAM is the flavor's multiplier times this.
CPU_SIZES = (8, 16, 32)
CPU_QUERY = """
query CpuFlavor {
  cpuFlavors(input: {id: "%s"}) {
    id ramMultiplier
    specifics(input: {instanceId: "%s"}) { stockStatus securePrice }
  }
}
"""


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunPodError("RUNPOD_API_KEY is not set")
        self.timeout = timeout

    def _request(self, method: str, url: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                text = response.read().decode()
        except urllib.error.HTTPError as error:
            raise RunPodError(f"{method} {url.split('?')[0]}: HTTP {error.code} "
                              f"{error.read().decode()[:300]}") from None
        return json.loads(text) if text else None

    def offers(self, vcpus_per_gpu: int = 8) -> list[Offer]:
        reply = self._request("POST", GRAPHQL, {"query": GPU_QUERY})
        if reply.get("errors"):
            raise RunPodError(f"gpuTypes: {reply['errors']}")
        out = []
        for gpu in reply["data"]["gpuTypes"]:
            stock = (gpu.get("lowestPrice") or {}).get("stockStatus")
            available = stock not in (None, "", "None")
            for cloud, flag, price, spot_price in (
                ("COMMUNITY", "communityCloud", "communityPrice", "communitySpotPrice"),
                ("SECURE", "secureCloud", "securePrice", "secureSpotPrice"),
            ):
                if not gpu.get(flag):
                    continue
                for spot, value in ((False, gpu.get(price)), (True, gpu.get(spot_price))):
                    if value:
                        out.append(Offer(gpu["id"], int(gpu.get("memoryInGb") or 0),
                                         vcpus_per_gpu, float(value), cloud, spot, available))
        return out + self.cpu_offers()

    def cpu_offers(self) -> list[Offer]:
        """CPU pods (secure cloud, on-demand only), one offer per flavor and size."""
        out = []
        for flavor in CPU_FLAVORS:
            ram = 2 if flavor.endswith("c") else 4
            for vcpus in CPU_SIZES:
                instance = f"{flavor}-{vcpus}-{vcpus * ram}"
                reply = self._request("POST", GRAPHQL, {"query": CPU_QUERY % (flavor, instance)})
                for row in (reply.get("data") or {}).get("cpuFlavors") or []:
                    info = row.get("specifics") or {}
                    if info.get("securePrice"):
                        out.append(Offer("cpu", 0, vcpus, float(info["securePrice"]), "SECURE",
                                         False, info.get("stockStatus") not in (None, "", "None"),
                                         flavor))
        return out

    def create_pod(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"{REST}/pods", spec)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"{REST}/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> None:
        self._request("POST", f"{REST}/pods/{pod_id}/stop")

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"{REST}/pods/{pod_id}")

    def balance(self) -> float | None:
        reply = self._request("POST", GRAPHQL, {"query": "query { myself { clientBalance } }"})
        return ((reply.get("data") or {}).get("myself") or {}).get("clientBalance")
