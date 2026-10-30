"""GCP provider adapter against a fake Compute Engine API (no network, no credentials)."""

import json

import pytest

from gpulayer.contracts import Accelerator
from gpulayer.errors import ValidationFailed
from gpulayer.providers.base import (
    CapacityUnavailable,
    InstanceState,
    ProviderError,
    QuotaExceeded,
    WorkerBootstrap,
)
from gpulayer.providers.config import load_providers
from gpulayer.providers.gcp import GCPConfig, GCPProvider


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = self.content.decode()

    def json(self):
        return self._body


class FakeCompute:
    """Answers in the shape of the Compute Engine v1 API; records every request."""

    def __init__(self):
        self.calls, self.instances, self.op_error = [], {}, None

    def request(self, method, url, timeout=None, params=None, json=None):
        self.calls.append((method, url, params, json))
        if url.endswith("/operations/op-1"):
            body = {"status": "DONE"}
            if self.op_error:
                body["error"] = {"errors": [self.op_error]}
            return Resp(200, body)
        if method == "POST" and url.endswith("/instances"):
            if not self.op_error:
                self.instances[json["name"]] = "PROVISIONING"
            return Resp(200, {"selfLink": url.rsplit("/instances", 1)[0] + "/operations/op-1"})
        name = url.rsplit("/", 1)[1]
        if method == "GET":
            if name not in self.instances:
                return Resp(404, {"error": {"code": 404, "message": "not found"}})
            return Resp(200, {"status": self.instances[name]})
        if method == "DELETE":
            if self.instances.pop(name, None) is None:
                return Resp(404, {"error": {"code": 404, "message": "not found"}})
            return Resp(200, {"status": "RUNNING"})
        raise AssertionError((method, url))


CONF = {
    "project": "p",
    "zone": "europe-west1-b",
    "operation_wait_seconds": 1,
    "offers": [
        {
            "id": "t4",
            "machine_type": "n1-standard-8",
            "gpu_type": "nvidia-tesla-t4",
            "gpu_count": 2,
            "spot": True,
            "packages": ["torch"],
            "frameworks": {"torch": "2.15.0", "torch.cuda": "2.15.0"},
            "price_per_hour": 0.5,
        },
        {"id": "cpu", "machine_type": "e2-standard-4"},
    ],
}


def provider():
    api = FakeCompute()
    return GCPProvider(GCPConfig.model_validate(CONF), session=api), api


def boot(ref="inst-abc123"):
    return WorkerBootstrap(
        control_plane_url="https://10.0.0.2:8470",
        tls_fingerprint="sha256:" + "0" * 64,
        instance_ref=ref,
        join_token="secret-token",
        labels={"reserved-for": "job-1"},
    )


def test_offers_describe_known_hardware_only():
    p, _ = provider()
    t4, cpu = p.list_offers()
    caps = t4.capabilities
    assert [d.accelerator for d in caps.devices(Accelerator.CUDA)] == [Accelerator.CUDA] * 2
    assert (
        caps.accelerators[0].memory_gb == 15.0 and caps.accelerators[0].compute_capability == "7.5"
    )
    assert caps.cpu_cores == 8 and caps.max_concurrent_tasks == 2
    assert caps.frameworks["torch.cuda"] == "2.15.0" and "nccl" in caps.distributed_backends
    assert t4.price.amount == 0.5 and "config" in t4.price.source
    assert (
        cpu.price is None
        and cpu.capabilities.accelerators == []
        and cpu.capabilities.cpu_cores == 4
    )
    with pytest.raises(ProviderError, match="unknown GPU type"):
        GCPProvider(
            GCPConfig.model_validate(
                {
                    **CONF,
                    "offers": [
                        {
                            "id": "x",
                            "machine_type": "a2",
                            "gpu_type": "nvidia-rtx-9000",
                            "gpu_count": 1,
                        }
                    ],
                }
            ),
            session=FakeCompute(),
        )


def test_provision_creates_a_spot_gpu_vm_that_joins_with_its_instance_token():
    p, api = provider()
    inst = p.provision("t4", boot())
    assert (
        inst.state == InstanceState.PROVISIONING
        and inst.provider_instance_id == "gpulayer-inst-abc123"
    )
    method, url, params, body = api.calls[0]
    assert method == "POST" and url.endswith("/projects/p/zones/europe-west1-b/instances")
    assert params["requestId"]  # the same name always gets the same id: a retry creates nothing new
    assert body["machineType"].endswith("machineTypes/n1-standard-8")
    assert body["guestAccelerators"][0]["acceleratorType"].endswith(
        "acceleratorTypes/nvidia-tesla-t4"
    )
    assert body["scheduling"]["provisioningModel"] == "SPOT"
    assert body["scheduling"]["onHostMaintenance"] == "TERMINATE"
    meta = {i["key"]: i["value"] for i in body["metadata"]["items"]}
    assert (
        meta["gpulayer-token"] == "secret-token" and meta["gpulayer-instance-ref"] == "inst-abc123"
    )
    assert (
        meta["gpulayer-fingerprint"].startswith("sha256:")
        and meta["install-nvidia-driver"] == "True"
    )
    assert (
        meta["gpulayer-packages"].startswith("gpulayer==") and "torch" in meta["gpulayer-packages"]
    )
    assert "gpulayer join" in meta["startup-script"] and "--instance-ref" in meta["startup-script"]
    assert body["labels"]["reserved-for"] == "job-1" and body["labels"]["gpulayer"] == "worker"
    assert "secret-token" not in json.dumps(body["labels"])


def test_states_and_idempotent_terminate():
    p, api = provider()
    p.provision("cpu", boot("inst-1"))
    assert p.get_instance("gpulayer-inst-1").state == InstanceState.PROVISIONING
    api.instances["gpulayer-inst-1"] = "RUNNING"
    assert p.get_instance("gpulayer-inst-1").state == InstanceState.RUNNING
    assert p.terminate("gpulayer-inst-1").state == InstanceState.TERMINATING
    assert p.terminate("gpulayer-inst-1").state == InstanceState.TERMINATED  # already gone
    assert p.get_instance("gpulayer-inst-1").state == InstanceState.TERMINATED


def test_capacity_and_quota_errors_are_typed():
    p, api = provider()
    api.op_error = {"code": "ZONE_RESOURCE_POOL_EXHAUSTED", "message": "no capacity"}
    with pytest.raises(CapacityUnavailable) as e:
        p.provision("t4", boot("inst-2"))
    assert e.value.retryable
    assert p.get_instance("gpulayer-inst-2").state == InstanceState.FAILED
    api.op_error = {"code": "QUOTA_EXCEEDED", "message": "GPUS_ALL_REGIONS"}
    with pytest.raises(QuotaExceeded):
        p.provision("t4", boot("inst-3"))


def test_load_providers_from_json(tmp_path):
    f = tmp_path / "providers.json"
    f.write_text(json.dumps({"gcp": {"type": "gcp", **CONF}}))
    (adapter,) = load_providers(f)
    assert adapter.name == "gcp" and not adapter.simulated
    f.write_text(json.dumps({"x": {"type": "nope"}}))
    with pytest.raises(ValidationFailed, match="unknown type"):
        load_providers(f)
