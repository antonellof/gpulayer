"""Google Cloud (Compute Engine) provider adapter.

Creates one VM per worker the control plane asks for, from offers you configure (machine type,
optional GPUs, image, Spot or standard). The VM's startup script installs GPUlayer from PyPI
and runs ``gpulayer join`` with the single-use instance token, so the worker registers as that
provisioned instance. Requires ``pip install 'gpulayer[gcp]'`` and Application Default
Credentials (``gcloud auth application-default login``, a service account, or workload
identity) allowed to create and delete instances in the project.

Prices are never invented: an offer has a price only when you put one in the config, and it
is labelled as coming from your config.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gpulayer.contracts import (
    Accelerator,
    AcceleratorDevice,
    GpuVendor,
    MemoryKind,
    WorkerCapabilities,
    WorkerOrigin,
)
from gpulayer.providers.base import (
    CapacityUnavailable,
    Instance,
    InstanceOffer,
    InstanceState,
    PriceQuote,
    ProviderAdapter,
    ProviderError,
    ProviderFeatures,
    QuotaExceeded,
    WorkerBootstrap,
)

API = "https://compute.googleapis.com/compute/v1"

# GPU memory (GB) and CUDA compute capability of the GCP accelerator types this adapter knows.
# Offers with another type are rejected rather than described with guessed numbers.
GPU_SPECS: dict[str, tuple[str, float, str]] = {
    "nvidia-tesla-t4": ("NVIDIA Tesla T4", 15.0, "7.5"),
    "nvidia-l4": ("NVIDIA L4", 22.5, "8.9"),
    "nvidia-tesla-v100": ("NVIDIA Tesla V100", 16.0, "7.0"),
    "nvidia-tesla-p100": ("NVIDIA Tesla P100", 16.0, "6.0"),
    "nvidia-tesla-p4": ("NVIDIA Tesla P4", 8.0, "6.1"),
    "nvidia-tesla-a100": ("NVIDIA A100 40GB", 40.0, "8.0"),
    "nvidia-a100-80gb": ("NVIDIA A100 80GB", 80.0, "8.0"),
    "nvidia-h100-80gb": ("NVIDIA H100 80GB", 80.0, "9.0"),
}

GPU_IMAGE = (
    "projects/deeplearning-platform-release/global/images/family/"
    "common-cu129-ubuntu-2204-nvidia-580"
)
CPU_IMAGE = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"


class GCPOffer(BaseModel):
    """One kind of VM the adapter may create."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,30}$")
    machine_type: str
    gpu_type: str | None = None
    gpu_count: int = Field(0, ge=0, le=16)
    spot: bool = False
    image: str | None = None  # default: a CUDA image for GPU offers, Ubuntu otherwise
    disk_gb: int = Field(100, ge=10)
    cpus: int | None = Field(None, description="vCPUs; default from the machine type name")
    memory_gb: float | None = None
    # Extra Python packages for jobs (pip specs), and the frameworks they provide, as the
    # scheduler should see them before the machine reports its real capabilities.
    packages: list[str] = Field(default_factory=list)
    frameworks: dict[str, str] = Field(default_factory=dict)
    price_per_hour: float | None = Field(None, ge=0, description="From your config, if any.")
    currency: str = "USD"


class GCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "gcp"
    project: str
    zone: str
    network: str = "global/networks/default"
    external_ip: bool = Field(True, description="needed to reach PyPI unless you have Cloud NAT")
    offers: list[GCPOffer]
    gpulayer_version: str | None = Field(
        None, description="GPUlayer version installed on new VMs; default: this server's"
    )
    gpulayer_package: str | None = Field(
        None,
        description="pip spec for GPUlayer on new VMs instead of gpulayer==<version>, e.g. "
        "'gpulayer @ git+https://github.com/antonellof/gpulayer@main'",
    )
    labels: dict[str, str] = Field(default_factory=dict)
    operation_wait_seconds: float = Field(20.0, ge=0)


def _cpus(machine_type: str) -> int | None:
    m = re.search(r"-(\d+)$", machine_type)
    return int(m.group(1)) if m else None


def _label(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "-", value.lower())[:63]


class GCPProvider(ProviderAdapter):
    simulated = False

    def __init__(self, config: GCPConfig, *, name: str = "gcp", session: Any = None) -> None:
        self.name = name
        self.config = config
        for o in config.offers:
            if o.gpu_count and o.gpu_type not in GPU_SPECS:
                raise ProviderError(
                    f"offer {o.id}: unknown GPU type {o.gpu_type!r}; known: {sorted(GPU_SPECS)}"
                )
        self._session = session
        self._ops: dict[str, str] = {}  # instance name -> insert operation URL

    # ------------------------------------------------------------------ http

    def _http(self) -> Any:
        if self._session is None:
            try:
                import google.auth
                from google.auth.transport.requests import AuthorizedSession
            except ImportError as e:
                raise ProviderError(
                    "the GCP provider needs google-auth: pip install 'gpulayer[gcp]'"
                ) from e
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._session = AuthorizedSession(creds)
        return self._session

    def _call(self, method: str, url: str, **kw: Any) -> tuple[int, dict[str, Any]]:
        resp = self._http().request(method, url, timeout=60, **kw)
        try:
            body = resp.json() if resp.content else {}
        except ValueError:
            body = {"error": {"message": resp.text[:500]}}
        return resp.status_code, body

    @property
    def _zone_url(self) -> str:
        return f"{API}/projects/{self.config.project}/zones/{self.config.zone}"

    # ---------------------------------------------------------------- offers

    def features(self) -> ProviderFeatures:
        return ProviderFeatures(
            list_offers=True,
            pricing=any(o.price_per_hour is not None for o in self.config.offers),
            provision=True,
            terminate=True,
            worker_bootstrap=True,
        )

    def list_offers(self) -> list[InstanceOffer]:
        return [self._offer(o) for o in self.config.offers]

    def _offer(self, o: GCPOffer) -> InstanceOffer:
        devices = []
        if o.gpu_count:
            assert o.gpu_type is not None
            name, mem, cc = GPU_SPECS[o.gpu_type]
            devices = [
                AcceleratorDevice(
                    index=i,
                    accelerator=Accelerator.CUDA,
                    vendor=GpuVendor.NVIDIA,
                    name=name,
                    memory_gb=mem,
                    memory_kind=MemoryKind.DEDICATED,
                    compute_capability=cc,
                )
                for i in range(o.gpu_count)
            ]
        torch = "torch" in o.frameworks
        caps = WorkerCapabilities(
            hostname=f"gcp-{o.id}",
            os="linux",
            cpu_arch="x86_64",
            cpu_cores=o.cpus or _cpus(o.machine_type) or 1,
            memory_gb=o.memory_gb,
            python_version="3.12",
            accelerators=devices,
            frameworks=dict(o.frameworks),
            distributed_backends=(["gloo", "nccl"] if o.gpu_count else ["gloo"]) if torch else [],
            launchers=["direct", "torchrun"] if torch else [],
            origin=WorkerOrigin.PROVIDER,
            provider=self.name,
            region=self.config.zone,
            max_concurrent_tasks=max(1, o.gpu_count),
        )
        price = None
        if o.price_per_hour is not None:
            price = PriceQuote(
                amount=o.price_per_hour, currency=o.currency, source="provider config (user)"
            )
        return InstanceOffer(
            offer_id=o.id,
            provider=self.name,
            instance_type=o.machine_type + (f"+{o.gpu_count}x{o.gpu_type}" if o.gpu_count else ""),
            region=self.config.zone,
            capabilities=caps,
            price=price,
        )

    # ------------------------------------------------------------- lifecycle

    def provision(self, offer_id: str, bootstrap: WorkerBootstrap) -> Instance:
        offer = next((o for o in self.config.offers if o.id == offer_id), None)
        if offer is None:
            raise ProviderError(f"unknown offer {offer_id}")
        if not bootstrap.control_plane_url:
            raise ProviderError("the control plane has no public URL for new workers to join")
        name = _label(f"gpulayer-{bootstrap.instance_ref}")
        status, body = self._call(
            "POST",
            f"{self._zone_url}/instances",
            params={"requestId": str(uuid.uuid5(uuid.NAMESPACE_URL, name))},  # retry-safe
            json=self._instance_body(name, offer, bootstrap),
        )
        if status >= 400:
            raise _error(body, f"create {name}")
        op_url = body.get("selfLink")
        if op_url:
            self._ops[name] = op_url
            self._wait(op_url, name)
        return Instance(
            provider_instance_id=name,
            provider=self.name,
            offer_id=offer_id,
            state=InstanceState.PROVISIONING,
        )

    def _wait(self, op_url: str, name: str) -> None:
        """Wait briefly for the insert to be accepted or refused (out of capacity, quota)."""
        deadline = time.monotonic() + self.config.operation_wait_seconds
        while time.monotonic() < deadline:
            status, op = self._call("GET", op_url)
            if status < 400 and op.get("status") == "DONE":
                if op.get("error"):
                    raise _error(op, f"create {name}")
                return
            time.sleep(2)

    def _instance_body(self, name: str, o: GCPOffer, b: WorkerBootstrap) -> dict[str, Any]:
        zone = self.config.zone
        version = self.config.gpulayer_version or _own_version()
        meta = {
            "startup-script": STARTUP_SCRIPT,
            "gpulayer-url": b.control_plane_url or "",
            "gpulayer-instance-ref": b.instance_ref,
            "gpulayer-token": b.join_token,
            "gpulayer-fingerprint": b.tls_fingerprint or "",
            "gpulayer-packages": "\n".join(
                [self.config.gpulayer_package or f"gpulayer=={version}", *o.packages]
            ),
        }
        if o.gpu_count:
            meta["install-nvidia-driver"] = "True"
        body: dict[str, Any] = {
            "name": name,
            "machineType": f"zones/{zone}/machineTypes/{o.machine_type}",
            "labels": {
                "gpulayer": "worker",
                "gpulayer-ref": _label(b.instance_ref),
                **{_label(k): _label(v) for k, v in {**self.config.labels, **b.labels}.items()},
            },
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "initializeParams": {
                        "sourceImage": o.image or (GPU_IMAGE if o.gpu_count else CPU_IMAGE),
                        "diskSizeGb": str(o.disk_gb),
                        "diskType": f"zones/{zone}/diskTypes/pd-balanced",
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "network": self.config.network,
                    **(
                        {"accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}]}
                        if self.config.external_ip
                        else {}
                    ),
                }
            ],
            "metadata": {"items": [{"key": k, "value": v} for k, v in meta.items()]},
            "scheduling": {
                "automaticRestart": False,
                "onHostMaintenance": "TERMINATE" if o.gpu_count or o.spot else "MIGRATE",
                **(
                    {"provisioningModel": "SPOT", "instanceTerminationAction": "DELETE"}
                    if o.spot
                    else {}
                ),
            },
        }
        if o.gpu_count:
            body["guestAccelerators"] = [
                {
                    "acceleratorType": f"zones/{zone}/acceleratorTypes/{o.gpu_type}",
                    "acceleratorCount": o.gpu_count,
                }
            ]
        return body

    def get_instance(self, provider_instance_id: str) -> Instance:
        status, body = self._call("GET", f"{self._zone_url}/instances/{provider_instance_id}")
        offer_id = ""
        if status == 404:
            # Gone: deleted, or its creation failed. The insert operation says which.
            op_url = self._ops.get(provider_instance_id)
            if op_url:
                s, op = self._call("GET", op_url)
                if s < 400 and op.get("error"):
                    msg = "; ".join(e.get("message", "") for e in op["error"].get("errors", []))
                    return Instance(
                        provider_instance_id=provider_instance_id,
                        provider=self.name,
                        offer_id=offer_id,
                        state=InstanceState.FAILED,
                        error=msg or "instance creation failed",
                    )
            return Instance(
                provider_instance_id=provider_instance_id,
                provider=self.name,
                offer_id=offer_id,
                state=InstanceState.TERMINATED,
            )
        if status >= 400:
            raise _error(body, f"describe {provider_instance_id}")
        state = {
            "PROVISIONING": InstanceState.PROVISIONING,
            "STAGING": InstanceState.PROVISIONING,
            "RUNNING": InstanceState.RUNNING,
            "STOPPING": InstanceState.TERMINATING,
            "SUSPENDING": InstanceState.TERMINATING,
            "SUSPENDED": InstanceState.TERMINATED,
            "STOPPED": InstanceState.TERMINATED,
            "TERMINATED": InstanceState.TERMINATED,  # GCP's name for a stopped VM
        }.get(body.get("status", ""), InstanceState.PROVISIONING)
        return Instance(
            provider_instance_id=provider_instance_id,
            provider=self.name,
            offer_id=offer_id,
            state=state,
        )

    def terminate(self, provider_instance_id: str) -> Instance:
        status, body = self._call("DELETE", f"{self._zone_url}/instances/{provider_instance_id}")
        if status >= 400 and status != 404:
            raise _error(body, f"delete {provider_instance_id}")
        return Instance(
            provider_instance_id=provider_instance_id,
            provider=self.name,
            offer_id="",
            state=InstanceState.TERMINATED if status == 404 else InstanceState.TERMINATING,
        )


def _error(body: dict[str, Any], what: str) -> ProviderError:
    err = body.get("error") or {}
    errors = err.get("errors") or [err]
    text = "; ".join(
        f"{e.get('reason') or e.get('code') or ''} {e.get('message', '')}".strip() for e in errors
    )
    blob = str(err).upper()
    if "RESOURCE_POOL_EXHAUSTED" in blob or "STOCKOUT" in blob or "DOES NOT HAVE ENOUGH" in blob:
        return CapacityUnavailable(f"GCP has no capacity to {what}: {text}")
    if "QUOTA" in blob:
        return QuotaExceeded(f"GCP quota prevents {what}: {text}")
    return ProviderError(f"GCP refused to {what}: {text}")


def _own_version() -> str:
    from gpulayer import __version__

    return __version__


# Runs as root at every boot. Reads its settings from instance metadata, installs GPUlayer into
# its own environment and runs `gpulayer join` as an unprivileged systemd service.
STARTUP_SCRIPT = r"""#!/bin/bash
set -euo pipefail
md() {
  curl -sf -H Metadata-Flavor:Google \
    "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}
if ! id gpulayer >/dev/null 2>&1; then useradd --create-home --shell /usr/sbin/nologin gpulayer; fi
if [ ! -x /opt/gpulayer/bin/gpulayer ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
  UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.12 /opt/gpulayer
  md attributes/gpulayer-packages > /tmp/gpulayer-requirements.txt  # one pip spec per line
  uv pip install --python /opt/gpulayer/bin/python -r /tmp/gpulayer-requirements.txt
fi
fp=$(md attributes/gpulayer-fingerprint || true)
umask 077
cat > /etc/gpulayer-worker.env <<EOF
GPULAYER_JOIN_TOKEN=$(md attributes/gpulayer-token)
GPULAYER_HOME=/var/lib/gpulayer
EOF
mkdir -p /var/lib/gpulayer && chown gpulayer /var/lib/gpulayer
cat > /etc/systemd/system/gpulayer-worker.service <<EOF
[Unit]
Description=GPUlayer worker
After=network-online.target
[Service]
User=gpulayer
EnvironmentFile=/etc/gpulayer-worker.env
ExecStart=/opt/gpulayer/bin/gpulayer join $(md attributes/gpulayer-url) \
  --instance-ref $(md attributes/gpulayer-instance-ref) \
  ${fp:+--fingerprint $fp} --advertise-address $(md network-interfaces/0/ip) --name $(hostname)
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now gpulayer-worker.service
"""
