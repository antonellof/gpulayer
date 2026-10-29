from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpulayer.artifacts.store import LocalArtifactStore
from gpulayer.contracts import (
    Accelerator,
    AcceleratorDevice,
    GpuVendor,
    MemoryKind,
    WorkerCapabilities,
    WorkerOrigin,
)
from gpulayer.control_plane.service import ControlPlane, ControlPlaneConfig
from gpulayer.storage.sqlite import SqliteStore


class Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


def make_caps(**kw) -> WorkerCapabilities:
    base = dict(
        hostname="h",
        os="linux",
        cpu_arch="x86_64",
        cpu_cores=8,
        memory_gb=32.0,
        disk_free_gb=100.0,
        python_version="3.11.5",
        origin=WorkerOrigin.LOCAL,
    )
    base.update(kw)
    return WorkerCapabilities(**base)


def cuda_device(i: int = 0, mem: float = 24.0, cc: str = "8.6") -> AcceleratorDevice:
    return AcceleratorDevice(
        index=i,
        accelerator=Accelerator.CUDA,
        vendor=GpuVendor.NVIDIA,
        name="NVIDIA RTX",
        memory_gb=mem,
        compute_capability=cc,
    )


def mps_device(mem: float = 32.0) -> AcceleratorDevice:
    return AcceleratorDevice(
        index=0,
        accelerator=Accelerator.MPS,
        vendor=GpuVendor.APPLE,
        name="Apple GPU",
        memory_gb=mem,
        memory_kind=MemoryKind.UNIFIED,
    )


TORCH_CAPS = dict(
    frameworks={"torch": "2.4.0"},
    distributed_backends=["gloo"],
    launchers=["torchrun"],
    advertise_address="127.0.0.1",
)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cp(tmp_path: Path, clock: Clock) -> ControlPlane:
    return ControlPlane(
        SqliteStore(tmp_path / "db.sqlite"),
        LocalArtifactStore(tmp_path / "artifacts"),
        config=ControlPlaneConfig(lease_ttl_seconds=10),
        clock=clock,
    )


@pytest.fixture
def real_cp(tmp_path: Path) -> ControlPlane:
    """Control plane on the wall clock with short leases, for real process execution."""
    return ControlPlane(
        SqliteStore(tmp_path / "db.sqlite"),
        LocalArtifactStore(tmp_path / "artifacts"),
        config=ControlPlaneConfig(heartbeat_interval_seconds=0.2, lease_ttl_seconds=10),
    )
