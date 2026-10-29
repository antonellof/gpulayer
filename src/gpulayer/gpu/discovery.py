"""Hardware and runtime capability discovery.

Detection is split into independent probes so each can be mocked in CPU-only CI:

* host probe      -- OS, CPU, RAM, disk (stdlib only)
* nvidia-smi      -- NVIDIA devices as seen by the driver (no CUDA runtime needed)
* torch probe     -- runs in a *subprocess* so a broken/missing torch install can never
                     crash or slow the worker; reports torch version, CUDA/MPS availability
                     and torch.distributed backends
* rocm-smi        -- presence only; ROCm execution is not supported (detection-only)

Discovery reports uncertainty in ``WorkerCapabilities.notes`` instead of guessing.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gpulayer.contracts import (
    Accelerator,
    AcceleratorDevice,
    GpuVendor,
    MemoryKind,
    WorkerCapabilities,
    WorkerOrigin,
)

CommandRunner = Callable[[list[str], float], "tuple[int, str]"]

_TORCH_PROBE = r"""
import json, sys
out = {"torch": None}
try:
    import torch
except Exception as e:  # ImportError or a broken install
    out["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out)); sys.exit(0)
out["torch"] = torch.__version__.split("+")[0]
out["cuda_built"] = torch.backends.cuda.is_built()
out["cuda_available"] = bool(torch.cuda.is_available())
out["cuda_runtime"] = torch.version.cuda
out["cuda_devices"] = []
if out["cuda_available"]:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["cuda_devices"].append({"name": p.name, "memory_bytes": p.total_memory,
                                    "cc": f"{p.major}.{p.minor}"})
mps = getattr(torch.backends, "mps", None)
out["mps_built"] = bool(mps and mps.is_built())
out["mps_available"] = bool(mps and mps.is_available())
try:
    import torch.distributed as dist
    out["dist_available"] = dist.is_available()
    out["gloo"] = dist.is_available() and dist.is_gloo_available()
    out["nccl"] = dist.is_available() and dist.is_nccl_available()
except Exception:
    out["dist_available"] = False
print(json.dumps(out))
"""


def run_command(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, PermissionError):
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    return proc.returncode, proc.stdout


@dataclass
class TorchProbe:
    torch_version: str | None = None
    error: str | None = None
    cuda_available: bool = False
    cuda_devices: list[dict[str, Any]] = field(default_factory=list)
    mps_built: bool = False
    mps_available: bool = False
    gloo: bool = False
    nccl: bool = False


def probe_torch(
    python: str = sys.executable, runner: CommandRunner = run_command, timeout: float = 60
) -> TorchProbe:
    """Probe torch in an isolated interpreter. Never raises."""
    code, out = runner([python, "-c", _TORCH_PROBE], timeout)
    if code != 0 or not out.strip():
        return TorchProbe(error=f"torch probe exited with {code}")
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return TorchProbe(error="torch probe returned unparseable output")
    if not data.get("torch"):
        return TorchProbe(error=data.get("error", "torch not installed"))
    return TorchProbe(
        torch_version=data["torch"],
        cuda_available=bool(data.get("cuda_available")),
        cuda_devices=list(data.get("cuda_devices", [])),
        mps_built=bool(data.get("mps_built")),
        mps_available=bool(data.get("mps_available")),
        gloo=bool(data.get("gloo")),
        nccl=bool(data.get("nccl")),
    )


def detect_nvidia(
    runner: CommandRunner = run_command,
) -> tuple[list[AcceleratorDevice], str | None]:
    """Query the NVIDIA driver. Returns (devices, note)."""
    code, out = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        15,
    )
    if code == 127:
        return [], None  # no driver tooling: simply no NVIDIA GPUs visible
    if code != 0:
        return [], f"nvidia-smi present but failed (exit {code}); NVIDIA devices not reported"
    devices = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mem_mib, cc, driver = parts[:5]
        try:
            mem_gb: float | None = round(float(mem_mib) / 1024, 2)
        except ValueError:
            mem_gb = None
        devices.append(
            AcceleratorDevice(
                index=int(idx),
                accelerator=Accelerator.CUDA,
                vendor=GpuVendor.NVIDIA,
                name=name,
                memory_gb=mem_gb,
                memory_kind=MemoryKind.DEDICATED,
                compute_capability=cc if cc and cc != "[N/A]" else None,
                driver_version=driver,
            )
        )
    return devices, None


def detect_rocm(runner: CommandRunner = run_command) -> list[AcceleratorDevice]:
    code, out = runner(["rocm-smi", "--showproductname", "--json"], 15)
    if code != 0:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    devices = []
    for i, (_card, info) in enumerate(sorted(data.items())):
        if not isinstance(info, dict):
            continue
        devices.append(
            AcceleratorDevice(
                index=i,
                accelerator=Accelerator.ROCM,
                vendor=GpuVendor.AMD,
                name=str(info.get("Card series") or info.get("Card model") or "AMD GPU"),
                execution_supported=False,
            )
        )
    return devices


def host_memory_gb(runner: CommandRunner = run_command) -> float | None:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024**2, 2)
        elif system == "Darwin":
            code, out = runner(["sysctl", "-n", "hw.memsize"], 5)
            if code == 0:
                return round(int(out.strip()) / 1024**3, 2)
    except (OSError, ValueError):
        pass
    return None


def default_advertise_address() -> str:
    """Best-effort routable address; overridable with --advertise-address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET; no packet is sent for UDP connect
            return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def detect_capabilities(
    *,
    origin: WorkerOrigin = WorkerOrigin.LOCAL,
    labels: dict[str, str] | None = None,
    advertise_address: str | None = None,
    workdir: str | None = None,
    runner: CommandRunner = run_command,
    torch_probe: Callable[[], TorchProbe] | None = None,
    max_concurrent_tasks: int = 1,
) -> WorkerCapabilities:
    notes: list[str] = []
    tp = torch_probe() if torch_probe else probe_torch(runner=runner)

    accelerators: list[AcceleratorDevice] = []
    nvidia, note = detect_nvidia(runner)
    if note:
        notes.append(note)
    if nvidia:
        accelerators.extend(nvidia)
        if not tp.cuda_available:
            # Hardware is visible but no usable CUDA runtime in this Python environment.
            for d in nvidia:
                d.execution_supported = False
            notes.append("NVIDIA GPU visible but torch CUDA runtime unavailable in worker env")
    elif tp.cuda_available:
        for i, cd in enumerate(tp.cuda_devices):
            accelerators.append(
                AcceleratorDevice(
                    index=i,
                    accelerator=Accelerator.CUDA,
                    vendor=GpuVendor.NVIDIA,
                    name=str(cd.get("name", "cuda")),
                    memory_gb=round(cd["memory_bytes"] / 1024**3, 2)
                    if cd.get("memory_bytes")
                    else None,
                    compute_capability=cd.get("cc"),
                )
            )

    mem_gb = host_memory_gb(runner)
    if tp.mps_available:
        accelerators.append(
            AcceleratorDevice(
                index=0,
                accelerator=Accelerator.MPS,
                vendor=GpuVendor.APPLE,
                name=f"Apple {platform.machine()} GPU (MPS)",
                # Unified memory: the GPU shares system RAM. Report it as such, never as VRAM.
                memory_gb=mem_gb,
                memory_kind=MemoryKind.UNIFIED,
            )
        )
    elif tp.mps_built:
        notes.append("torch built with MPS but MPS is not available on this machine")

    rocm = detect_rocm(runner)
    if rocm:
        accelerators.extend(rocm)
        notes.append("AMD ROCm device detected; ROCm execution is not supported yet")

    frameworks: dict[str, str] = {}
    backends: list[str] = []
    launchers: list[str] = []
    if tp.torch_version:
        frameworks["torch"] = tp.torch_version
        if tp.gloo:
            backends.append("gloo")
        if tp.nccl:
            backends.append("nccl")
        if backends:
            launchers.append("torchrun")
    elif tp.error:
        notes.append(f"torch unavailable: {tp.error}")

    disk = shutil.disk_usage(workdir or os.getcwd())
    return WorkerCapabilities(
        hostname=socket.gethostname(),
        os=platform.system().lower(),
        cpu_arch=platform.machine().lower(),
        cpu_cores=os.cpu_count() or 1,
        memory_gb=mem_gb,
        disk_free_gb=round(disk.free / 1024**3, 2),
        python_version=platform.python_version(),
        accelerators=accelerators,
        frameworks=frameworks,
        distributed_backends=backends,
        launchers=launchers,
        advertise_address=advertise_address,
        origin=origin,
        labels=dict(labels or {}),
        max_concurrent_tasks=max_concurrent_tasks,
        notes=notes,
    )
