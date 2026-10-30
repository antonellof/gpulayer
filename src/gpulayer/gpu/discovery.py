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

import contextlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
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
out = {"torch": None, "packages": {}}
try:
    from importlib.metadata import PackageNotFoundError, version
    for dist in %(packages)r:
        try:
            out["packages"][dist] = version(dist)
        except PackageNotFoundError:
            pass
except Exception:
    pass
# Printed before torch is imported, so a crashing torch cannot hide the other frameworks.
print("PACKAGES " + json.dumps(out["packages"]), flush=True)
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

# GPU-relevant packages reported as frameworks, read from installed-distribution metadata
# (nothing is imported). Distribution name -> framework name used in ResourceRequest.
GPU_PACKAGES: dict[str, str] = {
    "torch": "torch",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "tensorflow": "tensorflow",
    "mlx": "mlx",
    "cupy": "cupy",
    "cupy-cuda12x": "cupy",
    "cupy-cuda11x": "cupy",
    "onnxruntime": "onnxruntime",
    "onnxruntime-gpu": "onnxruntime",
    "tensorrt": "tensorrt",
    "triton": "triton",
    "vllm": "vllm",
    "transformers": "transformers",
    "numpy": "numpy",
}
_TORCH_PROBE = _TORCH_PROBE % {"packages": sorted(GPU_PACKAGES)}


# Can JAX and CuPy drive a CUDA GPU? Installed is not enough: a CPU-only jaxlib or a CuPy built
# for another CUDA major version imports fine and then fails on the GPU. Run only when one of
# them is installed, in its own interpreter; JAX must not preallocate GPU memory here.
_ACCEL_PROBE = r"""
import json, os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
out = {}
try:
    import jax
    if jax.devices("cuda"):
        out["jax.cuda"] = jax.__version__
except Exception:
    pass
try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        int((cupy.arange(4) * 2).sum())  # a kernel must actually run
        out["cupy.cuda"] = cupy.__version__
except Exception:
    pass
print("ACCEL " + json.dumps(out), flush=True)
"""


def probe_gpu_frameworks(
    packages: dict[str, str],
    python: str = sys.executable,
    runner: CommandRunner | None = None,
    timeout: float = 90,
) -> dict[str, str]:
    """``jax.cuda`` / ``cupy.cuda`` when those packages can run on a CUDA GPU. Never raises."""
    if not ({"jax", "jaxlib", "cupy", "cupy-cuda12x", "cupy-cuda11x"} & set(packages)):
        return {}
    code, out = (runner or run_command)([python, "-c", _ACCEL_PROBE], timeout)
    for line in out.splitlines():
        if line.startswith("ACCEL "):
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                found = json.loads(line[6:])
                return {str(k): str(v).split("+")[0] for k, v in found.items()}
    return {}


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
    packages: dict[str, str] = field(default_factory=dict)


def probe_torch(
    python: str = sys.executable, runner: CommandRunner = run_command, timeout: float = 60
) -> TorchProbe:
    """Probe torch in an isolated interpreter. Never raises."""
    code, out = runner([python, "-c", _TORCH_PROBE], timeout)
    lines = out.strip().splitlines()
    packages: dict[str, str] = {}
    for line in lines:
        if line.startswith("PACKAGES "):
            with contextlib.suppress(json.JSONDecodeError):
                packages = {k: str(v) for k, v in json.loads(line[9:]).items()}
    results = [line for line in lines if not line.startswith("PACKAGES ")]
    if code != 0 or not results:
        return TorchProbe(error=f"torch probe exited with {code}", packages=packages)
    try:
        data = json.loads(results[-1])
    except json.JSONDecodeError:
        return TorchProbe(error="torch probe returned unparseable output", packages=packages)
    packages = packages or {k: str(v) for k, v in (data.get("packages") or {}).items()}
    if not data.get("torch"):
        return TorchProbe(error=data.get("error", "torch not installed"), packages=packages)
    return TorchProbe(
        torch_version=data["torch"],
        cuda_available=bool(data.get("cuda_available")),
        cuda_devices=list(data.get("cuda_devices", [])),
        mps_built=bool(data.get("mps_built")),
        mps_available=bool(data.get("mps_available")),
        gloo=bool(data.get("gloo")),
        nccl=bool(data.get("nccl")),
        packages=packages,
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


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


def detect_capabilities(
    *,
    origin: WorkerOrigin = WorkerOrigin.LOCAL,
    labels: dict[str, str] | None = None,
    advertise_address: str | None = None,
    workdir: str | None = None,
    runner: CommandRunner = run_command,
    torch_probe: Callable[[], TorchProbe] | None = None,
    max_concurrent_tasks: int | None = None,
    gpus: Sequence[int] | None = None,
) -> WorkerCapabilities:
    """``max_concurrent_tasks`` defaults to one task per usable discrete GPU (at least 1).

    ``gpus`` limits the worker to those discrete GPU indices (as ``nvidia-smi`` numbers them);
    an empty list offers none. Use it to split one machine's GPUs between workers, so that two
    workers never offer the same GPU.
    """
    notes: list[str] = []
    tp = torch_probe() if torch_probe else probe_torch(runner=runner)

    accelerators: list[AcceleratorDevice] = []
    nvidia, note = detect_nvidia(runner)
    if note:
        notes.append(note)
    if nvidia:
        # The driver works, so CUDA workloads can run (CUDA binaries, CuPy, JAX, ...). Whether
        # PyTorch can use the GPU is reported separately as the 'torch.cuda' capability.
        accelerators.extend(nvidia)
        if tp.torch_version and not tp.cuda_available:
            notes.append("NVIDIA GPU usable, but torch in this environment has no CUDA support")
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
    if apple_silicon() or tp.mps_available:
        # The Apple GPU is reached through Metal (PyTorch MPS, MLX, Core ML, ...). Whether
        # PyTorch can use it is reported separately as the 'torch.mps' capability.
        accelerators.append(
            AcceleratorDevice(
                index=0,
                accelerator=Accelerator.MPS,
                vendor=GpuVendor.APPLE,
                name=f"Apple {platform.machine()} GPU (Metal)",
                # Unified memory: the GPU shares system RAM. Report it as such, never as VRAM.
                memory_gb=mem_gb,
                memory_kind=MemoryKind.UNIFIED,
            )
        )
        if tp.torch_version and not tp.mps_available:
            notes.append("Apple GPU present, but torch in this environment cannot use MPS")

    rocm = detect_rocm(runner)
    if rocm:
        accelerators.extend(rocm)
        notes.append("AMD ROCm device detected; ROCm execution is not supported yet")

    if gpus is not None:
        discrete = (Accelerator.CUDA, Accelerator.ROCM)
        present = {d.index for d in accelerators if d.accelerator in discrete}
        unknown = sorted(set(gpus) - present)
        if unknown:
            raise ValueError(
                f"GPU index {unknown} not found on this machine (found: {sorted(present)})"
            )
        accelerators = [
            d for d in accelerators if d.accelerator not in discrete or d.index in set(gpus)
        ]
        notes.append(f"limited to GPUs {sorted(gpus)}" if gpus else "GPUs not offered")

    frameworks: dict[str, str] = {}
    for dist, version in tp.packages.items():
        name = GPU_PACKAGES.get(dist)
        if name:
            frameworks[name] = version.split("+")[0]
    backends: list[str] = []
    launchers: list[str] = []
    if tp.torch_version:
        frameworks["torch"] = tp.torch_version
        # Pseudo-frameworks: "this torch can drive that GPU". The PyTorch layer requires them.
        if tp.cuda_available:
            frameworks["torch.cuda"] = tp.torch_version
        if tp.mps_available:
            frameworks["torch.mps"] = tp.torch_version
        if tp.gloo:
            backends.append("gloo")
        if tp.nccl:
            backends.append("nccl")
        if backends:
            launchers += ["direct", "torchrun"]
    elif tp.error:
        notes.append(f"torch unavailable: {tp.error}")
    if nvidia:
        # Like 'torch.cuda': this JAX / CuPy can run on the GPU, not just import.
        frameworks.update(probe_gpu_frameworks(tp.packages, runner=runner))

    disk = shutil.disk_usage(workdir or os.getcwd())
    caps = WorkerCapabilities(
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
        max_concurrent_tasks=max_concurrent_tasks or 1,
        notes=notes,
    )
    if max_concurrent_tasks is None:
        from gpulayer.control_plane.scheduler import default_max_tasks

        caps = caps.model_copy(update={"max_concurrent_tasks": default_max_tasks(caps)})
    return caps
