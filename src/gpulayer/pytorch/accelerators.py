"""Accelerator classes with Lightning's names. ``Trainer(accelerator=CUDAAccelerator())`` is the
same as ``Trainer(accelerator="cuda")``; the classes also answer what a machine has."""

from __future__ import annotations

import os
from typing import Any

import torch

from gpulayer.errors import ValidationFailed

_MB = 2**20


class Accelerator:
    """Base class: one device type (``name()``)."""

    @staticmethod
    def name() -> str:
        """The Trainer's accelerator string (``"cuda"``, ``"mps"``, ...)."""
        raise NotImplementedError

    @staticmethod
    def is_available() -> bool:
        """True when this machine can use the device type."""
        raise NotImplementedError

    @staticmethod
    def auto_device_count() -> int:
        """Devices ``devices="auto"`` uses on this machine."""
        return 1

    @staticmethod
    def parse_devices(devices: int | str | list[int]) -> Any:
        """``devices=`` as given (GPUlayer gives each process one device)."""
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[torch.device]:
        """The devices of the processes on this machine."""
        return [torch.device("cpu")]

    def setup_device(self, device: torch.device) -> None:
        """Prepare ``device`` before training (the Trainer's Runtime does this)."""

    def setup(self, trainer: Any) -> None:
        """Called when training starts."""

    def get_device_stats(self, device: torch.device | str | int) -> dict[str, float]:
        """Memory figures for ``device``, in MB."""
        return {}

    def teardown(self) -> None:
        """Release what ``setup`` took."""


class CPUAccelerator(Accelerator):
    @staticmethod
    def name() -> str:
        return "cpu"

    @staticmethod
    def is_available() -> bool:
        return True

    def get_device_stats(self, device: Any = None) -> dict[str, float]:
        try:
            import psutil
        except ImportError as e:
            raise ValidationFailed("CPU device stats need psutil (pip install psutil)") from e
        return {
            "cpu_percent": float(psutil.cpu_percent()),
            "cpu_vm_percent": float(psutil.virtual_memory().percent),
            "cpu_ram_used_mb": psutil.Process().memory_info().rss / _MB,
        }


class CUDAAccelerator(Accelerator):
    """NVIDIA GPUs (and AMD GPUs through a ROCm build of torch)."""

    @staticmethod
    def name() -> str:
        return "cuda"

    @staticmethod
    def is_available() -> bool:
        from gpulayer.pytorch.runtime import cuda_available

        return cuda_available()

    @staticmethod
    def auto_device_count() -> int:
        return torch.cuda.device_count()

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[torch.device]:
        ids = range(devices) if isinstance(devices, int) else devices
        return [torch.device("cuda", i) for i in ids]

    def setup_device(self, device: torch.device) -> None:
        torch.cuda.set_device(device)

    def get_device_stats(self, device: Any = None) -> dict[str, float]:
        dev = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        return {
            "memory_allocated_mb": torch.cuda.memory_allocated(dev) / _MB,
            "memory_reserved_mb": torch.cuda.memory_reserved(dev) / _MB,
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(dev) / _MB,
        }

    def teardown(self) -> None:
        torch.cuda.empty_cache()


class MPSAccelerator(Accelerator):
    """Apple GPUs (single process)."""

    @staticmethod
    def name() -> str:
        return "mps"

    @staticmethod
    def is_available() -> bool:
        return torch.backends.mps.is_available()

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[torch.device]:
        return [torch.device("mps")]

    def get_device_stats(self, device: Any = None) -> dict[str, float]:
        return {
            "memory_allocated_mb": torch.mps.current_allocated_memory() / _MB,
            "driver_memory_mb": torch.mps.driver_allocated_memory() / _MB,
        }


class XLAAccelerator(Accelerator):
    """Google TPUs through torch_xla (implemented, not run on a TPU)."""

    @staticmethod
    def name() -> str:
        return "tpu"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch_xla  # noqa: F401
        except ImportError:
            return False
        return os.environ.get("PJRT_DEVICE") == "TPU"

    @staticmethod
    def auto_device_count() -> int:
        import torch_xla.runtime as xr

        return int(xr.global_runtime_device_count())

    def get_device_stats(self, device: Any = None) -> dict[str, float]:
        import torch_xla.core.xla_model as xm

        info = xm.get_memory_info(device if device is not None else xm.xla_device())
        return {k: float(v) / _MB for k, v in info.items() if isinstance(v, (int, float))}


ACCELERATORS: dict[str, type[Accelerator]] = {
    "cpu": CPUAccelerator,
    "cuda": CUDAAccelerator,
    "gpu": CUDAAccelerator,
    "mps": MPSAccelerator,
    "tpu": XLAAccelerator,
}


def accelerator_name(accelerator: str | Accelerator) -> str:
    """The Trainer's accelerator string for an ``Accelerator`` instance or class."""
    if isinstance(accelerator, Accelerator) or (
        isinstance(accelerator, type) and issubclass(accelerator, Accelerator)
    ):
        return accelerator.name()
    return accelerator


def find_usable_cuda_devices(num_devices: int = -1) -> list[int]:
    """Indices of visible CUDA GPUs that accept an allocation (not held exclusively by
    another process), up to ``num_devices`` (-1: all). As in Lightning; on GPUlayer workers the
    scheduler already grants free GPUs."""
    if num_devices == 0:
        return []
    visible = list(range(torch.cuda.device_count()))
    if not visible:
        raise ValueError(f"requested {num_devices} devices but no CUDA devices are visible")
    if num_devices > len(visible):
        raise ValueError(
            f"requested {num_devices} devices but this machine has {len(visible)} GPUs"
        )
    usable, busy = [], []
    for i in visible:
        try:
            torch.tensor(0, device=torch.device("cuda", i))
        except RuntimeError:
            busy.append(i)
            continue
        usable.append(i)
        if len(usable) == num_devices:
            break
    if num_devices != -1 and len(usable) != num_devices:
        raise RuntimeError(
            f"requested {num_devices} devices but only {len(usable)} are free; {busy} are in use"
        )
    return usable
