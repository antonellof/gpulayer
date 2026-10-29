"""Explicit support matrix. The scheduler consults this; docs are generated from it.

Never infer support from a backend's name: if a combination is not listed here as
executable, the scheduler rejects it with the recorded reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpulayer.contracts import Accelerator, SupportLevel, WorkloadType


@dataclass(frozen=True)
class BackendSupport:
    independent: SupportLevel
    distributed: SupportLevel
    note: str

    def level(self, workload: WorkloadType) -> SupportLevel:
        return self.independent if workload == WorkloadType.INDEPENDENT else self.distributed


SUPPORT: dict[Accelerator, BackendSupport] = {
    Accelerator.CPU: BackendSupport(
        SupportLevel.TESTED,
        SupportLevel.TESTED,
        "Distributed CPU uses torchrun + gloo; tested multi-process on one host.",
    ),
    Accelerator.CUDA: BackendSupport(
        SupportLevel.IMPLEMENTED,
        SupportLevel.IMPLEMENTED,
        "Any CUDA workload (binaries, CuPy, JAX, PyTorch, ...) once the driver works. Detection "
        "and scheduling are unit-tested with mocks and execution shares the tested process "
        "path, but neither single-GPU nor NCCL multi-GPU runs have been made on NVIDIA "
        "hardware yet (tests/hardware/test_gpu_smoke.py is skipped without a GPU).",
    ),
    Accelerator.MPS: BackendSupport(
        SupportLevel.HARDWARE_VALIDATED,
        SupportLevel.UNSUPPORTED,
        "Apple GPU through Metal (PyTorch MPS, MLX, ...). Independent jobs only; "
        "multi-process and multi-Mac runs are out of scope (ADR-0006).",
    ),
    Accelerator.ROCM: BackendSupport(
        SupportLevel.DETECTION_ONLY,
        SupportLevel.UNSUPPORTED,
        "rocm-smi presence is detected; no execution path exists yet.",
    ),
}

EXECUTABLE = {SupportLevel.HARDWARE_VALIDATED, SupportLevel.TESTED, SupportLevel.IMPLEMENTED}


def unsupported_reason(accelerator: Accelerator, workload: WorkloadType) -> str | None:
    s = SUPPORT[accelerator]
    lvl = s.level(workload)
    if lvl in EXECUTABLE:
        return None
    return f"{accelerator.value} {workload.value} execution is {lvl.value}: {s.note}"
