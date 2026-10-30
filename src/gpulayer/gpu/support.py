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
        # A service replica runs like an independent job; only groups need the distributed path.
        return self.distributed if workload == WorkloadType.DISTRIBUTED else self.independent


SUPPORT: dict[Accelerator, BackendSupport] = {
    Accelerator.CPU: BackendSupport(
        SupportLevel.TESTED,
        SupportLevel.TESTED,
        "Distributed CPU uses torchrun + gloo; tested multi-process on one host.",
    ),
    Accelerator.CUDA: BackendSupport(
        SupportLevel.HARDWARE_VALIDATED,
        SupportLevel.IMPLEMENTED,
        "Any CUDA workload (binaries, CuPy, JAX, PyTorch, ...) once the driver works. Single-GPU "
        "jobs and the PyTorch Trainer (fp32, bf16, fp16, torch.compile) were validated on a "
        "Tesla T4 (scripts/validate_gpu.sh). NCCL across several GPUs has not been run on "
        "hardware yet.",
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
