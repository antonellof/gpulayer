# ADR-0006: Apple Metal / PyTorch MPS boundaries

- **Status:** Accepted (milestone 1)
- **Date:** 2026-10-29

## Context

Apple Silicon is a common developer machine. PyTorch exposes its GPU through the MPS
backend. MPS is **not** CUDA: operators, dtypes, custom extensions, memory semantics and
distributed collectives differ. Apple Silicon also uses unified memory shared with the
CPU.

## Decision

- MPS is detected **only** through PyTorch APIs (`torch.backends.mps.is_built()` and
  `is_available()`), inside the isolated torch probe (`gpu/discovery.py`). Missing torch,
  non-macOS hosts and built-but-unavailable MPS are handled gracefully and explained in
  `notes`.
- The MPS device is reported with `memory_kind=unified` and `memory_gb` equal to host
  RAM. It is never presented as dedicated VRAM. `ResourceRequest.gpu_memory_kind` lets a
  job demand `dedicated` memory, which then excludes Macs. Memory rejection reasons
  mention "unified memory shared with the CPU".
- **Supported:** independent jobs (`accelerator="mps"`). The workload itself selects
  `torch.device("mps")`. GpuLayer only routes and exposes `GPULAYER_ACCELERATOR`.
- **Unsupported:** distributed MPS. `JobSpec` validation rejects `DistributedJob` with
  `accelerator="mps"`, and the support matrix marks it unsupported. CPU/gloo distributed
  jobs *can* run on Macs, because they do not use MPS.
- A CUDA job is never placed on an MPS worker, and the reverse also holds.

## Validation

- Mock-based tests: `tests/unit/test_discovery.py` (available, built-but-unavailable,
  missing torch) and `tests/unit/test_scheduler.py` (MPS vs CUDA, unified memory). These
  run on CPU-only CI.
- Hardware: `tests/hardware/test_gpu_smoke.py::test_mps_independent_job` ran a 512×512
  matmul on MPS through the full SDK path on an Apple Silicon Mac (arm64, 32 GB, torch
  from the project venv). That covers one smoke workload only, not an operator coverage
  claim.

## Alternatives considered

- **Treating MPS as "a GPU" interchangeable with CUDA:** produces confusing runtime
  failures. Rejected.
- **Multi-Mac distributed with gloo over MPS tensors:** not validated, so it is out of
  scope.

## Consequences and trade-offs

- Workloads must be written for MPS (device selection, fallbacks such as
  `PYTORCH_ENABLE_MPS_FALLBACK`). GpuLayer does not translate CUDA code.
- Unified memory numbers overstate what a GPU workload can actually use.

## Security and operational implications

- None beyond ADR-0009. MPS jobs run as ordinary processes.

## Follow-up

- Optional operator and dtype probes for MPS capability detail.
- A separate design and validation effort before any distributed MPS claim.
