# ADR-0005: GPU capabilities and runtime profiles

- **Status:** Accepted (milestone 1) for the capability model, discovery and support matrix. Runtime profiles are Proposed.
- **Date:** 2026-10-29

## Context

"Has a GPU" says nothing about whether a workload can run. Hardware, driver, framework
build and Python environment are separate facts. Discovery must work on CPU-only CI
without GPU libraries, and it must report uncertain values as uncertain instead of guessing them.

## Decision

- `WorkerCapabilities` (`contracts/models.py`) separates the host (OS, arch, cores, RAM,
  free disk, Python), `accelerators` (`AcceleratorDevice`: accelerator, vendor, memory and
  `memory_kind`, compute capability, driver, `execution_supported`), `frameworks`,
  `distributed_backends`, `launchers`, network (`advertise_address`), origin/provider and
  `notes`.
- Discovery (`gpu/discovery.py`) is a set of independent, injectable probes:
  - `nvidia-smi` CSV query: devices as the driver sees them.
  - The torch probe runs in a subprocess with the *workload* interpreter. A missing
    or broken torch install cannot crash the worker or slow its import. It reports the
    torch version, CUDA/MPS built and available, and the gloo/nccl backends.
  - `rocm-smi`: presence only (`execution_supported=False`).
  - NVIDIA hardware with no usable torch CUDA runtime is reported but marked
    non-executable, with a note.
- The support matrix (`gpu/support.py`) is the single source of truth that the
  scheduler consults. Levels use the CLAUDE.md vocabulary:

  | Accelerator | Independent | Distributed |
  |---|---|---|
  | cpu | tested | tested (torchrun+gloo, one host) |
  | cuda | implemented (mock-based detection/scheduling tests; same process path as CPU) | implemented (planning tested; NCCL never run) |
  | mps | hardware-validated (smoke test on Apple Silicon) | unsupported |
  | rocm | detection-only | unsupported |

  `implemented` (added to `SupportLevel` for this reason) means an executable path exists
  but has not run on the relevant hardware; it is scheduled, but never advertised as tested. `tests/hardware/test_gpu_smoke.py`
  has a CUDA smoke test, but it has not been run on NVIDIA hardware in this repository.
- Workers get `CUDA_VISIBLE_DEVICES` restricted to the requested count. CPU jobs get an
  empty value, so they cannot grab a GPU by accident.

## Alternatives considered

- **Importing torch or pynvml in the worker process:** a heavy import, and failures are
  fatal. Rejected.
- **Trusting static configuration only:** drifts from reality. Configuration may add
  labels, but detection stays authoritative.
- **Inferring support from vendor names:** explicitly forbidden.

## Consequences and trade-offs

- Capabilities are detected once at worker start. A driver change needs a worker restart.
- Free memory and utilization are not reported, only capacity.
- The matrix must be edited by hand when validation status changes. This is intentional:
  every change in claimed support goes through a deliberate edit.

## Security and operational implications

- Capabilities are self-reported by the worker. A malicious worker can lie. Provider
  workers have identity fields overwritten from the control plane's record (ADR-0010),
  but not their hardware claims.

## Follow-up

- Run the CUDA smoke test on NVIDIA hardware, and promote the matrix entry only after
  that run.
- Runtime profiles: named, versioned combinations (interpreter, torch/CUDA build, OS)
  usable as a job requirement. A profile is a compatibility aid, not a guarantee.
- Periodic re-detection and resource-usage observations.

## Amendment (2026-10-29, ADR-0015)

GPU usability no longer depends on PyTorch. An NVIDIA GPU is usable when `nvidia-smi` works, and
an Apple Silicon Mac always reports its Metal GPU as `mps`. Installed GPU frameworks are reported
in `frameworks` from package metadata, and torch's ability to drive the GPU is reported as the
`torch.cuda` / `torch.mps` capability. The PyTorch integration requires those capabilities;
other jobs state their own framework requirements.
