# GPU compatibility, runtime profiles and Apple Metal/MPS boundaries

## Support matrix

The matrix is defined in `src/gpulayer/gpu/support.py`, and the scheduler enforces it. A
backend counts as supported only if it is listed there.

| Accelerator | Independent jobs | Distributed jobs | Notes |
|---|---|---|---|
| `cpu` | tested | tested (gloo, one host) | no GPU libraries required |
| `cuda` | implemented | implemented (nccl/gloo) | mocked detection + scheduling tests; not yet run on NVIDIA hardware |
| `mps` | hardware-validated | unsupported | Apple Silicon smoke test passes; see below |
| `rocm` | detection-only | unsupported | reported with `execution_supported=false`, never scheduled |

## Detection (`gpu/discovery.py`)

- NVIDIA devices come from `nvidia-smi --query-gpu=...` (name, memory, compute capability,
  driver). If the driver sees a GPU but the worker's Python has no CUDA-enabled torch, the
  devices are reported with `execution_supported=false` and a note, because a visible GPU can
  still lack a usable runtime.
- The torch probe runs `python -c ...` in a subprocess of the workload interpreter
  (`--python`), so a missing or broken torch cannot crash the worker. It reports the torch
  version, CUDA devices, `torch.backends.mps.is_built()/is_available()` and gloo/nccl
  availability.
- ROCm detection only checks for `rocm-smi`.
- Unknown values stay `None` and are listed in `capabilities.notes`; a requirement against an
  unknown value is not satisfied.

## Matching rules (scheduler)

- `accelerator` + `gpu_count` count only devices with `execution_supported=true`.
- `gpu_memory_gb` is per device. `gpu_memory_kind` can require `dedicated` or `unified`.
- `min_cuda_compute_capability` compares numerically (`8.6 >= 8.0`).
- `frameworks` and `python` are PEP 440 specifiers against reported versions.
- CPU jobs get `CUDA_VISIBLE_DEVICES=""`; CUDA jobs get the granted device indices.

## Apple Metal / PyTorch MPS

- MPS is a separate backend from CUDA. A CUDA job never lands on an MPS worker, and an MPS job
  never lands on a CUDA worker.
- Availability comes from `torch.backends.mps.is_available()` in the workload interpreter.
- Memory is unified. The reported figure is system RAM shared with the CPU, labelled
  `memory_kind=unified`, and it is never presented as VRAM.
- Operator coverage, dtypes (e.g. float64) and extensions differ from CUDA; validating a model
  on MPS is the workload author's responsibility.
- Distributed MPS (across Macs, or several processes sharing MPS) is unsupported and rejected
  at validation (ADR-0006).

## Runtime profiles

Not implemented yet. Today compatibility comes from live reported capabilities plus
`frameworks`/`python` constraints. Profiles, meaning named bundles of driver, framework and OS
versions, are planned as a convenience. Workers will still be checked against the job when
profiles exist.
