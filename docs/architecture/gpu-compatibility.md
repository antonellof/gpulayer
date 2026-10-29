# GPU compatibility, runtime profiles and Apple Metal/MPS boundaries

## Support matrix

Defined in code (`src/gpulayer/gpu/support.py`) and enforced by the scheduler. A backend is
never inferred to work from its name.

| Accelerator | Independent jobs | Distributed jobs | Notes |
|---|---|---|---|
| `cpu` | tested | tested (gloo, one host) | no GPU libraries required |
| `cuda` | implemented | implemented (nccl/gloo) | mocked detection + scheduling tests; not yet run on NVIDIA hardware |
| `mps` | **hardware-validated** | unsupported | Apple Silicon smoke test passes; see below |
| `rocm` | detection-only | unsupported | reported with `execution_supported=false`, never scheduled |

## Detection (`gpu/discovery.py`)

- **NVIDIA:** `nvidia-smi --query-gpu=...` for name, memory, compute capability, driver. If the
  driver sees a GPU but the worker's Python has no CUDA-enabled torch, devices are reported with
  `execution_supported=false` and a note — hardware visibility is not runtime readiness.
- **torch probe:** runs `python -c ...` in a **subprocess** of the *workload* interpreter
  (`--python`), so a missing or broken torch never crashes the worker. Reports torch version,
  CUDA devices, `torch.backends.mps.is_built()/is_available()`, and gloo/nccl availability.
- **ROCm:** `rocm-smi` presence only.
- Unknown values stay `None` and are listed in `capabilities.notes`; a requirement against an
  unknown value is not satisfied.

## Matching rules (scheduler)

- `accelerator` + `gpu_count` count only devices with `execution_supported=true`.
- `gpu_memory_gb` is per device. `gpu_memory_kind` can require `dedicated` or `unified`.
- `min_cuda_compute_capability` compares numerically (`8.6 >= 8.0`).
- `frameworks` and `python` are PEP 440 specifiers against reported versions.
- CPU jobs get `CUDA_VISIBLE_DEVICES=""`; CUDA jobs get the granted device indices.

## Apple Metal / PyTorch MPS

- MPS is a separate backend, **not a CUDA compatibility layer**. A CUDA job never lands on an
  MPS worker and vice versa.
- Availability comes from `torch.backends.mps.is_available()` in the workload interpreter.
- Memory is **unified**: the reported figure is system RAM shared with the CPU, labelled
  `memory_kind=unified`, never presented as VRAM.
- Operator coverage, dtypes (e.g. float64) and extensions differ from CUDA; validating a model
  on MPS is the workload author's responsibility.
- **Distributed MPS (multiple Macs, or multiple processes sharing MPS) is unsupported** and
  rejected at validation (ADR-0006).

## Runtime profiles

Not implemented yet. Today compatibility comes from live reported capabilities plus
`frameworks`/`python` constraints. Profiles (named driver/framework/OS bundles) are planned as a
convenience on top of — never instead of — worker checks.
