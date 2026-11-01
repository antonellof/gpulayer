# GPU compatibility, runtime profiles and Apple Metal/MPS boundaries

## Support matrix

The scheduler enforces the matrix in `src/gpulayer/gpu/support.py`. A backend is supported only
if it is listed there.

| Accelerator | Independent jobs | Distributed jobs | Notes |
|---|---|---|---|
| `cpu` | tested | tested (gloo, one host) | no GPU libraries required |
| `cuda` | hardware-validated (Tesla T4, RTX 3070, A100) | hardware-validated on one machine (nccl, up to 4 GPUs) and across up to 8 machines (one GPU each) | any CUDA workload; the Trainer validated on a T4, RTX 3070 and A100; NCCL across 8 machines with one GPU each (multi-GPU machines across the network not yet run) |
| `mps` | hardware-validated | unsupported | the Apple GPU through Metal (PyTorch MPS, MLX, ...); run on Apple Silicon |
| `rocm` | detection-only | unsupported | reported with `execution_supported=false`, never scheduled |

## Detection (`gpu/discovery.py`)

- NVIDIA: `nvidia-smi --query-gpu=...` (name, memory, compute capability, driver). A working
  driver makes the devices usable for any CUDA workload.
- Apple Silicon: the Metal GPU is reported as `mps` with unified memory, with or without a
  framework installed.
- A probe runs in a subprocess of the workload interpreter (`--python`), so a broken framework
  cannot crash the worker. It reads versions of `torch`, `jax`, `tensorflow`, `mlx`, `cupy`,
  `onnxruntime`, `tensorrt`, `triton`, `vllm`, `transformers` and `numpy` from package metadata
  without importing them, and asks torch, if present, whether it can use CUDA or MPS
  (`torch.cuda` / `torch.mps`) and which distributed backends it has.
- ROCm detection only checks for `rocm-smi`.
- Unknown values stay `None` and are listed in `capabilities.notes`. A requirement against an
  unknown value is not satisfied.

## Matching rules (scheduler)

- `accelerator` + `gpu_count` count only devices with `execution_supported=true`.
- `gpu_memory_gb` is per device. `gpu_memory_kind` can require `dedicated` or `unified`.
- `min_cuda_compute_capability` compares numerically (`8.6 >= 8.0`).
- `frameworks` and `python` are PEP 440 specifiers against reported versions.
- CPU jobs get `CUDA_VISIBLE_DEVICES=""`; CUDA jobs get the granted device indices.

## Apple Metal / PyTorch MPS

- MPS and CUDA are separate backends. A CUDA job never lands on an MPS worker, and vice versa.
- Availability comes from `torch.backends.mps.is_available()` in the workload interpreter.
- Memory is unified: the figure is system RAM shared with the CPU, labelled
  `memory_kind=unified`, never presented as VRAM.
- Operators, dtypes (e.g. float64) and extensions differ from CUDA. Validating a model on MPS is
  the workload author's job.
- Distributed MPS (across Macs, or several processes sharing MPS) is unsupported and rejected
  at validation (ADR-0006).

## Runtime profiles

Not implemented. Compatibility comes from live reported capabilities plus `frameworks`/`python`
constraints. Profiles (named bundles of driver, framework and OS versions) are planned as a
convenience; workers will still be checked against the job.
