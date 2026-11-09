# Framework requirements

Workers report their accelerators and the GPU frameworks installed in the Python that runs
jobs. Jobs say what they need, and the scheduler matches the two.

## What workers report

`gpulayer resources detect` shows the current machine:

```text
device    [0] mps Apple arm64 GPU (Metal) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

`gpulayer worker status --json` includes, for example:

```json
"frameworks": {"mlx": "0.32.4", "numpy": "2.4.6", "torch": "2.15.0", "torch.mps": "2.15.0"}
```

GPUs (`cuda` via `nvidia-smi`, `mps` on Apple Silicon, `rocm` via `rocm-smi`, reported but never
scheduled yet) are covered in [GPU compatibility](../architecture/gpu-compatibility.md). Framework
versions are read from package metadata without importing: `torch`, `jax`, `jaxlib`,
`tensorflow`, `mlx`, `cupy`, `onnxruntime`, `tensorrt`, `triton`, `vllm`, `transformers`, `numpy`.
Three derived entries say what actually works: `torch.cuda` and `torch.mps` (torch can drive the
GPU), and `torch.compile` (torch plus a C compiler: `cc`, `gcc`, `clang` or `$CC`; request with
`frameworks={"torch.compile": ">=0"}`). PyTorch jobs need `torch.cuda` or `torch.mps`, which the
PyTorch integration adds for you.

## What jobs ask for

```python
@gl.remote(accelerator="cuda", frameworks={"torch": ">=2.4"})
def train(lr: float) -> dict:
    import torch
    ...

@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})
def matmul_benchmark(n: int) -> dict:
    import mlx.core as mx
    ...
```

A [runtime profile](../reference/job-spec.md#runtime-profiles) such as `profile="pytorch-cuda"`
fills these in. A worker that lacks a framework or has the wrong version is rejected with a
reason such as `framework 'torch' not installed` or `mlx 0.9.0 does not satisfy >=0.10`.

In the job, `GPULAYER_ACCELERATOR` names the granted accelerator; for CUDA jobs
`CUDA_VISIBLE_DEVICES` lists the granted GPUs.

## Extra packages for one job

A job can list packages the worker lacks. The worker installs them once into its own environment
on top of the worker's Python and reuses it for later jobs with the same list
([ADR-0022](../decisions/ADR-0022-job-environments-and-dependency-installation.md)):

```bash
gpulayer submit --source ./app --requirement timm==1.0.30 -- python train.py
gpulayer submit --source ./app --requirements-file requirements.txt -- python train.py
```

```python
gl.command(["python", "train.py"], source=src, requirements=["timm==1.0.30"])
@gl.remote(accelerator="cuda", requirements=["timm"])
glt.launch(nodes=2, devices=4, accelerator="cuda", requirements=["timm"])
trainer = glt.Trainer(..., launch_options={"requirements": ["timm"]})
```

- **Once per worker.** The environment is keyed by the list (or file content) and the worker's
  Python. On an M2 Pro, `timm` took 6.6 s the first time and 0.00 s after.
- **Worker packages stay visible** and are not reinstalled: `timm` used the worker's torch. Pin
  versions that must match the worker's CUDA build.
- **CUDA libraries from pip are found.** On Linux the job's `LD_LIBRARY_PATH` starts with the
  `nvidia/*/lib` folders of its own packages, then the worker's. TensorFlow and ONNX Runtime
  need this: they install CUDA libraries with pip but do not find them, even in a plain
  virtualenv. Ask for `tensorflow[and-cuda]` and `onnxruntime-gpu[cuda,cudnn]`; the extras
  install what the worker's PyTorch lacks (ONNX Runtime needs cuRAND).
- **Reported apart from the run.** The task log says `environment built in 6.62s` or
  `environment reused in 0.00s`; the finish event carries `prepare_seconds` and
  `prepare_cached`. Job timings and benchmarks exclude it.
- A failed install fails the task before the job runs, with the installer's message.
- Environments live in the worker's directory (`<home>/envs` for local workers). Delete it to
  rebuild; automatic cleanup is not implemented yet.

## What has been run

PyTorch on NVIDIA CUDA, Apple MPS and CPU is validated on hardware, with automated tests. MLX on
the Apple GPU ([`examples/mlx_metal`](../../examples/mlx_metal/matmul.py)) has been run on Apple
Silicon; NumPy batch ([`examples/numpy_batch`](../../examples/numpy_batch/estimate.py)) and shell
tools ([`examples/shell_job`](../../examples/shell_job)) have been run. JAX, CuPy and
Transformers, TensorFlow and ONNX Runtime ran as GPUlayer jobs on an NVIDIA L4, each installed
with per-job requirements, and the worker's PyTorch still worked next to TensorFlow in the same
process ([results](../../benchmarks/results/2026-11-09-hf-strategies-speed/README.md)). Other reported packages have not been run as GPUlayer jobs on a GPU.
Distributed jobs start PyTorch process groups only.
