# Framework requirements

Workers report which accelerators they
have and which GPU frameworks are installed in the Python that runs jobs; jobs say what they
need; the scheduler matches the two.

## What workers report

`gpulayer resources detect` shows it for the current machine:

```text
device    [0] mps Apple arm64 GPU (Metal) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

and `gpulayer worker status --json` includes, for example:

```json
"frameworks": {"mlx": "0.32.4", "numpy": "2.4.6", "torch": "2.15.0", "torch.mps": "2.15.0"}
```

| Kind | How it is detected |
|---|---|
| NVIDIA GPU (`cuda`) | `nvidia-smi` works: name, memory, compute capability, driver |
| Apple GPU (`mps`, Metal) | an Apple Silicon Mac; memory is reported as unified |
| AMD GPU (`rocm`) | `rocm-smi` is present; reported but never scheduled yet |
| Frameworks | installed package versions, read from package metadata without importing: `torch`, `jax`, `jaxlib`, `tensorflow`, `mlx`, `cupy`, `onnxruntime`, `tensorrt`, `triton`, `vllm`, `transformers`, `numpy` |
| `torch.cuda`, `torch.mps` | torch in that environment can actually drive the GPU |
| `torch.compile` | torch is installed and a C compiler (`cc`, `gcc` or `clang`, or `$CC`) is present, so `torch.compile` can build its kernels; ask for it with `frameworks={"torch.compile": ">=0"}` |

A GPU counts as usable when its driver or platform is. PyTorch jobs also need `torch.cuda` or
`torch.mps`, which the PyTorch integration adds for you.

## What jobs ask for

Name the accelerator and the frameworks with version ranges:

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

A [runtime profile](../reference/job-spec.md#runtime-profiles) such as
`profile="pytorch-cuda"` fills in the accelerator and frameworks for you. A worker missing a
framework, or with the wrong version, is rejected with a reason such as
`framework 'torch' not installed` or `mlx 0.9.0 does not satisfy >=0.10`.

Inside the job, `GPULAYER_ACCELERATOR` says which accelerator was granted, and for CUDA jobs
`CUDA_VISIBLE_DEVICES` lists the granted GPUs.

## Extra packages for one job

A job can list packages its worker does not have. The worker installs them once, in an
environment of its own built on top of the worker's Python, and reuses it for every later job
with the same list ([ADR-0022](../decisions/ADR-0022-job-environments-and-dependency-installation.md)):

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

- **Installed once per worker, never per run.** The environment is keyed by the list (or the
  file's content) and the worker's Python. The first job on a worker builds it; later jobs
  reuse it. On an M2 Pro: `timm` took 6.6 s the first time and 0.00 s after.
- **The worker's packages stay.** PyTorch, the CUDA libraries and everything else the worker
  has are visible and are not installed again: `timm`, which depends on torch, used the
  worker's torch. Pin versions that must match the worker's CUDA build.
- **Reported apart from the run.** The task's log says `environment built in 6.62s` or
  `environment reused in 0.00s`, and its finish event carries `prepare_seconds` and
  `prepare_cached`; job timings and benchmarks never include it.
- A failed install fails the task before any of the job runs, with the installer's message.
- Environments live in the worker's directory (`<home>/envs` for local workers). Delete it to
  rebuild them; automatic cleanup is not implemented yet.

## What has been run

| Workload | Status |
|---|---|
| PyTorch on NVIDIA CUDA, Apple MPS and CPU | validated on hardware, with automated tests |
| MLX on the Apple GPU ([`examples/mlx_metal`](../../examples/mlx_metal/matmul.py)) | run on Apple Silicon |
| NumPy batch ([`examples/numpy_batch`](../../examples/numpy_batch/estimate.py)) | run |
| Shell tools ([`examples/shell_job`](../../examples/shell_job)) | run |

Workers also report other packages they find (JAX, CuPy, TensorFlow, ONNX Runtime and
others), but none of them has been run as a GPUlayer job on a GPU. Distributed jobs start
PyTorch process groups only.
