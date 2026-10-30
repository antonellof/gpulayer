# Any GPU framework

GPUlayer does not depend on a particular GPU library. Workers report which accelerators they
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

A GPU counts as usable when its driver or platform is, so a CUDA box without PyTorch can still
run CUDA binaries, CuPy or JAX jobs.

## What jobs ask for

Name the accelerator and the frameworks with version ranges:

```python
@gl.remote(accelerator="cuda", frameworks={"jax": ">=0.4.30"})
def train_jax(steps: int) -> dict:
    import jax
    ...

@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})
def matmul_benchmark(n: int) -> dict:
    import mlx.core as mx
    ...

@gl.remote(accelerator="cuda", gpus=1, frameworks={"cupy": ">=13"})
def fft(size: int) -> float:
    import cupy as cp
    ...
```

```bash
gpulayer submit --accelerator cuda --framework 'onnxruntime>=1.18' -- python infer.py
```

For common combinations, a [runtime profile](../reference/job-spec.md#runtime-profiles) such
as `profile="pytorch-cuda"`, `"jax-cuda"` or `"mlx"` fills in the accelerator and frameworks for
you.

A worker missing a framework, or with the wrong version, is rejected with a reason such as
`framework 'jax' not installed` or `mlx 0.9.0 does not satisfy >=0.10`. The PyTorch
integration adds `torch` plus `torch.cuda` or `torch.mps` automatically.

Inside the job, `GPULAYER_ACCELERATOR` says which accelerator was granted, and for CUDA jobs
`CUDA_VISIBLE_DEVICES` lists the granted GPUs, so any CUDA library uses the right devices.

## What has been run

| Workload | Status |
|---|---|
| MLX on the Apple GPU ([`examples/mlx_metal`](../../examples/mlx_metal/matmul.py)) | run on Apple Silicon |
| NumPy batch ([`examples/numpy_batch`](../../examples/numpy_batch/estimate.py)) | run |
| Shell tools ([`examples/shell_job`](../../examples/shell_job)) | run |
| PyTorch on CPU and MPS | run, with automated tests |
| JAX, CuPy, TensorFlow, ONNX Runtime on NVIDIA | the same mechanism; not run in this repository yet |

## Multi-process jobs in other frameworks

Distributed jobs start PyTorch-style process groups (the torchrun environment) today. Frameworks with their own launchers
(JAX multi-host, MPI, Horovod) are planned; until then, run them as independent jobs or
coordinate them yourself.
