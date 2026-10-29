# Apple Silicon (Metal)

Apple Silicon Macs have a GPU that frameworks reach through Metal: PyTorch through its MPS
backend, MLX natively, Core ML and others. GpuLayer reports the GPU of every Apple Silicon worker
as accelerator `mps` and schedules jobs that ask for it there. Whether torch on that machine can
use it is reported separately as the `torch.mps` capability.

## Check

```bash
gpulayer resources detect
```

```text
device    [0] mps Apple arm64 GPU (MPS) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

For PyTorch jobs the worker must also report `torch.mps`; install a recent torch in the Python
that runs jobs, or point the worker at another interpreter with `--python`.

## MLX

```python
@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})
def matmul_benchmark(n: int) -> dict:
    import mlx.core as mx
    ...
```

[`examples/mlx_metal/matmul.py`](../../examples/mlx_metal/matmul.py) runs this on the Apple GPU
(about 4.3 TFLOPS for 4096 x 4096 float32 matrix products on an M-series laptop).

## PyTorch on MPS

```python
trainer = glt.Trainer(max_epochs=5, accelerator="mps")    # or "auto", which picks MPS on a Mac
trainer.fit(model, train_loader, val_loader)
```

```bash
python examples/trainer_regression/train.py --accelerator mps
python examples/remote_sweep/sweep.py --accelerator mps
```

## Limits

| Topic | MPS behaviour in GpuLayer |
|---|---|
| Multi-process or multi-node | not supported; `devices > 1` or `num_nodes > 1` with MPS is rejected |
| Memory | unified with the CPU; reported as `memory_kind=unified`, never as dedicated VRAM |
| `gpu_memory_gb` requirement | compared against the unified memory size |
| Precision | `32-true` only; `bf16-mixed` and `16-mixed` raise an error |
| Operators and dtypes | not every CUDA operator or dtype (for example float64) exists on MPS; test your model |

A CUDA job never runs on an MPS worker, and an MPS job never runs on a CUDA worker. The design
reasons are in [ADR-0006](../decisions/ADR-0006-apple-metal-mps-boundaries.md).

## Validation

A real MPS smoke test (`tests/hardware/test_gpu_smoke.py`) and the examples in this
documentation have been run on an Apple Silicon Mac.

## Performance notes (measured)

Measured on an Apple Silicon Mac (10 CPU cores, 32 GB) with macOS 26.7, torch 2.15.0 and MLX
0.32.4:

- **One GPU job at a time is fastest.** Two training jobs sharing the Apple GPU finished no
  sooner than running them one after the other (1.0x for a small MLP), and were slower for
  larger ones (0.92x and 0.80x). Mac workers therefore default to one task at a time; raise it
  with `--concurrency` only for CPU-bound jobs.
- **Half precision does not speed up PyTorch training on MPS here.** For MLP training, fp16
  autocast was 3 to 8% slower than fp32 and bf16 about 50% slower, so `Trainer` keeps
  `bf16-mixed` and `16-mixed` disabled on MPS. Re-measure on your model before relying on it.
- **Avoid reading values every step.** The `Trainer` keeps logged values on the GPU until it
  prints them; this made MPS steps 1.6x faster. In your own loops, avoid `.item()` or
  `print(tensor)` on every step, and in MLX, call `mx.eval` once per step, not per value.
- **MLX uses unified memory**, so there is no host-to-device copy to optimise. `mx.compile`
  can help on stable step functions; capture model and optimiser state as the MLX docs
  describe rather than passing the model as an argument.
- **Docker cannot use the Apple GPU.** Run Mac workers natively (`gpulayer join` or the
  install script).
