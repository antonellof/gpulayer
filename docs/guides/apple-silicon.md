# Apple Silicon (Metal)

GPUlayer reports the GPU of every Apple Silicon worker as accelerator `mps`. PyTorch jobs also
need the `torch.mps` capability: a recent torch in the worker's Python (or set `--python`).

```bash
gpulayer resources detect
```

```text
device    [0] mps Apple arm64 GPU (MPS) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

## MLX

```python
@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})
def matmul_benchmark(n: int) -> dict:
    import mlx.core as mx
    ...
```

[`examples/mlx_metal/matmul.py`](../../examples/mlx_metal/matmul.py) reaches about 4.3 TFLOPS
(4096 x 4096 float32 matmul) on an M-series laptop.

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

| Topic | MPS behaviour in GPUlayer |
|---|---|
| Multi-process or multi-node | not supported; `devices > 1` or `num_nodes > 1` is rejected |
| Memory | unified, reported as `memory_kind=unified`; `gpu_memory_gb` is compared against it |
| Precision | `32-true` only; `bf16-mixed` and `16-mixed` raise an error |
| Operators and dtypes | some CUDA operators and dtypes (for example float64) are missing; test your model |
| Docker | cannot use the Apple GPU; run Mac workers natively |

CUDA and MPS jobs never run on each other's workers
([ADR-0006](../decisions/ADR-0006-apple-metal-mps-boundaries.md)).

## Validation

A real MPS smoke test (`tests/hardware/test_gpu_smoke.py`) and the examples in this
documentation have been run on an Apple Silicon Mac.

## Performance notes (measured)

Apple Silicon Mac (10 CPU cores, 32 GB), macOS 26.7, torch 2.15.0, MLX 0.32.4:

- **One GPU job at a time is fastest.** Two jobs sharing the GPU gave 1.0x (small MLP), 0.92x
  and 0.80x versus running in sequence, so Mac workers default to one task. Raise
  `--concurrency` only for CPU-bound jobs.
- **Half precision is slower.** fp16 autocast was 3 to 8% slower than fp32 for MLP training,
  bf16 about 50% slower. Re-measure on your model.
- **Avoid per-step reads.** Keeping logged values on the GPU made `Trainer` MPS steps 1.6x
  faster. Avoid per-step `.item()` or `print(tensor)`; in MLX call `mx.eval` once per step.
- **MLX:** unified memory means no host-to-device copy. `mx.compile` can help on stable step
  functions; capture model and optimiser state as the MLX docs describe.
