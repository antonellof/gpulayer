# Apple Silicon (MPS)

On Apple Silicon Macs, PyTorch can use the GPU through the Metal Performance Shaders (MPS)
backend. GpuLayer detects it and schedules MPS jobs on Macs that have it.

## Check

```bash
gpulayer resources detect
```

```text
device    [0] mps Apple arm64 GPU (MPS) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

If the line is missing, the Python that runs jobs has no MPS-enabled torch. Install a recent
torch in that environment, or point the worker at another interpreter with `--python`.

## Train on MPS

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
