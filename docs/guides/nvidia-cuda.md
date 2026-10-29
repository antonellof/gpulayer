# NVIDIA CUDA

## Requirements

- An NVIDIA driver (`nvidia-smi` works).
- A CUDA build of PyTorch in the interpreter that runs jobs. Check with
  `python -c "import torch; print(torch.cuda.is_available())"`.

```bash
gpulayer resources detect
```

Each GPU is listed with its memory and compute capability. If the driver sees GPUs but torch has
no CUDA support, the devices are listed with `execution_supported=false` and jobs that need
CUDA are not placed there.

## Train

```python
trainer = glt.Trainer(max_epochs=10, accelerator="cuda", devices=4, precision="bf16-mixed")
trainer.fit(model, train_loader, val_loader)
```

- `devices` is the number of GPUs (one process per GPU) on each node. The job gets
  `CUDA_VISIBLE_DEVICES` set to that many GPUs.
- Multi-GPU and multi-node runs use DDP with the `nccl` backend.
- `precision="16-mixed"` uses a gradient scaler; `bf16-mixed` needs a GPU with bfloat16 support.

## Placement constraints

Ask for specific hardware through `@gl.remote(...)`, `launch(...)` or a `JobSpec`:

```python
@gl.remote(accelerator="cuda", gpus=2, gpu_memory_gb=40)
def finetune(...): ...
```

`ResourceRequest(min_cuda_compute_capability="8.0")` restricts jobs to Ampere or newer.

## Validation status

CUDA detection and scheduling are tested with mocked `nvidia-smi` output, and execution uses
the same process path as CPU and MPS jobs. Neither single-GPU nor NCCL multi-GPU runs have
been validated on NVIDIA hardware in this repository yet. The hardware test
(`pytest -m hardware_cuda tests/hardware`) runs automatically on a machine with a GPU; please
report the result.
