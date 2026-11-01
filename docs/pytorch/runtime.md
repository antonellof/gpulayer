# Runtime helpers

`Runtime` is the Trainer's device, process-group, sharding and precision layer. Use it to keep
your own loop. It runs in a plain process, under `torchrun` and in [`launch()`](launch.md) jobs.

```python
import torch
import gpulayer.pytorch as glt

rt = glt.Runtime(accelerator="auto", precision="bf16-mixed").setup()
model = rt.module(MyModel())                       # on the device; DDP when WORLD_SIZE > 1
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loader = rt.dataloader(train_loader)               # DistributedSampler when WORLD_SIZE > 1

for epoch in range(10):
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    for batch in loader:
        x, y = rt.to_device(batch)
        with rt.autocast():
            loss = torch.nn.functional.cross_entropy(model(x), y)
        rt.backward(loss)
        rt.step(optimizer, model.parameters(), clip_val=1.0)
        optimizer.zero_grad()
    if rt.is_global_zero:
        print(epoch, rt.mean(loss.item()))

if rt.is_global_zero:
    torch.save(model.state_dict(), rt.output_dir() / "model.pt")
rt.teardown()
```

## API

| Member | Description |
|---|---|
| `Runtime(accelerator="auto", precision="32-true")` | resolves the device and checks the precision |
| `setup()`, `teardown()` | create and destroy the process group when `WORLD_SIZE > 1` |
| `module(model)` | moves to the device; DDP when distributed |
| `dataloader(loader, shuffle=None)` | adds a `DistributedSampler` when distributed |
| `to_device(batch)` | moves tensors and containers of tensors |
| `batches(loader, prefetch=True)` | yields batches on the device; prefetches on a side stream on CUDA |
| `autocast()` | the chosen mixed precision |
| `backward(loss)`, `step(optimizer, params, clip_val)` | with gradient scaling for `16-mixed` |
| `mean(value)`, `barrier()` | average a number across processes; synchronise |
| `output_dir(default)` | `$GPULAYER_OUTPUT_DIR` in a job, else `default` (`gpulayer_runs/local`) |
| `device`, `accelerator`, `world_size`, `global_rank`, `local_rank`, `node_rank`, `is_global_zero` | topology |

`glt.resolve_accelerator("auto")` and `glt.seed_everything(seed)` also work alone.

## Precision per backend

| Precision | CPU | CUDA | MPS |
|---|---|---|---|
| `32-true` | yes | yes | yes |
| `bf16-mixed` | yes | yes | error |
| `16-mixed` | error | yes (with `GradScaler`) | error |

`dataloader()` rejects a custom sampler or `batch_sampler`; pass your own `DistributedSampler`.
