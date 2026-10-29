# Runtime helpers

`Runtime` is what the Trainer uses internally. Use it directly when you want to keep your own
training loop but stop handling devices, process groups, data sharding and precision by hand.

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

The same loop runs in a plain process, under `torchrun`, and inside GPUlayer jobs. Combine it
with [`launch()`](launch.md) to choose where it runs.

## API

| Member | Description |
|---|---|
| `Runtime(accelerator="auto", precision="32-true")` | resolves the device and validates the precision for it |
| `setup()` | sets the CUDA device and initialises the process group when `WORLD_SIZE > 1` |
| `teardown()` | destroys the process group if `setup()` created it |
| `module(model)` | moves the model to the device; wraps it in DDP when distributed |
| `dataloader(loader, shuffle=None)` | rebuilds the loader with a `DistributedSampler` when distributed |
| `to_device(batch)` | moves tensors, and lists, tuples and dicts of tensors |
| `autocast()` | context manager for the chosen mixed precision (no-op for `32-true`) |
| `backward(loss)`, `step(optimizer, params, clip_val)` | backward and optimizer step, with gradient scaling for `16-mixed` |
| `mean(value)` | average of a Python number across all processes |
| `barrier()` | synchronise all processes |
| `output_dir(default)` | `$GPULAYER_OUTPUT_DIR` inside a job, else `default` (`gpulayer_runs/local`) |
| `device`, `accelerator`, `world_size`, `global_rank`, `local_rank`, `node_rank`, `is_global_zero` | topology |

`glt.resolve_accelerator("auto")` and `glt.seed_everything(seed)` are available on their own.

## Precision per backend

| Precision | CPU | CUDA | MPS |
|---|---|---|---|
| `32-true` | yes | yes | yes |
| `bf16-mixed` | yes | yes | error |
| `16-mixed` | error | yes (with `GradScaler`) | error |

`dataloader()` can shard loaders that use the default random or sequential sampler. A custom
sampler or `batch_sampler` raises an error; pass a `DistributedSampler` yourself in that case.
