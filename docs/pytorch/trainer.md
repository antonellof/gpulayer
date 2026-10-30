# Trainer and Module

`gpulayer.pytorch` gives you a small training API: put the model and its training logic in a
`Module`, and let a `Trainer` run the loop. The `Trainer` also decides where the loop runs.

```python
import torch
from torch import nn
import gpulayer.pytorch as glt


class Classifier(glt.Module):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", nn.functional.cross_entropy(logits, y))
        self.log("val_acc", (logits.argmax(1) == y).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = Classifier()
    trainer = glt.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
    print(trainer.callback_metrics)
```

## Module

`glt.Module` is an `nn.Module` with these hooks:

| Hook | Required | What to return |
|---|---|---|
| `training_step(batch, batch_idx)` | yes | the loss tensor |
| `validation_step(batch, batch_idx)` | no | nothing; call `self.log(...)` |
| `configure_optimizers()` | yes | an optimizer, `(optimizer, scheduler)`, or `{"optimizer": ..., "lr_scheduler": ..., "interval": "epoch" or "step"}` |

`self.log(name, value)` records a scalar (a float or a tensor). The Trainer averages each logged
name over the steps of an epoch and across all processes, then stores the result in
`trainer.callback_metrics` and `trainer.history`. If `training_step` logs nothing, the Trainer
logs the returned loss as `train_loss`.

Batches arrive on the right device already. The Trainer moves tensors, and lists, tuples and
dicts of tensors, before calling your step.

## Trainer arguments

| Argument | Default | Meaning |
|---|---|---|
| `max_epochs` | `1` | number of epochs |
| `max_steps` | `None` | stop after this many optimizer steps, even mid-epoch |
| `check_val_every_n_epoch` | `1` | validate every N epochs (and always after the last one) |
| `accelerator` | `"auto"` | `"cpu"`, `"cuda"`, `"mps"` or `"auto"` (the device granted to the job, else CUDA, then MPS, then CPU) |
| `devices` | `1` | processes per node; with CUDA, one GPU per process |
| `num_nodes` | `1` | number of workers for a distributed run |
| `precision` | `"32-true"` | `"32-true"`, `"bf16-mixed"` (CUDA, CPU) or `"16-mixed"` (CUDA, with a gradient scaler) |
| `target` | `None` | `None` runs here when possible; `"local"`, `"remote"` or `"cloud"` always submits a job |
| `gradient_clip_val` | `None` | clip the gradient norm to this value |
| `accumulate_grad_batches` | `1` | apply the optimizer every N batches; DDP exchanges gradients only on those steps |
| `callbacks` | `[]` | `Callback` objects, see [Callbacks](#callbacks) |
| `log_every_n_steps` | `50` | print logged values every N steps (`0` turns it off) |
| `limit_train_batches`, `limit_val_batches` | `None` | stop an epoch after N batches |
| `enable_checkpointing` | `True` | save `checkpoints/last.ckpt` after every epoch |
| `default_root_dir` | `"gpulayer_runs"` | where local checkpoints and downloaded job outputs go |
| `launch_options` | `{}` | extra arguments for [`launch()`](launch.md), for example `{"env": {...}, "timeout_seconds": 3600}` |

Unsupported combinations fail with an error instead of quietly doing something else. Examples:
`precision="16-mixed"` on MPS, `accelerator="cuda"` on a machine without CUDA, `devices=2` with
MPS, or `accelerator="auto"` with `target="remote"` (the remote device is unknown until the
job is placed, so name it).


## Callbacks

Callbacks hook into the training loop. Two are built in:

```python
from gpulayer.pytorch import EarlyStopping, ModelCheckpoint, Trainer

best = ModelCheckpoint(monitor="val_loss")                 # saves checkpoints/best.ckpt
trainer = Trainer(max_epochs=50, callbacks=[EarlyStopping(monitor="val_loss", patience=3), best])
trainer.fit(model, train_loader, val_loader)
model = MyModel.load_from_checkpoint(best.best_model_path)
```

- `EarlyStopping(monitor, mode="min", patience=3, min_delta=0.0)` stops when the metric has
  not improved for `patience` epochs.
- `ModelCheckpoint(monitor, mode="min", filename="best.ckpt")` saves a checkpoint each time
  the metric improves. `best_model_path` points at it, also when training ran on workers (the
  file comes back with the job's outputs).
- The monitored name must be logged with `self.log(...)`; otherwise the error lists the
  names that were logged.

Write your own by subclassing `Callback` and overriding any of `on_fit_start`,
`on_train_epoch_start`, `on_train_batch_end(trainer, model, loss, batch_idx)`,
`on_train_epoch_end` (after validation; `trainer.callback_metrics` holds the epoch's values)
and `on_fit_end`. Set `trainer.should_stop = True` to end training after the current epoch;
`trainer.save_checkpoint("name.ckpt")` writes a checkpoint.

In distributed runs callbacks run on every rank. Epoch metrics are averaged across ranks
first, and if any rank sets `should_stop`, every rank stops after the same epoch. When
`fit` launches training on workers, the callbacks run there; in the launching process only
`on_fit_end` runs.

## Where `fit` runs

```mermaid
flowchart TD
    F["trainer.fit(model, ...)"] --> Q{"inside a GPUlayer job<br/>or torchrun process?"}
    Q -- yes --> LOOP["run the training loop here<br/>(DDP when WORLD_SIZE > 1)"]
    Q -- no --> T{"target set, or<br/>devices x num_nodes > 1?"}
    T -- no --> LOOP
    T -- yes --> L["launch(): submit this script<br/>to GPUlayer workers"]
    L --> W["workers re-run the script;<br/>the same fit() call runs the loop"]
    W --> B["download checkpoint and metrics,<br/>load weights into the local model"]
```

When `fit` submits a job, it re-runs your script on the workers with the same command-line
arguments. On the workers, the same `fit` call runs the training loop. When the job finishes,
the launching process downloads the outputs and loads the final weights into your local
`model`. Code after `fit` therefore sees a trained model in both cases.

Because the script runs again on the workers, code at module level and after `fit` runs there
too. Guard work that should only happen once, on your machine:

```python
trainer.fit(model, train_loader, val_loader)
if glt.is_launcher_process():
    torch.save(model.state_dict(), "final.pt")
```

## After training

| Attribute | Contents |
|---|---|
| `trainer.callback_metrics` | metrics of the last epoch, e.g. `{"train_loss": 0.1, "val_loss": 0.08, ...}` |
| `trainer.history` | one dict per epoch |
| `trainer.checkpoint_path` | path of `last.ckpt` (downloaded, for a submitted job) |
| `trainer.launch_result` | the `LaunchResult` of a submitted job (job id, status, output files) |
| `trainer.global_step`, `trainer.current_epoch` | progress counters |

## Checkpoints and resuming

With checkpointing on, global rank 0 writes `checkpoints/last.ckpt` and `metrics.json` after
every epoch. Inside a job they go to `$GPULAYER_OUTPUT_DIR` and come back as artifacts
(`gpulayer_runs/<job-id>/...`); in-process they go to `gpulayer_runs/<directory>-<script>/`,
for example `gpulayer_runs/my_project-train/` for `my_project/train.py`.

```python
model = Classifier.load_from_checkpoint("gpulayer_runs/my_project-train/checkpoints/last.ckpt")
trainer = glt.Trainer(max_epochs=20)
trainer.fit(model, train_loader, val_loader,
            ckpt_path="gpulayer_runs/my_project-train/checkpoints/last.ckpt")
```

A checkpoint holds `state_dict`, `optimizer`, `epoch`, `global_step` and `history`.
`ckpt_path` restores the optimizer and continues from the next epoch.

## Validate, test and predict

```python
metrics = trainer.validate(model, val_loader)   # {"val_loss": ..., "val_acc": ...}
metrics = trainer.test(model, test_loader)      # runs Module.test_step: {"test_acc": ...}
outputs = trainer.predict(model, loader)        # one output per batch, on the CPU
```

`predict` calls `Module.predict_step(batch, batch_idx)`, which by default returns
`self(batch)` (or `self(batch[0])` for an `(inputs, targets)` batch). All three run in the
current process, without gradients and with the Trainer's precision.

## What the Trainer does not do yet

Logger integrations, learning-rate monitoring, keeping the best N checkpoints, and strategies
other than DDP (such as FSDP) are not implemented. [ADR-0018](../decisions/ADR-0018-trainer-scope.md)
lists what is planned. If you need them now, keep your existing training code and run it with
[`launch()`](launch.md).
