# Trainer and Module

Put the model and its training logic in a `Module`; a `Trainer` runs the loop and decides where
it runs. The names follow PyTorch Lightning, so Lightning code usually moves with an import
change ([Moving from Lightning](from-lightning.md)). Callbacks, loggers, profilers and the tuner
are on [their own page](callbacks.md); every signature is in the [API reference](api.md).

```python
import torch
from torch import nn
import gpulayer.pytorch as glt


class Classifier(glt.Module):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
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
        self.log("val_acc", (logits.argmax(1) == y).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    trainer = glt.Trainer(max_epochs=10, accelerator="auto")   # devices=4, num_nodes=2 to scale
    trainer.fit(Classifier(), train_loader, val_loader)
```

## Module

`glt.Module` is an `nn.Module` with Lightning's methods:

- `training_step` returns the loss (or a dict with `"loss"`; `None` skips the batch, which
  still counts as a step). `training_step(self, dataloader_iter)` pulls its own batches.
  `validation_step`, `test_step` and `predict_step` are optional.
- `configure_optimizers` returns an optimizer or any of Lightning's forms
  ([below](#optimizers-and-schedulers)).
- `self.log(name, value)` records a scalar. Epoch values are reduced over the epoch and across
  processes with `reduce_fx` (`"mean"`, `"sum"`, `"max"` or `"min"`; a mean is weighted by
  `batch_size=`, or by the size inferred from the batch as Lightning does). Processes may log
  different names. Logged values stay on the GPU until a log step, so logging does not slow
  the loop.
- Where a value goes follows Lightning. In `training_step` a name is logged per step and
  `trainer.callback_metrics` holds its last value; in evaluation and `on_*_epoch_end`, per
  epoch. With both (`on_step=True, on_epoch=True`, or `on_epoch=True` in `training_step`), the
  names are `<name>_step` and `<name>_epoch`, also for `monitor=`. `logger=False` keeps a value
  from the loggers, and `prog_bar=True` adds it to `trainer.progress_bar_metrics`. Loggers get
  Lightning's rows: steps counted from 0, an `epoch` column, and evaluation's per-step values
  under their own counter. `trainer.history` and the run's `metrics.csv` stay GPUlayer's own
  summary: every name's value for each epoch.
- `self.log(name, metric)` also takes a `torchmetrics.Metric`: it is computed once at the end
  of the epoch or evaluation, then reset, and syncs across processes itself. Every process
  must log it. `metric(preds, target)` also gives a value for the step; `metric.update(...)`
  does not.
- Every Lightning hook (`setup`, `on_train_batch_start`, `on_before_optimizer_step`,
  `on_save_checkpoint`, ...) is available; only the ones you override are called.
- `self.save_hyperparameters()` stores the `__init__` arguments in checkpoints, so
  `Classifier.load_from_checkpoint(path)` needs no arguments.
- Batches arrive on the right device. Override `transfer_batch_to_device` for custom types.

**Manual optimization.** Set `self.automatic_optimization = False` to step the optimizers
yourself (GANs, several optimizers):

```python
def training_step(self, batch, batch_idx):
    opt_g, opt_d = self.optimizers()
    opt_d.zero_grad()
    self.manual_backward(loss_d)          # gradient scaling with 16-mixed; averaged across DDP ranks
    self.clip_gradients(opt_d, gradient_clip_val=1.0)
    opt_d.step()
```

## DataModule

`glt.DataModule` keeps a project's data in one place:
`prepare_data()` runs once per machine, `setup(stage)` in every process, and
`train_dataloader()` and the other loaders return `DataLoader`s. Pass it with
`trainer.fit(model, datamodule=dm)`. It is built only in the processes that train.

## Trainer arguments

The common ones; all of them, with Lightning's names, are in the [API reference](api.md#trainer).

| Argument | Default | Meaning |
|---|---|---|
| `max_epochs`, `max_steps`, `max_time` | `None`, `None`, `None` | when to stop. With `max_steps` or `max_time`, epochs are not limited; with none of the three, one epoch (or `min_epochs`; Lightning's default is 1000). `-1` means no limit |
| `accelerator` | `"auto"` | `"cpu"`, `"cuda"`, `"mps"`, `"rocm"`, `"tpu"` or `"auto"` (the device granted to the job, else CUDA, then MPS, then CPU) |
| `devices`, `num_nodes` | `1`, `1` | processes per machine (one GPU each) and machines; more than one launches a job |
| `strategy` | `"ddp"` | `"ddp"`, `glt.DDPStrategy(...)`, `"fsdp"` or `glt.FSDPStrategy(...)` ([experimental](#fsdp-experimental)), `glt.ModelParallelStrategy(...)` ([tensor parallelism](#tensor-parallelism-experimental)), `"deepspeed_stage_2"` and the other DeepSpeed names ([validated on T4s](#deepspeed-and-tpu)) |
| `precision` | `"32-true"` | `"bf16-mixed"`, `"16-mixed"` (CUDA), or `"bf16-true"`, `"16-true"`, `"64-true"` |
| `target` | `None` | `None` runs here when possible; `"local"`, `"remote"` or `"cloud"` always submits a job |
| `callbacks`, `logger`, `profiler`, `plugins` | | see [Callbacks, loggers and profilers](callbacks.md) |
| `accumulate_grad_batches`, `gradient_clip_val` | `1`, `None` | accumulation and clipping; DDP exchanges gradients only on optimizer steps |
| `check_val_every_n_epoch`, `val_check_interval` | `1`, `None` | when to validate (every N epochs, or every N batches or fraction of an epoch; within an epoch only on epochs `check_val_every_n_epoch` selects; with `check_val_every_n_epoch=None` an integer `val_check_interval` counts batches across epochs) |
| `log_every_n_steps` | `50` | print logged values every N steps (`0`: off) |
| `limit_*_batches`, `fast_dev_run`, `overfit_batches` | | shorter runs for debugging |
| `sync_batchnorm`, `deterministic`, `inference_mode` | `False`, `False`, `True` | as in Lightning |
| `benchmark` | `None` | cuDNN autotuning; `None` turns it on for CUDA runs of 1,000 steps or more (5 to 9% faster CNN steps, 1 to 3 s to start) |
| `prefetch` | `False` | copy the next batch to the GPU on a side stream (helps compute-heavy models, slows short steps) |
| `gradient_sync` | `"auto"` | two-level gradient sync with gloo across machines ([ADR-0020](../decisions/ADR-0020-two-level-gradient-sync.md)) |
| `default_root_dir` | `"gpulayer_runs"` | where checkpoints and downloaded outputs go |
| `launch_options` | `{}` | extra [`launch()`](launch.md) arguments, e.g. `{"timeout_seconds": 3600}` |

`log_key_prefix`, `enable_autolog_hparams` and Lightning's other arguments are accepted with
the same meaning ([API reference](api.md#trainer)). A `torch.compile`d model trains compiled,
with its own compile options. Unsupported
combinations (for example `"16-mixed"` on MPS, or `accelerator="auto"` with `target="remote"`)
fail with an error instead of doing something else.

## Optimizers and schedulers

```python
return torch.optim.AdamW(self.parameters(), lr=1e-3)
return [opt_g, opt_d]                                  # manual optimization only
return [opt], [scheduler]
return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
return {"optimizer": opt, "lr_scheduler": ReduceLROnPlateau(opt), "monitor": "val_loss"}
```

Epoch schedulers step at the end of each epoch; step schedulers after every `frequency`
optimizer steps (at steps N, 2N, ...); `ReduceLROnPlateau` steps after validation with the
monitored value (with `interval="step"`, after each optimizer step). With manual
optimization the Trainer does not step schedulers, as in Lightning: call `sch.step()` in
`training_step` (`self.lr_schedulers()`). With a loader that has no length, the gradients
accumulated over the last batches of an epoch are applied at its end. Override `optimizer_step` or `lr_scheduler_step` on the Module to change
them (for example a warm-up).

## Where `fit` runs

With one device and no `target`, `fit` runs the loop in your process. Otherwise it submits a
job: GPUlayer re-runs your script on the workers, one process per rank, and the same `fit` call
runs the loop there. When the job ends, the checkpoint and metrics come back and the weights are
loaded into your local `model`.

Code at module level and after `fit` also runs on the workers. Guard what should happen once:

```python
trainer.fit(model, train_loader)
if glt.is_launcher_process():
    torch.save(model.state_dict(), "final.pt")
```

Pass loaders as functions (`trainer.fit(model, make_loader)`), or use a DataModule, so data is
built only where training runs and not thrown away in the launching process.

## Checkpoints and resuming

Rank 0 writes `checkpoints/last.ckpt`, `metrics.json` and `metrics.csv` after every epoch: in
`gpulayer_runs/<directory>-<script>/` in your process, or in the job's outputs, downloaded to
`gpulayer_runs/<job-id>/`. Checkpoints follow Lightning's layout, without its loop state.

```python
trainer.fit(model, train_loader, ckpt_path="gpulayer_runs/my_project-train/checkpoints/last.ckpt")
```

`ckpt_path` restores the model, optimizers, schedulers, callbacks and step counters; a local
checkpoint is uploaded with a launched job. In one process a resumed run also gives the same
weights as one that never stopped: checkpoints hold the random number generators (torch, Python,
NumPy, CUDA, MPS), a run resumed within an epoch rebuilds that epoch's order and skips the trained
batches in the sampler, without reading them, and a `torchdata` `StatefulDataLoader` continues
by itself. Across processes the data order is exact, but each rank's own random numbers are not
restored. A loader of an iterable dataset is read and thrown away up to where it stopped. Some
GPU operations are not deterministic unless `deterministic=True`. `ckpt_path="last"` picks the newest `last*.ckpt` of
this Trainer, its `ModelCheckpoint` folders and the run folder, or starts from scratch if there
is none, so one script can both start and continue a run; `"best"` needs a `ModelCheckpoint`.
A rerun of a launched script finds its checkpoint in `gpulayer_runs/<job-id>/`: pass that path. After `fit`, `trainer.callback_metrics`,
`trainer.history`, `trainer.checkpoint_path` and `trainer.launch_result` hold the results.

### Surviving preemption

```python
trainer = glt.Trainer(max_epochs=100, accelerator="cuda", devices=4, num_nodes=2,
                      launch_options={"max_attempts": 5})
trainer.fit(model, loader)
```

With `max_attempts` above 1, the worker stores each new `last.ckpt` on the control plane in the
background, without making training wait. When a worker is lost or the job gets a SIGTERM (a
Spot VM being reclaimed), the job runs again and `fit` continues from the newest checkpoint.
A group of machines restarts as a whole, after every rank of the old attempt has stopped. A
crash in your code is not retried. If the attempts run out, the checkpoint comes back in the
job's outputs as `resume/last.ckpt`.

What repeats is the work since the last `last.ckpt`, by default up to one epoch: add
`ModelCheckpoint(every_n_train_steps=..., save_last=True)` to lose less. Sharded checkpoints and
a remote `default_root_dir` are not kept this way ([ADR-0025](../decisions/ADR-0025-keeping-checkpoints-and-restarting-groups.md)).
Tested with real processes on one machine, with the job sending itself a SIGTERM; not yet on a
reclaimed cloud VM.

### Object stores

`default_root_dir`, `ModelCheckpoint(dirpath=...)`, `ckpt_path`, `save_checkpoint`,
`load_from_checkpoint` and `CSVLogger(save_dir=...)` take URLs such as `s3://bucket/run`, through
[fsspec](https://filesystem-spec.readthedocs.io):

```bash
pip install 'gpulayer[remote]' s3fs     # or gcsfs, adlfs
```

```python
trainer = glt.Trainer(default_root_dir="s3://bucket/run", max_epochs=20)
trainer.fit(model, loader, ckpt_path="last")       # continues from s3://bucket/run/checkpoints
```

With a remote `default_root_dir`, checkpoints and `metrics.json` go to `<url>/checkpoints` and
`<url>`, and a launched job reads its result from there, so workers need `fsspec` and the
store's package too (`launch_options={"requirements": ["fsspec", "s3fs"]}`) and credentials from
their environment, never in the URL (a URL with a password is refused, since paths are printed
in logs). A checkpoint is written to a local file first and uploaded whole. A URL nothing can
serve is an error when the Trainer is created. Sharded checkpoints need a local folder, and
`TensorBoardLogger` needs a local `save_dir`. Tested with fsspec's in-memory filesystem; not yet
run against S3, GCS or Azure.

## Validate, test and predict

```python
metrics = trainer.validate(model, val_loader)          # {"val_acc": ...}, also metrics[0]
metrics = trainer.test(ckpt_path="best", datamodule=dm)
outputs = trainer.predict(model, loader)               # one output per batch, on the CPU
```

They run under `torch.inference_mode`. With several devices or machines they run on workers
like `fit`: each process takes its shard, metrics are combined across processes, and
predictions come back in rank order (`trainer.predict_batch_indices` gives each batch's dataset
indices).

## Several dataloaders

Training takes a list or dict of loaders (or a `glt.CombinedLoader`; default mode
`max_size_cycle`). Validation, test and predict take a list: the step receives
`dataloader_idx`, and logged names get a `/dataloader_idx_N` suffix.

## Preemption and cancellation

On SIGTERM (a spot VM being reclaimed, a cancelled job) `fit` saves `checkpoints/last.ckpt`,
mid-epoch if needed, and raises `glt.SIGTERMException`; `fit(..., ckpt_path=...)` continues
from the next batch. One process checks after every batch; several agree at log steps and
epoch ends, so no step pays for an extra synchronisation.

## DDP options

`glt.DDPStrategy` takes Lightning's options: `ddp_comm_hook` (for example
`default_hooks.fp16_compress_hook`, half the bytes per gradient exchange; it replaces two-level
sync), `model_averaging_period` with a `PostLocalSGDState` (post-local SGD),
`process_group_backend` and `timeout`. Validated on two T4 machines over NCCL: fp16 compression
made steps 11% faster and post-local SGD 21% faster than plain DDP on that network
([results](../../benchmarks/results/2026-11-01-aws-2xt4-ddp-hooks/README.md)).

## FSDP (experimental)

`strategy="fsdp"` shards each submodule's parameters, gradients and optimizer state across
processes with PyTorch's FSDP2, so each holds about 1/N of the model. The Module's parameters
must live in submodules. It trained over NCCL on 2 and 8 T4 machines like DDP; its speed and
large models are not measured.

```python
glt.FSDPStrategy(
    sharding_strategy="HYBRID_SHARD",        # shard within a machine, replicate across them
    auto_wrap_policy={Block},                 # shard each Block as one unit
    activation_checkpointing_policy={Block},  # recompute Block activations in backward
    cpu_offload=False,
    state_dict_type="sharded",                # each rank writes its part: no gather on rank 0
)
```

Sharded checkpoints are folders (`last.ckpt/`); `ckpt_path` and `load_from_checkpoint` read
them, and a launched job ships them to its workers. After a job on several machines, each
node's shard files are merged into one folder, so no shared filesystem is needed. The options above were validated on two Tesla T4 machines over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-parallel/README.md)).

## Tensor parallelism (experimental)

`glt.ModelParallelStrategy(tensor_parallel_size=...)` gives the Module a
`(data_parallel, tensor_parallel)` device mesh, by default tensor parallel inside each machine.
Apply the plan in `configure_model`, as in Lightning:

```python
def configure_model(self):
    tp = self.device_mesh["tensor_parallel"]
    parallelize_module(self.block, tp, {"a": ColwiseParallel(), "b": RowwiseParallel()})
    fully_shard(self, mesh=self.device_mesh["data_parallel"])
```

Each data-parallel group reads its own shard of the data. With tensor parallelism only, keep
the `fully_shard` call: it makes every parameter a DTensor, which the optimizer's fused update
needs. Validated on two T4 machines: each GPU held half of every split layer and half the
memory of one GPU ([results](../../benchmarks/results/2026-11-01-aws-2xt4-parallel/README.md)).

## Large models: init, quantization, FP8

- `with trainer.init_module(): model = Model()` builds the weights on the training device in
  the precision's dtype; `empty_init=True` skips their initialisation before a checkpoint loads.
- `plugins=[glt.BitsandbytesPrecision("nf4-dq")]` quantizes `nn.Linear` layers to 4 or 8 bits
  (needs `bitsandbytes` and CUDA).
- `plugins=[glt.TransformerEnginePrecision()]` trains in FP8 on Hopper or Ada GPUs (needs
  `transformer_engine`).
- `model.to_tensorrt("model.ep", input_sample=x)` compiles with Torch-TensorRT for inference
  (validated on a T4: outputs within 1e-6 of PyTorch).

`BitsandbytesPrecision` was validated on a Tesla T4 (nf4: 145 MB to 27 MB of GPU memory,
outputs within 0.997 cosine of fp32; [results](../../benchmarks/results/2026-11-01-aws-2xt4-deepspeed/README.md)). `TransformerEnginePrecision` was
validated on an NVIDIA L4 with Transformer Engine 2.20: FP8 was on in the forward pass and the
loss ended within 2% of bf16-true. That version refuses to load the FP8 scaling state of a
checkpoint, which is a pickle, unless `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1` is set: set it
only for checkpoints you trust ([results](../../benchmarks/results/2026-11-09-hf-gpus/README.md)).

## DeepSpeed and TPU

- **DeepSpeed** (`strategy="deepspeed_stage_2"`, `glt.DeepSpeedStrategy(stage=3,
  offload_optimizer=True)` or `config=`) needs `pip install deepspeed` (and `ninja` for the
  CPU-offload optimizer) and CUDA GPUs. Validated on Tesla T4s: ZeRO stages 1 to 3, with and
  without CPU offload, on one machine and across two over NCCL; stage 3 with offload cut peak
  GPU memory from 1,036 MB to 458 MB ([results](../../benchmarks/results/2026-11-01-aws-2xt4-deepspeed/README.md)). One optimizer, automatic optimization;
  resuming a DeepSpeed run is not supported yet.
- **TPU** (`accelerator="tpu"`, implemented, not yet run on a TPU) needs `torch_xla` on a TPU VM. `devices="auto"` (or the VM's
  device count) trains on every TPU device: as in Lightning, the Trainer starts one process
  per device with `xmp.spawn`, feeds batches through `MpDeviceLoader`, averages gradients in
  `xm.optimizer_step`, reduces metrics with XLA collectives, and loads rank 0's checkpoint
  back into your process. On a TPU pod, start the script on every host (`gcloud compute tpus
  tpu-vm ssh ... --worker=all`), as with Lightning; `num_nodes` and `target` are not used.
  Tested with a simulated `torch_xla` (one VM and a two-host pod).
- **Habana HPU** is rejected with an error. Lightning removed it as well.
