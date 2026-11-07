# Moving from PyTorch Lightning

The GPUlayer Trainer uses Lightning's names
([ADR-0024](../decisions/ADR-0024-lightning-feature-parity.md)), so most code moves by changing
the imports.

## Side by side

```python
# Lightning
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

class Model(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(32, 1)
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.net(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=4, num_nodes=2,
                    callbacks=[EarlyStopping("val_loss"), ModelCheckpoint(monitor="val_loss")])
trainer.fit(Model(), datamodule=dm)
```

```python
# GPUlayer
import gpulayer.pytorch as glt
from gpulayer.pytorch import EarlyStopping, ModelCheckpoint

class Model(glt.Module):                   # L.LightningModule in Lightning
    ...                                    # unchanged

trainer = glt.Trainer(max_epochs=10, accelerator="cuda", devices=4, num_nodes=2,
                      callbacks=[EarlyStopping("val_loss"), ModelCheckpoint(monitor="val_loss")])
trainer.fit(Model(), datamodule=dm)        # GPUlayer picks two machines and starts 8 processes
```

Lightning needs torchrun, SLURM or a cluster to start processes on each machine. GPUlayer's
`fit` starts them on its workers and returns the trained weights to `model`.

## Renamed or different

| Lightning | GPUlayer |
|---|---|
| `L.LightningModule`, `L.LightningDataModule` | `glt.Module`, `glt.DataModule` |
| `LightningCLI`, `LightningArgumentParser` | `glt.GPULayerCLI`, `glt.GPULayerArgumentParser` |
| `trainer.lightning_module` | `trainer.module` |
| `accelerator="gpu"` | accepted; GPUlayer's name is `"cuda"` (or `"rocm"`, `"mps"`) |
| no limit given means 1000 epochs | no limit given means one epoch; `max_steps` and `max_time` work as in Lightning |
| progress bar and model summary on by default | off by default: `enable_progress_bar=True`, `enable_model_summary=True` |
| `logger=True` means TensorBoard (or CSV) | `logger=None` writes `metrics.csv` in the run folder; pass `glt.TensorBoardLogger(...)` |
| `ModelCheckpoint()` names files `epoch=..-step=..ckpt` | `best.ckpt`; pass `filename="{epoch}-{step}"` for Lightning's names |
| `trainer.validate/test` return a list of dicts | a dict that also answers `result[0]` |
| `benchmark=None` leaves cuDNN autotuning off | `None` turns it on for CUDA runs of 1,000 steps or more |
| `L.seed_everything` | `glt.seed_everything` |
| `lightning.pytorch.utilities` (`grad_norm`, `rank_zero_only`, `measure_flops`, ...) | `gpulayer.pytorch.utilities` |
| SIGTERM checked every step on every rank | one process: every step; several: at log steps and epoch ends (no extra sync per step) |
| `model_registry=` (Lightning AI cloud) | rejected: checkpoints come back to `default_root_dir` |

## Not run on hardware yet, or not available

Every public name in Lightning's `Trainer`, `LightningModule`, hooks, callbacks, loggers,
profilers, plugins, strategies, accelerators and utilities has a GPUlayer counterpart (compared
with `scripts/compare_api.py` against Lightning's source). How far each has run:

- DeepSpeed (ZeRO 1 to 3, CPU offload) and `BitsandbytesPrecision` were validated on Tesla
  T4s; TPU (XLA) is implemented but not yet run on a TPU. See
  [DeepSpeed and TPU](trainer.md#deepspeed-and-tpu).
- `ModelParallelStrategy` (tensor parallelism), the FSDP options (sharding strategies,
  activation checkpointing, CPU offload, sharded checkpoints) and `to_tensorrt` were validated
  on Tesla T4s across two machines ([results](../../benchmarks/results/2026-11-01-aws-2xt4-parallel/README.md)).
- `TransformerEnginePrecision`, `XLAProfiler`, and the
  Comet, W&B, MLflow, TensorBoard and `LitLogger` loggers are tested against fake packages: the
  real ones need NVIDIA GPUs, TPUs or accounts.
- `glt.GPULayerCLI` ([Command line](cli.md)) uses `argparse`, not `jsonargparse`.
- Habana HPUs are not supported. Current Lightning removed them too (`lightning-habana` is
  archived).

## Lightning's speed features

Each of Lightning's performance mechanisms, found in its source, and GPUlayer's equivalent:

| Lightning | GPUlayer |
|---|---|
| calls only hooks a model or callback overrides | same |
| keeps logged values on the device; syncs at log steps | same |
| `inference_mode` for evaluation (`no_grad` with gloo, FSDP, XLA) | same |
| `zero_grad(set_to_none=True)`; `no_sync` while accumulating | same |
| DDP `static_graph`, `gradient_as_bucket_view`, communication hooks | same |
| one CPU thread share per process | same, set by the worker (`OMP_NUM_THREADS`) |
| hint to enable TF32 on Tensor Core GPUs; hint when `num_workers` is low | same hints, once per process |
| keeps every `DataLoader` setting when adding a `DistributedSampler` | same |
| `benchmark=` for cuDNN autotuning | on by default for CUDA runs of 1,000 steps or more |
| `torch.compile`, `AsyncCheckpointIO`, `MpDeviceLoader` on TPUs | same |
| FSDP, activation checkpointing, tensor parallelism, 4-bit and FP8 plugins | same (validated on T4s except FP8, which needs Hopper or Ada GPUs) |
| SLURM auto-requeue on SIGUSR1 | not needed: GPUlayer retries tasks itself; SIGTERM saves a checkpoint |

Each row has a behaviour test in `tests/pytorch/test_speed_mechanisms.py` (for example, the
number of host-device synchronisations does not grow with the number of steps). The
device-dependent ones run on every accelerator present and passed on the CPU and an Apple M2 Pro
GPU (MPS); on CUDA machines they run on the GPU too. cuDNN
autotuning and `MpDeviceLoader` are covered by the T4 measurements and the simulated TPU tests.
GPUlayer's own additions are in [Performance](../architecture/performance.md): one autocast
object reused across steps, two-level gradient sync with gloo, the direct process launcher and
placement on the best-connected GPUs.

## Validated on GPUs

`sync_batchnorm`, 16-mixed with manual optimization, `torch.compile` with Inductor and
`PyTorchProfiler`'s CUDA tables were validated on Tesla T4s (2026-11-01), the first two across
up to 8 machines over NCCL. FSDP is experimental: it trained over NCCL on 2 and 8 T4 machines; its
speed and larger models are not measured. See [Project status](../project/status.md).
