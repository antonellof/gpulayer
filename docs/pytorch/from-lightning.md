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

class Model(glt.LightningModule):          # the same class as glt.Module
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
| `accelerator="gpu"` | `accelerator="cuda"` (or `"rocm"`, `"mps"`) |
| `max_epochs=None` means 1000 | the default is `max_epochs=1`; pass a number |
| progress bar and model summary on by default | off by default: `enable_progress_bar=True`, `enable_model_summary=True` |
| `logger=True` means TensorBoard (or CSV) | `logger=None` writes `metrics.csv` in the run folder; pass `glt.TensorBoardLogger(...)` |
| `ModelCheckpoint()` names files `epoch=..-step=..ckpt` | `best.ckpt`; pass `filename="{epoch}-{step}"` for Lightning's names |
| `trainer.validate/test` return a list of dicts | a dict that also answers `result[0]` |
| `benchmark=None` leaves cuDNN autotuning off | `None` turns it on for CUDA runs of 1,000 steps or more |
| `L.seed_everything` | `glt.seed_everything` |

## Not run on hardware yet, or not available

- DeepSpeed and TPU (XLA) are implemented but not yet run on hardware; see
  [DeepSpeed and TPU](trainer.md#deepspeed-and-tpu).
- `glt.LightningCLI` ([Command line](cli.md)) cannot link arguments between sections.
- Not available: Habana HPUs (Lightning ships them in the separate `lightning-habana`
  package) and TPU pods across several TPU VMs. Run Lightning itself for those with
  [`launch()`](launch.md).

## Validated on GPUs

`sync_batchnorm`, 16-mixed with manual optimization, `torch.compile` with Inductor and
`PyTorchProfiler`'s CUDA tables were validated on Tesla T4s (2026-11-01), the first two across
up to 8 machines over NCCL. FSDP is experimental: it trained over NCCL on 2 and 8 T4 machines; its
speed and larger models are not measured. See [Project status](../project/status.md).
