# Callbacks, loggers and profilers

Callbacks add behaviour to the [Trainer](trainer.md) without changing the model. Names and
signatures follow PyTorch Lightning; full signatures are in the [API reference](api.md#callbacks).

## Built-in callbacks

```python
from gpulayer.pytorch import EarlyStopping, ModelCheckpoint, Trainer

best = ModelCheckpoint(monitor="val_loss", filename="{epoch}-{val_loss:.3f}", save_top_k=3)
trainer = Trainer(max_epochs=50, callbacks=[EarlyStopping(monitor="val_loss", patience=3), best])
trainer.fit(model, train_loader, val_loader)
model = MyModel.load_from_checkpoint(best.best_model_path)
trainer.test(model, test_loader, ckpt_path="best")
```

| Callback | What it does |
|---|---|
| `ModelCheckpoint(monitor, filename, save_top_k, save_last, every_n_epochs, ...)` | keeps the best checkpoints by `monitor` (`None`: the latest). Default file `best.ckpt`; `save_top_k=-1` keeps all. Sets `best_model_path`, `best_model_score`, `last_model_path`, also when training ran on workers |
| `EarlyStopping(monitor, min_delta, patience, mode, ...)` | stops on no improvement, a threshold or NaN |
| `LearningRateMonitor(logging_interval, log_key_prefix)` | records `lr` (`lr-<i>` per group) |
| `Timer(duration)` | stops after a duration |
| `GradientAccumulationScheduler({0: 8, 4: 4})` | changes `accumulate_grad_batches` by epoch |
| `StochasticWeightAveraging(swa_lrs, swa_epoch_start)` | SWA with `SWALR` |
| `EMAWeightAveraging(decay)`, `WeightAveraging(...)` | averaged weights, kept in the model and checkpoints |
| `BackboneFinetuning(unfreeze_backbone_at_epoch)`, `BaseFinetuning` | freezes `model.backbone`, then unfreezes it |
| `ModelPruning(pruning_fn, amount, use_lottery_ticket_hypothesis, ...)` | `torch.nn.utils.prune` at epoch ends, structured or not |
| `ThroughputMonitor(batch_size_fn, length_fn)` | samples, batches or tokens per second |
| `DeviceStatsMonitor(filter_keys)` | CUDA or MPS memory (CPU with `psutil`) |
| `SpikeDetection(window, rtol)` | raises `TrainingSpikeException` when the loss jumps |
| `ModelSummary(max_depth)`, `RichModelSummary` | layers and parameter counts |
| `ProgressBar()` (`TQDMProgressBar`), `RichProgressBar` | a progress line per epoch (`rich` for the Rich ones) |
| `BasePredictionWriter(write_interval)` | writes predictions per batch or epoch |
| `OnExceptionCheckpoint(dirpath)`, `ModelCheckpoint(save_on_exception=True)` | checkpoints on an error or Ctrl+C (one process); both are `Checkpoint` subclasses |
| `LambdaCallback(on_train_epoch_end=...)` | a callback from functions |
| `LearningRateFinder()`, `BatchSizeFinder()` | run the [Tuner](#tuner) when `fit` starts (one process) |

## Writing a callback

```python
class PrintLoss(glt.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 100 == 0:
            print(trainer.global_step, float(outputs["loss"]))  # reads the GPU: keep it rare

    def state_dict(self):              # saved in checkpoints, restored on resume
        return {}
```

Override any Lightning hook. Set `trainer.should_stop = True` to end training;
`trainer.save_checkpoint("name.ckpt")` writes a checkpoint. A monitored name must be logged
with `self.log(...)`. Callbacks written for 0.7 with `on_train_batch_end(trainer, model, loss,
batch_idx)` still work.

In distributed runs callbacks run on every rank, and if any rank sets `should_stop` all ranks
stop after that epoch. When `fit` runs on workers, only `on_fit_end` runs in your process.

## Loggers

```python
trainer = glt.Trainer(logger=[glt.CSVLogger("logs"), glt.TensorBoardLogger("tb")],
                      profiler="simple")
```

`CSVLogger`, `TensorBoardLogger`, `WandbLogger`, `MLFlowLogger`, `CometLogger` and `LitLogger`
(Lightning AI's service) log from rank 0 and add no device synchronisation. All but
`CSVLogger` need their package installed and are tested in CI against fake packages only.

## Profilers

`profiler="simple"` times each loop action, `"advanced"` adds `cProfile`, `"pytorch"`
(`glt.PyTorchProfiler(export_to_chrome=True)`) uses `torch.profiler`, and `"xla"`
(`glt.XLAProfiler`) starts torch_xla's profiler server on TPUs.

Lightning's helpers (`grad_norm`, `rank_zero_only`, `measure_flops`, `move_data_to_device`,
...) are in `gpulayer.pytorch.utilities`.

## Tuner

```python
tuner = glt.Tuner(trainer)
finder = tuner.lr_find(model, train_loader)        # sets model.lr (or hparams.lr)
size = tuner.scale_batch_size(model, datamodule=dm)  # sets dm.batch_size
```

Both run in this process and restore the weights. `scale_batch_size` doubles until out of
memory; `mode="binsearch"` then narrows it.

## Cost per step

Monitors read the device only on logged steps. `SpikeDetection` and the tuner read the loss
every step, which makes the CPU wait for the GPU: use them for debugging, not long runs.
