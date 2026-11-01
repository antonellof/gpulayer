# Callbacks, loggers and profilers

Callbacks add behaviour to the [Trainer](trainer.md) without changing the model: checkpoints,
early stopping, weight averaging, monitoring. Loggers send metrics to an experiment tracker,
profilers time the loop, and the tuner searches for a learning rate or batch size. All of them
follow PyTorch Lightning's names and signatures. Signatures of every class are in the
[API reference](api.md#callbacks).

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
| `ModelCheckpoint(dirpath, filename, monitor, save_top_k, save_last, every_n_train_steps, every_n_epochs, train_time_interval, save_weights_only, mode)` | keeps the best checkpoints by `monitor` (`monitor=None`: the latest). `filename` is `best.ckpt` by default or a template such as `"{epoch}-{val_loss:.2f}"`; `save_top_k=-1` keeps all, `0` none. `best_model_path`, `best_model_score`, `best_k_models`, `last_model_path` afterwards, also when training ran on workers |
| `EarlyStopping(monitor, min_delta, patience, mode, strict, check_finite, stopping_threshold, divergence_threshold)` | stops when the metric stops improving, reaches a threshold or becomes NaN |
| `LearningRateMonitor(logging_interval, log_momentum, log_weight_decay)` | records `lr` (or `lr-<i>` per group) per epoch, or per logged step with `"step"` |
| `Timer(duration, interval)` | stops after a duration; `time_elapsed()`, `time_remaining()` |
| `GradientAccumulationScheduler({0: 8, 4: 4})` | changes `accumulate_grad_batches` by epoch |
| `StochasticWeightAveraging(swa_lrs, swa_epoch_start)` | SWA with `SWALR`; recomputes BatchNorm statistics at the end |
| `EMAWeightAveraging(decay)`, `WeightAveraging(...)` | an average of the weights (one fused multi-tensor update per step), used for validation and left in the model and checkpoints |
| `BackboneFinetuning(unfreeze_backbone_at_epoch)`, `BaseFinetuning` | freeze `model.backbone`, then unfreeze it with its own learning rate |
| `ModelPruning(pruning_fn, amount)` | `torch.nn.utils.prune` at epoch ends; plain weights in checkpoints |
| `ThroughputMonitor(batch_size_fn, length_fn)` | samples, batches or tokens per second on logged steps, timed on the host (no device sync) |
| `DeviceStatsMonitor()` | CUDA or MPS memory (and CPU with `psutil`) on logged steps |
| `SpikeDetection(window, rtol)` | raises `TrainingSpikeException` when the loss jumps (reads the loss every step; for debugging) |
| `ModelSummary(max_depth)` | prints layers and parameter counts |
| `ProgressBar()` (`TQDMProgressBar`) | a progress line per epoch on a terminal (uses `tqdm` when installed) |
| `BasePredictionWriter(write_interval)` | writes predictions per batch or epoch instead of keeping them |
| `OnExceptionCheckpoint(dirpath)` | saves a checkpoint when training stops with an error or Ctrl+C (one process) |
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

Write your own by subclassing `Callback` and overriding any Lightning hook, for example
`on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)`; `state_dict` and
`load_state_dict` are saved in checkpoints. Callbacks written for GPUlayer 0.7 with
`on_train_batch_end(trainer, model, loss, batch_idx)` still work. Set
`trainer.should_stop = True` to end training; `trainer.save_checkpoint("name.ckpt")` writes a
checkpoint. The monitored name must be logged with `self.log(...)`; otherwise the error lists
the names that were logged.

In distributed runs callbacks run on every rank. Epoch metrics are averaged across ranks
first, and if any rank sets `should_stop`, every rank stops after the same epoch. When
`fit` launches training on workers, the callbacks run there; in the launching process only
`on_fit_end` runs.

## Loggers

```python
trainer = glt.Trainer(logger=[glt.CSVLogger("logs"), glt.TensorBoardLogger("tb")],
                      profiler="simple")
```

`CSVLogger`, `TensorBoardLogger`, `WandbLogger` and `MLFlowLogger` receive the hyperparameters,
the values of each logged step and each epoch's metrics, from rank 0. They are given values the
Trainer already read for its log line, so they add no device synchronisation. TensorBoard,
Weights & Biases and MLflow need their packages installed (only `CSVLogger` is tested in CI).

## Profilers

`profiler="simple"` prints the time of each loop action (data loading, training batch,
backward, optimizer step, validation) at the end; `"advanced"` adds `cProfile` per action;
`"pytorch"` (`glt.PyTorchProfiler(export_to_chrome=True)`) records operators with
`torch.profiler`. Without a profiler the loop has no profiling code on its path.

## Tuner

```python
tuner = glt.Tuner(trainer)
finder = tuner.lr_find(model, train_loader)        # sets model.lr (or hparams.lr)
size = tuner.scale_batch_size(model, datamodule=dm)  # sets dm.batch_size
```

`lr_find` trains a few steps with a rising learning rate and suggests the rate where the loss
falls fastest; `scale_batch_size` doubles the batch size until a step runs out of memory
(`mode="binsearch"` then narrows it down). Both run in this process and restore the weights.

## Cost per step

Only hooks a callback overrides are called. Callbacks that act per step but only need values on
logged steps (`LearningRateMonitor(logging_interval="step")`, `ThroughputMonitor`,
`DeviceStatsMonitor`) read nothing from the device in between. `SpikeDetection` and the tuner
read the loss every step, which makes the CPU wait for the GPU; use them for debugging and
searches, not for long runs. `EMAWeightAveraging` adds one fused multi-tensor update per
optimizer step.
