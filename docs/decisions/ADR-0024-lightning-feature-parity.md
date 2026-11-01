# ADR-0024: Lightning feature parity for the GPUlayer Trainer

- **Status:** Accepted
- **Date:** 2026-11-01
- **Amends:** [ADR-0018](ADR-0018-trainer-scope.md) (point 3 is replaced; points 1, 4 and 5 stay)

## Context

ADR-0018 kept the Trainer small: it adopted the features most scripts need and left loggers,
model summaries, progress bars, tuners and many callbacks to users' own Lightning scripts run
with `launch()`. In practice that sends anyone with an existing `LightningModule` back to
Lightning as soon as their code uses one hook or callback the Trainer lacks. The product focus
(ADR-0023) is one Trainer from a laptop to many machines, so code written for Lightning should
run on it with an import change.

We cloned PyTorch Lightning (master, 2026-09-21, commit `84df182`) and went through its
Trainer arguments, `LightningModule` and `LightningDataModule` methods, hooks, callbacks,
loggers, profilers, tuner and the code paths that affect speed.

Lightning's speed-related choices are few, and the GPUlayer Trainer already had most of them
(`zero_grad(set_to_none=True)`, cuDNN autotuning, no per-step host reads):

- `torch.inference_mode` for `validate`, `test` and `predict`; `no_grad` for validation
  inside `fit` and with gloo or FSDP;
- DDP options `find_unused_parameters`, `static_graph`, `gradient_as_bucket_view`;
- `sync_batchnorm`;
- "-true" precisions (`bf16-true`, `16-true`, `64-true`) that keep weights in that dtype;
- `torch.compile`d models trained compiled.

## Decision

1. **Adopt Lightning's public surface where it does not need a dependency we cannot test.**
   Implemented in `gpulayer.pytorch`:
   - Trainer arguments: `min_epochs`, `min_steps`, `limit_test_batches`,
     `limit_predict_batches`, fractional `limit_*`, `overfit_batches`,
     `reload_dataloaders_every_n_epochs`, `use_distributed_sampler`, `inference_mode`,
     `detect_anomaly`, `sync_batchnorm`, `barebones`, `logger`, `profiler`,
     `enable_progress_bar`, `enable_model_summary`, `deterministic="warn"`, `devices="auto"`,
     `max_time` as a duration, precision short forms (`32`, `"bf16"`, ...) and "-true" modes,
     strategy strings (`"ddp_find_unused_parameters_true"`) and `DDPStrategy(...)`.
   - Module: every Lightning hook (`setup`, `on_train_batch_start` returning -1, `on_before_backward`,
     `on_after_backward`, `on_before_optimizer_step`, `on_before_zero_grad`, validation/test/
     predict hooks, `on_save_checkpoint`, `on_load_checkpoint`, batch-transfer hooks,
     `configure_model`, `configure_callbacks`, `configure_gradient_clipping`, `optimizer_step`,
     `optimizer_zero_grad`, `lr_scheduler_step`), manual optimization
     (`automatic_optimization=False`, `optimizers()`, `lr_schedulers()`, `manual_backward`,
     `clip_gradients`, `toggle_optimizer`), several optimizers, every `configure_optimizers`
     return form including scheduler dicts with `interval`, `frequency` and `monitor`
     (ReduceLROnPlateau), `save_hyperparameters`/`hparams` (saved in checkpoints, used by
     `load_from_checkpoint`), `self.log` options (`reduce_fx`, `batch_size`, `prog_bar`,
     `on_step`), `log_dict`, `print`, `all_gather`, `freeze`, `to_torchscript`, `to_onnx`,
     `LightningModule`/`LightningDataModule` aliases.
   - `DataModule` with `prepare_data` (once per machine), `setup(stage)`, the four loaders,
     transfer hooks and state.
   - Callbacks: `ModelCheckpoint` (Lightning's arguments: templates, `every_n_train_steps`,
     `train_time_interval`, `save_last`, `save_top_k` of -1 and 0, `monitor=None`,
     `save_weights_only`), `EarlyStopping` (thresholds, `check_finite`, `strict`), `Timer`,
     `LambdaCallback`, `GradientAccumulationScheduler`, `ThroughputMonitor`,
     `DeviceStatsMonitor`, `SpikeDetection`, `ModelSummary`, `ProgressBar`,
     `BasePredictionWriter`, `OnExceptionCheckpoint`, `StochasticWeightAveraging`,
     `WeightAveraging`, `EMAWeightAveraging`, `BaseFinetuning`, `BackboneFinetuning`,
     `ModelPruning`, `LearningRateFinder`, `BatchSizeFinder`; callback `state_dict` saved in
     checkpoints.
   - Loggers: `CSVLogger`, `TensorBoardLogger`, `WandbLogger`, `MLFlowLogger` (the last three
     need their packages). Profilers: `SimpleProfiler`, `AdvancedProfiler`, `PyTorchProfiler`.
   - `Tuner.lr_find` and `Tuner.scale_batch_size`.
   - Checkpoints in Lightning's layout (`optimizer_states`, `lr_schedulers`, `callbacks`,
     `hyper_parameters`, scaler), resuming within an epoch from step checkpoints, and
     Ctrl+C ending `fit` gracefully after `on_exception` hooks.
2. **Still not adopted:** DeepSpeed, TPU/HPU/XLA accelerators, `plugins=`, `LightningCLI`,
   several dataloaders per stage, distributed `predict`, Lightning Fabric. Users who need
   them keep Lightning and run it with `launch()`.
3. **Defaults that differ from Lightning, on purpose:** `max_epochs=1` (Lightning: 1000),
   `enable_progress_bar=False` and `enable_model_summary=False` (GPUlayer prints one log line
   per epoch; job logs are not terminals), `benchmark=None` (ADR-0018 amendment),
   `logger=None` keeps the built-in `metrics.csv`. `validate`/`test` return a dict that also
   answers `result[0]`, Lightning's list form.
4. **The per-step path stays lean (ADR-0018 point 4).** Hooks are resolved once per run and
   only overridden ones are called; `self.log` has a fast path for the default call; epoch
   metrics from all names are averaged across ranks in one all-reduce instead of one per
   name; loggers and monitoring callbacks act only on log steps with values already read;
   profiling code runs only with a profiler. Measured on an Apple M2 Pro (MLP on CPU, A/B
   against 0.7.1 in alternating fresh processes): 0.589 ms per step before, 0.597 after
   (+1.4%, within run-to-run spread); MPS and 2-process DDP unchanged
   (`benchmarks/results/2026-11-01-m2pro-parity/`).
5. **Validation status.** Every feature has a CPU test (`tests/pytorch/test_lightning_parity.py`);
   manual optimization with DDP and the DDP options run in a two-process test. CUDA-only
   features (`sync_batchnorm`, 16-mixed with manual optimization, `torch.compile` with
   Inductor, `PyTorchProfiler` CUDA tables) are implemented and named as such until a GPU run
   validates them.

## Consequences

- A `LightningModule` usually moves by changing `import lightning as L` to
  `import gpulayer.pytorch as glt` and `L.LightningModule` to `glt.Module` (or
  `glt.LightningModule`). Differences are listed in the Trainer docs.
- Breaking for 0.7 callbacks only in argument order: `ModelCheckpoint`'s first positional
  argument is now `dirpath`, as in Lightning (`monitor` was first). Callbacks with the 0.7
  `on_train_batch_end(trainer, model, loss, batch_idx)` signature keep working.
- The Trainer is larger; ADR-0018's rule that loop changes come with a comparison run still
  holds.
