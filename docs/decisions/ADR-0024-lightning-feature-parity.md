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
5. **Validation status.** Every feature has a CPU test (`tests/pytorch/test_trainer_api.py`);
   manual optimization with DDP and the DDP options run in a two-process test. CUDA-only
   features (`sync_batchnorm`, 16-mixed with manual optimization, `torch.compile` with
   Inductor, `PyTorchProfiler` CUDA tables) were validated on 2026-11-01 on Tesla T4s, across
   up to 8 AWS machines over NCCL (`benchmarks/results/2026-11-01-aws-8xt4/`). That run found
   one bug, fixed in the same change: `clip_gradients` with manual optimization and 16-mixed
   unscaled the gradients twice.

## Consequences

- A `LightningModule` usually moves by changing `import lightning as L` to
  `import gpulayer.pytorch as glt` and `L.LightningModule` to `glt.Module` (or
  `glt.LightningModule`). Differences are listed in the Trainer docs.
- Breaking for 0.7 callbacks only in argument order: `ModelCheckpoint`'s first positional
  argument is now `dirpath`, as in Lightning (`monitor` was first). Callbacks with the 0.7
  `on_train_batch_end(trainer, model, loss, batch_idx)` signature keep working.
- The Trainer is larger; ADR-0018's rule that loop changes come with a comparison run still
  holds.

## Amendment (2026-11-01): the rest of the surface

Point 2 listed what was not adopted. Most of it is now implemented, with Lightning's names:

- Several dataloaders per stage: `CombinedLoader` (`max_size_cycle`, `min_size`, `max_size`,
  `sequential`) for training; lists for validate/test/predict, with `dataloader_idx` and
  `/dataloader_idx_N` metric names.
- Distributed `validate`, `test` and `predict`: with several devices or machines they run on
  the workers, each rank on its shard; metrics are averaged across ranks and predictions come
  back in rank order. A job launched by an earlier `fit` of the same Trainer is reused, since it
  already ran the rest of the script.
- `LightningCLI` as `GPULayerCLI` (standard library only; no argument linking).
- `plugins=`: precision plugins, `CheckpointIO`, `TorchCheckpointIO`, `AsyncCheckpointIO`
  (background checkpoint writes) and `TorchSyncBatchNorm`. Cluster-environment plugins are not
  needed because GPUlayer starts the processes.
- `DeepSpeedStrategy` (ZeRO 1–3, offload, Lightning's strategy names) and `accelerator="tpu"`
  (one device per process, `xm.optimizer_step`, `xm.mark_step`). Both are implemented and
  unit-tested for their configuration and error paths, but not run: they need DeepSpeed with
  CUDA GPUs and torch_xla on a TPU VM.

Still not available: resuming DeepSpeed runs from a checkpoint. Habana HPUs are rejected with
an error: Lightning removed them (`_graveyard/hpu.py`; `lightning-habana` archived).

**Several TPU devices (2026-11-01).** `accelerator="tpu"` with `devices="auto"` (or the VM's
device count) now trains on every TPU device of one TPU VM, following Lightning's
`XLAStrategy`: the launching process calls `xmp.spawn` (fork, one process per device, no
GPUlayer job), each process takes its rank from `torch_xla.runtime`, the model is not wrapped
in DDP, `xm.optimizer_step` all-reduces the gradients over the TPU interconnect, a gloo group
carries the small CPU reductions (metrics, stop flags, barriers), and rank 0 saves CPU-tensor
checkpoints that the launching process loads. Tested with a fake `torch_xla` package
(`tests/pytorch/test_tpu_spawn.py`); not run on a TPU.

Follow-up the same day, closer to Lightning: no gloo group on TPUs. Metrics, stop flags,
barriers and `all_gather` use torch_xla's collectives (`xm.all_reduce`, `xm.rendezvous`,
`xm.all_gather`), which also span the hosts of a TPU pod, so a pod is one run when the script
is started on every host, as Lightning documents. Batches go through `MpDeviceLoader` (next
batches copied to the device while the step runs), and evaluation uses `torch.no_grad`, as
Lightning does on XLA. The simulated test covers one VM and a two-host pod.

**Full public API (2026-11-01).** A comparison of Lightning's source (commit `84df182`) with
`gpulayer.pytorch`, now `scripts/compare_api.py`, found the remaining gaps; all are
closed, and the script reports none missing:

- Trainer: `log_key_prefix`, `enable_autolog_hparams`, `suggest_integrations`, `init_module`
  (with `empty_init`), `print`, `device_ids`, `distributed_sampler_kwargs`, the batch counts,
  `received_sigterm`. `model_registry` (Lightning AI's cloud) is rejected with an error.
- SIGTERM: `fit` saves a resumable checkpoint (mid-epoch if needed) and raises
  `SIGTERMException`. Unlike Lightning, which broadcasts a flag every step, several processes
  agree at log steps and epoch ends, so steps carry no extra synchronisation (ADR-0016).
- Module: the eval/train mode hooks, `on_validation_model_zero_grad`, `strict_loading`,
  `configure_sharded_model`, `to_tensorrt`. As in Lightning, the training loop no longer calls
  `model.train()` every epoch: submodules keep their mode and evaluation restores each one.
- Strategies: Lightning's `DDPStrategy` options (communication hooks, post-local SGD, backend,
  timeout), `FSDPStrategy` options mapped onto FSDP2 (sharding strategies, wrap and activation
  checkpointing policies, CPU offload, mixed-precision policy, sharded checkpoints),
  `ModelParallelStrategy` (2D device mesh for tensor parallelism with FSDP2), DeepSpeed's
  configuration arguments, and the base classes.
- Plugins: `BitsandbytesPrecision`, `TransformerEnginePrecision`, `XLAPrecision`,
  `FSDPPrecision`, `DeepSpeedPrecision`, `XLACheckpointIO`, `LayerSync`.
- Callbacks, loggers, profilers: `Checkpoint`, `RichProgressBar`, `RichModelSummary`, the
  missing arguments of `ModelCheckpoint`, `LearningRateMonitor`, `DeviceStatsMonitor` and
  `ModelPruning`; `CometLogger`, `LitLogger`, the missing W&B, MLflow and TensorBoard options;
  `XLAProfiler` and `PyTorchProfiler`'s options.
- CLI: Lightning's constructor arguments and overridable methods, argument links, saved
  configs, optimizers from the command line, subclass mode (standard library `argparse`).
- Accelerator classes and `gpulayer.pytorch.utilities`.

Habana HPUs stay out: Lightning removed them. Tests: the FSDP options, sharded checkpoints,
tensor parallelism and DDP hooks run with two CPU processes; bitsandbytes, Transformer Engine,
TensorRT, torch_xla and the hosted loggers run against fake packages.

**Hardware (2026-11-01).** DeepSpeed (ZeRO 1 to 3, CPU offload; one machine and two over
NCCL) and `BitsandbytesPrecision` (nf4, int8) were validated on Tesla T4s, after three fixes
the run found (a process group and `LOCAL_RANK` for single-process DeepSpeed, the job Python's
`bin` on `PATH`). Results: `benchmarks/results/2026-11-01-aws-2xt4-deepspeed/`. A second run on two
T4 machines validated the FSDP options, sharded checkpoints merged across machines, tensor
parallelism and `to_tensorrt` (`benchmarks/results/2026-11-01-aws-2xt4-parallel/`), and found
that outputs over the worker's size cap were dropped silently (fixed). A third run validated
DDP communication hooks and post-local SGD (`benchmarks/results/2026-11-01-aws-2xt4-ddp-hooks/`).
Transformer Engine (FP8 needs Hopper or Ada GPUs) and TPUs have not run on hardware.

**No Lightning names in GPUlayer's code (0.9.0, 2026-11-01).** The aliases
`LightningModule`, `LightningDataModule`, `LightningCLI` and `LightningArgumentParser`, the
`Trainer.lightning_module` property and the other Lightning-named identifiers are removed or
renamed (`Module`, `DataModule`, `GPULayerCLI`, `GPULayerArgumentParser`, `Trainer.module`,
`CaseInsensitiveEnum`, `add_component_args`); files and test names follow. The API stays
compatible in shape: code moves with an import change and these renames
(docs/pytorch/from-lightning.md). Lightning is named only where GPUlayer is compared with it
(benchmarks, the compare page) or where the docs explain a migration.

