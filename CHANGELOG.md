# Changelog

All notable changes to GPUlayer. Versions follow [semantic versioning](https://semver.org);
before 1.0, minor versions may change public APIs, with migration notes here.

## Unreleased

- Fix: a job that called the Trainer more than once (fit, then predict or test) could fail
  intermittently with "Gloo connectFullMesh failed ... Connection refused": the process group
  was destroyed after each call, and a rank that finished first joined the next call through
  rank 0's old store. A multi-process group now stays up until the process exits (seen twice
  in CI; regression test added).
- Home page: dark wireframe scenes with one glowing element for the six features (gentle
  motion, none with reduced motion), and a dashboard section with
  screenshots of the overview, a distributed job and the workers.
- Tests: every public name of the reference trainer API is exercised by at least one behaviour
  test (`tests/pytorch/test_feature_coverage.py`, `test_speed_mechanisms.py`); check it with
  `python scripts/compare_api.py <reference-source> --tests`. The speed-mechanism tests run on
  every accelerator present (CPU, Apple GPU, CUDA).
- Fix: `trainer.test_dataloaders`, `val_dataloaders` and `predict_dataloaders` are set by
  `test`, `validate` and `predict` (they stayed None).

## 0.9.0 (2026-11-01)

The Trainer now covers the rest of the reference training API, validated on Tesla T4s for
DeepSpeed, FSDP options, tensor parallelism, DDP hooks, quantized training and TensorRT.

**Migrating from 0.8** (breaking changes):

- GPUlayer's code carries no Lightning names. Removed aliases: `glt.LightningModule` (use
  `glt.Module`), `glt.LightningDataModule` (`glt.DataModule`), `glt.LightningCLI`
  (`glt.GPULayerCLI`), `glt.LightningArgumentParser` (`glt.GPULayerArgumentParser`). Renamed:
  `Trainer.lightning_module` is `Trainer.module`; the CLI parser's `add_lightning_class_args`
  is `add_component_args`; `utilities.LightningEnum` is `utilities.CaseInsensitiveEnum`;
  `LitLogger`'s default folder is `./gpulayer_logs`. The benchmark scripts moved from
  `benchmarks/lightning_comparison/` to `benchmarks/trainer_comparison/`.
- The training loop no longer calls `model.train()` every epoch (frozen submodules stay in
  eval mode); call `model.train()` before `fit` if a model arrives in eval mode (`fit` warns).
- `PyTorchProfiler`'s positional arguments after `filename` follow the reference order; pass
  them by keyword.
- `WandbLogger(log_model=...)` uploads checkpoints as W&B artifacts instead of `run.save`.
- A bare class name on the CLI (`--data Data`) that matches classes in two modules is an error
  naming both; give `module.Class`.

- Validated on 2 Tesla T4 machines over NCCL: DDP communication hooks (fp16 compression, 11%
  faster steps than plain DDP on that network) and post-local SGD (21% faster; replicas
  identical after each averaging step) (`benchmarks/results/2026-11-01-aws-2xt4-ddp-hooks/`).
- Home page: no Lightning names outside the benchmark comparisons.
- Validated on 2 Tesla T4 machines over NCCL (`benchmarks/results/2026-11-01-aws-2xt4-parallel/`):
  FSDP sharding strategies, activation checkpointing, CPU offload and sharded checkpoints
  (merged across machines, resumed in a second job); tensor parallelism with
  `ModelParallelStrategy`; `Module.to_tensorrt`.
- Fix: a task whose outputs passed the worker's 512 MiB cap returned none of them (the loop
  stopped at the first file over the cap, so an 800 MB checkpoint also dropped the small
  files after it) and said nothing. A file over a limit is now skipped and named in the task's
  system log; the others still upload. The default cap is 4 GiB per task
  (`GPULAYER_MAX_OUTPUT_BYTES`), 2 GiB per file. Found on AWS T4s with a 67M-parameter model.
- Sharded checkpoints written on several machines are merged into one folder after the job,
  so they load and resume without a shared filesystem.
- DeepSpeed validated on Tesla T4s (AWS): ZeRO stages 1 to 3, with and without CPU offload, on
  one machine and across two over NCCL; `BitsandbytesPrecision` nf4 and int8 on one T4
  (`benchmarks/results/2026-11-01-aws-2xt4-deepspeed/`). Three bugs found and fixed there:
  DeepSpeed in one process needs a process group (it tried MPI discovery) and `LOCAL_RANK`,
  which the Trainer now provides for the run; and jobs did not have their Python's `bin`
  directory on `PATH`, so DeepSpeed could not find `ninja` to build its CPU Adam. Workers now
  put it first, as an activated venv does.
- Fix: sharding a DataLoader across processes dropped `persistent_workers`,
  `prefetch_factor`, `worker_init_fn`, `generator`, `timeout` and the multiprocessing context,
  so distributed runs lost the user's loader tuning. They are kept now, as in Lightning.
- Trainer: Lightning's two speed hints (enable TF32 on Tensor Core GPUs; raise a low
  `num_workers`), once per process, silenced by
  `gpulayer.pytorch.utilities.disable_possible_user_warnings()`.
- Trainer: the rest of Lightning's public API (ADR-0024, "Full public API"), checked with the
  new `scripts/compare_api.py`. Highlights: SIGTERM saves a resumable checkpoint
  and raises `SIGTERMException`; `trainer.init_module()`; FSDP options (sharding strategies,
  activation checkpointing, CPU offload, sharded checkpoints) and `ModelParallelStrategy`
  (tensor parallelism); DDP communication hooks and post-local SGD; `BitsandbytesPrecision`
  and `TransformerEnginePrecision`; `RichProgressBar`, `CometLogger`, `LitLogger`,
  `XLAProfiler`; Lightning's CLI hooks, argument links and saved configs; accelerator classes
  and `gpulayer.pytorch.utilities`. Tested on CPU (two processes for the distributed parts);
  the GPU-only plugins and hosted loggers against fake packages.
- Behaviour change, as in Lightning: the training loop no longer calls `model.train()` every
  epoch, so frozen submodules stay in eval mode; evaluation restores each submodule's mode.
  `fit` warns when the model is in eval mode at the start.
- Trainer: `accelerator="tpu"` with `devices="auto"` trains on every TPU device of a TPU VM,
  as Lightning's `XLAStrategy` does (one process per device through `xmp.spawn`, gradients
  averaged by `xm.optimizer_step`, rank 0's checkpoint and metrics loaded in your process).
  Batches go through `MpDeviceLoader`; metrics, barriers and `all_gather` use XLA collectives,
  so a TPU pod is one run when the script is started on every host, as in Lightning. Manual
  optimization on TPUs steps through `xm.optimizer_step` and `xm.mark_step`, and TPU
  checkpoints hold CPU tensors. Tested with a simulated `torch_xla` (one VM and a two-host
  pod); not run on a TPU. `accelerator="hpu"` gives an error, as Lightning removed Habana.
- Docs and site made shorter: the guides, architecture, reference, PyTorch and get-started
  pages lost 30 to 60% of their words with the same commands, settings and status; the
  Trainer page points to the API reference for the full argument list. The compare page is
  now one training-focused capability matrix (GPUlayer, Lightning, Accelerate, Ray, SkyPilot,
  dstack, Modal, Slurm/Kubernetes) with the measured speed against Lightning and when to choose
  something else. The home page drops the quickstart, placement and architecture sections.
  Corrected two stale statements: Trainer DDP on GPUs (validated on RTX 3070s and T4 machines)
  and NCCL across machines in the GPU compatibility matrix.

- Fix: a worker that finished a task while the control plane was unreachable (a server pod
  restarting) lost the outcome on the refused connection, so the server kept the task running
  and the worker busy. Outcomes are now retried with backoff for up to 2 minutes; a rejection
  is final. Found by the Kubernetes CI job; regression test added.
- Fix: manual optimization with DDP could hang (seen on a 2-vCPU CI runner). The step no
  longer relies on DDP's reducer internals: `manual_backward` averages the gradients across
  ranks in one coalesced all-reduce per dtype. Re-validated on 2 Tesla T4 machines over NCCL
  (identical weights on both ranks).
- Trainer, the rest of Lightning's surface (ADR-0024 amendment): several dataloaders
  (`glt.CombinedLoader` for training; lists for validate/test/predict with `dataloader_idx`);
  `validate`, `test` and `predict` on several devices or machines (each rank on its shard,
  results back in the launching process); `glt.GPULayerCLI` (subcommands,
  `--trainer.*`/`--model.*`/`--data.*`, JSON/YAML configs, `class_path`); `plugins=` with
  precision plugins, `CheckpointIO`, `TorchCheckpointIO`, `AsyncCheckpointIO` and
  `TorchSyncBatchNorm`; `DeepSpeedStrategy` and `accelerator="tpu"` (implemented, not run on
  hardware). New guide page: [Command line](docs/pytorch/cli.md).
- Install script at a short URL: `curl -fsSL https://gpulayer.dev/install.sh | sh` (the site
  proxies the repository's `install.sh`). The home hero no longer shows a `gpulayer up` button.
- Site: the home page hero has an animated wireframe landscape (mountains, lakes, hills) behind
  the headline, drawn with three.js loaded after the page renders. It pauses offscreen and in
  hidden tabs, shows one still frame with reduced motion, uses a lighter grid on phones and
  follows the light/dark theme.

## 0.8.1 (2026-11-01)

- Benchmarks re-measured with the 0.8 Trainer (2026-11-01) and every Trainer chart regenerated:
  one T4 on AWS (GPUlayer Trainer 1.08 to 1.38x faster than Lightning), 4x RTX 3070 on Vast.ai
  (1.26 to 1.60x in 5 of 6 cases, 1.12x slower on the 1-GPU CNN), Apple M2 Pro (1.02 to
  1.80x), and a new chart, one T4 per machine on 2, 4 and 8 AWS machines over NCCL (MLP 1.08
  to 1.29x faster, CNN 1.01 to 1.09x slower), and strong scaling with gloo on 2, 4 and 8 AWS
  machines (17.8% faster on 8). `multinode.py` and `scaling_run.py` take the
  worker layout from `MN_*` variables.

- Validated on hardware (2026-11-01, 8 AWS `g4dn.xlarge`, one Tesla T4 each,
  `benchmarks/results/2026-11-01-aws-8xt4/`): NCCL across machines (DDP on 2, 4 and 8 machines
  with 16-mixed and `sync_batchnorm`, identical weights on every rank; manual optimization on
  8; FSDP on 2 and 8), and on one T4 Inductor `torch.compile`, 16-mixed manual optimization,
  `16-true`/`bf16-true`, `PyTorchProfiler` CUDA columns, EMA and device stats.
- Fix: `clip_gradients` with manual optimization and 16-mixed unscaled the gradients twice
  (it received the wrapper from `self.optimizers()`), so such runs did not learn. Found on the
  8-machine run; regression test added.
- Docs: the NVIDIA guide explains `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` from a system
  `LD_LIBRARY_PATH` (AWS Deep Learning AMIs) and how to start workers without it.

- Docs: a generated [PyTorch API reference](docs/pytorch/api.md) with every public class,
  method, property and hook of `gpulayer.pytorch` (`scripts/make_api_reference.py`; a test
  fails when it is out of date or a public member has no docstring). The Trainer guide gains
  a hook-order diagram; callbacks, loggers, profilers and the tuner moved to
  [their own page](docs/pytorch/callbacks.md), and [Moving from Lightning](docs/pytorch/from-lightning.md)
  has a side-by-side example and a table of differences.
- Docstrings for every public member of `gpulayer.pytorch`; hooks share one description of when
  they run (`gpulayer.pytorch.hooks`). `BackboneFinetuning`'s default `lambda_func` is now the
  named function `multiplicative`, as in Lightning.
- `benchmarks/cloud/aws.sh`: instances run in a VPC of their own (`gpulayer-bench`, with its
  own internet gateway; the default VPC is not used), install uv, PyTorch, Lightning and the
  current GPUlayer commit on first boot (user-data, `aws-boot.sh`; `ready` waits for it), and
  take `GL_AWS_SPOT=1`, `GL_AWS_AZ=...`, `allow`/`revoke` for one outside IP, `ssh`, `ip`,
  `private-ip` and `down-network`.
- `scripts/validate_trainer_gpu.py` (one GPU: Inductor `torch.compile`, 16-mixed manual
  optimization, "-true" precisions, `PyTorchProfiler` CUDA columns, EMA, device stats) and
  `scripts/validate_nccl.py` (DDP with `sync_batchnorm`, manual optimization and FSDP across
  GPU machines over NCCL).

## 0.8.0 (2026-11-01)

Lightning feature parity for the GPUlayer Trainer ([ADR-0024](docs/decisions/ADR-0024-lightning-feature-parity.md)).

- Module: every Lightning hook, manual optimization (`automatic_optimization = False`,
  `optimizers()`, `manual_backward`, `clip_gradients`, `toggle_optimizer`; also with DDP),
  several optimizers, all `configure_optimizers` forms including scheduler dicts with
  `interval`, `frequency` and `monitor` (ReduceLROnPlateau), `save_hyperparameters`/`hparams`
  (used by `load_from_checkpoint`), `self.log` options (`reduce_fx`, `batch_size`,
  `prog_bar`, `on_step`), `log_dict`, `all_gather`, `freeze`, `to_onnx`, `to_torchscript`,
  model-defined dataloaders, `LightningModule` alias.
- `DataModule` (`LightningDataModule`): `prepare_data` once per machine, `setup(stage)`,
  loaders, transfer hooks; `fit/validate/test/predict(datamodule=...)`.
- Trainer arguments: `min_epochs`, `min_steps`, `limit_test_batches`,
  `limit_predict_batches`, fractional limits, `overfit_batches`,
  `reload_dataloaders_every_n_epochs`, `use_distributed_sampler`, `inference_mode` (on for
  validate/test/predict), `detect_anomaly`, `sync_batchnorm` (CUDA/ROCm), `barebones`,
  `logger`, `profiler`, `enable_progress_bar`, `enable_model_summary`,
  `deterministic="warn"`, `devices="auto"`, `max_time` as a duration; precisions
  `bf16-true`, `16-true`, `64-true` and Lightning's short forms; strategy strings and
  `DDPStrategy(find_unused_parameters, static_graph, gradient_as_bucket_view, ...)`;
  `torch.compile`d models train compiled with their own options.
- Callbacks: `ModelCheckpoint` with Lightning's arguments (templates, `every_n_train_steps`,
  `train_time_interval`, `save_last`, `save_top_k=-1/0`, `monitor=None`), `EarlyStopping`
  thresholds, `Timer`, `LambdaCallback`, `GradientAccumulationScheduler`,
  `ThroughputMonitor`, `DeviceStatsMonitor`, `SpikeDetection`, `ModelSummary`,
  `ProgressBar`, `BasePredictionWriter`, `OnExceptionCheckpoint`,
  `StochasticWeightAveraging`, `WeightAveraging`, `EMAWeightAveraging`, `BaseFinetuning`,
  `BackboneFinetuning`, `ModelPruning`, `LearningRateFinder`, `BatchSizeFinder`.
- Loggers `CSVLogger`, `TensorBoardLogger`, `WandbLogger`, `MLFlowLogger`; profilers
  `SimpleProfiler`, `AdvancedProfiler`, `PyTorchProfiler`; `Tuner.lr_find` and
  `Tuner.scale_batch_size`.
- Checkpoints use Lightning's layout (`optimizer_states`, `lr_schedulers`, `callbacks`,
  `hyper_parameters`, scaler; `optimizer` is still written) and resume within an epoch from
  step checkpoints. Ctrl+C ends `fit` gracefully after `on_exception` hooks.
- Speed: only overridden hooks are called, `self.log` has a fast path, epoch metrics are
  averaged across ranks in one all-reduce. Step time measured within 1.4% of 0.7.1 on an
  Apple M2 Pro CPU (A/B, `benchmarks/results/2026-11-01-m2pro-parity/`).
- Migration: `ModelCheckpoint`'s first positional argument is now `dirpath`, as in
  Lightning (pass `monitor=` by name). `save_top_k=0` is now valid (keeps none).
  `validate`/`test` print their results (`verbose=False` to silence). Callbacks with the 0.7
  `on_train_batch_end(trainer, model, loss, batch_idx)` signature keep working.
- Not validated on GPUs yet: `sync_batchnorm`, Inductor `torch.compile`, 16-mixed with manual
  optimization. TensorBoard, W&B and MLflow loggers are untested (optional packages).

- Deprecated the `jax-cuda` and `cupy-cuda` runtime profiles (never run on a GPU). They still
  work and warn; they are removed in 0.9.0. Migration: replace `profile="jax-cuda"` with
  `accelerator="cuda", frameworks={"jax": ">=0.4", "jaxlib": ">=0.4"}` (and `cupy>=12` for
  `cupy-cuda`).
- Fix: `gpulayer up --name NAME` printed the hostname as the local worker's name; the banner
  now shows NAME, as `gpulayer worker status` does.
- Docs and site: the home page terminal has a "Server + client" tab, and the quickstart (site,
  docs and README) walks through `gpulayer up`, a joined worker and a client training on two
  nodes, with output from a run on one Mac in three terminals.

## 0.7.1 (2026-10-31)

- Fix: after a server restart behind the same address (a Kubernetes server pod), a worker's
  kept-open connection could go silent and hold a heartbeat for 60 s, past the 20 s lease, so
  its tasks were marked lost. Heartbeats now time out after a quarter of the lease and
  acquires 10 s after their long-poll wait; a test reproduces the silent connection.
- Tagline is now "Train PyTorch on every GPU you can reach: one Trainer, from your laptop to
  many machines." The comparison with PyTorch Lightning appears only in performance sections,
  always as the GPUlayer Trainer.
- Site, README and docs rewritten shorter and around training: quickstart starts with the
  Trainer, the home page drops the positioning section and non-training features, the
  architecture diagrams show training paths only.
- Removed unverified claims: JAX, CuPy, TensorFlow and ONNX Runtime examples and mentions
  (detected on workers, never run on a GPU), the ffmpeg/NVENC illustration, and "simulated"
  cloud workers in the site diagram (Google Cloud VMs are created per run).

## 0.7.0 (2026-10-31)

GPUlayer now has one focus: **train PyTorch on every GPU you can reach, with one Trainer, from
your laptop to many machines, faster than Lightning** (ADR-0023).

- Against PyTorch Lightning, with the same model, data and optimizer, the Trainer was faster in
  10 of 11 GPU benchmarks (Tesla T4; 1, 2 and 4 RTX 3070s; Apple M2 Pro GPU) and in every CPU
  benchmark. New on an Apple M2 Pro: faster in all 5 cases (MLP 1.62x on the CPU and 1.75x on
  the GPU, CNN 1.05x on both, DDP on 2 CPU processes 1.21x).
- AMD ROCm execution path (implemented, not yet run on AMD hardware): ROCm builds of torch are
  recognised (`torch.rocm` capability; AMD GPUs are no longer reported as CUDA devices), jobs
  get their GPUs with `HIP_VISIBLE_DEVICES`, and the Trainer accepts `accelerator="rocm"`
  (RCCL for DDP).
- Zone placement measured with the scheduler choosing: the same 4-machine job took 42.5 s with
  `group_locality="prefer"` (one zone, every run) and 58.8 s with `"any"` (two regions, every
  run), median of five on GCP.
- Site, README, docs, package metadata and repository description lead with PyTorch training;
  the job engine (commands, functions, services) is presented second. "Works with" lists only
  what has been run as GPUlayer jobs.
- Removed: the Blender, ffmpeg and JAX/CuPy examples (never validated on NVIDIA), the Kaggle
  runner and the library-validation script.

## 0.6.0 (2026-10-31)

GPUs first: validated on several GPUs per machine, benchmarked against PyTorch Lightning on
1 to 4 GPUs, with faster start-up, placement by network and GPU interconnect, and per-job
packages.

**GPU results**

- Against Lightning (same model, data and optimizer, each trainer's defaults), the Trainer was
  faster in 8 of 9 GPU cases: 1.14 to 1.39x per step on a Tesla T4, and 1.11 to 1.76x in 5 of 6
  cases on 1, 2 and 4 RTX 3070s (DDP over NCCL); 1.14x slower on the 1-GPU CNN on that host,
  where both run the same GPU kernels. Charts and logs on the performance page.
- Validated on rented multi-GPU machines (Vast.ai): NCCL across 4x RTX 3070 (PCIe) and over
  NVLink on 2x A100 SXM4; the full test suite and every CUDA test passed on a 4x RTX 3070.
  New `scripts/validate_multigpu.py` (run by `validate_gpu.sh` on 2+ GPUs) and
  `benchmarks/cloud/vast.sh`, `aws.sh`, `kaggle.sh`.

**Placement**

- Multi-GPU tasks get the best-connected free GPUs on a machine, read from
  `nvidia-smi topo -m` (`AcceleratorDevice.links`, `numa_node`). On a 4x RTX 3070 server a
  2-GPU job got GPUs 2-3 behind one PCIe switch instead of 0-1 across CPU sockets.
- Distributed groups stay in one network group, else zone, else region (ADR-0021):
  `PlacementPolicy.group_locality` (`--group-locality`), `WorkerCapabilities.zone`, the label
  `gpulayer.dev/network-group`, `--zone`/`--region`/`--network-group` on `up`, `join` and
  `worker start` (read from GCE metadata on Google Cloud). On GCP a 4-machine job split across
  two regions took 56.2 s against 45.0 to 48.6 s in one region.
- Workers report `torch.compile` only when a C compiler is present, so such jobs are not placed
  where they would fail mid-run (rented GPU containers often have none).

**Trainer and start-up**

- Start-up on several GPUs is now faster than Lightning's in most cases: the local worker reads
  torch in-process instead of importing it again (0.69 s to 0.01 s on an M2 Pro), and cuDNN
  autotuning (`benchmark=None`) turns on only for runs of 1,000+ steps (a 5-step 2-GPU run went
  from 14.9 to 10.7 s).
- `fit`, `validate`, `test` and `predict` accept a function that returns the loader, built only
  in the processes that train; data built in the launching process is otherwise thrown away.
- CPU threads per process follow the platform: 1 for CPU training, the cores divided among the
  processes for GPU training.
- `torch.autocast` is built once per run instead of once per step (each build queried the CUDA
  driver).
- `prefetch` (default `False`): a background thread loads and pins batches and copies them on a
  side stream; it helped a CNN by 5% and slowed an MLP by 36% on an RTX 3070, so it is opt-in.
  `Runtime.batches(loader)` exposes it.
- `num_sanity_val_steps` (default 2), `fast_dev_run`, `max_time`,
  `gradient_clip_algorithm="value"` and `deterministic`.

**Jobs**

- Per-job requirements (ADR-0022): `JobSpec.environment` (`requirements`, `requirements_file`),
  `requirements=` in the SDK, `launch()` and the Trainer's `launch_options`, and
  `gpulayer submit --requirement/--requirements-file`. Each worker builds the environment once
  on top of its own Python and reuses it; PyTorch and the CUDA libraries are not installed
  again. Preparation is reported apart from the run (`prepare_seconds`, `prepare_cached`).
- Fix: jobs receive `LD_LIBRARY_PATH` and the other library and CUDA paths. Where the GPU
  driver is found through them (Kaggle), a job's processes saw no GPUs. Proxy and package-index
  URLs are still not passed (they can carry credentials).

**Migration**

- New optional fields on `JobSpec` (`environment`), `PlacementPolicy` (`group_locality`),
  `WorkerCapabilities` (`zone`), `AcceleratorDevice` (`links`, `numa_node`) and `TaskOutcome`
  (`prepare_seconds`, `prepare_cached`); older control planes and workers ignore them.
- The GCP adapter reports `region="europe-west1"` and `zone="europe-west1-b"` (it used to put
  the zone in `region`); `placement.regions` matches either.
- The Trainer's `prefetch` defaults to `False`.

**Other**

- Lightning comparison, DDP on 2 CPU processes rerun on an `n1-standard-8`: 3.20 against 9.56
  ms per step.
- Examples run on an Apple M2 Pro (Blender fails the job when the GPU device is missing, ffmpeg
  uses VideoToolbox on a Mac, the inference service uses MPS).

## 0.5.0 (2026-10-30)

- Two-level gradient sync (ADR-0020): with gloo and several processes per machine, the
  Trainer sums gradients inside each machine before all-reducing across machines
  (`Trainer(gradient_sync="auto" | "flat" | "two_level")`). On 8 GCP machines x 4 processes a
  DDP step went from 117 to 102 ms, and the strong-scaling job ran 11.9% faster than
  Lightning (50.5 s vs 57.3 s). NCCL jobs are unchanged.
- Workers report `jax.cuda` and `cupy.cuda` only when JAX or CuPy actually ran on the GPU (a
  probe in its own interpreter, run only when they are installed), as `torch.cuda` already
  was. Before, an installed CPU-only `jax` was enough for a `--accelerator cuda --framework
  jax` job to be placed there.
- Google Cloud GPU provisioning validated: a `--provider gcp --accelerator cuda` job made
  GPUlayer create a Spot T4 VM, which joined with its GPU and `torch.cuda` 78 s after boot
  started, ran a `Trainer` job on CUDA, and was deleted when it ended.
- Examples for remote NVIDIA GPUs: Blender Cycles rendering, ffmpeg NVENC transcoding, JAX and
  CuPy, and a PyTorch inference service. Not yet run on a GPU.

## 0.4.0 (2026-10-30)

- Long-running services (ADR-0019): `gpulayer submit --service`, `gl.service(...)` and
  `Job.endpoints()`. A service runs `replicas` copies on distinct workers, gives each a free
  `$PORT`, reports `host:port` once it accepts connections, restarts replicas that exit (back-off
  1 s to 30 s, `--restart`, `--max-restarts`), is placed again after a worker loss, and stops on
  cancel. New worker route `POST /v1/worker/tasks/{id}/endpoint` and `TaskStatus.endpoint`;
  the `tasks.endpoint` column is added to existing databases on start. No load balancer yet.
- Less server work per job: active-task queries use a partial index instead of scanning every
  task ever run (2.2 ms to 0.002 ms per scheduling pass at 13,000 tasks), the HTTP client
  shares kept-open connections between threads (a worker no longer opens a TLS connection per
  job), and worker capabilities are parsed once. Server CPU per job in
  `benchmarks/server_throughput.py`: 6.7-6.8 ms to 5.2-6.0 ms.
- `POST /v1/jobs/batch` and `Client.submit_batch`: many jobs in one request, one transaction
  and one scheduling pass (all or none); server CPU per job 5.6-6.2 to 4.1 ms.
- Site: sitemap.xml, robots.txt, canonical URLs, link previews and structured data.
- `Trainer(strategy="fsdp")` (experimental): FSDP2 sharding of parameters, gradients and
  optimizer state, full-state checkpoints; on CPU it trains exactly like DDP. Not yet run on
  CUDA GPUs.
- Multi-node training against PyTorch Lightning, both trainers on the same machines: the same
  speed within 2% on 1 to 8 machines (40.5 s vs 41.2 s on 8). The earlier comparison, which
  ran the trainers on different VMs, overstated GPUlayer's lead.
- On 8 GCP workers after the server changes, 2,000 tiny jobs ran at 83 jobs/s, or 93 with
  `submit_batch`, which submits all of them in 1.1 s. The workers now set the limit.
- Google Cloud provider adapter (`gpulayer[gcp]`, `gpulayer up --providers FILE`): when no worker
  fits a `--provider gcp` job, GPUlayer creates a Compute Engine VM from a configured offer, the
  VM installs GPUlayer and joins with a single-use token, the job runs, and the VM is deleted.
  Validated end to end on GCP with a CPU VM. `gpulayer join --instance-ref` for provisioned
  machines.
- Kubernetes: a Helm chart (`deploy/kubernetes/helm/gpulayer`) with the server as a
  StatefulSet and CPU and GPU worker pools; tokens and TLS generated into Secrets and kept on
  upgrade. Tested in CI on kind, including a distributed job across worker pods. Guide:
  Kubernetes.
- Releases push container images to `ghcr.io/antonellof/gpulayer` (`<version>` and
  `<version>-torch`).
- `gpulayer up` takes its tokens from `GPULAYER_UP_CLIENT_TOKEN` / `GPULAYER_UP_JOIN_TOKEN`;
  `--ca-file` defaults to `$GPULAYER_CA_FILE`; `gpulayer join` waits for a server that is not
  reachable yet instead of exiting.
- `fit(ckpt_path=...)` resumes launched (remote) runs: the local checkpoint is uploaded with
  the job. `launch(inputs={...})` ships local files to `$GPULAYER_INPUT_DIR`.

## 0.3.0 (2026-10-30)

First release on PyPI (`pip install gpulayer`), under the Apache-2.0 licence. Tested on 30
machines in two GCP regions, with a multi-machine training comparison against PyTorch
Lightning.

### Scale and speed

- Scheduling no longer slows down as the queue grows: a submit costs 0.5 ms at 2,000 queued
  jobs instead of 44 ms (`benchmarks/submit_queue.py`).
- Behind `gpulayer up`, idle workers wait until something changes instead of re-checking
  every 0.2 s, and a change wakes only the workers or job waiters it concerns. On 30 workers
  a 480-run training sweep finished in 45 s instead of 102 s.
- SQLite runs with `synchronous=NORMAL` (WAL): commits no longer wait for the disk (2.25 ms
  to 0.02 ms on a GCP standard disk). Consistent after a crash; a power loss can roll back
  the last commits.
- The Trainer does less work per step (0.60 to 0.585 ms on a small CPU model; a plain
  PyTorch loop is 0.55).
- Validated on 30 workers in two regions: distributed jobs on up to 16 machines (32
  processes), fail-fast across 8 machines in 4.9 s, retries after killing workers mid-run.
  One training job on 8 machines ran 3.4x faster than on one, and 7% sooner than PyTorch
  Lightning under torchrun (benchmarks/results/2026-10-30-gcp-fleet).
- Known limit: bursts of very short jobs reach about 25 jobs/s, bounded by the server using
  one CPU core (ADR-0016, item 6).

### Features and fixes

- `--gpus 0,1|none` on `gpulayer up`, `join` and `worker start`: a worker offers only those
  GPUs, so several workers on one machine no longer offer the same GPU (the known issue in
  0.2.0). `--gpus none` makes a CPU-only worker on a GPU machine.
- CUDA jobs get `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `CUDA_VISIBLE_DEVICES` selects the GPUs
  that were reserved (`nvidia-smi` numbering) on machines with mixed GPU models.
- Trainer: `max_steps`, `check_val_every_n_epoch`, `trainer.test()` with `Module.test_step`
  and `trainer.predict()` with `Module.predict_step` (ADR-0018).
- `ModelCheckpoint(save_top_k=N)` keeps the N best epochs (also after remote runs), and a
  `LearningRateMonitor` callback records each epoch's learning rate.
- `val_check_interval` (batches or a fraction of the epoch) and `metrics.csv` next to
  `metrics.json`, with step, mid-epoch validation and epoch rows.
- A launched training run returns its metrics even with checkpointing off (`metrics.json` is
  written every epoch).
- Apache-2.0 licence (ADR-0012), PyPI package metadata, and a trusted-publishing release
  workflow (`.github/workflows/publish.yml`).
- Benchmarks: `benchmarks/cloud/gcp.sh` builds an image and starts a server plus a fleet of
  workers; `benchmarks/scale_test.py`; `benchmarks/trainer_comparison/scaling_run.py` and
  `multinode.py` compare multi-machine training with PyTorch Lightning.

## 0.2.0 (2026-10-30)

Validated on an NVIDIA GPU and across machines, faster distributed start-up and HTTP API, and
a measured comparison with PyTorch Lightning.

### Migration from 0.1.0

- Distributed jobs default to `launcher="direct"`. Upgrade the server and the workers
  together: 0.1.0 workers are not chosen for `direct` jobs, and a 0.1.0 server rejects them.
  Pass `launcher="torchrun"` (`--launcher torchrun`) to keep the old behaviour.

### Changes

- `gpulayer mcp`: a Model Context Protocol server on stdio, so AI and coding agents can list
  resources, submit commands (with a project directory and inputs), wait, read logs, cancel
  and download outputs; `--read-only` for monitoring. Tested with the official MCP Python SDK.
- Agent Skill for the CLI in `integrations/skills/gpulayer`.
- Guide: AI and coding agents.
- Trainer callbacks: `Callback` hooks, `EarlyStopping` and `ModelCheckpoint` (best checkpoint,
  also returned from remote runs); when one DDP rank asks to stop, every rank stops after the
  same epoch.
- Faster distributed jobs: `DistributedJob.launcher="direct"` is the new default. Workers
  start the ranks themselves, with the same environment torchrun sets, instead of through
  torchrun's elastic agent: a 2 x 2 job on one Mac takes 0.93 s instead of about 2.1 s.
  `launcher="torchrun"` (`--launcher torchrun`) keeps the old behaviour.
  Migration: workers and the server must both be upgraded to run `direct` jobs. v0.1.0
  workers report only `torchrun` and are not chosen for them; a v0.1.0 server rejects the
  new value, so submit with `launcher="torchrun"` there.
- Faster HTTP API: clients and workers keep their connection open, and the server sets
  `TCP_NODELAY` and runs TLS handshakes per connection. On Linux a request dropped from 43 ms
  to 0.6 ms, and 500 small jobs over two machines ran 7.1x faster (7.8 to 55.6 jobs/s).
- Validated on NVIDIA: a Tesla T4 on GCP (CUDA detection, CUDA jobs, Trainer in fp32, bf16,
  fp16 and with `torch.compile`, GPU reservation), and on two machines over a real network
  (distributed jobs across both, fail-fast, hundreds of jobs). `scripts/validate_gpu.sh`
  repeats the single-machine part on any NVIDIA machine.
- Performance and comparison: a docs page and README section with charts
  (`scripts/make_perf_charts.py`), the GPUlayer vs PyTorch Lightning harness
  (`benchmarks/trainer_comparison/`), `benchmarks/multi_machine.py`, raw results in
  `benchmarks/results/2026-10-30-gcp/`, and `benchmarks/cloud/gcp.sh` to rerun them on Google
  Cloud.
- NVIDIA CUDA independent jobs are now `hardware-validated` in the support matrix.

## 0.1.0 (2026-10-29)

The first release. GPUlayer runs commands, Python functions and distributed jobs on the
machines you can reach, from a laptop to servers that joined a control plane.

### Run work

- Job types: commands in any language, Python functions (`@gl.remote`, `.map()`), and
  distributed jobs launched with torchrun (static rendezvous, gang scheduling, fail-fast).
- PyTorch integration: `Module`, `Trainer` (in-process, multi-process DDP, multi-node,
  remote), `Runtime` and `launch()`; gradient accumulation with DDP `no_sync`.
- Runtime profiles (`pytorch-cuda`, `pytorch-mps`, `jax-cuda`, `cupy-cuda`, `mlx`, ...).

### Place and run it well

- Capability-aware scheduler with per-worker reasons when a job cannot run.
- Several tasks per worker, each with its own reserved GPUs (default one slot per NVIDIA or
  AMD GPU); `spread` placement keeps machines evenly busy.
- Event-driven dispatch: long-poll worker acquire and job waits, background log shipping.
  One job on an idle worker: 626 ms to 31 ms; burst throughput 27 to 87 jobs/s (one Mac).
- Per-rank log streams for distributed jobs.

### Servers and workers

- `gpulayer up` (control plane with TLS and a pinned self-signed certificate) and
  `gpulayer join`; named, expiring, revocable tokens with roles (admin, submitter, join,
  metrics); rate-limited logins; worker artifact scoping; retention and `gpulayer gc`.

### Observe

- Web dashboard: jobs, live logs, placement reasons, workers, and activity charts (jobs,
  CPU load, memory, task processes, NVIDIA GPU utilisation).
- Prometheus `/metrics`, a Grafana dashboard and a Prometheus/Grafana Compose setup.

### Install

- `install.sh` (curl one-liner, private virtualenv, no sudo), a Docker image for server and
  workers (CPU or CUDA base) with Compose, and CI that tests all of it.

### Validation status

- Tested on CPU and Apple Silicon (PyTorch MPS and MLX), with several workers on one host,
  and in Linux CI including a Docker server/worker test.
- NVIDIA CUDA execution and multi-machine runs are implemented but not validated on hardware.
- Cloud provisioning is simulated (`FakeProvider`); cloud VMs can join with `gpulayer join`.
- Not a sandbox for untrusted code. See docs/project/status.md.
