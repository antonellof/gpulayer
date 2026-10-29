# ADR-0014: PyTorch integration and a training API

- Status: Accepted (milestone 2, first slice)
- Date: 2026-10-29

## Context

GPUlayer is workload-agnostic (ADR-0015); this ADR covers its first framework integration.

Milestone 1 made GPUlayer able to place and run jobs, but a PyTorch user still had to split
code into a training script and a submission script, and had to handle devices, DDP wrapping,
data sharding and checkpoints by hand. That offered too little over plain PyTorch.

Training frameworks commonly hide the device and the parallel strategy behind a module with
step hooks and a trainer object. They work inside the machine or cluster they are started on.
They do not choose which machine, cluster or cloud runs the job, and they do not provision,
queue, or collect results across machines. GPUlayer does those things, so its PyTorch API
should cover both: how a model trains and where it trains.

## Decision

Integrate with PyTorch in three layers, each usable on its own, all in `gpulayer.pytorch`.
Torch is imported lazily, so installs without torch keep working. Jobs submitted by this layer
require the `torch` framework and, for GPUs, the `torch.cuda` or `torch.mps` capability
(ADR-0015), so they only land on workers whose torch can drive the requested GPU.

1. Launcher (framework agnostic). `gpulayer.pytorch.launch(nodes=, devices=, accelerator=,
   target=, ...)` goes at the top of any script. Outside a GPUlayer job it packages the
   script's directory and submits the same script with the same arguments: a `CommandJob` for
   one process, otherwise a torchrun `DistributedJob`. It then streams logs, downloads outputs
   and exits with the job's exit code. Inside the job it returns immediately and the script
   continues. Any code that reads the standard torchrun environment (`RANK`, `WORLD_SIZE`,
   `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`) works with it, including other training
   frameworks.
2. Runtime helpers. `Runtime` resolves `accelerator="auto"` (the device the job was granted,
   else cuda, then mps, then cpu). It initialises the process group when `WORLD_SIZE > 1`,
   moves modules to the device and wraps them in DDP, rebuilds dataloaders with a
   `DistributedSampler`, applies a precision policy, and exposes rank helpers and an output
   directory whose files come back as artifacts.
3. Trainer. `gpulayer.pytorch.Module` provides `training_step`, `validation_step`,
   `configure_optimizers` and `self.log`. `gpulayer.pytorch.Trainer(max_epochs, accelerator,
   devices, num_nodes, precision, target, ...)` runs `trainer.fit(model, train_loader,
   val_loader)` in-process when it can. Otherwise it uses the launcher: the script is
   re-executed on the chosen workers, where the same `fit` call runs the loop. When the remote
   run finishes, the launching process loads the final weights into its local `model` object,
   so code after `fit` behaves the same in both cases.

Re-executing the script per process is how torchrun-based multi-process training already
works, and it keeps ADR-0002: code travels as source and nothing is pickled.

## Alternatives considered

- Make an existing third-party training framework a dependency and ship only a plugin for it.
  This is the least code, but it makes that framework mandatory and ties GPUlayer to another
  project's release cycle. Interop stays available through the launcher instead.
- Ship models, dataloaders and closures to workers with pickle or cloudpickle. Convenient, but
  it breaks across Python and torch versions and turns deserialization into remote code
  execution on workers. Rejected by ADR-0002.
- Build a large trainer with callbacks, logger integrations, many strategies and dozens of
  flags. That is a big surface to maintain. The GPUlayer Trainer stays small; users who need
  more can run their framework of choice through the launcher.

## Consequences

- One training file runs unchanged on CPU, MPS or CUDA, locally or remotely, on one device or
  across nodes. Only the `Trainer` arguments change.
- Module-level code in the script runs on every rank and also in the launching process, as
  with any torchrun script. Expensive setup belongs after `launch()` or inside `fit`.
  `gpulayer.pytorch.is_launcher_process()` tells the two apart.
- Precision is explicit per backend: `bf16-mixed` on CUDA and CPU, and `16-mixed` on CUDA with
  a gradient scaler. Unsupported combinations, such as `16-mixed` on MPS or CPU, raise an
  error instead of silently training in another precision.
- MPS stays single-process (ADR-0006), so `devices > 1` or `num_nodes > 1` with MPS is
  rejected.

## Security and operational implications

No new trust boundaries. The launcher uses the existing submission API, source bundles and
artifact transfer. Re-executing a script on a worker runs arbitrary code there, exactly as
any `CommandJob` does (ADR-0009).

## Follow-up

- Interop tests with third-party training frameworks, skipped when they are not installed.
- Callbacks, early stopping, gradient accumulation and learning-rate monitoring.
- FSDP and other strategies after DDP is hardware-validated on CUDA.
- Resuming from a checkpoint artifact after worker loss, with explicit user opt-in.
