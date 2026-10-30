# ADR-0018: Trainer scope relative to PyTorch Lightning

- **Status:** Accepted (milestone 2)
- **Date:** 2026-10-30

## Context

GPUlayer's `Trainer` (ADR-0014) exists so that a PyTorch training script can choose where it
runs: in this process, as DDP on one machine, or on several workers. Users compare it with
PyTorch Lightning, the most common PyTorch trainer. On 2026-10-30 a measured comparison on a
Tesla T4 (`benchmarks/lightning_comparison/`) found GPUlayer's loop 1.3 to 1.4x faster per
step on small models, equal on a compute-bound CNN, and quicker to start
([Performance and comparison](../get-started/performance-and-comparison.md)). Lightning has
far more features. We need a rule for which ones the Trainer adopts, so that it stays small
and fast without making users leave it for common needs.

## Decision

1. **Adopt Lightning's names and semantics where we adopt a feature.** A user moving from
   Lightning should recognise `max_epochs`, `max_steps`, `precision="bf16-mixed"`,
   `accumulate_grad_batches`, `check_val_every_n_epoch`, `fit/validate/test/predict`,
   `training_step/validation_step/test_step/predict_step`, `self.log`, `Callback`,
   `EarlyStopping` and `ModelCheckpoint`. Behaviour may be simpler, never different under the
   same name.
2. **Adopt what most training scripts need**, in this order:
   - done: the four loops, step limits, validation interval, gradient clipping and
     accumulation, mixed precision, schedulers per step or epoch, callbacks, early stopping,
     the best N checkpoints (`save_top_k`), a learning-rate monitor, resume from a checkpoint;
   - planned: a metrics file per step (CSV), validation within an epoch
     (`val_check_interval`);
   - planned, needs hardware first: FSDP for models that do not fit one GPU, validated on
     several CUDA GPUs before it is documented.
3. **Do not adopt** logger integrations (TensorBoard, W&B, MLflow), DeepSpeed, model
   summaries, progress bars beyond the printed log line, tuners, or plugins. Users who need
   them keep their Lightning or custom script and run it with `launch()`, which gives it the
   same placement and the standard torchrun environment.
4. **The per-step path stays lean.** Features must not add work to every step by default:
   no host-device synchronisation (`.item()`, `.cpu()`), no per-step collectives, no
   per-step file writes. Anything per step (for example the planned CSV metrics) is opt-in
   and buffered. Changes to the loop come with a run of the comparison harness, and a
   regression against the published numbers blocks the change.
5. **Distributed correctness first.** A feature that ends or skips work (early stopping,
   `max_steps`, validation intervals) must make every rank take the same branch, or combine
   the decision with a collective (`Runtime.any`), so DDP never hangs.

## Consequences

- The Trainer covers common training scripts; projects that rely on Lightning's ecosystem
  keep it and gain GPUlayer's placement through `launch()`.
- Each adopted feature needs a CPU test, and a distributed test when it changes control flow.
- The comparison harness is part of the definition of done for loop changes.
- The planned items above are not available until listed in `docs/pytorch/trainer.md`.
