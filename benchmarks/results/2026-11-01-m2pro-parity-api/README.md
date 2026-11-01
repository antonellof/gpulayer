# Hot-path check: Lightning public API parity (2026-11-01)

The change that completed Lightning's public API (ADR-0024, "Full public API") touched the
training loop: a SIGTERM flag check after every batch, a precision-plugin check in
`Runtime.autocast()`, the batch count computed before `on_train_epoch_start`, and no
`model.train()` per epoch. This compares GPUlayer's step time before (commit `23036b1`) and
after, nothing else running.

- Hardware: Apple M2 Pro (arm64), macOS; torch 2.15.0, Python 3.12.
- Method: `benchmarks/trainer_comparison/bench.py gl CASE STEPS` from each tree, 100 and
  600 steps, five runs each, before and after alternating; ms per step is the difference of
  the median `fit` times divided by 500 (`hotpath.py`).

| Case | Before | After |
|---|---|---|
| `mlp-cpu` (small MLP, batch 64, CPU) | 0.631 ms | 0.625 ms |
| `mlp-mps` (MLP, batch 256, Apple GPU) | 3.920 ms | 3.927 ms |

No measurable change (differences of 1% and 0.2%, inside run-to-run noise).

Raw `fit` seconds per run (100 steps; 600 steps):

- mlp-cpu before: 0.659, 0.56, 0.556, 0.553, 0.54; 0.871, 0.845, 0.868, 0.882, 0.875
- mlp-cpu after: 0.539, 0.546, 0.543, 0.554, 0.549; 0.858, 0.872, 0.84, 0.867, 0.841
- mlp-mps before: 1.05, 1.05, 1.044, 1.048, 1.073; 3.009, 3.017, 3.01, 3.003, 3.027
- mlp-mps after: 1.042, 1.045, 1.039, 1.049, 1.041; 3.024, 3.005, 2.998, 3.012, 3.002
