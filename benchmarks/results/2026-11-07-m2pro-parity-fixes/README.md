# Trainer step time with the 0.10 Lightning fixes

Apple M2 Pro (10 CPU cores, 32 GB), macOS, Python 3.12.13, torch 2.15.0, 2026-11-07.

What changed on the per-step path: `self.log` options and checks (the plain call stays on its
fast path), gradients freed after `on_train_batch_end` instead of inside the optimizer step, a
check for `self.log`'s hook phase when batch-end hooks exist, and the `dataloader_iter` branch
of the epoch. Once per epoch: reading the random generators' state, and the epoch-end results.

Method: `benchmarks/results/2026-11-01-m2pro-parity/ab.py` (alternating fresh processes, 5
repeats each; ms/step = (median wall(2200 steps) - median wall(200 steps)) / 2000). "old" is a
git worktree of `b860457` (0.9.2) on `PYTHONPATH`; "new" is `main` with every item merged.

| Run | old (0.9.2) | new |
|---|---|---|
| 1 | 0.633 ms | 0.631 ms (-0.3%) |
| 2 | 0.627 ms | 0.629 ms (+0.3%) |
| 3 | 0.629 ms | 0.626 ms (-0.5%) |
| 4 | 0.636 ms | 0.625 ms (-1.7%) |

No measurable change. CPU, one process, `mlp-cpu`. Not measured on a GPU or with several
processes. Automatic resume adds nothing to a step: the worker uploads on its own thread, and
the Trainer prints one line per checkpoint. The time an upload takes is not measured here.
