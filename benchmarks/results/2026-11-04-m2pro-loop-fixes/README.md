# Trainer step time with the loop-semantics fixes

Apple M2 Pro (10 CPU cores, 32 GB), macOS, Python 3.12.13, torch 2.15.0, 2026-11-04.

What changed on the per-step path: the scheduler step moved after the step count, the
validation schedule check, a batch-size lookup per step for weighting epoch means
(shapes only, no device synchronisation), and, only for loaders without a length, reading
one batch ahead.

Method: `benchmarks/results/2026-11-01-m2pro-parity/ab.py` (alternating fresh processes, 5
repeats each; ms/step = (median wall(2200 steps) - median wall(200 steps)) / 2000). "old" is
a git worktree of `5915ee4` (0.9.1 plus site changes) on `PYTHONPATH`; "new" the working tree.

| Run | old | new |
|---|---|---|
| 1 | 0.649 ms | 0.614 ms (-5.4%) |
| 2 | 0.631 ms | 0.633 ms (+0.3%) |
| 3 | 0.619 ms | 0.599 ms (-3.2%) |
| 4 | 0.596 ms | 0.587 ms (-1.5%) |

No measurable change. A first version that wrapped every loader in a look-ahead generator
measured +2.6% to +16% in four runs (a fifth, 3.363 ms old against 0.404 ms new, was an
outlier) and was replaced by the version measured
above, which reads ahead only when the loader has no length.

CPU, one process. Not measured on a GPU or with several processes; the cross-rank changes
run once per epoch (and per validation), not per step.
