# GPUlayer Trainer step time before and after Lightning feature parity (ADR-0024)

Apple M2 Pro (10 CPU cores, 32 GB), macOS, Python 3.12, torch 2.15.0, 2026-11-01.

Method: `benchmarks/lightning_comparison/bench.py gl <case>`, fresh process per run, wall time
from outside the process. ms/step = (median wall(2200 steps) - median wall(200 steps)) / 2000.

## A/B, alternating runs (ab.py, 5 repeats each)

The "old" side imports 0.7.1 from a git worktree of `2a3943b` via `PYTHONPATH`; the "new" side
the working tree with ADR-0024's changes.

| Case | 0.7.1 | with parity features |
|---|---|---|
| MLP, CPU (`mlp-cpu`) | 0.589 ms | 0.597 ms (+1.4%) |

## run.py (TRAINERS=gl), before and after, not interleaved

| Case | before | after |
|---|---|---|
| `mlp-cpu` | 0.55 ms | 0.67 ms (first run after the rewrite; the A/B above shows the drift was not the code) |
| `mlp-mps` | 3.89 ms | 3.85 ms |
| `ddp-cpu-2` | 2.09 ms | 2.01 ms |

Changes on the per-step path: hooks resolved once per run (only overridden ones called), a
fast path in `Module.log` for the default call, scheduler and optimizer-hook checks hoisted
out of the step, and one all-reduce for all epoch metrics instead of one per name.
