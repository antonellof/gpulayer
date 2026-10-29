"""PyTorch integration (requires ``torch``). See ADR-0014.

Three layers, each usable alone:

* ``launch()``  -- run any PyTorch script, whatever framework it uses, on GPUlayer workers
* ``Runtime``   -- device, DDP, data sharding and precision helpers
* ``Module`` + ``Trainer`` -- a small training API that also picks where training runs
"""

from gpulayer.pytorch.launch import LaunchResult, in_gpulayer_job, is_launcher_process, launch
from gpulayer.pytorch.module import Module
from gpulayer.pytorch.runtime import Runtime, resolve_accelerator, seed_everything
from gpulayer.pytorch.trainer import Trainer

__all__ = [
    "LaunchResult",
    "Module",
    "Runtime",
    "Trainer",
    "in_gpulayer_job",
    "is_launcher_process",
    "launch",
    "resolve_accelerator",
    "seed_everything",
]
