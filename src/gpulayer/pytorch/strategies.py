"""Training strategies: how the model is replicated or sharded across processes.

    Trainer(strategy="ddp")                                   # the default
    Trainer(strategy="ddp_find_unused_parameters_true")       # Lightning's string form
    Trainer(strategy=DDPStrategy(static_graph=True, gradient_as_bucket_view=True))
    Trainer(strategy="fsdp")                                  # experimental

The names and options follow PyTorch Lightning. Extra ``DDPStrategy`` keyword arguments go to
``torch.nn.parallel.DistributedDataParallel`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gpulayer.errors import ValidationFailed


@dataclass
class DDPStrategy:
    """DistributedDataParallel with its options.

    - ``find_unused_parameters``: needed when some parameters get no gradient in a step
      (costs a graph traversal per step, so off by default).
    - ``static_graph``: the set of used parameters never changes; DDP can then skip work
      and overlap more communication.
    - ``gradient_as_bucket_view``: gradients are views into DDP's communication buckets,
      which saves one copy of the gradients in memory and a copy per step.
    """

    find_unused_parameters: bool = False
    static_graph: bool = False
    gradient_as_bucket_view: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        find_unused_parameters: bool = False,
        static_graph: bool = False,
        gradient_as_bucket_view: bool = False,
        **kwargs: Any,
    ) -> None:
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.kwargs = kwargs

    name = "ddp"

    def ddp_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to ``DistributedDataParallel``."""
        out = dict(self.kwargs)
        if self.find_unused_parameters:
            out["find_unused_parameters"] = True
        if self.static_graph:
            out["static_graph"] = True
        if self.gradient_as_bucket_view:
            out["gradient_as_bucket_view"] = True
        return out


@dataclass
class FSDPStrategy:
    """Fully sharded data parallel (FSDP2), experimental: see the Trainer docs."""

    name = "fsdp"

    def ddp_kwargs(self) -> dict[str, Any]:
        """Always empty: FSDP does not use DistributedDataParallel."""
        return {}


@dataclass
class DeepSpeedStrategy:
    """DeepSpeed ZeRO (needs the ``deepspeed`` package and CUDA GPUs; implemented, not yet
    validated on hardware). ``stage`` 1 shards optimizer state, 2 also gradients, 3 also
    parameters; ``offload_optimizer``/``offload_parameters`` move them to CPU memory.
    ``config`` (a dict or a JSON file) replaces the generated configuration. Checkpoints hold
    full weights (gathered from all ranks with stage 3); resuming a DeepSpeed run from a
    checkpoint is not supported yet."""

    stage: int = 2
    offload_optimizer: bool = False
    offload_parameters: bool = False
    config: Any = None
    name = "deepspeed"

    def __post_init__(self) -> None:
        if self.stage not in (0, 1, 2, 3):
            raise ValidationFailed("DeepSpeedStrategy stage must be 0, 1, 2 or 3")

    def ddp_kwargs(self) -> dict[str, Any]:
        """Always empty: DeepSpeed wraps the model itself."""
        return {}

    def deepspeed_config(
        self, precision: str, accumulate: int, clip: float | None, micro_batch: int | None
    ) -> dict[str, Any]:
        """The DeepSpeed configuration for this Trainer's settings."""
        import json
        from pathlib import Path

        if isinstance(self.config, (str, Path)):
            return dict(json.loads(Path(self.config).read_text()))
        if isinstance(self.config, dict):
            return dict(self.config)
        zero: dict[str, Any] = {"stage": self.stage}
        if self.offload_optimizer:
            zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if self.offload_parameters:
            zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        cfg: dict[str, Any] = {
            "zero_optimization": zero,
            "gradient_accumulation_steps": accumulate,
            "train_micro_batch_size_per_gpu": micro_batch or 1,
            "zero_allow_untested_optimizer": True,
        }
        if clip:
            cfg["gradient_clipping"] = clip
        if precision in ("16-mixed", "16-true"):
            cfg["fp16"] = {"enabled": True}
        elif precision in ("bf16-mixed", "bf16-true"):
            cfg["bf16"] = {"enabled": True}
        return cfg


_STRINGS: dict[str, Any] = {
    "auto": lambda: DDPStrategy(),
    "ddp": lambda: DDPStrategy(),
    "ddp_find_unused_parameters_false": lambda: DDPStrategy(find_unused_parameters=False),
    "ddp_find_unused_parameters_true": lambda: DDPStrategy(find_unused_parameters=True),
    "fsdp": lambda: FSDPStrategy(),
    "deepspeed": lambda: DeepSpeedStrategy(),
    "deepspeed_stage_1": lambda: DeepSpeedStrategy(stage=1),
    "deepspeed_stage_2": lambda: DeepSpeedStrategy(stage=2),
    "deepspeed_stage_2_offload": lambda: DeepSpeedStrategy(stage=2, offload_optimizer=True),
    "deepspeed_stage_3": lambda: DeepSpeedStrategy(stage=3),
    "deepspeed_stage_3_offload": lambda: DeepSpeedStrategy(
        stage=3, offload_optimizer=True, offload_parameters=True
    ),
}


Strategy = DDPStrategy | FSDPStrategy | DeepSpeedStrategy


def resolve_strategy(strategy: Any) -> Any:
    if isinstance(strategy, (DDPStrategy, FSDPStrategy, DeepSpeedStrategy)):
        return strategy
    if strategy in _STRINGS:
        return _STRINGS[strategy]()
    if isinstance(strategy, str) and strategy.startswith(("ddp_spawn", "ddp_fork", "ddp_notebook")):
        raise ValidationFailed(
            f"strategy {strategy!r}: GPUlayer starts one process per rank itself, so spawn, "
            "fork and notebook launchers are not needed; use 'ddp'"
        )
    raise ValidationFailed(
        f"strategy must be one of {sorted(_STRINGS)}, or a DDPStrategy, FSDPStrategy or "
        "DeepSpeedStrategy, "
        f"got {strategy!r}"
    )
