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


_STRINGS: dict[str, Any] = {
    "auto": lambda: DDPStrategy(),
    "ddp": lambda: DDPStrategy(),
    "ddp_find_unused_parameters_false": lambda: DDPStrategy(find_unused_parameters=False),
    "ddp_find_unused_parameters_true": lambda: DDPStrategy(find_unused_parameters=True),
    "fsdp": lambda: FSDPStrategy(),
}


def resolve_strategy(strategy: str | DDPStrategy | FSDPStrategy) -> DDPStrategy | FSDPStrategy:
    if isinstance(strategy, (DDPStrategy, FSDPStrategy)):
        return strategy
    if strategy in _STRINGS:
        return _STRINGS[strategy]()
    if isinstance(strategy, str) and strategy.startswith(("ddp_spawn", "ddp_fork", "ddp_notebook")):
        raise ValidationFailed(
            f"strategy {strategy!r}: GPUlayer starts one process per rank itself, so spawn, "
            "fork and notebook launchers are not needed; use 'ddp'"
        )
    raise ValidationFailed(
        f"strategy must be one of {sorted(_STRINGS)}, a DDPStrategy or an FSDPStrategy, "
        f"got {strategy!r}"
    )
