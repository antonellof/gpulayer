"""Lightning's ``lightning.pytorch.utilities`` helpers, with the same names and behaviour.

from gpulayer.pytorch.utilities import grad_norm, rank_zero_only, measure_flops
"""

from __future__ import annotations

import functools
import os
import pickle
import time
import warnings
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

from gpulayer.pytorch.combined import CombinedLoader
from gpulayer.pytorch.module import AttributeDict

T = TypeVar("T")


# ------------------------------------------------------------------ rank zero


def _rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def rank_zero_only(fn: Callable[..., T]) -> Callable[..., T | None]:
    """Decorator: call ``fn`` only in the process with global rank 0 (others return None)."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> T | None:
        if getattr(rank_zero_only, "rank", _rank()) == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped


@rank_zero_only
def rank_zero_info(*args: Any, **kwargs: Any) -> None:
    """``print`` from global rank 0 only."""
    print(*args, **kwargs, flush=True)


@rank_zero_only
def rank_zero_warn(
    message: str | Warning, category: type[Warning] = UserWarning, **kw: Any
) -> None:
    """``warnings.warn`` from global rank 0 only."""
    warnings.warn(message, category, stacklevel=kw.pop("stacklevel", 3), **kw)


def rank_zero_deprecation(message: str | Warning, **kwargs: Any) -> None:
    """A ``DeprecationWarning`` from global rank 0 only."""
    rank_zero_warn(message, DeprecationWarning, **kwargs)


def disable_possible_user_warnings(module: str = "") -> None:
    """Silence the warnings GPUlayer raises about possible mistakes (as Lightning's
    ``PossibleUserWarning``), for ``module`` or everywhere."""
    warnings.filterwarnings("ignore", category=UserWarning, module=module or "gpulayer")


# ------------------------------------------------------------------ data


def move_data_to_device(batch: Any, device: torch.device | str | int) -> Any:
    """Move every tensor in ``batch`` (nested lists, tuples, dicts, dataclasses' tensors are
    followed) to ``device``; CPU tensors bound for CUDA are copied without blocking."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    non_blocking = device.type == "cuda"
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return type(batch)((k, move_data_to_device(v, device)) for k, v in batch.items())
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return type(batch)(*(move_data_to_device(v, device) for v in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_data_to_device(v, device) for v in batch)
    if hasattr(batch, "to") and callable(batch.to):
        return batch.to(device)
    return batch


def suggested_max_num_workers(local_world_size: int) -> int:
    """An upper bound for ``DataLoader(num_workers=...)``: the CPU cores this process may use,
    shared by the processes on this machine, minus one for the training process."""
    if local_world_size < 1:
        raise ValueError(f"local_world_size should be >= 1, got {local_world_size}")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, (cpus or 1) // local_world_size - 1)


def is_picklable(obj: object) -> bool:
    """True when ``obj`` can be pickled."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
        return False


def is_shared_filesystem(
    strategy: Any = None, path: str | Path | None = None, timeout: int = 3
) -> bool:
    """True when ``path`` (the working directory by default) is the same filesystem in every
    process of the run: rank 0 writes a file and the others look for it. ``strategy`` is the
    Trainer (or its ``Runtime``); call it on every rank."""
    if path is not None and "://" in str(path) and not str(path).startswith("file://"):
        return True  # object stores (s3://, gs://, ...) are shared
    rt = getattr(strategy, "runtime", strategy)
    p = Path(Path.cwd() if path is None else path).resolve()
    if rt is None or not getattr(rt, "distributed", False):
        return True
    folder = p.parent if p.is_file() else p
    check = folder / ".gpulayer_shared_fs_check"
    if rt.is_global_zero:
        check.unlink(missing_ok=True)
    rt.barrier()
    found = True
    if rt.is_global_zero:
        check.touch()
    rt.barrier()
    if not rt.is_global_zero:
        deadline = time.perf_counter() + timeout
        found = check.exists()
        while not found and time.perf_counter() < deadline:
            time.sleep(0.05)  # not a hot path: a one-off check
            found = check.exists()
    shared = not rt.any(not found)
    rt.barrier()
    if rt.is_global_zero:
        check.unlink(missing_ok=True)
    return shared


# ------------------------------------------------------------------ models


def grad_norm(
    module: nn.Module, norm_type: float | int | str, group_separator: str = "/"
) -> dict[str, torch.Tensor]:
    """Each parameter's gradient norm and the total norm (all gradients as one vector), keyed
    ``grad_{p}_norm/{name}`` and ``grad_{p}_norm_total``: log it in
    ``on_before_optimizer_step``. The values stay on the device (no synchronisation)."""
    p = float(norm_type)
    if p <= 0:
        raise ValueError(f"norm_type must be a positive number or 'inf', got {norm_type}")
    norms = {
        f"grad_{p}_norm{group_separator}{name}": param.grad.detach().norm(p)
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    if norms:
        norms[f"grad_{p}_norm_total"] = torch.stack(list(norms.values())).norm(p)
    return norms


def measure_flops(
    model: nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> int:
    """FLOPs of one forward (and backward, with ``loss_fn``), counted by
    ``torch.utils.flop_counter``. Build the model on ``torch.device("meta")`` to count without
    allocating memory."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    return int(counter.get_total_flops())


def find_shared_parameters(module: nn.Module) -> list[list[str]]:
    """Groups of parameter names that are the same tensor (tied weights)."""
    seen: dict[nn.Parameter, list[str]] = {}
    for mod_name, mod in module.named_modules(remove_duplicate=False):
        for name, param in mod._parameters.items():
            if param is not None:
                full = f"{mod_name}.{name}" if mod_name else name
                if full not in seen.setdefault(param, []):
                    seen[param].append(full)
    return [names for names in seen.values() if len(names) > 1]


def set_shared_parameters(module: nn.Module, shared_params: list[list[str]]) -> nn.Module:
    """Tie the parameters again after something (``.to(device)`` on XLA, a copy) untied them."""
    for group in shared_params:
        ref = module.get_parameter(group[0])
        for path in group[1:]:
            owner_name, _, attr = path.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            setattr(owner, attr, ref)
    return module


# ------------------------------------------------------------------ enums


class CaseInsensitiveEnum(str, Enum):
    """A string enum that compares equal to its value, case-insensitively."""

    @classmethod
    def from_str(cls, value: str) -> CaseInsensitiveEnum | None:
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None

    def __eq__(self, other: object) -> bool:
        other = other.value if isinstance(other, Enum) else other
        return isinstance(other, str) and self.value.lower() == other.lower()

    def __hash__(self) -> int:
        return hash(self.value.lower())


class GradClipAlgorithmType(CaseInsensitiveEnum):
    """``gradient_clip_algorithm`` values."""

    VALUE = "value"
    NORM = "norm"


__all__ = [
    "AttributeDict",
    "CombinedLoader",
    "GradClipAlgorithmType",
    "CaseInsensitiveEnum",
    "disable_possible_user_warnings",
    "find_shared_parameters",
    "grad_norm",
    "is_picklable",
    "is_shared_filesystem",
    "measure_flops",
    "move_data_to_device",
    "rank_zero_deprecation",
    "rank_zero_info",
    "rank_zero_only",
    "rank_zero_warn",
    "set_shared_parameters",
    "suggested_max_num_workers",
]


def to_lightning_checkpoint(source: str | Path, target: str | Path) -> Path:
    """Write a copy of a GPUlayer checkpoint for PyTorch Lightning's ``Trainer.fit``, without
    what only GPUlayer reads: callback states (their fields differ), run history and random
    states. Weights, optimizers, schedulers and hyperparameters stay; Lightning restarts its
    epoch count, as for any checkpoint without loop state. ``load_from_checkpoint`` needs no
    export. Returns ``target``."""
    from gpulayer.pytorch.storage import as_path, load_file

    state = load_file(source, map_location="cpu", weights_only=False)
    for key in (
        "gpulayer_version",
        "history",
        "callback_metrics",
        "rng_states",
        "next_epoch",
        "batches_done",
        "dataloaders",
        "optimizer",
    ):
        state.pop(key, None)
    state["callbacks"] = {}
    out = as_path(target)
    from gpulayer.pytorch.plugins import TorchCheckpointIO

    TorchCheckpointIO().save_checkpoint(state, out)
    return Path(out) if isinstance(out, Path) else out
