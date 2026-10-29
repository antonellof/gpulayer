"""Device, process-group, data-sharding and precision helpers.

Works the same in a plain local process, inside a GPUlayer job, and under torchrun:

    rt = Runtime(accelerator="auto", precision="bf16-mixed")
    rt.setup()
    model = rt.module(model)            # moved to the device; DDP-wrapped when world_size > 1
    loader = rt.dataloader(loader)      # DistributedSampler when world_size > 1
    with rt.autocast():
        loss = ...
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from gpulayer.errors import ValidationFailed

PRECISIONS = ("32-true", "bf16-mixed", "16-mixed")


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def resolve_accelerator(accelerator: str = "auto") -> str:
    """'auto' -> the accelerator GPUlayer granted this job, else cuda > mps > cpu."""
    if accelerator == "auto":
        granted = os.environ.get("GPULAYER_ACCELERATOR")
        if granted:
            return granted
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if _mps_available() else "cpu"
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise ValidationFailed("accelerator='cuda' requested but torch.cuda is not available")
    if accelerator == "mps" and not _mps_available():
        raise ValidationFailed("accelerator='mps' requested but torch MPS is not available")
    if accelerator not in ("cpu", "cuda", "mps"):
        raise ValidationFailed(f"unknown accelerator {accelerator!r}")
    return accelerator


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed)


class Runtime:
    def __init__(self, accelerator: str = "auto", precision: str = "32-true") -> None:
        if precision not in PRECISIONS:
            raise ValidationFailed(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self.accelerator = resolve_accelerator(accelerator)
        self.precision = precision
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.node_rank = int(
            os.environ.get("GPULAYER_NODE_RANK", os.environ.get("GROUP_RANK", "0"))
        )
        if self.accelerator == "mps" and self.world_size > 1:
            raise ValidationFailed("MPS supports a single process only (ADR-0006)")
        if precision == "16-mixed" and self.accelerator != "cuda":
            raise ValidationFailed("precision '16-mixed' is only supported on CUDA")
        if precision == "bf16-mixed" and self.accelerator == "mps":
            raise ValidationFailed("precision 'bf16-mixed' is not supported on MPS yet")
        if self.accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device(self.accelerator)
        self._owns_group = False
        self.scaler = torch.amp.GradScaler("cuda") if precision == "16-mixed" else None

    # ---------------------------------------------------------------- topology

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    def setup(self) -> Runtime:
        if self.accelerator == "cuda":
            torch.cuda.set_device(self.device)
        if self.distributed and not dist.is_initialized():
            backend = os.environ.get("GPULAYER_COMM_BACKEND") or (
                "nccl" if self.accelerator == "cuda" else "gloo"
            )
            dist.init_process_group(backend)
            self._owns_group = True
        return self

    def teardown(self) -> None:
        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_group = False

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------- wrapping

    def module(self, model: nn.Module) -> nn.Module:
        model.to(self.device)
        if not self.distributed:
            return model
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[self.local_rank] if self.accelerator == "cuda" else None)

    def dataloader(self, loader: DataLoader, *, shuffle: bool | None = None) -> DataLoader:
        """Return a loader that gives each rank its own shard (unchanged when not distributed)."""
        if not self.distributed:
            return loader
        if isinstance(loader.sampler, DistributedSampler):
            return loader
        if shuffle is None:
            if isinstance(loader.sampler, RandomSampler):
                shuffle = True
            elif isinstance(loader.sampler, SequentialSampler):
                shuffle = False
            else:
                raise ValidationFailed(
                    "cannot shard a DataLoader with a custom sampler; pass a DistributedSampler"
                )
        if loader.batch_size is None:
            raise ValidationFailed("cannot shard a DataLoader that uses a batch_sampler")
        sampler: DistributedSampler[Any] = DistributedSampler(
            loader.dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=shuffle
        )
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )

    def no_sync(
        self, model: nn.Module, enabled: bool = True
    ) -> contextlib.AbstractContextManager[Any]:
        """Skip DDP's gradient all-reduce for this backward (gradient accumulation)."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        if enabled and isinstance(model, DDP):
            return model.no_sync()
        return contextlib.nullcontext()

    def to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        return batch

    # ------------------------------------------------------------- precision

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if self.precision == "32-true":
            yield
            return
        dtype = torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=dtype):
            yield

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(
        self, optimizer: torch.optim.Optimizer, params: Any = None, clip_val: float | None = None
    ) -> None:
        if self.scaler is not None:
            if clip_val:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, clip_val)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if clip_val:
                nn.utils.clip_grad_norm_(params, clip_val)
            optimizer.step()

    # ------------------------------------------------------------- reductions

    def any(self, flag: bool) -> bool:
        """True on every rank if ``flag`` is true on any rank."""
        if not (self.distributed and dist.is_initialized()):
            return bool(flag)
        t = torch.tensor(
            [1.0 if flag else 0.0], device=self.device if self.accelerator == "cuda" else "cpu"
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def mean(self, value: float) -> float:
        if not (self.distributed and dist.is_initialized()):
            return float(value)
        t = torch.tensor(
            [float(value)], device=self.device if self.accelerator == "cuda" else "cpu"
        )
        dist.all_reduce(t)
        return float(t.item()) / self.world_size

    # ------------------------------------------------------------- outputs

    def output_dir(self, default: str | Path = "gpulayer_runs/local") -> Path:
        """$GPULAYER_OUTPUT_DIR inside a job (returned as artifacts), else ``default``."""
        p = Path(os.environ.get("GPULAYER_OUTPUT_DIR") or default)
        p.mkdir(parents=True, exist_ok=True)
        return p
