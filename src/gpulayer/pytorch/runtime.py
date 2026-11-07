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
import math
import os
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.storage import as_path

PRECISIONS = ("32-true", "bf16-mixed", "16-mixed", "bf16-true", "16-true", "64-true")
# Lightning's short forms, accepted with the same meaning.
_PRECISION_ALIASES = {
    "32": "32-true",
    "16": "16-mixed",
    "bf16": "bf16-mixed",
    "64": "64-true",
    "transformer-engine": None,
}
_TRUE_DTYPES = {"bf16-true": "bfloat16", "16-true": "float16", "64-true": "float64"}


def normalize_precision(precision: Any) -> str:
    """'32-true', or a Lightning short form such as 32, "16" or "bf16", to its full name."""
    name = str(precision)
    if name in _PRECISION_ALIASES:
        full = _PRECISION_ALIASES[name]
        if full is None:
            raise ValidationFailed(f"precision {name!r} is not supported by the GPUlayer Trainer")
        return full
    if name not in PRECISIONS:
        raise ValidationFailed(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return name


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def _rocm_build() -> bool:
    return bool(getattr(torch.version, "hip", None))


XLA_SPAWNED_ENV = "GPULAYER_XLA_SPAWNED"
"""Set in the TPU processes the Trainer starts with xmp.spawn (one per TPU device)."""


def _xla_available() -> bool:
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_accelerator(accelerator: str = "auto") -> str:
    """'auto' -> the accelerator GPUlayer granted this job, else a TPU (when PJRT_DEVICE=TPU
    and torch_xla is installed), else cuda/rocm > mps > cpu."""
    if accelerator == "xla":
        accelerator = "tpu"
    if accelerator == "auto":
        granted = os.environ.get("GPULAYER_ACCELERATOR")
        if granted:
            return granted
        if os.environ.get("PJRT_DEVICE") == "TPU" and _xla_available():
            return "tpu"
        if torch.cuda.is_available():
            return "rocm" if _rocm_build() else "cuda"
        return "mps" if _mps_available() else "cpu"
    if accelerator == "cuda" and not (torch.cuda.is_available() and not _rocm_build()):
        raise ValidationFailed("accelerator='cuda' requested but torch.cuda is not available")
    if accelerator == "rocm" and not (torch.cuda.is_available() and _rocm_build()):
        raise ValidationFailed(
            "accelerator='rocm' requested but this torch is not a ROCm build with an AMD GPU"
        )
    if accelerator == "mps" and not _mps_available():
        raise ValidationFailed("accelerator='mps' requested but torch MPS is not available")
    if accelerator == "tpu" and not _xla_available():
        raise ValidationFailed(
            "accelerator='tpu' needs torch_xla (pip install torch_xla[tpu]) on a TPU VM"
        )
    if accelerator in ("hpu", "gaudi"):
        # Lightning removed its Habana support too (lightning-habana is archived).
        raise ValidationFailed(
            "accelerator='hpu' (Habana Gaudi) is not supported, as in PyTorch Lightning, "
            "which removed it; use 'cuda', 'rocm', 'mps', 'tpu' or 'cpu'"
        )
    if accelerator not in ("cpu", "cuda", "rocm", "mps", "tpu"):
        raise ValidationFailed(f"unknown accelerator {accelerator!r}")
    return accelerator


class _ReusedAutocast:
    """torch.autocast, built once per Runtime instead of once per step.

    Constructing torch.autocast for CUDA asks the driver how many devices there are; per step
    that cost 0.5 ms on an RTX 3070 host (1,000 calls to cudaGetDeviceCount in a 1,000-step
    profile). One instance is reused for sequential steps; a nested use gets its own instance,
    since an autocast object keeps the state it restores on exit.
    """

    def __init__(self, device_type: str, dtype: torch.dtype) -> None:
        self._make = lambda: torch.autocast(device_type=device_type, dtype=dtype)
        self._ctx = self._make()
        self._stack: list[Any] = []

    def __enter__(self) -> Any:
        ctx = self._make() if self._stack else self._ctx
        self._stack.append(ctx)
        return ctx.__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._stack.pop().__exit__(*exc)


def _pin(batch: Any) -> Any:
    """Page-locked copies of the CPU tensors in a batch, so the GPU copy can run asynchronously."""
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory() if batch.device.type == "cpu" and not batch.is_pinned() else batch
    if isinstance(batch, (list, tuple)):
        return type(batch)(_pin(b) for b in batch)
    if isinstance(batch, dict):
        return {k: _pin(v) for k, v in batch.items()}
    return batch


def _record_stream(batch: Any, stream: Any) -> None:
    if isinstance(batch, torch.Tensor):
        if batch.device.type == "cuda":
            batch.record_stream(stream)
    elif isinstance(batch, (list, tuple)):
        for b in batch:
            _record_stream(b, stream)
    elif isinstance(batch, dict):
        for b in batch.values():
            _record_stream(b, stream)


def seed_everything(seed: int | None = None, workers: bool = False) -> int:
    """Seed Python, NumPy and torch (every device) and return the seed. Without ``seed``,
    ``$PL_GLOBAL_SEED`` or 0. ``workers`` is accepted for Lightning compatibility: torch
    already derives each DataLoader worker's seed from the base seed."""
    import random

    if seed is None:
        seed = int(os.environ.get("PL_GLOBAL_SEED", "0"))
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed % 2**32)
    return seed


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Tensors in a batch (and in lists, tuples and dicts of them) moved to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


_NOOP = contextlib.nullcontext()


def _fsdp_modules(model: nn.Module) -> list[Any]:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # torch without FSDP2
        return []
    return [m for m in model.modules() if isinstance(m, FSDPModule)]


_EXIT_HOOK = False


def _destroy_group_at_exit() -> None:
    """Destroy the default process group when the process exits (registered once)."""
    global _EXIT_HOOK
    if _EXIT_HOOK:
        return
    import atexit

    def destroy() -> None:
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()

    atexit.register(destroy)
    _EXIT_HOOK = True


def _has_dtensor(model: nn.Module) -> bool:
    """True when some parameter is a DTensor (tensor parallelism)."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return any(isinstance(p, DTensor) for p in model.parameters())


@contextlib.contextmanager
def _fsdp_no_sync(modules: list[Any]) -> Iterator[None]:
    for m in modules:
        m.set_requires_gradient_sync(False, recurse=False)
    try:
        yield
    finally:
        for m in modules:
            m.set_requires_gradient_sync(True, recurse=False)


GRADIENT_SYNC = ("auto", "flat", "two_level")


class Runtime:
    def __init__(
        self,
        accelerator: str = "auto",
        precision: str = "32-true",
        gradient_sync: str = "auto",
        ddp_kwargs: dict[str, Any] | None = None,
    ) -> None:
        precision = normalize_precision(precision)
        # Options passed to DistributedDataParallel (find_unused_parameters, static_graph,
        # gradient_as_bucket_view, bucket_cap_mb, ...).
        self.ddp_kwargs = dict(ddp_kwargs or {})
        if gradient_sync not in GRADIENT_SYNC:
            raise ValidationFailed(
                f"gradient_sync must be one of {GRADIENT_SYNC}, got {gradient_sync!r}"
            )
        self.gradient_sync = gradient_sync
        self._two_level: Any = None
        # Set by the Trainer from its strategy: a DDPStrategy with a communication hook, and
        # (size, rank) of this process's data-parallel group when it differs from the world
        # (ModelParallelStrategy: tensor-parallel ranks read the same data).
        self.comm_strategy: Any = None
        self._env_set: list[str] = []  # variables set for this run, removed by teardown
        self.data_parallel: tuple[int, int] | None = None
        self.accelerator = resolve_accelerator(accelerator)
        self.precision = precision
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self.node_rank = int(
            os.environ.get("GPULAYER_NODE_RANK", os.environ.get("GROUP_RANK", "0"))
        )
        if self.accelerator == "mps" and self.world_size > 1:
            raise ValidationFailed("MPS supports a single process only (ADR-0006)")
        # ROCm builds of torch drive AMD GPUs through the same torch.cuda API and devices.
        self.gpu = self.accelerator in ("cuda", "rocm")
        if precision == "16-mixed" and not self.gpu:
            raise ValidationFailed("precision '16-mixed' is only supported on CUDA and ROCm")
        if precision == "bf16-mixed" and self.accelerator == "mps":
            raise ValidationFailed("precision 'bf16-mixed' is not supported on MPS yet")
        if precision == "64-true" and self.accelerator == "mps":
            raise ValidationFailed("precision '64-true' is not supported on MPS (no float64)")
        # "-true" precisions keep the weights and the inputs in that dtype (no autocast).
        self.param_dtype: torch.dtype | None = (
            getattr(torch, _TRUE_DTYPES[precision]) if precision in _TRUE_DTYPES else None
        )
        self.xla = self.accelerator == "tpu"
        if self.xla and self.world_size > 1 and os.environ.get(XLA_SPAWNED_ENV) != "1":
            raise ValidationFailed(
                "several TPU processes are started by the Trainer (devices=N or 'auto' on one "
                "TPU VM, through torch_xla's xmp.spawn); TPU pods across machines are not "
                "supported yet"
            )
        if self.gpu:
            self.device = torch.device("cuda", self.local_rank)
        elif self.xla:
            import torch_xla.core.xla_model as xm

            self.device = xm.xla_device()
        else:
            self.device = torch.device(self.accelerator)
        self._owns_group = False
        self._autocast: _ReusedAutocast | None = None
        # Set by a precision plugin that runs forwards in its own context (Transformer Engine's
        # fp8_autocast); called once per forward.
        self.forward_context: Any = None
        self.scaler = torch.amp.GradScaler("cuda") if precision == "16-mixed" else None

    # ---------------------------------------------------------------- topology

    @property
    def distributed(self) -> bool:
        """True when more than one process trains (``WORLD_SIZE`` > 1)."""
        return self.world_size > 1

    @property
    def is_global_zero(self) -> bool:
        """True in the process with global rank 0."""
        return self.global_rank == 0

    def setup(self, backend: str | None = None, timeout: Any = None) -> Runtime:
        """Select the GPU of this process and start the process group when distributed (NCCL on
        GPUs, gloo otherwise, or ``backend``; ``timeout`` for its collectives). Returns self."""
        if self.gpu:
            torch.cuda.set_device(self.device)
        if self.distributed and not self.xla and not dist.is_initialized():
            # TPU processes need no process group: reductions use torch_xla's collectives
            # (xla_all_reduce), which also span the hosts of a TPU pod, as in Lightning.
            backend = (
                backend
                or os.environ.get("GPULAYER_COMM_BACKEND")
                or ("nccl" if self.gpu else "gloo")
            )
            kwargs = {"timeout": timeout} if timeout is not None else {}
            dist.init_process_group(backend, **kwargs)
            self._owns_group = True
        return self

    def teardown(self) -> None:
        """Release the process group if this Runtime started it, and remove the environment
        variables it set.

        A group of several processes stays up for the next Trainer call in the same process
        (fit, then test or predict) and is destroyed when the process exits. Destroying it
        between calls raced: a rank that finished first could join the next call through the
        old group's store, still alive on rank 0, and read stale addresses (seen in CI as
        "Gloo connectFullMesh failed ... Connection refused"). Keeping it also saves setting
        the group up again. A one-process group (DeepSpeed's) is destroyed at once."""
        if self._owns_group and dist.is_initialized():
            if self.distributed:
                _destroy_group_at_exit()
            else:
                dist.destroy_process_group()
            self._owns_group = False
        for name in self._env_set:
            os.environ.pop(name, None)
        self._env_set.clear()

    def barrier(self) -> None:
        """Wait until every process reaches this point (no-op in one process)."""
        if self.distributed and self.xla:
            import torch_xla.core.xla_model as xm

            xm.rendezvous("gpulayer-barrier")
        elif self.distributed and dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------- wrapping

    def module(self, model: nn.Module) -> nn.Module:
        """Move ``model`` to the device and wrap it in DDP when distributed (with two-level
        gradient sync when it applies)."""
        model.to(self.device)
        if not self.distributed or self.xla:
            # On TPUs xm.optimizer_step all-reduces the gradients itself (as in Lightning's
            # XLAStrategy), over the TPU interconnect; DDP is not used.
            return model
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp = DDP(model, device_ids=[self.local_rank] if self.gpu else None, **self.ddp_kwargs)
        strategy = self.comm_strategy
        if strategy is not None and strategy.register_comm_hook(ddp):
            return ddp  # the user's hook replaces two-level sync (DDP takes one hook)
        if self.uses_two_level_sync():
            from gpulayer.pytorch.hierarchical import HierarchicalAllReduce

            if self._two_level is None:  # new_group is a collective: create the groups once
                self._two_level = HierarchicalAllReduce.create(self.local_world_size)
            ddp.register_comm_hook(self._two_level, HierarchicalAllReduce.hook)
        return ddp

    def uses_two_level_sync(self) -> bool:
        """Sum gradients inside each machine before crossing the network (ADR-0020). "auto"
        does it for gloo with several machines and several processes per machine; NCCL plans
        its own topology-aware rings."""
        if self.gradient_sync == "flat" or not self.distributed:
            return False
        local = self.local_world_size
        if self.gradient_sync == "two_level":
            return True
        return (
            dist.is_initialized()
            and dist.get_backend() == "gloo"
            and 1 < local < self.world_size
            and self.world_size % local == 0
        )

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
        replicas, rank = self.data_parallel or (self.world_size, self.global_rank)
        sampler: DistributedSampler[Any] = DistributedSampler(
            loader.dataset, num_replicas=replicas, rank=rank, shuffle=shuffle
        )
        # Keep every loader setting (persistent workers, prefetching, seeding), as Lightning
        # does: dropping them would silently slow sharded runs.
        extra: dict[str, Any] = {
            "worker_init_fn": loader.worker_init_fn,
            "timeout": loader.timeout,
            "generator": loader.generator,
            "multiprocessing_context": loader.multiprocessing_context,
            "pin_memory_device": getattr(loader, "pin_memory_device", ""),
        }
        if loader.num_workers > 0:
            extra["persistent_workers"] = loader.persistent_workers
            extra["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            **extra,
        )

    def shard(self, model: nn.Module, options: Any = None) -> nn.Module:
        """FSDP (experimental): shard the parameters of every submodule that owns them across
        all ranks, so each rank holds about 1/world_size of the model, and make ``model`` the
        FSDP root. Call ``model`` (its forward) every step, so the root's hooks run and the
        submodules release their gathered parameters after their forward. Call this before
        creating the optimizer. A no-op in a single process and on TPUs. ``options`` is an
        ``FSDPStrategy`` (sharding strategy, wrap and activation checkpointing policies, CPU
        offload, mixed precision)."""
        model.to(self.device)
        if not self.distributed or self.xla:
            return model
        if (options is None or options.auto_wrap_policy is None) and next(
            model.parameters(recurse=False), None
        ) is not None:
            raise ValidationFailed(
                "strategy='fsdp' shards submodules; move parameters the model owns directly "
                "(nn.Parameter attributes) into a submodule such as nn.Linear"
            )
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        kwargs: dict[str, Any] = dict(getattr(options, "kwargs", {}) or {})
        strategy = getattr(options, "sharding_strategy", "FULL_SHARD")
        if strategy in ("HYBRID_SHARD", "NO_SHARD"):
            # 2D mesh: replicate across the first dimension, shard across the second.
            shape = getattr(options, "device_mesh", None)
            if strategy == "NO_SHARD":
                shape = (self.world_size, 1)
            elif shape is None:
                shape = (self.world_size // self.local_world_size, self.local_world_size)
            if not isinstance(shape, tuple):
                mesh = shape  # a DeviceMesh given by the user
            else:
                if shape[0] * shape[1] != self.world_size:
                    raise ValidationFailed(
                        f"device_mesh {shape} does not match {self.world_size} processes"
                    )
                mesh = init_device_mesh(
                    self.device.type, shape, mesh_dim_names=("replicate", "shard")
                )
        else:
            # An explicit mesh: without one, FSDP picks the accelerator device itself.
            mesh = init_device_mesh(self.device.type, (self.world_size,))
        if strategy == "SHARD_GRAD_OP":
            kwargs.setdefault("reshard_after_forward", False)
        if getattr(options, "cpu_offload", False):
            from torch.distributed.fsdp import CPUOffloadPolicy

            kwargs.setdefault("offload_policy", CPUOffloadPolicy())
        if getattr(options, "mixed_precision", None) is not None:
            kwargs.setdefault("mp_policy", options.mixed_precision)
        ckpt_classes = getattr(options, "activation_checkpointing_policy", None)
        if ckpt_classes:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
            )

            check = (
                (lambda m: isinstance(m, ckpt_classes))
                if isinstance(ckpt_classes, tuple)
                else ckpt_classes
            )
            apply_activation_checkpointing(model, check_fn=check)
        should = (
            options.should_shard
            if options is not None
            else (lambda m: next(m.parameters(recurse=False), None) is not None)
        )
        # Children before parents (reverse of pre-order).
        for m in reversed(list(model.modules())[1:]):
            if should(m):
                fully_shard(m, mesh=mesh, **kwargs)
        fully_shard(model, mesh=mesh, **kwargs)  # the root: runs every step
        return model

    def save_sharded(
        self, model: nn.Module, optimizers: list[Any], path: Path, meta: dict[str, Any]
    ) -> None:
        """A sharded checkpoint: the folder ``path`` with each rank's part of the model and
        optimizer state (``torch.distributed.checkpoint``) and ``meta.pt`` with everything else.
        A collective: call it on every rank."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        msd, osd = get_state_dict(model, optimizers)
        dcp.save({"model": msd, "optimizers": osd}, checkpoint_id=str(path))
        if self.is_global_zero:
            torch.save(meta, path / "meta.pt")
        self.barrier()

    def load_sharded(self, model: nn.Module, optimizers: list[Any], path: Path) -> dict[str, Any]:
        """Load a sharded checkpoint into ``model`` and ``optimizers`` (each rank reads its own
        part) and return ``meta.pt``. A collective."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        msd, osd = get_state_dict(model, optimizers)
        state: dict[str, Any] = {"model": msd, "optimizers": osd}
        dcp.load(state, checkpoint_id=str(path))
        set_state_dict(
            model, optimizers, model_state_dict=state["model"], optim_state_dict=state["optimizers"]
        )
        return dict(torch.load(path / "meta.pt", map_location="cpu", weights_only=False))

    def no_sync(
        self, model: nn.Module, enabled: bool = True
    ) -> contextlib.AbstractContextManager[Any]:
        """Skip the gradient exchange for this backward (gradient accumulation), with DDP or
        FSDP."""
        if not (enabled and self.distributed):
            return _NOOP  # called every step: no import or allocation on the common path
        from torch.nn.parallel import DistributedDataParallel as DDP

        if isinstance(model, DDP):
            return model.no_sync()
        sharded = _fsdp_modules(model)
        return _fsdp_no_sync(sharded) if sharded else _NOOP

    def full_state(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Unsharded model and optimizer state on the CPU. With FSDP this is a collective:
        call it on every rank."""
        if not _fsdp_modules(model) and not _has_dtensor(model):
            return model.state_dict(), optimizer.state_dict()
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return (
            get_model_state_dict(model, options=opts),
            get_optimizer_state_dict(model, optimizer, options=opts),
        )

    def load_full_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_state: dict[str, Any],
        optim_state: dict[str, Any],
    ) -> None:
        """Load an unsharded checkpoint into a (possibly sharded) model and optimizer."""
        if not _fsdp_modules(model) and not _has_dtensor(model):
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
            return
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        opts = StateDictOptions(full_state_dict=True)
        set_model_state_dict(model, model_state, options=opts)
        set_optimizer_state_dict(model, optimizer, optim_state, options=opts)

    def convert_module(self, model: nn.Module) -> nn.Module:
        """With a "-true" precision, cast the weights to that dtype (a no-op otherwise)."""
        if self.param_dtype is not None:
            model.to(self.param_dtype)
        return model

    def to_device(self, batch: Any) -> Any:
        """A batch on this process's device (float tensors in the "-true" precision's dtype)."""
        if self.param_dtype is not None:
            return self._to_device(batch)
        if self.device.type == "cpu":
            return batch  # DataLoader batches are already on the CPU
        return self._to_device(batch)

    def batches(self, loader: Any, *, prefetch: bool = True) -> Iterator[Any]:
        """Iterate ``loader`` with each batch on the device.

        On CUDA (with ``prefetch``) a background thread loads the next batches and pins them,
        and each one is copied to the GPU on a side stream while the current step runs. The
        training loop then only starts the copy: collating, pinning and copying overlap with
        GPU work. On TPUs torch_xla's MpDeviceLoader does the same. Elsewhere this is
        ``to_device`` per batch.
        """
        if self.xla:
            # As Lightning's XLAStrategy: MpDeviceLoader loads the next batches in background
            # threads and copies them to the TPU while the current step runs.
            from torch_xla.distributed.parallel_loader import MpDeviceLoader

            for batch in MpDeviceLoader(loader, self.device):
                yield self.to_device(batch)  # only casts with the "-true" precisions
            return
        if not prefetch or self.device.type != "cuda":
            for batch in loader:
                yield self.to_device(batch)
            return
        import queue
        import threading

        stream = torch.cuda.Stream(self.device)
        compute = torch.cuda.current_stream(self.device)
        ready: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2)
        stop = threading.Event()

        def produce() -> None:
            try:
                for batch in loader:
                    ready.put(("batch", _pin(batch)))
                    if stop.is_set():
                        return
                ready.put(("end", None))
            except BaseException as e:  # re-raised in the training loop
                ready.put(("error", e))

        thread = threading.Thread(target=produce, name="gpulayer-prefetch", daemon=True)
        thread.start()

        def copy_next() -> Any:
            kind, value = ready.get()
            if kind == "error":
                raise value
            if kind == "end":
                return None
            with torch.cuda.stream(stream):
                return self._to_device(value), True

        try:
            nxt = copy_next()
            while nxt is not None:
                compute.wait_stream(stream)  # the copy of this batch is done before it is used
                cur = nxt[0]
                _record_stream(cur, compute)  # memory from the side stream, used on this one
                nxt = copy_next()
                yield cur
        finally:
            stop.set()
            while True:  # unblock the producer if it waits on a full queue
                try:
                    ready.get_nowait()
                except queue.Empty:
                    break

    def _to_device(self, batch: Any, pin: bool = False) -> Any:
        if isinstance(batch, torch.Tensor):
            if pin and batch.device.type == "cpu" and not batch.is_pinned():
                batch = batch.pin_memory()
            if self.param_dtype is not None and batch.is_floating_point():
                # "-true" precision: float inputs in the weights' dtype, as Lightning does.
                return batch.to(self.device, self.param_dtype, non_blocking=True)
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(b, pin) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v, pin) for k, v in batch.items()}
        return batch

    # ------------------------------------------------------------- precision

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """The mixed-precision context for the forward pass (a no-op for 32-true and "-true"
        precisions), inside a precision plugin's context when one is set."""
        if self.forward_context is not None:
            stack = contextlib.ExitStack()
            stack.enter_context(self.forward_context())
            if self.precision not in ("32-true",) and self.param_dtype is None:
                stack.enter_context(self._mixed())
            return stack
        if self.precision == "32-true" or self.param_dtype is not None:
            return _NOOP
        return self._mixed()

    def _mixed(self) -> contextlib.AbstractContextManager[Any]:
        if self._autocast is None:
            dtype = torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16
            self._autocast = _ReusedAutocast(self.device.type, dtype)
        return self._autocast

    def backward(self, loss: torch.Tensor) -> None:
        """``loss.backward()``, through the gradient scaler with 16-mixed."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        params: Any = None,
        clip_val: float | None = None,
        clip_algorithm: str = "norm",
    ) -> None:
        """Optimizer step, clipping gradients first: by total norm, or each value to
        [-clip_val, clip_val] with ``clip_algorithm="value"``."""
        clip = nn.utils.clip_grad_value_ if clip_algorithm == "value" else nn.utils.clip_grad_norm_
        if self.xla:
            import torch_xla.core.xla_model as xm

            if clip_val:
                clip(params, clip_val)
            xm.optimizer_step(optimizer)
            xm.mark_step()  # run the step's graph now, as torch_xla expects once per step
        elif self.scaler is not None:
            if clip_val:
                self.scaler.unscale_(optimizer)
                clip(params, clip_val)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if clip_val:
                clip(params, clip_val)
            optimizer.step()

    # ------------------------------------------------------------- reductions

    def xla_all_reduce(self, t: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce a small CPU tensor across TPU processes (``"sum"`` or ``"max"``) with
        torch_xla's collectives; returns a CPU tensor."""
        import torch_xla.core.xla_model as xm

        kind = xm.REDUCE_MAX if op == "max" else xm.REDUCE_SUM
        return xm.all_reduce(kind, t.to(self.device)).cpu()

    def any(self, flag: bool) -> bool:
        """True on every rank if ``flag`` is true on any rank."""
        if self.distributed and self.xla:
            return bool(self.xla_all_reduce(torch.tensor([float(flag)]), "max").item() > 0)
        if not (self.distributed and dist.is_initialized()):
            return bool(flag)
        t = torch.tensor([1.0 if flag else 0.0], device=self.device if self.gpu else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def mean_dict(self, values: dict[str, float]) -> dict[str, float]:
        """Average every value across ranks in one all-reduce (keys sorted, so ranks agree)."""
        if self.distributed and self.xla and values:
            keys = sorted(values)
            t = self.xla_all_reduce(torch.tensor([float(values[k]) for k in keys]))
            return dict(zip(keys, (t / self.world_size).tolist(), strict=True))
        if not (self.distributed and dist.is_initialized()) or not values:
            return {k: float(v) for k, v in values.items()}
        keys = sorted(values)
        t = torch.tensor(
            [float(values[k]) for k in keys], device=self.device if self.gpu else "cpu"
        )
        dist.all_reduce(t)
        t /= self.world_size
        return dict(zip(keys, t.tolist(), strict=True))

    def _all_reduce(self, t: torch.Tensor, op: str) -> torch.Tensor:
        """``op`` is "sum" or "max"; returns a CPU tensor."""
        if self.xla:
            return self.xla_all_reduce(t.float(), op)
        t = t.to(self.device if self.gpu else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
        return t.cpu()

    def reduce_metrics(
        self,
        means: dict[str, tuple[float, float]],
        sums: dict[str, float],
        maxes: dict[str, float],
        mins: dict[str, float],
    ) -> dict[str, float]:
        """Epoch values across ranks, each by its ``reduce_fx``: a mean is the weighted
        total over the weights of every rank (``(total, weight)`` pairs), a sum adds up,
        max and min take the extreme.

        Ranks may log different names (``self.log`` behind a condition): the name lists
        are compared first (one small all-reduce), and only when they differ are they
        exchanged, so no rank enters a reduction of a different length (which would hang)."""

        def local() -> dict[str, float]:
            out = {k: (n / d if d else math.nan) for k, (n, d) in means.items()}
            return {**out, **sums, **maxes, **mins}

        if not self.distributed or not (self.xla or dist.is_initialized()):
            return local()
        layout = [sorted(means), sorted(sums), sorted(maxes), sorted(mins)]
        crc = zlib.crc32(repr(layout).encode())
        hi, lo = float(crc >> 16), float(crc & 0xFFFF)  # exact in float32
        f = self._all_reduce(torch.tensor([hi, lo, -hi, -lo]), "max")
        if not (f[0] == -f[2] and f[1] == -f[3]):
            if self.xla:
                raise ValidationFailed(
                    "processes logged different metric names; log the same names on every process"
                )
            gathered: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, layout)
            layout = [sorted(set().union(*(g[i] for g in gathered))) for i in range(4)]
        mean_keys, sum_keys, max_keys, min_keys = layout
        dtype = torch.float32 if self.xla else torch.float64
        packed = torch.tensor(
            [means.get(k, (0.0, 0.0))[0] for k in mean_keys]
            + [means.get(k, (0.0, 0.0))[1] for k in mean_keys]
            + [sums.get(k, 0.0) for k in sum_keys],
            dtype=dtype,
        )
        out: dict[str, float] = {}
        if packed.numel():
            total = self._all_reduce(packed, "sum").tolist()
            n = len(mean_keys)
            for i, k in enumerate(mean_keys):
                out[k] = total[i] / total[n + i] if total[n + i] else math.nan
            for i, k in enumerate(sum_keys):
                out[k] = total[2 * n + i]
        if max_keys or min_keys:
            ext = torch.tensor(
                [maxes.get(k, -math.inf) for k in max_keys]
                + [-mins.get(k, math.inf) for k in min_keys],
                dtype=dtype,
            )
            got = self._all_reduce(ext, "max").tolist()
            for i, k in enumerate(max_keys):
                out[k] = got[i]
            for i, k in enumerate(min_keys):
                out[k] = -got[len(max_keys) + i]
        return out

    def mean(self, value: float) -> float:
        """The average of ``value`` across processes (one all-reduce)."""
        if self.distributed and self.xla:
            return float(self.xla_all_reduce(torch.tensor([float(value)])).item()) / self.world_size
        if not (self.distributed and dist.is_initialized()):
            return float(value)
        t = torch.tensor([float(value)], device=self.device if self.gpu else "cpu")
        dist.all_reduce(t)
        return float(t.item()) / self.world_size

    # ------------------------------------------------------------- outputs

    def output_dir(self, default: Any = "gpulayer_runs/local") -> Any:
        """$GPULAYER_OUTPUT_DIR inside a job (returned as artifacts), else ``default``."""
        p = as_path(os.environ.get("GPULAYER_OUTPUT_DIR") or default)
        p.mkdir(parents=True, exist_ok=True)
        return p
