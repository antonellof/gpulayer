"""Training strategies: how the model is replicated or sharded across processes.

    Trainer(strategy="ddp")                                   # the default
    Trainer(strategy="ddp_find_unused_parameters_true")       # Lightning's string form
    Trainer(strategy=DDPStrategy(static_graph=True, gradient_as_bucket_view=True))
    Trainer(strategy=DDPStrategy(ddp_comm_hook=default_hooks.fp16_compress_hook))
    Trainer(strategy=FSDPStrategy(activation_checkpointing_policy={Block}))  # experimental
    Trainer(strategy=ModelParallelStrategy(tensor_parallel_size=4))         # experimental

The names and options follow PyTorch Lightning. Extra ``DDPStrategy`` keyword arguments go to
``torch.nn.parallel.DistributedDataParallel`` unchanged. GPUlayer starts every process itself,
so Lightning's launcher arguments (``accelerator``, ``parallel_devices``,
``cluster_environment``, ``start_method``) are accepted for compatibility and not used.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from gpulayer.errors import ValidationFailed

DEFAULT_TIMEOUT = timedelta(minutes=30)  # torch.distributed's default


class Strategy:
    """Base class of the strategies."""

    name = "strategy"
    process_group_backend: str | None = None
    timeout: timedelta | None = DEFAULT_TIMEOUT
    checkpoint_io: Any = None
    precision_plugin: Any = None

    trainer: Any = None  # set by the Trainer

    def ddp_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``DistributedDataParallel`` (empty unless DDP is used)."""
        return {}

    # What code usually reads through ``trainer.strategy``. GPUlayer's strategies describe
    # how to train; the Runtime does it, so these ask the Runtime.

    @property
    def _runtime(self) -> Any:
        return self.trainer.runtime if self.trainer is not None else None

    @property
    def _distributed(self) -> bool:
        rt = self._runtime
        return rt is not None and rt.distributed

    @property
    def root_device(self) -> Any:
        """The device of this process."""
        import torch

        rt = self._runtime
        return rt.device if rt is not None else torch.device("cpu")

    @property
    def global_rank(self) -> int:
        """This process's rank across all machines."""
        return self.trainer.global_rank if self.trainer is not None else 0

    @property
    def local_rank(self) -> int:
        """This process's rank on its machine."""
        return self.trainer.local_rank if self.trainer is not None else 0

    @property
    def node_rank(self) -> int:
        """This machine's rank."""
        return self.trainer.node_rank if self.trainer is not None else 0

    @property
    def world_size(self) -> int:
        """Processes in the run."""
        return self.trainer.world_size if self.trainer is not None else 1

    @property
    def is_global_zero(self) -> bool:
        """True in the first process."""
        return self.global_rank == 0

    @property
    def lightning_module(self) -> Any:
        """The Module being trained."""
        return self.trainer.module if self.trainer is not None else None

    @property
    def model(self) -> Any:
        """The Module as the loop calls it (wrapped in DDP when distributed)."""
        return self.trainer.model if self.trainer is not None else None

    @property
    def optimizers(self) -> list[Any]:
        """The optimizers from ``configure_optimizers``."""
        return list(self.trainer.optimizers) if self.trainer is not None else []

    def barrier(self, name: str | None = None) -> None:
        """Wait until every process reaches this point."""
        if self._distributed:
            self._runtime.barrier()

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        """Process ``src``'s ``obj``, on every process. Not available on TPUs."""
        if not self._distributed:
            return obj
        if self._runtime.xla:
            raise ValidationFailed("strategy.broadcast is not available on TPUs")
        import torch.distributed as dist

        box = [obj]
        dist.broadcast_object_list(box, src=src)
        return box[0]

    def reduce(self, tensor: Any, group: Any = None, reduce_op: str | None = "mean") -> Any:
        """``tensor`` combined across processes: ``"mean"``, ``"sum"``, ``"max"`` or
        ``"min"``."""
        import torch

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device=self.root_device, dtype=torch.float32)
        if not self._distributed:
            return tensor
        op = (reduce_op or "mean").lower()
        if op not in ("mean", "sum", "max", "min"):
            raise ValidationFailed(f"reduce_op must be mean, sum, max or min, got {reduce_op!r}")
        value = tensor.detach().float().reshape(-1)
        rt = self._runtime
        if op == "min":
            out = -rt._all_reduce(-value, "max")
        else:
            out = rt._all_reduce(value, "max" if op == "max" else "sum")
        if op == "mean":
            out = out / self.world_size
        return out.to(tensor.device).reshape(tensor.shape)

    def all_gather(self, tensor: Any, group: Any = None, sync_grads: bool = False) -> Any:
        """Gather from every process, with a leading world-size dimension."""
        return self.trainer.module.all_gather(tensor, sync_grads=sync_grads)

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:  # noqa: A002
        """Whether every process (``all=True``) or any process decided ``True``."""
        if not self._distributed:
            return bool(decision)
        rt = self._runtime
        return not rt.any(not decision) if all else rt.any(decision)


class SingleDeviceStrategy(Strategy):
    """One process on one device (what ``devices=1`` does). ``device`` names it, for example
    ``"cuda:1"``, as in Lightning."""

    name = "single_device"

    def __init__(
        self,
        device: Any = "cpu",
        accelerator: Any = None,
        checkpoint_io: Any = None,
        precision_plugin: Any = None,
    ) -> None:
        import torch

        self.device = (
            torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
        )
        self.checkpoint_io = checkpoint_io
        self.precision_plugin = precision_plugin


class ParallelStrategy(Strategy):
    """Base class of the strategies that train in several processes."""

    def __init__(
        self,
        accelerator: Any = None,
        parallel_devices: Any = None,
        cluster_environment: Any = None,
        checkpoint_io: Any = None,
        precision_plugin: Any = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = DEFAULT_TIMEOUT,
    ) -> None:
        # GPUlayer places and starts the processes: accelerator, devices and the cluster
        # environment come from the Trainer and the job, so these are not used.
        self.checkpoint_io = checkpoint_io
        self.precision_plugin = precision_plugin
        self.process_group_backend = process_group_backend
        self.timeout = timeout


class DDPStrategy(ParallelStrategy):
    """DistributedDataParallel with its options.

    - ``find_unused_parameters``: needed when some parameters get no gradient in a step
      (costs a graph traversal per step, so off by default).
    - ``static_graph``: the set of used parameters never changes; DDP can then skip work
      and overlap more communication.
    - ``gradient_as_bucket_view``: gradients are views into DDP's communication buckets,
      which saves one copy of the gradients in memory and a copy per step.
    - ``ddp_comm_hook`` (with ``ddp_comm_state``, optionally wrapped by ``ddp_comm_wrapper``):
      a gradient communication hook, for example
      ``torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook`` to send
      gradients in half precision (half the bytes on slow networks). It replaces GPUlayer's
      two-level gradient sync.
    - ``model_averaging_period``: post-local SGD (with ``ddp_comm_hook=post_localSGD_hook`` and
      a ``PostLocalSGDState``): after the warm-up, each process steps on its own and the models
      are averaged every N steps, trading exactness for far less communication.
    - ``process_group_backend`` (``"nccl"``, ``"gloo"``) and ``timeout`` of the process group.
    """

    name = "ddp"

    def __init__(
        self,
        accelerator: Any = None,
        parallel_devices: Any = None,
        cluster_environment: Any = None,
        checkpoint_io: Any = None,
        precision_plugin: Any = None,
        ddp_comm_state: Any = None,
        ddp_comm_hook: Callable[..., Any] | None = None,
        ddp_comm_wrapper: Callable[..., Any] | None = None,
        model_averaging_period: int | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = DEFAULT_TIMEOUT,
        start_method: str = "popen",
        find_unused_parameters: bool = False,
        static_graph: bool = False,
        gradient_as_bucket_view: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator, parallel_devices, cluster_environment, checkpoint_io, precision_plugin,
            process_group_backend, timeout,
        )  # fmt: skip
        if start_method not in ("popen", "spawn", "fork", "forkserver"):
            raise ValidationFailed(f"unknown start_method {start_method!r}")
        if model_averaging_period is not None and model_averaging_period < 1:
            raise ValidationFailed("model_averaging_period must be >= 1")
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.ddp_comm_state = ddp_comm_state
        self.ddp_comm_hook = ddp_comm_hook
        self.ddp_comm_wrapper = ddp_comm_wrapper
        self.model_averaging_period = model_averaging_period
        self.start_method = start_method  # GPUlayer starts processes itself: not used
        self.kwargs = kwargs

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

    def register_comm_hook(self, ddp: Any) -> bool:
        """Register ``ddp_comm_hook`` on ``ddp`` (returns False when there is none)."""
        if self.ddp_comm_hook is None:
            return False
        hook = self.ddp_comm_hook
        if self.ddp_comm_wrapper is not None:
            hook = self.ddp_comm_wrapper(hook)
        ddp.register_comm_hook(state=self.ddp_comm_state, hook=hook)
        return True


_SHARDING = ("FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD")


class FSDPStrategy(ParallelStrategy):
    """Fully sharded data parallel with PyTorch's FSDP2 (``fully_shard``), experimental.

    - ``sharding_strategy``: ``"FULL_SHARD"`` (the default: parameters, gradients and optimizer
      state sharded; parameters gathered for each layer's forward and backward),
      ``"SHARD_GRAD_OP"`` (parameters kept after the forward: less communication, more
      memory), ``"HYBRID_SHARD"`` (sharded inside each machine, replicated across machines:
      only gradients cross the network) or ``"NO_SHARD"`` (replicated, as DDP).
    - ``auto_wrap_policy``: a set of module classes (for example a transformer block) to shard
      as units; by default every module that owns parameters is one.
    - ``activation_checkpointing_policy``: a set of module classes whose activations are
      recomputed in the backward pass instead of kept (less memory, about one more forward).
    - ``cpu_offload``: keep sharded parameters, gradients and optimizer state in CPU memory.
    - ``mixed_precision``: a ``torch.distributed.fsdp.MixedPrecisionPolicy``; by default it
      follows ``precision`` (``bf16-mixed``: bf16 compute, fp32 gradient reduction).
    - ``state_dict_type``: ``"full"`` (one file gathered on rank 0) or ``"sharded"`` (a
      folder where each rank writes its shard, for models too large to gather).
    - ``device_mesh``: ``(replicas, shards)`` for ``HYBRID_SHARD``; by default
      ``(machines, processes per machine)``.
    """

    name = "fsdp"

    def __init__(
        self,
        accelerator: Any = None,
        parallel_devices: Any = None,
        cluster_environment: Any = None,
        checkpoint_io: Any = None,
        precision_plugin: Any = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = DEFAULT_TIMEOUT,
        cpu_offload: bool | Any = None,
        mixed_precision: Any = None,
        auto_wrap_policy: Any = None,
        activation_checkpointing: Any = None,
        activation_checkpointing_policy: Any = None,
        sharding_strategy: str = "FULL_SHARD",
        state_dict_type: str = "full",
        device_mesh: tuple[int, int] | Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator, parallel_devices, cluster_environment, checkpoint_io, precision_plugin,
            process_group_backend, timeout,
        )  # fmt: skip
        sharding_strategy = getattr(sharding_strategy, "name", sharding_strategy)
        if str(sharding_strategy).upper() not in _SHARDING:
            raise ValidationFailed(f"sharding_strategy must be one of {_SHARDING}")
        if state_dict_type not in ("full", "sharded"):
            raise ValidationFailed("state_dict_type must be 'full' or 'sharded'")
        if activation_checkpointing is not None and activation_checkpointing_policy is not None:
            raise ValidationFailed(
                "pass activation_checkpointing_policy (activation_checkpointing is its old name)"
            )
        self.cpu_offload = bool(cpu_offload)
        self.mixed_precision = mixed_precision
        self.auto_wrap_policy = _classes(auto_wrap_policy)
        policy = activation_checkpointing_policy or activation_checkpointing
        self.activation_checkpointing_policy = _classes(policy)
        self.sharding_strategy = str(sharding_strategy).upper()
        self.state_dict_type = state_dict_type
        self.device_mesh = device_mesh
        self.kwargs = kwargs  # extra fully_shard arguments

    def should_shard(self, module: Any) -> bool:
        """Whether ``module`` becomes its own FSDP unit."""
        policy = self.auto_wrap_policy
        if policy is None:
            return next(module.parameters(recurse=False), None) is not None
        if isinstance(policy, tuple):
            return isinstance(module, policy)
        # an FSDP1-style callable policy
        n = sum(p.numel() for p in module.parameters())
        return bool(policy(module=module, recurse=False, nonwrapped_numel=n))


def _classes(policy: Any) -> Any:
    """A set or list of module classes as a tuple (for isinstance), or a callable unchanged."""
    if policy is None or callable(policy) and not isinstance(policy, type):
        return policy
    if isinstance(policy, type):
        return (policy,)
    if hasattr(policy, "_module_classes"):  # torch's ModuleWrapPolicy
        return tuple(policy._module_classes)
    return tuple(policy)


class ModelParallelStrategy(ParallelStrategy):
    """Tensor parallelism with FSDP2 (2D parallelism), experimental, as Lightning's.

    The processes form a ``(data_parallel, tensor_parallel)`` device mesh, by default
    ``(machines, processes per machine)`` so tensor-parallel traffic stays inside a machine.
    The Module gets it as ``self.device_mesh`` and must apply the parallelism itself in
    ``configure_model``, for example with
    ``torch.distributed.tensor.parallel.parallelize_module(self.model,
    self.device_mesh["tensor_parallel"], plan)`` and then ``fully_shard(self.model,
    mesh=self.device_mesh["data_parallel"])``. Each data-parallel group reads its own shard of
    the data. ``save_distributed_checkpoint=True`` writes a folder with each rank's shard;
    ``False`` gathers full weights on rank 0."""

    name = "model_parallel"

    def __init__(
        self,
        data_parallel_size: int | str = "auto",
        tensor_parallel_size: int | str = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: str | None = None,
        timeout: timedelta | None = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(process_group_backend=process_group_backend, timeout=timeout)
        for name, v in (
            ("data_parallel_size", data_parallel_size),
            ("tensor_parallel_size", tensor_parallel_size),
        ):
            if v != "auto" and (not isinstance(v, int) or v < 1):
                raise ValidationFailed(f"{name} must be 'auto' or a number >= 1")
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.save_distributed_checkpoint = save_distributed_checkpoint
        self.state_dict_type = "sharded" if save_distributed_checkpoint else "full"

    def mesh_shape(self, world_size: int, local_world_size: int) -> tuple[int, int]:
        """``(data_parallel, tensor_parallel)`` sizes for this run."""
        dp, tp = self.data_parallel_size, self.tensor_parallel_size
        if tp == "auto":
            tp = local_world_size if dp == "auto" else world_size // int(dp)
        if dp == "auto":
            dp = world_size // int(tp)
        if int(dp) * int(tp) != world_size:
            raise ValidationFailed(
                f"data_parallel_size={dp} times tensor_parallel_size={tp} must equal the "
                f"number of processes ({world_size})"
            )
        return int(dp), int(tp)


class DeepSpeedStrategy(ParallelStrategy):
    """DeepSpeed ZeRO (needs the ``deepspeed`` package and CUDA GPUs; implemented, not run on
    hardware). ``stage`` 1 shards optimizer state, 2 also gradients, 3 also parameters;
    ``offload_optimizer``/``offload_parameters`` move them to CPU memory (or NVMe with
    ``offload_*_device="nvme"`` and ``nvme_path``). The other arguments are Lightning's and go
    into the generated DeepSpeed configuration; ``config`` (a dict or a JSON file) replaces it.
    Checkpoints hold full weights (gathered from all ranks with stage 3); resuming a DeepSpeed
    run from a checkpoint is not supported yet."""

    name = "deepspeed"

    def __init__(
        self,
        accelerator: Any = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: str | None = None,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = "cpu",
        nvme_path: str = "/local_nvme",
        params_buffer_count: int = 5,
        params_buffer_size: int = 100_000_000,
        max_in_cpu: int = 1_000_000_000,
        offload_optimizer_device: str = "cpu",
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1_000_000_000_000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: str | int = "auto",
        config: str | Path | dict[str, Any] | None = None,
        logging_level: int = logging.WARN,
        parallel_devices: Any = None,
        cluster_environment: Any = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
        precision_plugin: Any = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = DEFAULT_TIMEOUT,
        exclude_frozen_parameters: bool = False,
    ) -> None:
        super().__init__(
            accelerator, parallel_devices, cluster_environment, None, precision_plugin,
            process_group_backend, timeout,
        )  # fmt: skip
        if stage not in (0, 1, 2, 3):
            raise ValidationFailed("DeepSpeedStrategy stage must be 0, 1, 2 or 3")
        for name, dev in (
            ("offload_params_device", offload_params_device),
            ("offload_optimizer_device", offload_optimizer_device),
        ):
            if dev not in ("cpu", "nvme"):
                raise ValidationFailed(f"{name} must be 'cpu' or 'nvme'")
        self.zero_optimization = zero_optimization
        self.stage = stage
        self.remote_device = remote_device
        self.offload_optimizer = offload_optimizer
        self.offload_parameters = offload_parameters
        self.config = config
        self.logging_level = logging_level
        self.load_full_weights = load_full_weights
        self.exclude_frozen_parameters = exclude_frozen_parameters
        self.logging_batch_size_per_gpu = logging_batch_size_per_gpu
        self.loss_scale = loss_scale
        self.initial_scale_power = initial_scale_power
        self.loss_scale_window = loss_scale_window
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale
        self.zero_allow_untested_optimizer = zero_allow_untested_optimizer
        self._offload = {
            "nvme_path": nvme_path, "offload_params_device": offload_params_device,
            "params_buffer_count": params_buffer_count, "params_buffer_size": params_buffer_size,
            "max_in_cpu": max_in_cpu, "offload_optimizer_device": offload_optimizer_device,
            "optimizer_buffer_count": optimizer_buffer_count, "pin_memory": pin_memory,
        }  # fmt: skip
        self._zero = {
            "sub_group_size": sub_group_size, "contiguous_gradients": contiguous_gradients,
            "overlap_comm": overlap_comm, "allgather_partitions": allgather_partitions,
            "reduce_scatter": reduce_scatter, "allgather_bucket_size": allgather_bucket_size,
            "reduce_bucket_size": reduce_bucket_size,
        }  # fmt: skip
        self._aio = {
            "block_size": block_size, "queue_depth": queue_depth, "single_submit": single_submit,
            "overlap_events": overlap_events, "thread_count": thread_count,
        }  # fmt: skip
        self._activation = {
            "partition_activations": partition_activations,
            "cpu_checkpointing": cpu_checkpointing,
            "contiguous_memory_optimization": contiguous_memory_optimization,
            "synchronize_checkpoint_boundary": synchronize_checkpoint_boundary,
        }

    def deepspeed_config(
        self, precision: str, accumulate: int, clip: float | None, micro_batch: int | None
    ) -> dict[str, Any]:
        """The DeepSpeed configuration for this Trainer's settings (Lightning's layout)."""
        if isinstance(self.config, (str, Path)):
            cfg = dict(json.loads(Path(self.config).read_text()))
        elif isinstance(self.config, dict):
            cfg = dict(self.config)
        else:
            o = self._offload
            cfg = {"activation_checkpointing": dict(self._activation), "aio": dict(self._aio)}
            if self.zero_optimization:
                zero: dict[str, Any] = {"stage": self.stage, **self._zero}
                if self.offload_optimizer:
                    zero["offload_optimizer"] = {
                        "device": o["offload_optimizer_device"], "nvme_path": o["nvme_path"],
                        "buffer_count": o["optimizer_buffer_count"], "pin_memory": o["pin_memory"],
                    }  # fmt: skip
                if self.offload_parameters:
                    zero["offload_param"] = {
                        "device": o["offload_params_device"], "nvme_path": o["nvme_path"],
                        "buffer_count": o["params_buffer_count"],
                        "buffer_size": o["params_buffer_size"], "max_in_cpu": o["max_in_cpu"],
                        "pin_memory": o["pin_memory"],
                    }  # fmt: skip
                cfg["zero_optimization"] = zero
                cfg["zero_allow_untested_optimizer"] = self.zero_allow_untested_optimizer
        if (
            "gradient_accumulation_steps" in cfg
            and cfg["gradient_accumulation_steps"] != accumulate
        ):
            raise ValidationFailed(
                "set gradient accumulation with Trainer(accumulate_grad_batches=...), not in the "
                "DeepSpeed config"
            )
        cfg["gradient_accumulation_steps"] = accumulate
        if "train_micro_batch_size_per_gpu" not in cfg:
            lb = self.logging_batch_size_per_gpu
            cfg["train_micro_batch_size_per_gpu"] = lb if lb != "auto" else (micro_batch or 1)
        if clip and "gradient_clipping" not in cfg:
            cfg["gradient_clipping"] = clip
        if precision in ("16-mixed", "16-true") and "fp16" not in cfg:
            cfg["fp16"] = {
                "enabled": True, "loss_scale": self.loss_scale,
                "initial_scale_power": self.initial_scale_power,
                "loss_scale_window": self.loss_scale_window, "hysteresis": self.hysteresis,
                "min_loss_scale": self.min_loss_scale,
            }  # fmt: skip
        elif precision in ("bf16-mixed", "bf16-true") and "bf16" not in cfg:
            cfg["bf16"] = {"enabled": True}
        return cfg


_STRINGS: dict[str, Any] = {
    "auto": lambda: DDPStrategy(),
    "ddp": lambda: DDPStrategy(),
    "ddp_find_unused_parameters_false": lambda: DDPStrategy(find_unused_parameters=False),
    "ddp_find_unused_parameters_true": lambda: DDPStrategy(find_unused_parameters=True),
    "fsdp": lambda: FSDPStrategy(),
    "fsdp_cpu_offload": lambda: FSDPStrategy(cpu_offload=True),
    "model_parallel": lambda: ModelParallelStrategy(),
    "single_device": lambda: SingleDeviceStrategy(),
    "deepspeed": lambda: DeepSpeedStrategy(),
    "deepspeed_stage_1": lambda: DeepSpeedStrategy(stage=1),
    "deepspeed_stage_1_offload": lambda: DeepSpeedStrategy(stage=1, offload_optimizer=True),
    "deepspeed_stage_2": lambda: DeepSpeedStrategy(stage=2),
    "deepspeed_stage_2_offload": lambda: DeepSpeedStrategy(stage=2, offload_optimizer=True),
    "deepspeed_stage_3": lambda: DeepSpeedStrategy(stage=3),
    "deepspeed_stage_3_offload": lambda: DeepSpeedStrategy(
        stage=3, offload_optimizer=True, offload_parameters=True
    ),
    "deepspeed_stage_3_offload_nvme": lambda: DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        offload_params_device="nvme",
        offload_optimizer_device="nvme",
    ),  # fmt: skip
}


def resolve_strategy(strategy: Any) -> Any:
    """A strategy object from ``Trainer(strategy=...)``."""
    if isinstance(strategy, Strategy):
        return strategy
    if strategy in _STRINGS:
        return _STRINGS[strategy]()
    if isinstance(strategy, str) and strategy.startswith("ddp_spawn"):
        raise ValidationFailed(
            f"strategy {strategy!r}: GPUlayer starts one process per rank itself, so the spawn "
            "launcher is not needed; use 'ddp'"
        )
    if isinstance(strategy, str) and strategy.startswith(("ddp_fork", "ddp_notebook")):
        # In a notebook the Trainer forks its ranks whatever the name; elsewhere it is ddp.
        return resolve_strategy(strategy.replace("ddp_fork", "ddp").replace("ddp_notebook", "ddp"))
    if strategy in ("xla", "xla_debug", "single_xla"):
        return DDPStrategy()  # accelerator="tpu" picks the XLA path (see the Trainer)
    raise ValidationFailed(
        f"strategy must be one of {sorted(_STRINGS)}, or a Strategy object, got {strategy!r}"
    )
