"""``Trainer``: runs a ``Module``'s training loop here, or launches it on GPUlayer workers.

Arguments, hooks and callbacks follow PyTorch Lightning's names and meaning (ADR-0018,
ADR-0024). The per-step path stays lean: hooks a model or callback does not override are not
called, logged values stay on the device until a log step, and nothing is profiled, logged
or synchronised per step unless a feature that needs it is turned on.
"""

from __future__ import annotations

import contextlib
import csv
import inspect
import itertools
import json
import math
import multiprocessing
import multiprocessing.connection
import os
import random
import shutil
import signal
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, SequentialSampler

from gpulayer.contracts.models import CHECKPOINT_MARKER, RESUME_INPUT
from gpulayer.errors import GpuLayerError, ValidationFailed
from gpulayer.pytorch.accelerators import Accelerator, accelerator_name
from gpulayer.pytorch.callbacks import Callback, Checkpoint
from gpulayer.pytorch.callbacks.base import legacy_batch_end, overrides
from gpulayer.pytorch.callbacks.monitor import EarlyStopping, ModelCheckpoint
from gpulayer.pytorch.callbacks.utility import (
    ModelSummary,
    ProgressBar,
    RichModelSummary,
    RichProgressBar,
    _parse_duration,
)
from gpulayer.pytorch.combined import CombinedLoader, loaders_of
from gpulayer.pytorch.launch import (
    LaunchResult,
    in_gpulayer_job,
    in_notebook,
    in_slurm_task,
    is_launcher_process,
    launch,
)
from gpulayer.pytorch.loggers import Logger
from gpulayer.pytorch.module import DataModule, Module, _TrainingStep, overridden
from gpulayer.pytorch.plugins import CheckpointIO, TorchCheckpointIO, apply_plugins
from gpulayer.pytorch.profilers import Profiler, resolve_profiler
from gpulayer.pytorch.runtime import (
    GRADIENT_SYNC,
    XLA_SPAWNED_ENV,
    Runtime,
    normalize_precision,
    resolve_accelerator,
)
from gpulayer.pytorch.storage import as_path, is_remote
from gpulayer.pytorch.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    ModelParallelStrategy,
    Strategy,
    resolve_strategy,
)
from gpulayer.sdk.remote import RemoteError, default_client, default_target, explain_failure

CHECKPOINT = "checkpoints/last.ckpt"
# cuDNN autotuning pays off from about this many optimizer steps (see Trainer.fit).
AUTOTUNE_MIN_STEPS = 1000
UNLIMITED_EPOCHS = 10**9
LIGHTNING_VERSION_KEY = "pytorch-lightning_version"
LIGHTNING_FORMAT = "2.5.0"  # the checkpoint layout written here; Lightning migrates from it


@dataclass
class LRSchedulerConfig:
    """A learning-rate scheduler and when to step it, as in Lightning."""

    scheduler: Any
    name: str | None = None
    interval: str = "epoch"
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: str | None = None
    strict: bool = True


@dataclass
class TrainerState:
    fn: str | None = None  # "fit", "validate", "test", "predict"
    stage: str | None = None  # "train", "sanity_check", "validate", "test", "predict"
    status: str = "initializing"  # "running", "finished", "interrupted"


class _Results(dict):
    """``validate``/``test`` results: a dict, and also ``results[0]`` like Lightning's list."""

    def __getitem__(self, key: Any) -> Any:
        if key == 0 and 0 not in self.keys():
            return self
        return super().__getitem__(key)

    def __len__(self) -> int:
        return super().__len__()


class _ManualOptimizer:
    """What ``Module.optimizers()`` returns with manual optimization: the optimizer, with
    ``step`` going through the gradient scaler (16-mixed) and counting ``global_step``."""

    def __init__(self, optimizer: torch.optim.Optimizer, trainer: Trainer) -> None:
        self.__dict__["_opt"] = optimizer
        self.__dict__["_trainer"] = trainer

    def __getattr__(self, name: str) -> Any:
        if name == "_opt":
            raise AttributeError(name)
        return getattr(self._opt, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._opt, name, value)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._opt

    def step(self, closure: Callable[[], Any] | None = None, **kwargs: Any) -> Any:
        t = self._trainer
        t._call_hook("on_before_optimizer_step", self._opt)
        scaler = t.runtime.scaler if t.runtime is not None else None
        if scaler is not None:
            out = scaler.step(self._opt, closure, **kwargs) if closure else scaler.step(self._opt)
            scaler.update()
        elif t.runtime is not None and t.runtime.xla:
            import torch_xla.core.xla_model as xm

            # all-reduces the gradients across TPU processes, then runs the step's graph
            out = xm.optimizer_step(self._opt, optimizer_args=kwargs)
            xm.mark_step()
        else:
            out = self._opt.step(closure, **kwargs) if closure else self._opt.step()
        t.global_step += 1
        t._stepped = True
        return out

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._trainer._call_hook("on_before_zero_grad", self._opt)
        self._opt.zero_grad(set_to_none=set_to_none)


def _limit(value: int | float | None, n: int | None, name: str) -> int | None:
    """``limit_*_batches``: a number of batches, or a fraction of the loader."""
    if value is None:
        return None
    if isinstance(value, float):
        if not 0.0 <= value <= 1.0:
            raise ValidationFailed(f"{name} as a float must be between 0 and 1")
        if value == 1.0:
            return None
        if n is None:
            raise ValidationFailed(f"a fractional {name} needs a loader with a length")
        return max(1, int(n * value)) if value > 0 else 0
    if value < 0:
        raise ValidationFailed(f"{name} must be >= 0")
    return int(value)


def _len(loader: Any) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


class Trainer:
    """Train a ``Module``: ``Trainer(max_epochs=5).fit(model, train_loader, val_loader)``.

    With one device and no ``target`` the loop runs in this process. With ``devices`` > 1,
    ``num_nodes`` > 1 or a ``target``, ``fit`` submits a job and the same script runs on
    GPUlayer workers, one process per rank; the trained weights come back to ``model``.

    Every argument is described in the guide's table:
    [Trainer arguments](trainer.md#trainer-arguments). Names and meanings follow PyTorch
    Lightning; the GPUlayer-only ones are ``target``, ``gradient_sync``, ``prefetch`` and
    ``launch_options``.
    """

    def __init__(
        self,
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "ddp",
        devices: int | str = 1,
        num_nodes: int = 1,
        precision: str | int = "32-true",
        logger: Logger | list[Logger] | bool | None = None,
        callbacks: Callback | list[Callback] | None = None,
        fast_dev_run: bool | int = False,
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int | None = None,
        min_steps: int | None = None,
        max_time: float | timedelta | dict[str, int] | str | None = None,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        overfit_batches: int | float = 0.0,
        val_check_interval: int | float | None = None,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int = 2,
        log_every_n_steps: int = 50,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        deterministic: bool | str = False,
        benchmark: bool | None = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: str | Profiler | None = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Any = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: str | Path = "gpulayer_runs",
        log_key_prefix: str | None = None,
        enable_autolog_hparams: bool = True,
        model_registry: str | None = None,
        suggest_integrations: bool = True,
        # GPUlayer's own arguments
        target: str | None = None,
        gradient_sync: str = "auto",
        prefetch: bool = False,
        launch_options: dict[str, Any] | None = None,
    ) -> None:
        if model_registry is not None:
            raise ValidationFailed(
                "model_registry uploads checkpoints to Lightning AI's cloud registry, which "
                "GPUlayer does not use; checkpoints come back to default_root_dir"
            )
        accelerator = accelerator_name(accelerator)
        if accelerator == "gpu":  # Lightning's alias
            accelerator = "cuda"
        plugged = apply_plugins(plugins) if plugins is not None else {}
        precision = plugged.get("precision", precision)
        sync_batchnorm = plugged.get("sync_batchnorm", sync_batchnorm)
        self.checkpoint_io: CheckpointIO = plugged.get("checkpoint_io") or TorchCheckpointIO()
        self._custom_scaler = plugged.get("scaler")
        self._precision_plugin = plugged.get("precision_plugin")
        if max_steps == -1:
            max_steps = None
        if max_epochs == -1 or (
            max_epochs is None and (max_steps is not None or max_time is not None)
        ):
            max_epochs = UNLIMITED_EPOCHS  # another limit ends the run
        elif max_epochs is None:
            max_epochs = max(1, min_epochs or 1)  # Lightning's default is 1000
        # On TPUs, "auto" means every TPU device of the VM (counted by torch_xla when the
        # processes start).
        self._all_devices = devices in ("auto", -1)
        if devices in ("auto", -1):
            devices = max(1, torch.cuda.device_count())
        if not isinstance(devices, int):
            raise ValidationFailed(
                "devices must be a number of processes per node, 'auto' or -1; to pick GPU "
                "indices, set CUDA_VISIBLE_DEVICES or use launch_options={'gpus': [...]}"
            )
        if min(devices, num_nodes, max_epochs, accumulate_grad_batches) < 1 or (
            check_val_every_n_epoch is not None and check_val_every_n_epoch < 1
        ):
            raise ValidationFailed(
                "devices, num_nodes, max_epochs, accumulate_grad_batches and "
                "check_val_every_n_epoch must be >= 1"
            )
        if max_steps is not None and max_steps < 1:
            raise ValidationFailed("max_steps must be >= 1 (or None/-1 for no limit)")
        if gradient_clip_algorithm not in ("norm", "value"):
            raise ValidationFailed(
                "gradient_clip_algorithm must be 'norm' or 'value', "
                f"got {gradient_clip_algorithm!r}"
            )
        if isinstance(max_time, (timedelta, dict, str)):
            parsed = _parse_duration(max_time)
            max_time = parsed.total_seconds() if parsed is not None else None
        if num_sanity_val_steps < -1 or (max_time is not None and max_time <= 0):
            raise ValidationFailed("num_sanity_val_steps must be >= -1 and max_time > 0")
        if barebones:
            # Everything that may cost speed is off: checkpoints, logging, summaries, bars.
            enable_checkpointing = enable_progress_bar = enable_model_summary = False
            logger = False
            log_every_n_steps = 0
            num_sanity_val_steps = 0
        self.barebones = barebones
        if fast_dev_run:
            # A quick end-to-end check: n training and n validation batches, one epoch, no
            # checkpoints. True means 1.
            n = 1 if fast_dev_run is True else int(fast_dev_run)
            if n < 1:
                raise ValidationFailed("fast_dev_run must be True or a number of batches >= 1")
            max_epochs, max_steps, enable_checkpointing = 1, None, False
            limit_train_batches, limit_val_batches, num_sanity_val_steps = n, n, 0
            limit_test_batches = limit_predict_batches = n
            log_every_n_steps = 1
        if overfit_batches:
            limit_train_batches = limit_val_batches = overfit_batches
        self.overfit_batches = overfit_batches
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.num_sanity_val_steps = num_sanity_val_steps
        self.fast_dev_run = bool(fast_dev_run)
        self.max_time = max_time
        if deterministic not in (True, False, "warn"):
            raise ValidationFailed("deterministic must be True, False or 'warn'")
        self.deterministic = deterministic
        if benchmark and deterministic:
            raise ValidationFailed(
                "benchmark=True picks kernels by speed; it conflicts with deterministic=True"
            )
        self.benchmark = benchmark
        self.prefetch = prefetch
        # Lightning's names: a prefix for the keys the Trainer itself logs ("epoch", ...);
        # whether the hyperparameters go to the loggers when fit starts. suggest_integrations
        # only prints tips in Lightning and is accepted for compatibility.
        self.log_key_prefix = log_key_prefix or ""
        self.enable_autolog_hparams = enable_autolog_hparams
        self.suggest_integrations = suggest_integrations
        self.received_sigterm = False
        self._sigterm_stop = False
        if gradient_sync not in GRADIENT_SYNC:
            raise ValidationFailed(
                f"gradient_sync must be one of {GRADIENT_SYNC}, got {gradient_sync!r}"
            )
        self.gradient_sync = gradient_sync
        self._strategy = resolve_strategy(strategy)
        self._strategy.trainer = self
        precision = normalize_precision(precision)
        if isinstance(self._strategy, FSDPStrategy) and precision == "16-mixed":
            raise ValidationFailed(
                "strategy='fsdp' supports precision '32-true' and 'bf16-mixed' (16-mixed needs a "
                "sharded gradient scaler that has not been validated yet)"
            )
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self._max_steps = max_steps
        self.min_steps = min_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch
        if val_check_interval is not None and (
            val_check_interval <= 0
            or (isinstance(val_check_interval, float) and val_check_interval > 1.0)
        ):
            raise ValidationFailed(
                "val_check_interval must be a number of batches (int >= 1) or a fraction of "
                "the epoch (0 < float <= 1)"
            )
        self.val_check_interval = val_check_interval
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.precision = precision
        self.target = target
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps
        self.limit_train_batches = 1.0 if limit_train_batches is None else limit_train_batches
        self.limit_val_batches = 1.0 if limit_val_batches is None else limit_val_batches
        self.limit_test_batches = 1.0 if limit_test_batches is None else limit_test_batches
        self.limit_predict_batches = 1.0 if limit_predict_batches is None else limit_predict_batches
        self.enable_checkpointing = enable_checkpointing
        self.inference_mode = inference_mode
        self.use_distributed_sampler = use_distributed_sampler
        self.detect_anomaly = detect_anomaly
        self.sync_batchnorm = sync_batchnorm
        if reload_dataloaders_every_n_epochs < 0:
            raise ValidationFailed("reload_dataloaders_every_n_epochs must be >= 0")
        self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.default_root_dir = as_path(default_root_dir)
        self.launch_options = launch_options or {}
        self.profiler: Profiler | None = resolve_profiler(profiler)

        # Loggers: None/True keep the built-in metrics.csv; False turns it off.
        self._csv = logger is not False
        if isinstance(logger, Logger):
            self.loggers: list[Logger] = [logger]
        elif isinstance(logger, (list, tuple)):
            self.loggers = list(logger)
        else:
            self.loggers = []

        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callbacks: list[Callback] = list(callbacks or [])
        if enable_progress_bar and not any(isinstance(c, ProgressBar) for c in self.callbacks):
            self.callbacks.append(ProgressBar())
        if enable_model_summary and not any(isinstance(c, ModelSummary) for c in self.callbacks):
            # With a RichProgressBar the summary is a rich table too, as in Lightning.
            rich = any(isinstance(c, RichProgressBar) for c in self.callbacks)
            self.callbacks.insert(0, RichModelSummary() if rich else ModelSummary())
        self.callbacks = _checkpoints_last(self.callbacks)
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary

        self.state = TrainerState()
        self.callback_metrics: dict[str, float] = {}
        self.logged_metrics: dict[str, float] = {}
        self.progress_bar_metrics: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
        # Validation results within epochs (val_check_interval), with their step and epoch.
        self.val_history: list[dict[str, float]] = []
        # Step rows for metrics.csv, taken when the step log line is printed.
        self._step_rows: list[dict[str, float]] = []
        self.global_step = 0
        self.current_epoch = 0
        self.launch_result: LaunchResult | None = None
        self.runtime: Runtime | None = None
        self.checkpoint_path: Path | None = None
        self.should_stop = False  # set by a callback (e.g. EarlyStopping) to end training
        self.interrupted = False
        self.ckpt_path: str | Path | None = None
        self.num_training_batches: int | float = math.inf
        self.num_val_batches: list[int | float] = []
        self.is_last_batch = False
        self.datamodule: DataModule | None = None
        self.train_dataloader: Any = None
        self.val_dataloaders: Any = None
        self.test_dataloaders: Any = None
        self.predict_dataloaders: Any = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.lr_scheduler_configs: list[LRSchedulerConfig] = []
        self._out_dir: Path | None = None
        self._model: Module | None = None
        self._ddp: Any = None
        self._is_log_step = False
        self._stepped = False
        self._train_epoch_log: dict[str, float] = {}
        self._last_batch_idx = -1
        self._epoch_rng: dict[str, Any] | None = None
        self._restored_rng: dict[str, Any] | None = None
        self._restored_loaders: list[Any] | None = None
        self._eval_log_step = 0
        self._weights_changed_at_end = False
        self._predictions: list[Any] = []
        self._predict_indices: list[Any] | None = None
        self.predict_batch_indices: Any = None
        self._unscaled: set[int] = set()
        self._hook_cache: dict[str, list[Callable[..., Any]]] = {}
        self._batch_counter = 0
        self._total_batches = 0  # training batches run in this fit (all epochs)
        self._custom_opt = (False, False, False)
        self._step_scheds = False
        self._manual_opts: list[_ManualOptimizer] = []
        self._pending_monitors: list[Callback] = []
        self._train_src: Any = None
        self._eval_calls: dict[str, int] = defaultdict(int)
        self._ds_engine: Any = None
        self._reloaded_train: Any = None
        self._num_test_batches: list[int | float] = []
        self._num_predict_batches: list[int | float] = []

    # ---------------------------------------------------------------- properties

    @property
    def strategy(self) -> Strategy:
        """The strategy object (``DDPStrategy``, ``FSDPStrategy``, ...) from ``strategy=``."""
        return self._strategy

    @property
    def max_steps(self) -> int:
        """The limit on optimizer steps; -1 for none, as in Lightning."""
        return -1 if self._max_steps is None else self._max_steps

    @max_steps.setter
    def max_steps(self, value: int | None) -> None:
        self._max_steps = None if value in (None, -1) else value

    @property
    def lightning_module(self) -> Module | None:
        """The Module being trained: Lightning's name for ``module``."""
        return self.module

    @property
    def val_check_batch(self) -> int | float:
        """Training batches between validations."""
        n = self.num_training_batches
        every = self._val_every(None if n == math.inf else int(n))
        return every if every is not None else n

    @property
    def fit_loop(self) -> _FitLoop:
        """Lightning's most used loop state; ``fit_loop.max_epochs`` can be set."""
        return _FitLoop(self)

    @property
    def validate_loop(self) -> _EvaluationLoop:
        """Progress of the last validation."""
        return _EvaluationLoop(self.num_val_batches)

    @property
    def test_loop(self) -> _EvaluationLoop:
        """Progress of the last test."""
        return _EvaluationLoop(self.num_test_batches)

    @property
    def predict_loop(self) -> _EvaluationLoop:
        """Progress of the last prediction."""
        return _EvaluationLoop(self.num_predict_batches)

    @property
    def module(self) -> Module | None:
        """The Module being trained or evaluated (unwrapped; ``model`` is what the loop calls)."""
        return self._model

    @property
    def model(self) -> Any:
        """The Module as the loop calls it: wrapped in DDP when distributed, else the Module."""
        return self._ddp if self._ddp is not None else self._model

    @property
    def logger(self) -> Logger | None:
        """The first logger, or None."""
        return self.loggers[0] if self.loggers else None

    @property
    def log_dir(self) -> str | None:
        """The first logger's directory, else the run's output directory."""
        if self.loggers and self.loggers[0].log_dir:
            return self.loggers[0].log_dir
        return str(self._out_dir) if self._out_dir is not None else None

    @property
    def is_global_zero(self) -> bool:
        """True in the process with global rank 0 (and in a single process)."""
        return self.runtime.is_global_zero if self.runtime else True

    @property
    def global_rank(self) -> int:
        """This process's rank among all processes (0 in a single process)."""
        return self.runtime.global_rank if self.runtime else 0

    @property
    def local_rank(self) -> int:
        """This process's rank on its machine."""
        return self.runtime.local_rank if self.runtime else 0

    @property
    def node_rank(self) -> int:
        """This machine's rank among the machines of the job."""
        return self.runtime.node_rank if self.runtime else 0

    @property
    def world_size(self) -> int:
        """Number of processes training together."""
        return self.runtime.world_size if self.runtime else 1

    @property
    def num_devices(self) -> int:
        """Processes (devices) per node, from ``devices=``."""
        return int(self.devices)

    @property
    def training(self) -> bool:
        """True while training batches run."""
        return self.state.stage == "train"

    @property
    def validating(self) -> bool:
        """True while validation runs (inside fit or ``validate``)."""
        return self.state.stage == "validate"

    @property
    def testing(self) -> bool:
        """True while ``test`` runs."""
        return self.state.stage == "test"

    @property
    def predicting(self) -> bool:
        """True while ``predict`` runs."""
        return self.state.stage == "predict"

    @property
    def sanity_checking(self) -> bool:
        """True during the validation check before training."""
        return self.state.stage == "sanity_check"

    @property
    def checkpoint_callbacks(self) -> list[Checkpoint]:
        """Every checkpointing callback (``ModelCheckpoint``, ``OnExceptionCheckpoint``, ...)."""
        return [c for c in self.callbacks if isinstance(c, Checkpoint)]

    @property
    def checkpoint_callback(self) -> ModelCheckpoint | None:
        """The first ``ModelCheckpoint``, or None."""
        found = [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]
        return found[0] if found else None

    @property
    def early_stopping_callbacks(self) -> list[EarlyStopping]:
        """Every ``EarlyStopping`` among the callbacks."""
        return [c for c in self.callbacks if isinstance(c, EarlyStopping)]

    @property
    def early_stopping_callback(self) -> EarlyStopping | None:
        """The first ``EarlyStopping``, or None."""
        found = self.early_stopping_callbacks
        return found[0] if found else None

    # ------------------------------------------------ Lightning's other members

    @property
    def device_ids(self) -> list[int]:
        """Index of this process's device (one per process, as GPUlayer starts one process
        per GPU)."""
        rt = self.runtime
        if rt is not None and rt.device.index is not None:
            return [rt.device.index]
        return [0] if rt is None or rt.device.type != "cpu" else []

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        """What the Trainer gives ``DistributedSampler`` (``None`` in one process)."""
        rt = self.runtime
        if rt is None or not rt.distributed:
            return None
        return {"num_replicas": rt.world_size, "rank": rt.global_rank}

    @property
    def enable_validation(self) -> bool:
        """True when ``fit`` will validate (validation data and batches to run)."""
        return (
            self.val_dataloaders is not None
            and self.limit_val_batches != 0
            and self.check_val_every_n_epoch is not None
        )

    @property
    def evaluating(self) -> bool:
        """True while validating or testing."""
        return self.validating or self.testing

    @property
    def num_sanity_val_batches(self) -> list[int | float]:
        """Validation batches the sanity check runs, per loader."""
        loaders = self.val_dataloaders
        if loaders is None or not self.num_sanity_val_steps:
            return []
        out: list[int | float] = []
        for loader in loaders if isinstance(loaders, list) else [loaders]:
            n = _len(loader) or math.inf
            out.append(n if self.num_sanity_val_steps == -1 else min(self.num_sanity_val_steps, n))
        return out

    @property
    def num_test_batches(self) -> list[int | float]:
        """Test batches per loader of the last ``test``."""
        return self._num_test_batches

    @property
    def num_predict_batches(self) -> list[int | float]:
        """Prediction batches per loader of the last ``predict``."""
        return list(self._num_predict_batches)

    @property
    def precision_plugin(self) -> Any:
        """The precision plugin in use (a ``Precision`` for plain precision strings)."""
        from gpulayer.pytorch.plugins import Precision

        return self._precision_plugin or Precision(self.precision)

    @property
    def scaler(self) -> Any:
        """The gradient scaler with ``16-mixed`` (else None)."""
        return self.runtime.scaler if self.runtime is not None else self._custom_scaler

    @property
    def progress_bar_callback(self) -> ProgressBar | None:
        """The progress bar callback, if any."""
        return next((cb for cb in self.callbacks if isinstance(cb, ProgressBar)), None)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """``print`` from the first process of each machine only."""
        rt = self.runtime
        if rt is None or rt.local_rank == 0:
            print(*args, **kwargs)

    @contextlib.contextmanager
    def init_module(self, empty_init: bool | None = None) -> Iterator[None]:
        """Create the model inside this block to build its tensors on the training device and
        in the precision's dtype (``"bf16-true"`` and the other "-true" precisions) directly,
        without a copy in CPU memory first. With ``empty_init=True`` the weights are left
        uninitialised (faster for a model whose checkpoint is loaded next).

        In the launching process of a job (several devices, machines or a target), and on
        TPUs, tensors stay on the CPU: the model is built again where training runs."""
        device = torch.device("cpu")
        if not self.will_launch and not self._uses_tpu() and not self._xla_spawn_wanted():
            accel = resolve_accelerator(self.accelerator)
            if accel in ("cuda", "rocm"):
                count = torch.cuda.device_count()
                device = torch.device(
                    "cuda", int(os.environ.get("LOCAL_RANK", "0")) if count > 1 else 0
                )
            elif accel == "mps":
                device = torch.device("mps")
        from gpulayer.pytorch.runtime import _TRUE_DTYPES

        dtype_name = _TRUE_DTYPES.get(self.precision)
        previous = torch.get_default_dtype()
        with contextlib.ExitStack() as stack:
            stack.enter_context(device)
            if dtype_name is not None:
                torch.set_default_dtype(getattr(torch, dtype_name))
                stack.callback(torch.set_default_dtype, previous)
            if empty_init:
                stack.enter_context(_EmptyInit())
            yield

    def _xla_spawn_wanted(self) -> bool:
        return self._uses_tpu() and (self._all_devices or self.devices > 1)

    @property
    def estimated_stepping_batches(self) -> int | float:
        """Optimizer steps this fit plans (``inf`` when the loader's length is unknown)."""
        n = self._planned_steps(self.train_dataloader)
        return n if n else math.inf

    # --------------------------------------------------------------- placement

    @property
    def will_launch(self) -> bool:
        """True when ``fit`` will run on GPUlayer workers instead of this process."""
        if self._uses_tpu():
            return False  # several TPU devices: xmp.spawn in this process (_fit_xla_spawn)
        return (
            is_launcher_process()
            and not self._notebook_fork()
            and (self.target is not None or self.devices * self.num_nodes > 1)
        )

    def _notebook_fork(self) -> bool:
        """Several devices of this machine from a notebook: its ranks are forked, since cells
        cannot be submitted as a script."""
        return (
            self.target is None
            and self.devices > 1
            and self.num_nodes == 1
            and is_launcher_process()
            and in_notebook()
        )

    def _fit_notebook(self, model: Module, *args: Any) -> None:
        """Train in one forked process per device, then load what rank 0 saved, as after a
        launched job. A fork inherits the cells' classes and data, so nothing is pickled."""
        accelerator = resolve_accelerator(self.accelerator)
        if accelerator == "mps":
            raise ValidationFailed("forked processes cannot use MPS; use accelerator='cpu'")
        if sys.platform == "darwin":
            # A Jupyter kernel has threads, and on macOS a child forked from one crashes in
            # gloo: an Objective-C abort, or with OBJC_DISABLE_INITIALIZE_FORK_SAFETY a segfault.
            raise ValidationFailed(
                "macOS crashes processes forked from a Jupyter kernel: use devices=1 here, or "
                "export the notebook to a .py file and run it, which starts real processes"
            )
        if torch.cuda.is_initialized():
            raise ValidationFailed(
                "CUDA was already used in this notebook, and a forked process cannot use it: "
                "restart the kernel and keep CUDA calls out of the cells before fit"
            )
        out = Path(self.default_root_dir) / "notebook"
        shutil.rmtree(out / "checkpoints", ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        from gpulayer.runtime.distributed import free_port

        port = free_port("127.0.0.1")
        context = multiprocessing.get_context("fork")
        ranks = [
            context.Process(
                target=_notebook_rank, args=(r, self.devices, port, out, self, model, args)
            )
            for r in range(self.devices)
        ]
        for rank in ranks:
            rank.start()
        failed = _wait_for_ranks(ranks)
        if failed is not None:
            raise GpuLayerError(f"a forked training process failed (exit {failed.exitcode})")
        self._load_outputs(out, model)

    def _uses_tpu(self) -> bool:
        return self.accelerator in ("tpu", "xla") or (
            self.accelerator == "auto" and os.environ.get("PJRT_DEVICE") == "TPU"
        )

    def _xla_spawn_needed(self) -> bool:
        """True in the launching process when the run wants several TPU devices: the
        Trainer then starts one process per device with torch_xla's xmp.spawn, as
        Lightning's XLAStrategy does."""
        if not self._uses_tpu() or os.environ.get(XLA_SPAWNED_ENV) == "1":
            return False
        if self.num_nodes > 1 or self.target is not None:
            raise ValidationFailed(
                "TPU training runs on the TPU VMs it is started on, so num_nodes and target= "
                "are not used with accelerator='tpu': on a TPU pod, start the script on every "
                "host (as with Lightning: gcloud compute tpus tpu-vm ssh ... --worker=all) and "
                "torch_xla joins them into one run"
            )
        return is_launcher_process() and (self._all_devices or self.devices > 1)

    def _fit_xla_spawn(self, model: Module, *args: Any) -> None:
        """Run ``fit`` in one process per TPU device and load rank 0's results here."""
        import torch_xla.distributed.xla_multiprocessing as xmp

        out = (self.default_root_dir / _run_name()).resolve()
        env = {XLA_SPAWNED_ENV: "1", "GPULAYER_OUTPUT_DIR": str(out)}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            # nprocs=None: one process per TPU device (PJRT supports only all devices or 1).
            # fork, as Lightning's XLA launcher, so the model and loaders need no pickling.
            xmp.spawn(_xla_spawned_fit, args=(self, model, *args), nprocs=None, start_method="fork")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._load_outputs(out, model)

    def _launch(self, model: Module, ckpt_path: str | Path | None = None) -> None:
        result = self._launch_job(ckpt_path)
        self._load_outputs(result.output_dir, model)

    def _load_outputs(self, output_dir: Path, model: Module) -> None:
        """Load the final weights and metrics that the training processes wrote."""
        ckpts = sorted(output_dir.rglob("last.ckpt"))
        if len(ckpts) > 1 and all(c.is_dir() for c in ckpts):
            # A sharded checkpoint written on several machines comes back as one folder per
            # node, each with that node's shard files: put them together, so it loads (and
            # resumes) without a shared filesystem.
            merged = output_dir / "last.ckpt"
            merged.mkdir(exist_ok=True)
            for part in ckpts:
                if part != merged:
                    for f in part.rglob("*"):
                        if f.is_file():
                            dest = merged / f.relative_to(part)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            if not dest.exists():
                                shutil.copy2(f, dest)
            ckpts = [merged]
        state: dict[str, Any] | None = None
        if ckpts:
            self.checkpoint_path = ckpts[0]  # node0/... sorts first
            try:
                state = self.checkpoint_io.load_checkpoint(self.checkpoint_path)
            except BaseException as e:  # torch's CheckpointException is a BaseException
                if (
                    isinstance(e, (KeyboardInterrupt, SystemExit))
                    or not self.checkpoint_path.is_dir()
                ):
                    raise
                # Should not happen after the merge above; keep the metrics if it does.
                self._say(f"[gpulayer] sharded checkpoint not loaded here ({e}); metrics only")
        elif is_remote(self.default_root_dir) or any(
            is_remote(getattr(cb, "dirpath", None)) for cb in self.checkpoint_callbacks
        ):
            # The workers wrote checkpoints to the store, so they are not among the outputs.
            self.checkpoint_path = self._last_checkpoint()
            if self.checkpoint_path is not None:
                state = self.checkpoint_io.load_checkpoint(self.checkpoint_path)
        if state is not None:
            model.load_state_dict(state["state_dict"], strict=model.strict_loading)
            self.history = state.get("history", [])
            self.callback_metrics = state.get("callback_metrics") or _metrics_of(self.history)
            self.global_step = state.get("global_step", 0)
            self.current_epoch = state.get("next_epoch", state.get("epoch", 0))
        else:
            # Without checkpoints, the metrics still come back (weights do not).
            found = sorted(output_dir.rglob("metrics.json"))
            if found:
                self.history = json.loads(found[0].read_text())
                self.callback_metrics = _metrics_of(self.history)
                if self.history:
                    self.current_epoch = int(self.history[-1].get("epoch", 0)) + 1

    def _launch_job(self, ckpt_path: str | Path | None = None) -> LaunchResult:
        """Submit this script as a job on GPUlayer workers and wait for it."""
        options = dict(self.launch_options)
        if ckpt_path is not None and not is_remote(ckpt_path):
            # The checkpoint is on this machine: ship it with the job as an input; the copy of
            # this script on the workers finds it there (see _resume_path).
            local = Path(ckpt_path)
            if local.is_dir():
                # A sharded checkpoint folder: ship each file; every node gets them all.
                shipped = {
                    f"{RESUME_INPUT}.d/{f.relative_to(local).as_posix()}": f
                    for f in local.rglob("*")
                    if f.is_file()
                }
                options["inputs"] = {**options.get("inputs", {}), **shipped}
            elif local.is_file():
                options["inputs"] = {**options.get("inputs", {}), RESUME_INPUT: local}
            else:
                raise ValidationFailed(f"checkpoint not found: {local}")
        client = options.pop("client", None) or default_client()
        target = self.target or default_target(client)
        accel = self.accelerator
        if accel == "auto":
            if target != "local":
                raise ValidationFailed(
                    f"accelerator='auto' cannot be resolved for target '{target}' before the "
                    "job is placed; pass accelerator='cuda', 'mps' or 'cpu'"
                )
            accel = resolve_accelerator("auto")
        result = launch(
            nodes=self.num_nodes,
            devices=self.devices,
            accelerator=accel,
            target=target,
            client=client,
            output_dir=None if is_remote(self.default_root_dir) else self.default_root_dir,
            exit=False,
            **options,
        )
        assert result is not None
        self.launch_result = result
        if not result.succeeded:
            raise RemoteError(
                f"remote run {result.status.state.value}: {explain_failure(result.status)}",
                result.job,
            )
        return result

    # ------------------------------------------------------------------ hooks

    def _build_hooks(self, model: Module) -> None:
        """Which hooks are overridden, decided once per run (nothing is called per step for a
        hook nobody overrides)."""
        cache: dict[str, list[Callable[..., Any]]] = {}
        names = [n for n in dir(Callback) if n.startswith("on_") or n in ("setup", "teardown")]
        for name in names:
            fns: list[Callable[..., Any]] = []
            for cb in self.callbacks:
                if overrides(cb, name):
                    fn = getattr(cb, name)
                    if name == "on_train_batch_end" and legacy_batch_end(cb):
                        fns.append(_legacy_batch_end(fn, self, model))
                    else:
                        fns.append(_with_trainer(fn, self, model))
            if overridden(model, name, Module):
                fns.append(getattr(model, name))
            cache[name] = fns
        self._hook_cache = cache
        self._custom_opt = (
            overridden(model, "configure_gradient_clipping", Module),
            overridden(model, "optimizer_step", Module),
            overridden(model, "optimizer_zero_grad", Module),
        )

    def _call_hook(self, name: str, *args: Any) -> None:
        with _no_log(self._model, name):
            for fn in self._hook_cache.get(name, ()):
                fn(*args)

    def _call_epoch_end(self, name: str, model: Module) -> dict[str, float]:
        """``on_*_epoch_end``: other callbacks, then the model (whose ``self.log`` calls are
        kept), then the monitoring callbacks, which see what the model logged (as in
        Lightning)."""
        model._gl_step_logs = {}
        monitors = (EarlyStopping, ModelCheckpoint)
        ran = overridden(model, name, Module)
        for cb in self.callbacks:
            if not isinstance(cb, monitors) and overrides(cb, name):
                getattr(cb, name)(self, model)
                ran = True
        if overridden(model, name, Module):
            getattr(model, name)()
        rt = self.runtime
        logs = {k: [v] for k, v in model._gl_step_logs.items()}
        if ran and rt is not None and rt.distributed:
            # Every rank ran the same hooks, so they all reach this collective.
            extra = self._aggregate(rt, logs, {}, model._gl_log_options, 1)
        else:
            extra = {k: float(v[0]) for k, v in logs.items()}
        extra.update(_compute_metrics(model._gl_metrics))
        model._gl_step_logs = {}
        if extra:
            self.callback_metrics.update(extra)
            opts = model._gl_log_options
            self.progress_bar_metrics.update(
                {k: v for k, v in extra.items() if k in opts and opts[k].prog_bar}
            )
        self._pending_monitors = [
            cb for cb in self.callbacks if isinstance(cb, monitors) and overrides(cb, name)
        ]
        return extra

    def _call_monitors(self, name: str, model: Module) -> None:
        for cb in getattr(self, "_pending_monitors", []):
            getattr(cb, name)(self, model)
        self._pending_monitors = []

    def _merge_model_callbacks(self, model: Module) -> None:
        extra = model.configure_callbacks()
        if isinstance(extra, Callback):
            extra = [extra]
        extra = list(extra or [])
        if not extra:
            return
        kinds = {type(c) for c in extra}
        kept = [c for c in self.callbacks if type(c) not in kinds]
        self.callbacks = _checkpoints_last(kept + extra)

    # ---------------------------------------------------------------- loaders

    def _source(self, given: Any, name: str, model: Module | None) -> Any:
        """A loader given to fit/validate/test/predict, or the DataModule's or model's."""
        if given is not None:
            return given
        dm = self.datamodule
        hook = f"{name}_dataloader"
        if dm is not None and overridden(dm, hook, DataModule):
            return getattr(dm, hook)
        if model is not None and overridden(model, hook, Module):
            return getattr(model, hook)
        return None

    def _prepare_loader(self, rt: Runtime, loader: Any, shuffle: bool | None = None) -> Any:
        """The training loader: one loader, or a list/dict of loaders (or a CombinedLoader),
        each sharded across processes and iterated together (``max_size_cycle`` by default)."""
        if loader is None:
            return None
        loader = _materialize(loader)
        if isinstance(loader, CombinedLoader):
            loader.flattened = [self._prepare_one(rt, x, shuffle) for x in loader.flattened]
            return loader
        if isinstance(loader, dict) or (isinstance(loader, (list, tuple)) and len(loader) > 1):
            parts = (
                {k: self._prepare_one(rt, v, shuffle) for k, v in loader.items()}
                if isinstance(loader, dict)
                else [self._prepare_one(rt, x, shuffle) for x in loader]
            )
            return CombinedLoader(parts, mode="max_size_cycle")
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        return self._prepare_one(rt, loader, shuffle)

    def _prepare_loaders(self, rt: Runtime, src: Any) -> list[Any] | None:
        """Validation, test or predict loaders: always a list (one entry per dataloader_idx)."""
        if src is None:
            return None
        src = _materialize(src)
        items = loaders_of(src)
        return [self._prepare_one(rt, _materialize(x), False) for x in items]

    def _prepare_one(self, rt: Runtime, loader: Any, shuffle: bool | None = None) -> Any:
        if self.overfit_batches and isinstance(loader, DataLoader) and shuffle is not False:
            loader = _without_shuffle(loader)
        if self.use_distributed_sampler and isinstance(loader, DataLoader):
            loader = rt.dataloader(loader, shuffle=shuffle)
        return loader

    def _device_batches(
        self, rt: Runtime, model: Module, loader: Any, prefetch: bool, dataloader_idx: int = 0
    ) -> Iterator[Any]:
        """Batches on the device, through the model's and DataModule's transfer hooks when
        they are overridden (then without prefetch)."""
        dm = self.datamodule
        custom = (
            overridden(model, "transfer_batch_to_device", Module)
            or overridden(model, "on_before_batch_transfer", Module)
            or overridden(model, "on_after_batch_transfer", Module)
            or (dm is not None and overridden(dm, "on_before_batch_transfer", DataModule))
            or (dm is not None and overridden(dm, "on_after_batch_transfer", DataModule))
        )
        if not custom:
            yield from rt.batches(loader, prefetch=prefetch)
            return
        i = dataloader_idx
        for batch in loader:
            if dm is not None:
                batch = dm.on_before_batch_transfer(batch, i)
            batch = model.on_before_batch_transfer(batch, i)
            batch = model.transfer_batch_to_device(batch, rt.device, i)
            if rt.param_dtype is not None:
                batch = rt._to_device(batch)
            if dm is not None:
                batch = dm.on_after_batch_transfer(batch, i)
            yield model.on_after_batch_transfer(batch, i)

    # --------------------------------------------------------------- fitting

    def fit(
        self,
        model: Module,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: DataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        """Train ``model``. The loaders may be given as functions that return them, or come
        from a ``DataModule`` or the model's ``train_dataloader``/``val_dataloader``: then
        they are built only in the processes that train. When the Trainer launches a job, the
        launching process only waits, so data it built would be thrown away (measured: a 6 s
        dataset built for nothing made a 2-GPU run look 25% slower per step)."""
        model = self._unwrap(model)
        if isinstance(train_dataloaders, DataModule):
            datamodule, train_dataloaders = train_dataloaders, None
        if datamodule is not None and (
            train_dataloaders is not None or val_dataloaders is not None
        ):
            raise ValidationFailed("pass either dataloaders or a datamodule, not both")
        self.datamodule = datamodule
        ckpt_path = self.ckpt_path = self._fit_checkpoint(ckpt_path)
        if self._xla_spawn_needed() or self._notebook_fork():
            self._merge_model_callbacks(model)
            start = self._fit_xla_spawn if self._xla_spawn_needed() else self._fit_notebook
            start(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
            for cb in self.callbacks:
                if overrides(cb, "on_fit_end"):
                    cb.on_fit_end(self, model)
            return
        if in_slurm_task() and self.devices * self.num_nodes != int(os.environ["SLURM_NTASKS"]):
            raise ValidationFailed(
                f"srun started {os.environ['SLURM_NTASKS']} tasks, but devices={self.devices} "
                f"and num_nodes={self.num_nodes} make {self.devices * self.num_nodes}: match "
                "them, with --ntasks-per-node=devices and --nodes=num_nodes"
            )
        if self.will_launch:
            # Training runs on workers (the callbacks run there); here only on_fit_end runs,
            # so callbacks can pick up what came back (e.g. ModelCheckpoint's best file).
            self._merge_model_callbacks(model)
            self._launch(model, ckpt_path)
            for cb in self.callbacks:
                if overrides(cb, "on_fit_end"):
                    cb.on_fit_end(self, model)
            return
        self.state = TrainerState("fit", None, "running")
        self._total_batches = self._eval_log_step = 0
        self.callback_metrics = {}
        self._merge_model_callbacks(model)
        if self.deterministic is True:
            # Same results on every run (with seed_everything), at some speed cost. cuBLAS
            # reads its setting when CUDA starts, so set it first.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        elif self.deterministic == "warn":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
        rt_precision = self.precision
        if isinstance(self._strategy, DeepSpeedStrategy):
            # DeepSpeed runs mixed precision itself; the inputs only need the model's dtype.
            rt_precision = {"16-mixed": "16-true", "bf16-mixed": "bf16-true"}.get(
                self.precision, self.precision
            )
        strat = self._strategy
        rt = Runtime(self.accelerator, rt_precision, self.gradient_sync, strat.ddp_kwargs())
        if isinstance(strat, DDPStrategy):
            rt.comm_strategy = strat
        rt.setup(backend=strat.process_group_backend, timeout=strat.timeout)
        if self._custom_scaler is not None:
            rt.scaler = self._custom_scaler
        self.runtime = rt
        anomaly = torch.autograd.set_detect_anomaly(True) if self.detect_anomaly else _NOOP
        self.received_sigterm = self._sigterm_stop = False
        previous_sigterm = None
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.signal(signal.SIGTERM, self._sigterm_handler)
        try:
            with anomaly:
                self._run_fit(rt, model, train_dataloaders, val_dataloaders, ckpt_path)
            self.state.status = "finished"
        except KeyboardInterrupt as e:
            # As in Lightning: Ctrl+C ends training gracefully (callbacks see it first).
            self.interrupted = True
            self.state.status = "interrupted"
            self._on_exception(model, e)
            self._say("[gpulayer] training interrupted")
        except BaseException as e:
            self.state.status = "interrupted"
            self._on_exception(model, e)
            raise
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
            model.__dict__.pop("training_step", None)
            self.checkpoint_io.teardown()
            self._finalize_loggers("success" if self.state.status == "finished" else "failed")
            rt.teardown()
        if is_launcher_process():
            # Same result as a submitted run: the trained model comes back on the CPU.
            model.cpu()

    def _on_exception(self, model: Module, e: BaseException) -> None:
        for cb in self.callbacks:
            if overrides(cb, "on_exception"):
                with contextlib.suppress(Exception):
                    cb.on_exception(self, model, e)
        if overridden(model, "on_exception", Module):
            with contextlib.suppress(Exception):
                model.on_exception(e)
        if self.datamodule is not None:
            with contextlib.suppress(Exception):
                self.datamodule.on_exception(e)

    def _finalize_loggers(self, status: str) -> None:
        if self.runtime is not None and not self.runtime.is_global_zero:
            return
        for lg in self.loggers:
            lg.finalize(status)

    @staticmethod
    def _unwrap(model: Any) -> Module:
        """A ``torch.compile``d Module is trained compiled, with the options it was compiled
        with: the Module's call is compiled in place (as ``nn.Module.compile`` does), so
        ``self(x)`` in the steps runs the compiled graph and the hooks see the Module."""
        orig = getattr(model, "_orig_mod", None)
        if isinstance(orig, Module):
            ctx = getattr(model, "dynamo_ctx", None)
            if ctx is not None:
                orig._compiled_call_impl = ctx(orig._call_impl)
            else:
                orig.compile()
            return orig
        if not isinstance(model, Module):
            raise ValidationFailed("Trainer expects a gpulayer.pytorch.Module")
        return model

    def _planned_steps(self, train: Any) -> int:
        """Optimizer steps this fit plans (0 when the loader's length is unknown)."""
        if self._max_steps is not None:
            return self._max_steps
        n = _len(train) if train is not None else None
        if n is None:
            return 0
        limit = _limit(self.limit_train_batches, n, "limit_train_batches")
        if limit is not None:
            n = min(n, limit)
        return self.max_epochs * n // max(1, self.accumulate_grad_batches)

    def _optimizers(
        self, model: Module
    ) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerConfig]]:
        with _no_log(model, "configure_optimizers"):
            cfg = model.configure_optimizers()
        monitor = None
        if isinstance(cfg, torch.optim.Optimizer):
            opts, scheds = [cfg], []
        elif isinstance(cfg, dict):
            if "optimizer" not in cfg:
                raise ValidationFailed("the dict from configure_optimizers needs 'optimizer'")
            opts = [cfg["optimizer"]]
            monitor = cfg.get("monitor")
            sched = cfg.get("lr_scheduler")
            scheds = [sched] if sched is not None else []
            if sched is not None and not isinstance(sched, dict) and "interval" in cfg:
                # GPUlayer 0.7 form: {"optimizer", "lr_scheduler", "interval"}
                scheds = [{"scheduler": sched, "interval": cfg["interval"]}]
        elif (
            isinstance(cfg, (list, tuple))
            and len(cfg) == 2
            and isinstance(cfg[0], (list, tuple))
            and all(isinstance(o, torch.optim.Optimizer) for o in cfg[0])
        ):
            opts = list(cfg[0])
            scheds = list(cfg[1]) if isinstance(cfg[1], (list, tuple)) else [cfg[1]]
        elif (
            isinstance(cfg, (list, tuple))
            and len(cfg) == 2
            and isinstance(cfg[0], torch.optim.Optimizer)
            and not isinstance(cfg[1], torch.optim.Optimizer)
        ):
            opts, scheds = [cfg[0]], [cfg[1]]  # (optimizer, scheduler)
        elif isinstance(cfg, (list, tuple)) and cfg and all(isinstance(d, dict) for d in cfg):
            opts = [d["optimizer"] for d in cfg]
            scheds = [d["lr_scheduler"] for d in cfg if d.get("lr_scheduler") is not None]
        elif (
            isinstance(cfg, (list, tuple))
            and cfg
            and all(isinstance(o, torch.optim.Optimizer) for o in cfg)
        ):
            opts, scheds = list(cfg), []
        else:
            raise ValidationFailed(
                "configure_optimizers must return an optimizer, a list of optimizers, "
                "([optimizers], [schedulers]), (optimizer, scheduler) or a dict with "
                "'optimizer' (and optionally 'lr_scheduler')"
            )
        configs = []
        for s in scheds:
            if isinstance(s, dict):
                if "scheduler" not in s:
                    raise ValidationFailed("a scheduler dict needs the key 'scheduler'")
                known = {f for f in LRSchedulerConfig.__dataclass_fields__}
                c = LRSchedulerConfig(**{k: v for k, v in s.items() if k in known})
            else:
                c = LRSchedulerConfig(s)
            if c.interval not in ("step", "epoch"):
                raise ValidationFailed("a scheduler's interval must be 'step' or 'epoch'")
            c.reduce_on_plateau = c.reduce_on_plateau or isinstance(
                c.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            )
            if c.reduce_on_plateau:
                c.monitor = c.monitor or monitor
                if c.monitor is None:
                    raise ValidationFailed(
                        "ReduceLROnPlateau needs a metric: add 'monitor': 'val_loss' to the "
                        "scheduler dict or to the dict returned by configure_optimizers"
                    )
            configs.append(c)
        if len(opts) > 1 and model.automatic_optimization:
            raise ValidationFailed(
                "several optimizers need manual optimization, as in Lightning: set "
                "self.automatic_optimization = False and call opt.step() in training_step"
            )
        return opts, configs

    def _say(self, msg: str) -> None:
        if self.is_global_zero:
            print(msg, flush=True)

    def _setup_model(self, rt: Runtime, model: Module, stage: str) -> None:
        """prepare_data (once per machine), then setup in the DataModule, callbacks and model."""
        model.trainer = self
        dm = self.datamodule
        if dm is not None:
            dm.trainer = self
        for owner, base in ((dm, DataModule), (model, Module)):
            rank = (
                rt.local_rank if getattr(owner, "prepare_data_per_node", True) else rt.global_rank
            )
            if owner is not None and rank == 0 and overridden(owner, "prepare_data", base):
                owner.prepare_data()
        rt.barrier()
        with _no_log(model, "setup"):
            if dm is not None:
                dm.setup(stage)
            for cb in self.callbacks:
                if overrides(cb, "setup"):
                    cb.setup(self, model, stage)
            model.setup(stage)

    def _teardown_model(self, model: Module, stage: str) -> None:
        if self.datamodule is not None:
            self.datamodule.teardown(stage)
        with _no_log(model, "teardown"):
            for cb in self.callbacks:
                if overrides(cb, "teardown"):
                    cb.teardown(self, model, stage)
            model.teardown(stage)

    def _run_fit(
        self,
        rt: Runtime,
        model: Module,
        train_src: Any,
        val_src: Any,
        ckpt_path: str | Path | None,
    ) -> None:
        self._setup_model(rt, model, "fit")
        if isinstance(self._strategy, ModelParallelStrategy):
            self._setup_model_parallel(rt, model)
        for hook in ("configure_sharded_model", "configure_model"):
            if overridden(model, hook, Module):
                with rt.device:
                    getattr(model, hook)()
        train_src = self._source(train_src, "train", model)
        if train_src is None:
            raise ValidationFailed(
                "fit needs training data: pass train_dataloaders, a datamodule, or define "
                "train_dataloader() on the model"
            )
        val_src = self._source(val_src, "val", model)
        self._train_src = train_src
        train = self._prepare_loader(rt, train_src)
        val = self._prepare_loaders(rt, val_src)
        self.train_dataloader = train
        self.val_dataloaders = val[0] if isinstance(val, list) and len(val) == 1 else val
        if rt.is_global_zero:
            _performance_hints(rt, loaders_of(train), self.num_devices)
        if rt.device.type == "cuda" and not self.deterministic:
            # cuDNN times its algorithms for each new input shape and keeps the fastest: 5 to
            # 9% faster CNN steps measured, for 1 to 3 s of timing at the start. Worth it only
            # on longer runs, so by default it is on from AUTOTUNE_MIN_STEPS planned steps.
            torch.backends.cudnn.benchmark = (
                self.benchmark
                if self.benchmark is not None
                else self._planned_steps(train) >= AUTOTUNE_MIN_STEPS
            )
        self._fit_loop(rt, model, train, val, _resume_path(ckpt_path))

    def _fit_loop(
        self,
        rt: Runtime,
        model: Module,
        train: Any,
        val: Any,
        ckpt_path: str | Path | None,
    ) -> None:
        model.trainer = self
        self._apply_precision_plugin(rt, model)
        model.to(rt.device)
        rt.convert_module(model)
        fsdp = isinstance(self._strategy, FSDPStrategy) and rt.distributed
        if self.sync_batchnorm and rt.distributed and not fsdp:
            if not rt.gpu:
                raise ValidationFailed(
                    "sync_batchnorm needs CUDA or ROCm GPUs (PyTorch's SyncBatchNorm does not "
                    "run on CPU processes)"
                )
            model = _convert_sync_batchnorm(model)
        sharded: torch.nn.Module | None = None
        if fsdp:
            # Shard before the optimizer, which must see the sharded parameters. The wrapper
            # (whose forward calls training_step) is the FSDP root, called every step.
            if next(model.parameters(recurse=False), None) is not None:
                raise ValidationFailed(
                    "strategy='fsdp' shards submodules; move parameters the Module owns "
                    "directly (nn.Parameter attributes) into a submodule such as nn.Linear"
                )
            sharded = rt.shard(_TrainingStep(model), self._strategy)
        optimizers, sched_configs = self._optimizers(model)
        if isinstance(self._strategy, DeepSpeedStrategy):
            if ckpt_path:
                raise ValidationFailed(
                    "resuming a DeepSpeed run from a checkpoint is not supported yet"
                )
            if len(optimizers) > 1 or not model.automatic_optimization:
                raise ValidationFailed(
                    "DeepSpeedStrategy supports one optimizer and automatic optimization"
                )
            try:
                import deepspeed
            except ImportError as e:
                raise ValidationFailed("strategy='deepspeed' needs: pip install deepspeed") from e
            import torch.distributed as dist

            if not dist.is_initialized():
                # DeepSpeed needs a process group even for one process (without one it tries
                # MPI discovery): start a one-process group, without rank variables that
                # would change how GPUlayer sees this process.
                from gpulayer.runtime.distributed import free_port

                dist.init_process_group(
                    "nccl" if rt.gpu else "gloo",
                    init_method=f"tcp://127.0.0.1:{free_port('127.0.0.1')}",
                    rank=0,
                    world_size=1,
                )
                rt._owns_group = True
            if "LOCAL_RANK" not in os.environ:
                # DeepSpeed reads LOCAL_RANK; set it for this run only (Runtime.teardown
                # removes it, so is_launcher_process() is unchanged afterwards).
                os.environ["LOCAL_RANK"] = str(rt.local_rank)
                rt._env_set.append("LOCAL_RANK")
            micro = getattr(train, "batch_size", None)
            cfg = self._strategy.deepspeed_config(
                self.precision, self.accumulate_grad_batches, self.gradient_clip_val, micro
            )
            engine, ds_opt, _, _ = deepspeed.initialize(
                model=_TrainingStep(model), optimizer=optimizers[0], config=cfg
            )
            self._ds_engine = engine
            optimizers = [ds_opt]
        if fsdp and len(optimizers) > 1:
            raise ValidationFailed("strategy='fsdp' supports one optimizer")
        averaging = getattr(self._strategy, "model_averaging_period", None)
        if averaging and rt.distributed:
            optimizers = _post_local_sgd(optimizers, self._strategy, rt)
        self.optimizers, self.lr_scheduler_configs = optimizers, sched_configs
        self._model, self.should_stop = model, False
        _take_dataloader_iter(model)
        automatic = model.automatic_optimization
        self._manual_opts = [_ManualOptimizer(o, self) for o in optimizers]

        # DDP only syncs gradients for calls to its forward, so wrap training_step for DDP;
        # FSDP hooks into each sharded submodule instead, and a single process needs neither
        # (one module call fewer per step). With manual optimization the step runs outside
        # DDP's forward: DDP only broadcasts the initial weights, and manual_backward averages
        # the gradients across ranks itself (no reliance on DDP's reducer internals).
        step_fn: Any
        if self._ds_engine is not None:
            ddp, step_fn = self._ds_engine, self._ds_engine
        elif sharded is not None:
            ddp, step_fn = sharded, sharded
        elif rt.distributed and isinstance(self._strategy, ModelParallelStrategy):
            # configure_model applied tensor parallelism and FSDP: the Module syncs itself.
            ddp, step_fn = model, model.training_step
        elif rt.distributed:
            ddp = rt.module(_TrainingStep(model))
            step_fn = ddp if automatic else model.training_step
        else:
            ddp, step_fn = model, model.training_step
        self._ddp = ddp if rt.distributed else None
        if not rt.distributed and self._ds_engine is None:
            model.to(rt.device)
        root = self.default_root_dir
        out_dir = rt.output_dir(root if is_remote(root) else root / _run_name())
        self._out_dir = out_dir
        self._build_hooks(model)
        if self.profiler is not None:
            self.profiler.setup("fit", rt.local_rank, str(out_dir))

        # Resume.
        start_epoch, skip_batches = self.current_epoch, 0  # a second fit continues
        if ckpt_path:
            start_epoch, skip_batches = self._restore(rt, model, ckpt_path)

        where = f"{rt.accelerator}" + (
            f", rank {rt.global_rank}/{rt.world_size}" if rt.distributed else ""
        )
        if sharded is None and rt.uses_two_level_sync():
            where += f", two-level gradient sync ({rt.local_world_size} per machine)"
        planned = f"{self.max_epochs} epoch(s)"
        if self.max_epochs == UNLIMITED_EPOCHS:
            planned = f"{self._max_steps} step(s)" if self._max_steps else "until stopped"
        self._say(f"[gpulayer] training on {where}, precision {rt.precision}, {planned}")
        if rt.is_global_zero and self.loggers and self.enable_autolog_hparams:
            hp = {
                **dict(model.hparams),
                **(dict(self.datamodule.hparams) if self.datamodule else {}),
            }
            for lg in self.loggers:
                lg.log_hyperparams(hp)

        self._call_hook("on_fit_start")
        fit_t0 = time.perf_counter()
        n_sanity = self.num_sanity_val_steps
        if val is not None and n_sanity and not ckpt_path:
            # Run a few validation batches first, so a broken validation_step fails now and
            # not after the first training epoch. Results are discarded.
            self.state.stage = "sanity_check"
            self._call_hook("on_sanity_check_start")
            self._evaluate(
                rt, model, val, "validation_step", limit=None if n_sanity == -1 else n_sanity
            )
            self._call_hook("on_sanity_check_end")
            self.callback_metrics = {}
            self.progress_bar_metrics = {}
        self.state.stage = "train"
        if not model.training:
            self._say(
                "[gpulayer] warning: the model is in eval mode at the start of training "
                "(dropout and BatchNorm updates are off); call model.train() before fit if "
                "that is not intended"
            )
        finished = self.current_epoch = start_epoch
        done = start_epoch >= self.max_epochs or (
            self._max_steps is not None and self.global_step >= self._max_steps
        )
        if done:
            self._say("[gpulayer] nothing to train: the limits were reached already")
            self.state.stage = None
            self._call_hook("on_fit_end")
            self._teardown_model(model, "fit")
            return
        self._call_hook("on_train_start")
        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch
            finished = epoch + 1
            n = self.reload_dataloaders_every_n_epochs
            if n and epoch > start_epoch and epoch % n == 0 and callable(self._train_src):
                train = self._prepare_loader(rt, self._train_src)
                self.train_dataloader = train
            if self._reloaded_train is not None:
                train, self._reloaded_train = self._reloaded_train, None
            for part in loaders_of(train):
                if isinstance(getattr(part, "sampler", None), DistributedSampler):
                    part.sampler.set_epoch(epoch)
            # No model.train() here, as in Lightning: submodules keep the mode they were given
            # (a frozen backbone stays in eval); evaluation restores each one's mode after.
            n_all = _len(train)
            n_lim = _limit(self.limit_train_batches, n_all, "limit_train_batches")
            n_run = n_all if n_lim is None else (min(n_all, n_lim) if n_all else n_lim)
            self.num_training_batches = n_run if n_run is not None else math.inf  # for hooks
            self._call_hook("on_train_epoch_start")
            self.state.stage = "train"
            t0 = time.perf_counter()
            metrics = self._train_epoch(
                rt, model, train, val, step_fn, ddp, automatic, fit_t0, skip_batches
            )
            if not self._sigterm_stop:  # every rank decides together at the epoch's end
                self._sigterm_stop = (
                    rt.any(self.received_sigterm) if rt.distributed else self.received_sigterm
                )
            if self._sigterm_stop:
                self._stop_on_sigterm(rt, out_dir)
            skip_batches = 0
            metrics["epoch"] = epoch
            metrics["epoch_seconds"] = round(time.perf_counter() - t0, 3)
            self.history.append(metrics)
            extra = self._call_epoch_end("on_train_epoch_end", model)
            metrics.update(extra)
            shown = " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k not in ("epoch", "epoch_seconds")
            )
            self._say(f"epoch {epoch} done in {metrics['epoch_seconds']}s: {shown}")
            self._log_to_loggers({**self._train_epoch_log, **self._for_loggers(extra, model)})
            self._call_monitors("on_train_epoch_end", model)
            # ReduceLROnPlateau steps after validation, on the monitored value.
            if automatic:
                self._step_schedulers("epoch", plateau=True)
            if self.enable_checkpointing:
                # Every rank calls it: with FSDP, gathering the full state is a collective.
                saved = self.save_checkpoint(Path(CHECKPOINT).name)
                if saved is not None:
                    self.checkpoint_path = saved
            if rt.is_global_zero:
                # Metrics are written even without checkpoints: a remote run returns them.
                self._write_metrics(out_dir)
            if self.max_time is not None and time.perf_counter() - fit_t0 >= self.max_time:
                self.should_stop = True
                self._say(f"[gpulayer] max_time of {self.max_time:g}s reached")
            if self._max_steps is not None and self.global_step >= self._max_steps:
                self.should_stop = True
            # Every rank must leave the loop together, or the next collective would hang.
            self.should_stop = rt.any(self.should_stop)
            rt.barrier()
            if self.should_stop and self._mins_met():
                break
        self.state.stage = None
        self.current_epoch = finished  # after fit it counts epochs, as in Lightning
        self._call_hook("on_train_end")
        if self._weights_changed_at_end and self.enable_checkpointing:
            # SWA, weight averaging or pruning replaced the weights after the last epoch.
            saved = self.save_checkpoint(Path(CHECKPOINT).name)
            if saved is not None:
                self.checkpoint_path = saved
        self._call_hook("on_fit_end")
        self._teardown_model(model, "fit")
        if self.profiler is not None:
            self.profiler.teardown("fit")
            if rt.is_global_zero:
                report = self.profiler.describe()
                if report:
                    print(report, flush=True)

    def _setup_model_parallel(self, rt: Runtime, model: Module) -> None:
        """ModelParallelStrategy: build the (data_parallel, tensor_parallel) mesh for
        ``configure_model`` and read data per data-parallel group."""
        if not overridden(model, "configure_model", Module):
            raise ValidationFailed(
                "ModelParallelStrategy needs configure_model() on the Module, applying tensor "
                "parallelism and fully_shard with self.device_mesh"
            )
        if not rt.distributed:
            return
        from torch.distributed.device_mesh import init_device_mesh

        dp, tp = self._strategy.mesh_shape(rt.world_size, rt.local_world_size)
        mesh = init_device_mesh(
            rt.device.type, (dp, tp), mesh_dim_names=("data_parallel", "tensor_parallel")
        )
        model.device_mesh = mesh
        rt.data_parallel = (dp, mesh["data_parallel"].get_local_rank())

    def _apply_precision_plugin(self, rt: Runtime, model: Module) -> None:
        """Layer-replacing precision plugins (bitsandbytes, Transformer Engine): convert the
        model once, before it moves to the device, and run forwards in their context."""
        plugin = self._precision_plugin
        if plugin is None:
            return
        if hasattr(plugin, "convert_module") and not model.__dict__.get("_gl_converted"):
            plugin.convert_module(model)
            model.__dict__["_gl_converted"] = True
        if hasattr(plugin, "forward_context"):
            rt.forward_context = plugin.forward_context

    def _stop_on_sigterm(self, rt: Runtime, out_dir: Path) -> None:
        """Save a checkpoint a resumed run continues from (mid-epoch when stopped mid-epoch),
        then end ``fit`` with ``SIGTERMException``, as Lightning does."""
        self._say(
            f"[gpulayer] SIGTERM: stopping at step {self.global_step}"
            + (", saving checkpoints/last.ckpt" if self.enable_checkpointing else "")
        )
        if self.enable_checkpointing:
            saved = self.save_checkpoint(Path(CHECKPOINT).name)
            if saved is not None:
                self.checkpoint_path = saved
        if rt.is_global_zero:
            self._write_metrics(out_dir)
        raise SIGTERMException(f"SIGTERM received; stopped at step {self.global_step}")

    def _sigterm_handler(self, signum: int, frame: Any) -> None:
        self.received_sigterm = True

    def _reconfigure_optimizers(self, model: Module) -> None:
        """Build the optimizers again (after a callback changed the learning rate)."""
        self.optimizers, self.lr_scheduler_configs = self._optimizers(model)
        self._manual_opts = [_ManualOptimizer(o, self) for o in self.optimizers]

    def _reload_train_loader(self) -> None:
        """Build the training loader again (after a callback changed the batch size)."""
        if self.runtime is not None and callable(self._train_src) or self.datamodule is not None:
            src = self._source(None if self.datamodule else self._train_src, "train", self._model)
            assert self.runtime is not None
            self.train_dataloader = self._prepare_loader(self.runtime, src)
            self._reloaded_train = self.train_dataloader

    def _mins_met(self) -> bool:
        if self.min_epochs is not None and self.current_epoch + 1 < self.min_epochs:
            return False
        return not (self.min_steps is not None and self.global_step < self.min_steps)

    def _train_epoch(
        self,
        rt: Runtime,
        model: Module,
        train: Any,
        val: Any,
        step_fn: Any,
        ddp: Any,
        automatic: bool,
        fit_t0: float,
        skip_batches: int,
    ) -> dict[str, float]:
        epoch = self.current_epoch
        # Logged values stay on the device (see Module.log) and are added up once per epoch,
        # not with one tensor operation per step.
        logged: dict[str, list[Any]] = defaultdict(list)
        weights: dict[str, list[float]] = defaultdict(list)
        count = 0
        accumulate = self.accumulate_grad_batches
        n_total = _len(train)
        limit = _limit(self.limit_train_batches, n_total, "limit_train_batches")
        n_batches = n_total if limit is None else (min(n_total, limit) if n_total else limit)
        self.num_training_batches = n_batches if n_batches is not None else math.inf
        val_every = self._val_every(n_batches) if val is not None else None
        log_every = self.log_every_n_steps
        prof = self.profiler
        hooks = self._hook_cache
        batch_start = hooks.get("on_train_batch_start", [])
        batch_end = hooks.get("on_train_batch_end", [])
        before_backward = hooks.get("on_before_backward", [])
        after_backward = hooks.get("on_after_backward", [])
        single = not rt.distributed
        # Lightning clears gradients before backward. That is the order when a model or callback
        # can see it. Otherwise they are freed as soon as nothing can read them, after
        # on_train_batch_end, so the next forward pass runs with less memory in use.
        zero_in_step = automatic and (bool(hooks.get("on_before_zero_grad")) or self._custom_opt[2])
        # Lightning steps schedulers only with automatic optimization; with manual
        # optimization the model calls sch.step() itself.
        self._step_scheds = automatic and any(
            c.interval == "step" for c in self.lr_scheduler_configs
        )
        plateau_step = automatic and any(
            c.interval == "step" and c.reduce_on_plateau for c in self.lr_scheduler_configs
        )
        if not automatic:
            model.__dict__["_gl_manual"] = self._manual_opts
        every = self.check_val_every_n_epoch
        # Validate within this epoch only on epochs check_val_every_n_epoch selects. With
        # check_val_every_n_epoch=None an integer val_check_interval counts batches across
        # epochs, as in Lightning, so it may be longer than one epoch.
        val_epoch = every is None or (epoch + 1) % every == 0
        count_across = every is None and isinstance(self.val_check_interval, int)
        validated_last: dict[str, float] | None = None
        restored_rng, self._restored_rng = self._restored_rng, None
        if restored_rng is not None:
            set_rng_states(restored_rng["epoch_start" if skip_batches else "now"])
        self._epoch_rng = restored_rng["epoch_start"] if skip_batches and restored_rng else (
            None if rt.distributed or rt.xla else _rng_states()
        )  # fmt: skip
        source, first_idx = self._resumed_source(train, skip_batches)
        stateful = any(_is_stateful(d) for d in loaders_of(train))
        batches: Any = self._device_batches(rt, model, source, self.prefetch and not stateful)
        if skip_batches and restored_rng is not None:
            # The sampler drew its permutation on the first batch; training continues with
            # the numbers it had when the checkpoint was written.
            batches = _after_first(batches, lambda: set_rng_states(restored_rng["now"]))
        if prof is not None:
            batches = _profiled(batches, prof, "train_dataloader_next")
        # A loader without a length gives no batch count: look one batch ahead to know the
        # last batch, so its accumulated gradients are applied in this epoch (Lightning).
        iter_step = "training_step" in model.__dict__  # takes dataloader_iter
        if iter_step:
            batches = _IterSteps(batches)
        ahead = _LookAhead(batches) if n_total is None and not iter_step else None
        if ahead is not None:
            batches = ahead
        loss: Any = None
        out: Any = None
        last = False
        self._batch_counter = 0
        for batch_idx, batch in enumerate(batches, first_idx):
            if limit is not None and batch_idx >= limit:
                break
            if batch_idx < skip_batches:
                continue  # resuming within an epoch, and the loader could not skip for us
            last = (n_batches is not None and batch_idx + 1 == n_batches) or (
                ahead is not None and ahead.last
            )
            self.is_last_batch = last
            self._total_batches += 1
            if batch_start:
                ended = False
                for fn in batch_start:
                    if fn(batch, batch_idx) == -1:
                        ended = True
                if ended:
                    last = True
                    break  # on_train_batch_start returned -1: end the epoch (as in Lightning)
            self._batch_counter = batch_idx + 1
            model._gl_step_logs = {}
            if batch_end:
                model._gl_in_step = True
            self._stepped = False
            if prof is not None:
                prof.start("run_training_batch")
            if self._ds_engine is not None:
                engine = self._ds_engine
                out = engine(batch, batch_idx)
                loss = out["loss"] if isinstance(out, dict) else out
                for fn in before_backward:
                    fn(loss)
                engine.backward(loss)  # DeepSpeed scales, accumulates and reduces
                for fn in after_backward:
                    fn()
                boundary = engine.is_gradient_accumulation_boundary()
                engine.step()
                if boundary:
                    self._stepped = True
                    self.global_step += 1
                    if self._step_scheds:
                        self._step_schedulers("step")
            elif automatic:
                # Apply the optimizer every `accumulate` batches (and on the last one). On the
                # other batches DDP skips its gradient all-reduce.
                apply = (batch_idx + 1) % accumulate == 0 or last
                with rt.no_sync(ddp, enabled=not apply):
                    with rt.autocast():
                        out = step_fn(batch, batch_idx)
                    loss = out["loss"] if isinstance(out, dict) else out
                    if loss is not None and not isinstance(loss, torch.Tensor):
                        raise ValidationFailed(
                            "training_step must return the loss tensor (or a dict with 'loss')"
                        )
                    if zero_in_step and batch_idx % accumulate == 0:
                        self._zero_grad(model, self.optimizers[0], batch_idx)
                    if loss is not None:
                        for fn in before_backward:
                            fn(loss)
                        if prof is not None:
                            prof.start("backward")
                        rt.backward(loss / accumulate if accumulate > 1 else loss)
                        if prof is not None:
                            prof.stop("backward")
                        for fn in after_backward:
                            fn()
                if apply:
                    if prof is not None:
                        prof.start("optimizer_step")
                    # None skips the batch, as in Lightning: the step still counts, and with
                    # no gradients an optimizer changes nothing. A GradScaler refuses to run.
                    if loss is not None or rt.scaler is None:
                        self._optimizer_step(rt, model, self.optimizers[0], batch_idx)
                    if prof is not None:
                        prof.stop("optimizer_step")
                    self._stepped = True
                    self.global_step += 1
                    if self._step_scheds:  # after the count, so frequency=N steps at N, 2N
                        self._step_schedulers("step")
            else:
                with rt.autocast():
                    out = step_fn(batch, batch_idx)
                loss = out["loss"] if isinstance(out, dict) else out
            if prof is not None:
                prof.stop("run_training_batch")
            logs = model._gl_step_logs or (
                {"train_loss": loss.detach().float()} if isinstance(loss, torch.Tensor) else {}
            )
            opts = model._gl_log_options
            size: float | None = None
            for k, v in logs.items():
                logged[k].append(v)
                o = opts.get(k)
                if o is not None and o[3] is not None:
                    weights[k].append(o[3])
                else:  # weight the epoch mean by the inferred batch size (Lightning)
                    if size is None:
                        size = _batch_size(batch)
                    weights[k].append(size)
            count += 1
            self._is_log_step = bool(
                self._stepped and log_every and self.global_step % log_every == 0
            )
            if self._is_log_step:
                values = {k: float(v) for k, v in logs.items() if k not in opts or opts[k].step}
                shown = " ".join(f"{k}={v:.4f}" for k, v in values.items())
                self._say(f"epoch {epoch} step {self.global_step} {shown}")
                sent = self._publish_step(values, opts)
                self._step_rows.append({"epoch": epoch, "step": self.global_step, **sent})
            if batch_end:
                model._gl_in_step = False
                if loss is None and automatic:
                    outputs: Any = {}
                elif isinstance(out, dict):
                    outputs = out
                else:
                    outputs = {"loss": loss.detach() if isinstance(loss, torch.Tensor) else loss}
                for fn in batch_end:
                    fn(outputs, batch, batch_idx)
            if self._stepped and not zero_in_step and automatic:
                self.optimizers[0].zero_grad(set_to_none=True)
            if val_every and val_epoch:
                if count_across:
                    seen = epoch * n_batches + batch_idx + 1 if n_batches else self._total_batches
                else:
                    seen = batch_idx + 1
                # With an interval, validation runs at its multiples only, as in Lightning.
                if seen % val_every == 0:
                    # Same batch index on every rank, so the collectives inside line up.
                    mid = self._run_validation(
                        rt, model, val, zero_grad=(batch_idx + 1) % accumulate == 0 or last
                    )
                    self.val_history.append({"epoch": epoch, "step": self.global_step, **mid})
                    self.state.stage = "train"
                    if last:
                        validated_last = mid
            if plateau_step and self._stepped:
                # ReduceLROnPlateau with interval="step": after any validation of this batch.
                self._step_schedulers("step", plateau=True)
            # SIGTERM (preemption, cancel): one process stops at once; several agree at a log
            # step, where values are read anyway, so no step pays for an extra synchronisation.
            if (single and self.received_sigterm) or (
                not single and self._is_log_step and rt.any(self.received_sigterm)
            ):
                self._sigterm_stop = True
                return {}  # keeps _batch_counter: the checkpoint resumes after this batch
            if self._max_steps is not None and self.global_step >= self._max_steps:
                break  # same step count on every rank, so no extra synchronisation
            if single and (
                self.max_time is not None and time.perf_counter() - fit_t0 >= self.max_time
            ):
                self.should_stop = True
            if single and self.should_stop and self._mins_met():
                break  # one process: stop now (several processes stop at the epoch end)
        self._last_batch_idx = self._batch_counter - 1
        self._batch_counter = 0
        if automatic:
            self._step_schedulers("epoch")
        metrics = self._aggregate(rt, logged, weights, model._gl_log_options, count)
        metrics.update(_compute_metrics(model._gl_metrics))
        self._train_epoch_log = self._publish_train_epoch(rt, metrics, logged, model)
        if validated_last is not None:
            metrics.update(validated_last)  # the last batch's validation was this epoch's
        elif val is not None and val_every is None and self._epoch_end_validation(epoch, last):
            metrics.update(self._run_validation(rt, model, val))
        return metrics

    def _resumed_source(self, train: Any, skip_batches: int) -> tuple[Any, int]:
        """What to iterate in the first epoch of a run resumed within an epoch, and the index
        of its first batch: a stateful loader continues by itself, and any other ``DataLoader``
        skips in its sampler, so trained batches are never read again. Other loaders are read
        and thrown away, by the caller."""
        restored, self._restored_loaders = self._restored_loaders, None
        if not skip_batches:
            return train, 0
        loaders = [d for d in loaders_of(train) if _is_stateful(d)]
        if restored and len(restored) == len(loaders):
            for loader, state in zip(loaders, restored, strict=True):
                loader.load_state_dict(state)
            return train, skip_batches
        skipping = _skipping_loader(train, skip_batches)
        return (train, 0) if skipping is None else (skipping, skip_batches)

    def _epoch_end_validation(self, epoch: int, last_batch: bool) -> bool:
        """Validate after the last batch of an epoch ``check_val_every_n_epoch`` selects, or
        when stopping early. ``max_steps`` ending the run within an epoch does not validate."""
        every = self.check_val_every_n_epoch
        selected = every is not None and (epoch + 1) % every == 0
        return selected and (last_batch or self.should_stop)

    def _publish_step(self, values: dict[str, float], opts: dict[str, Any]) -> dict[str, float]:
        """Send a log step's values where ``self.log`` asked; returns what the loggers got."""
        sent: dict[str, float] = {}
        for k, v in values.items():
            o = opts.get(k)
            key = f"{k}_step" if o is not None and o.on_epoch else k
            self.callback_metrics[k] = self.callback_metrics[key] = v
            if o is None or o.logger:
                sent[key] = v
            if o is not None and o.prog_bar:
                self.progress_bar_metrics[key] = v
        self._log_to_loggers(sent)
        return sent

    def _publish_train_epoch(
        self, rt: Runtime, metrics: dict[str, float], logged: dict[str, list[Any]], model: Module
    ) -> dict[str, float]:
        """At an epoch's end: names logged per epoch get their reduced value; names logged per
        step keep their last one, the default in ``training_step``. Returns what the loggers
        get at the end of the epoch (``trainer.history`` keeps every name's epoch value)."""
        opts = model._gl_log_options
        sent: dict[str, float] = {}
        last = {k: (float(v[-1]), 1.0) for k, v in logged.items() if v}
        lasts = rt.reduce_metrics(last, {}, {}, {}) if rt.distributed else {
            k: v[0] for k, v in last.items()
        }  # fmt: skip
        for k, v in metrics.items():
            o = opts.get(k)
            if o is None or not (o.on_epoch or not o.step or k not in last):
                self.callback_metrics[k] = lasts.get(k, v)
                continue
            both = bool(o.on_epoch) and o.step
            key = f"{k}_epoch" if both else k
            self.callback_metrics[k] = self.callback_metrics[key] = v
            if both and k in lasts:
                self.callback_metrics[f"{k}_step"] = lasts[k]
            if o.logger:
                sent[key] = v
            if o.prog_bar:
                self.progress_bar_metrics[key] = v
        return sent

    @staticmethod
    def _for_loggers(values: dict[str, float], model: Module) -> dict[str, float]:
        opts = model._gl_log_options
        return {k: v for k, v in values.items() if k not in opts or opts[k].logger}

    def _aggregate(
        self,
        rt: Runtime,
        logged: dict[str, list[Any]],
        weights: dict[str, list[float]],
        opts: dict[str, Any],
        count: int,
    ) -> dict[str, float]:
        """Epoch values: each name reduced with its reduce_fx over the steps and across
        processes (a mean weighted by batch size, as in Lightning)."""
        means: dict[str, tuple[float, float]] = {}
        sums: dict[str, float] = {}
        maxes: dict[str, float] = {}
        mins: dict[str, float] = {}
        for k, values in logged.items():
            fx = opts[k][0] if k in opts else "mean"
            if fx == "mean":
                w = weights.get(k)
                if w and len(w) == len(values):
                    means[k] = (_weighted_total(values, w), float(sum(w)))
                else:
                    means[k] = (_total(values), float(len(values)))
            elif fx == "sum":
                sums[k] = _total(values)
            elif fx == "max":
                maxes[k] = _extreme(values, fx)
            else:
                mins[k] = _extreme(values, fx)
        return rt.reduce_metrics(means, sums, maxes, mins)

    def _optimizer_step(
        self, rt: Runtime, model: Module, optimizer: torch.optim.Optimizer, batch_idx: int
    ) -> None:
        hooks = self._hook_cache
        before = hooks.get("on_before_optimizer_step", [])
        custom_clip, custom_step, _ = self._custom_opt
        if not (before or custom_clip or custom_step):
            rt.step(
                optimizer, model.parameters(), self.gradient_clip_val, self.gradient_clip_algorithm
            )
        else:
            scaler = rt.scaler
            self._unscaled.clear()
            if scaler is not None:
                scaler.unscale_(optimizer)  # hooks and clipping see the real gradients
                self._unscaled.add(id(optimizer))
            for fn in before:
                fn(optimizer)
            model.configure_gradient_clipping(
                optimizer, self.gradient_clip_val, self.gradient_clip_algorithm
            )
            if custom_step:
                model.optimizer_step(
                    self.current_epoch,
                    batch_idx,
                    _StepThrough(optimizer, scaler),  # type: ignore[arg-type]
                )
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

    def _zero_grad(self, model: Module, optimizer: torch.optim.Optimizer, batch_idx: int) -> None:
        """Before backward, on the first batch of an accumulation, as in Lightning."""
        for fn in self._hook_cache.get("on_before_zero_grad", ()):
            fn(optimizer)
        if self._custom_opt[2]:
            model.optimizer_zero_grad(self.current_epoch, batch_idx, optimizer)
        else:
            optimizer.zero_grad(set_to_none=True)

    def _clip(self, optimizer: torch.optim.Optimizer, value: float, algorithm: str) -> None:
        # With manual optimization the model holds _ManualOptimizer wrappers: unscale the real
        # optimizer, which GradScaler.step checks (unscaling the wrapper made step unscale the
        # gradients a second time, so 16-mixed manual runs with clipping did not learn).
        if isinstance(optimizer, _ManualOptimizer):
            optimizer = optimizer.optimizer
        rt = self.runtime
        if rt is not None and rt.scaler is not None and id(optimizer) not in self._unscaled:
            rt.scaler.unscale_(optimizer)
            self._unscaled.add(id(optimizer))
        params = [p for g in optimizer.param_groups for p in g["params"]]
        if algorithm == "value":
            torch.nn.utils.clip_grad_value_(params, value)
        else:
            torch.nn.utils.clip_grad_norm_(params, value)

    def _manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        rt = self.runtime
        assert rt is not None and self._model is not None
        self._call_hook("on_before_backward", loss)
        self._unscaled.clear()
        if rt.scaler is not None:
            loss = rt.scaler.scale(loss)
        self._model.backward(loss, *args, **kwargs)
        if (
            self._ddp is not None
            and self._ds_engine is None
            and not rt.xla  # xm.optimizer_step all-reduces TPU gradients in step()
            and not isinstance(self._strategy, ModelParallelStrategy)
            and not _fsdp_wrapped(self._ddp)
        ):
            _average_gradients(self._model)
        self._call_hook("on_after_backward")

    def _step_schedulers(self, interval: str, plateau: bool = False) -> None:
        for c in self.lr_scheduler_configs:
            if c.interval != interval or c.reduce_on_plateau != plateau:
                continue
            count = self.global_step if interval == "step" else self.current_epoch + 1
            if count % c.frequency:
                continue
            metric = None
            step_logs = self._model._gl_step_logs if self._model is not None else {}
            if c.reduce_on_plateau and interval == "step" and c.monitor in step_logs:
                metric = float(step_logs[c.monitor])  # this step's value (read only here)
            elif c.reduce_on_plateau:
                if c.monitor not in self.callback_metrics:
                    if c.strict:
                        known = ", ".join(sorted(self.callback_metrics)) or "none"
                        raise ValidationFailed(
                            f"ReduceLROnPlateau monitors {c.monitor!r}, which was not logged "
                            f"(logged: {known})"
                        )
                    continue
                metric = self.callback_metrics[c.monitor]
            if self._model is not None and overridden(self._model, "lr_scheduler_step", Module):
                self._model.lr_scheduler_step(c.scheduler, metric)
            elif metric is None:
                c.scheduler.step()
            else:
                c.scheduler.step(metric)

    def _read_step_metrics(self) -> None:
        """Put this step's logged values into callback_metrics (reads them from the device;
        only for callbacks that act on step values). With several processes the values are
        averaged across them, so every rank decides the same way (a ModelCheckpoint save
        is a collective with FSDP; ranks deciding apart would hang). Every rank calls this
        at the same step."""
        model = self._model
        if model is None:
            return
        rt = self.runtime
        if rt is not None and rt.distributed:
            means = {k: (float(v), 1.0) for k, v in model._gl_step_logs.items()}
            values = rt.reduce_metrics(means, {}, {}, {})
        else:
            values = {k: float(v) for k, v in model._gl_step_logs.items()}
        opts = model._gl_log_options
        self.callback_metrics.update(values)
        self.callback_metrics.update(
            {f"{k}_step": v for k, v in values.items() if k in opts and opts[k].on_epoch}
        )

    def _broadcast_flag(self, flag: bool) -> bool:
        rt = self.runtime
        if rt is None or not rt.distributed:
            return flag
        import torch.distributed as dist

        if rt.xla:  # rank 0's flag: everyone else contributes 0 to the sum
            mine = 1.0 if flag and rt.is_global_zero else 0.0
            return bool(rt.xla_all_reduce(torch.tensor([mine])).item() > 0)
        t = torch.tensor([1.0 if flag else 0.0], device=rt.device if rt.gpu else "cpu")
        dist.broadcast(t, 0)
        return bool(t.item() > 0)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record extra values for the current step (used by monitoring callbacks): they go to
        metrics.csv and the loggers, from rank 0."""
        step = self.global_step if step is None else step
        if self._step_rows and self._step_rows[-1].get("step") == step:
            self._step_rows[-1].update(metrics)
        else:
            self._step_rows.append({"epoch": self.current_epoch, "step": step, **metrics})
        self._log_to_loggers(metrics, step)

    def _log_to_loggers(
        self, metrics: dict[str, float], step: int | None = None, epoch: bool = True
    ) -> None:
        """Send values to the loggers, with Lightning's step numbers: counted from 0 while
        fitting (the first optimizer step is step 0), ``global_step`` for a ``validate`` or
        ``test`` after it. ``epoch`` adds the epoch; evaluation's per-step rows have none."""
        if not metrics:
            return
        self.logged_metrics.update(metrics)
        if not self.loggers or not self.is_global_zero:
            return
        if step is None:
            fitting = self.state.fn == "fit"
            step = max(0, self.global_step - 1) if fitting else self.global_step
        out = {**metrics, "epoch": self.current_epoch} if epoch else dict(metrics)
        if self.log_key_prefix:
            for key in ("epoch", "epoch_seconds"):
                if key in out:
                    out[f"{self.log_key_prefix}{key}"] = out.pop(key)
        for lg in self.loggers:
            lg.log_metrics(out, step)

    # ------------------------------------------------------------- evaluation

    def _run_validation(
        self, rt: Runtime, model: Module, val: Any, zero_grad: bool = True
    ) -> dict[str, float]:
        if zero_grad:
            # Free the gradients' memory during validation (Lightning's hook), except in the
            # middle of an accumulation.
            model.on_validation_model_zero_grad()
        self.state.stage = "validate"
        out = self._evaluate(rt, model, val, "validation_step")
        self.state.stage = "train"
        return out

    def _evaluate(
        self,
        rt: Runtime,
        model: Module,
        loaders: Any,
        step: str = "validation_step",
        limit: int | None = None,
        grad_context: Any = None,
    ) -> dict[str, float]:
        """Run ``step`` over one loader or a list of them. With several loaders, the step and
        the batch hooks receive ``dataloader_idx`` and logged names get a
        ``/dataloader_idx_N`` suffix (unless logged with ``add_dataloader_idx=False``)."""
        stage = {"validation_step": "validation", "test_step": "test"}[step]
        loaders = loaders if isinstance(loaders, list) else [loaders]
        many = len(loaders) > 1
        hooks = self._hook_cache
        batch_start = hooks.get(f"on_{stage}_batch_start", [])
        batch_end = hooks.get(f"on_{stage}_batch_end", [])
        self._call_hook(f"on_{stage}_start")
        modes = {m: m.training for m in model.modules()}
        getattr(model, f"on_{stage}_model_eval")()
        self._call_hook(f"on_{stage}_epoch_start")
        step_fn = getattr(model, step)
        logged: dict[str, list[Any]] = defaultdict(list)
        weights: dict[str, list[float]] = defaultdict(list)
        options: dict[str, Any] = {}
        # A validation inside a training epoch must not compute the training metrics.
        outer_metrics, model._gl_metrics = model._gl_metrics, {}
        metric_objects: dict[str, Any] = {}
        count = 0
        lim_name = "limit_val_batches" if stage == "validation" else "limit_test_batches"
        ctx = grad_context if grad_context is not None else torch.no_grad()
        prof = self.profiler
        if prof is not None:
            prof.start(f"run_{stage}")
        sizes: list[int | float] = []
        with ctx, rt.autocast():
            for dl_idx, loader in enumerate(loaders):
                configured = _limit(getattr(self, lim_name), _len(loader), lim_name)
                lim = (
                    configured
                    if limit is None
                    else (limit if configured is None else min(limit, configured))
                )
                sizes.append(lim if lim is not None else (_len(loader) or math.inf))
                extra = (dl_idx,) if many else ()
                for batch_idx, batch in enumerate(
                    self._device_batches(rt, model, loader, self.prefetch, dl_idx)
                ):
                    if lim is not None and batch_idx >= lim:
                        break
                    for fn in batch_start:
                        fn(batch, batch_idx, *extra)
                    model._gl_step_logs = {}
                    out = step_fn(batch, batch_idx, *extra)
                    opts = model._gl_log_options
                    no_idx = model.__dict__.get("_gl_no_idx", ())
                    size: float | None = None
                    step_values: dict[str, float] = {}
                    for k, v in model._gl_step_logs.items():
                        name = f"{k}/dataloader_idx_{dl_idx}" if many and k not in no_idx else k
                        logged[name].append(v)
                        o = opts.get(k)
                        if o is not None:
                            options[name] = o
                            if o.on_step and o.logger:  # asked for: reads the value every step
                                step_values[_suffixed(name, "step")] = float(v)
                        if o is not None and o[3] is not None:
                            weights[name].append(o[3])
                        else:  # weight the epoch mean by the inferred batch size (Lightning)
                            if size is None:
                                size = _batch_size(batch)
                            weights[name].append(size)
                    if step_values and self.state.stage != "sanity_check":
                        self._log_to_loggers(step_values, self._eval_log_step, epoch=False)
                        self._eval_log_step += 1
                    if model._gl_metrics:
                        for k, m in model._gl_metrics.items():
                            metric_objects[
                                f"{k}/dataloader_idx_{dl_idx}" if many and k not in no_idx else k
                            ] = m
                        model._gl_metrics = {}
                    count += 1
                    for fn in batch_end:
                        fn(out, batch, batch_idx, *extra)
        if stage == "test":
            self._num_test_batches = sizes
        else:
            self.num_val_batches = sizes
        if prof is not None:
            prof.stop(f"run_{stage}")
        values = self._aggregate(rt, logged, weights, options, count)
        values.update(_compute_metrics(metric_objects))
        metrics: dict[str, float] = {}
        sent: dict[str, float] = {}
        for k, v in values.items():
            o = options.get(k)
            if o is not None and o.on_step and o.on_epoch is False:
                continue  # per step only
            key = _suffixed(k, "epoch") if o is not None and o.on_step else k
            metrics[key] = self.callback_metrics[k] = self.callback_metrics[key] = v
            if o is None or o.logger:
                sent[key] = v
            if o is not None and o.prog_bar:
                self.progress_bar_metrics[key] = v
        extra_logs = self._call_epoch_end(f"on_{stage}_epoch_end", model)
        metrics.update(extra_logs)
        if self.state.stage != "sanity_check":
            self._log_to_loggers({**sent, **self._for_loggers(extra_logs, model)})
        self._call_monitors(f"on_{stage}_epoch_end", model)
        self._call_hook(f"on_{stage}_end")
        model._gl_metrics = outer_metrics
        if overridden(model, f"on_{stage}_model_train", Module):
            getattr(model, f"on_{stage}_model_train")()
        else:  # each submodule back to its mode before (frozen parts stay in eval), as Lightning
            for m, training in modes.items():
                m.training = training
        return metrics

    def _eval_context(self, rt: Runtime) -> Any:
        """``torch.inference_mode`` for validate/test/predict (faster than no_grad: no version
        counters or view tracking), except with gloo or FSDP, where Lightning also uses
        no_grad."""
        import torch.distributed as dist

        if not self.inference_mode:
            return torch.no_grad()
        if self.runtime is not None and self.runtime.xla:
            return torch.no_grad()  # as Lightning does on XLA devices
        if dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo":
            return torch.no_grad()
        if isinstance(self._strategy, FSDPStrategy):
            return torch.no_grad()
        return torch.inference_mode()

    def _resolve_ckpt(self, ckpt_path: str | Path | None) -> str | Path | None:
        if ckpt_path == "best":
            cb = self.checkpoint_callback
            if cb is None or cb.best_model_path is None:
                raise ValidationFailed("ckpt_path='best' needs a ModelCheckpoint that saved a file")
            return cb.best_model_path
        if ckpt_path == "last":
            last = self._last_checkpoint()
            if last is None:
                raise ValidationFailed("ckpt_path='last': no checkpoint was saved")
            return last
        return ckpt_path

    def _last_checkpoint(self) -> Any:
        """The newest ``last*.ckpt`` of this Trainer, its ``ModelCheckpoint`` folders and the
        run folder, so a rerun of a script can continue with ``fit(ckpt_path="last")``."""
        root = self.default_root_dir
        folders = {root / "checkpoints", root / _run_name() / "checkpoints"}
        found: list[Any] = [] if self.checkpoint_path is None else [as_path(self.checkpoint_path)]
        for cb in self.checkpoint_callbacks:
            if isinstance(cb, ModelCheckpoint):
                if cb.dirpath is not None:
                    folders.add(as_path(cb.dirpath))
                if cb.last_model_path:
                    found.append(as_path(cb.last_model_path))
        for folder in folders:
            found += folder.glob("last*.ckpt")
        existing = [p for p in found if p.exists()]
        return max(existing, key=lambda p: p.stat().st_mtime) if existing else None

    def _fit_checkpoint(self, ckpt_path: str | Path | None) -> str | Path | None:
        """Resolve ``fit``'s ``ckpt_path``. Inside a launched job the checkpoint found by the
        launching process has arrived as an input; the job must use it. With ``"last"`` and
        nothing to resume from, training starts from the beginning, as in Lightning."""
        kept = _kept_checkpoint()
        if kept is not None:
            self._say(f"[gpulayer] attempt {os.environ['GPULAYER_ATTEMPT']}: resuming")
            return kept
        shipped = _resume_path(ckpt_path)
        if shipped != ckpt_path:
            return shipped
        if ckpt_path == "last" and self._last_checkpoint() is None:
            self._say("[gpulayer] ckpt_path='last': no checkpoint found, starting from scratch")
            return None
        return self._resolve_ckpt(ckpt_path)

    def _eval_entry(
        self,
        stage: str,
        model: Module | None,
        dataloaders: Any,
        ckpt_path: str | Path | None,
        datamodule: DataModule | None,
    ) -> tuple[Runtime, Module, Any]:
        if isinstance(dataloaders, DataModule):
            datamodule, dataloaders = dataloaders, None
        if model is None:
            if self._model is None:
                raise ValidationFailed(f"{stage} needs a model (none was fitted with this Trainer)")
            model = self._model
        model = self._unwrap(model)
        if datamodule is not None:
            self.datamodule = datamodule
        strat = self._strategy
        rt = Runtime(self.accelerator, self.precision, self.gradient_sync).setup(
            backend=strat.process_group_backend, timeout=strat.timeout
        )
        self.runtime = rt
        self._model = model
        self.state = TrainerState(stage, stage, "running")
        self._setup_model(rt, model, stage)
        path = _resume_path(self._resolve_ckpt(ckpt_path))
        if path is not None:
            ckpt = self.checkpoint_io.load_checkpoint(path)
            model.on_load_checkpoint(ckpt)
            model.load_state_dict(ckpt["state_dict"], strict=model.strict_loading)
        self._apply_precision_plugin(rt, model)
        model.to(rt.device)
        rt.convert_module(model)
        self._build_hooks(model)
        name = {"validate": "val", "test": "test", "predict": "predict"}[stage]
        src = self._source(dataloaders, name, model)
        if src is None:
            raise ValidationFailed(
                f"{stage} needs data: pass dataloaders, a datamodule, or define "
                f"{name}_dataloader() on the model"
            )
        loaders = self._prepare_loaders(rt, src)
        # The loaders of this call, under the reference attribute names.
        setattr(
            self,
            {"val": "val_dataloaders", "test": "test_dataloaders"}.get(name, "predict_dataloaders"),
            loaders,
        )
        return rt, model, loaders

    def validate(
        self,
        model: Module | None = None,
        dataloaders: Any = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
    ) -> dict[str, float]:
        """Run ``validation_step`` over ``dataloaders`` in this process and return the averaged
        logs, e.g. ``{"val_loss": 0.21}`` (``result[0]`` also works, as in Lightning)."""
        return self._run_eval(
            "validate", "validation_step", model, dataloaders, ckpt_path, verbose, datamodule
        )

    def test(
        self,
        model: Module | None = None,
        dataloaders: Any = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
    ) -> dict[str, float]:
        """Run ``test_step`` over ``dataloaders`` in this process and return the averaged logs,
        e.g. ``{"test_acc": 0.93}``."""
        return self._run_eval(
            "test", "test_step", model, dataloaders, ckpt_path, verbose, datamodule
        )

    def _run_eval(
        self,
        stage: str,
        step: str,
        model: Module | None,
        dataloaders: Any,
        ckpt_path: str | Path | None,
        verbose: bool,
        datamodule: DataModule | None,
    ) -> dict[str, float]:
        self._eval_calls[stage] += 1
        call = self._eval_calls[stage]
        if self.will_launch:
            metrics = self._launched_metrics(stage, call, ckpt_path)
            self.callback_metrics.update(metrics)
            if verbose and metrics:
                width = max(len(k) for k in metrics)
                for k, v in metrics.items():
                    print(f"{k:<{width}}  {v:.6g}")
            return _Results(metrics)
        rt, model, loaders = self._eval_entry(stage, model, dataloaders, ckpt_path, datamodule)
        self.callback_metrics = {}
        self._eval_log_step = 0
        try:
            metrics = self._evaluate(rt, model, loaders, step, grad_context=self._eval_context(rt))
            if in_gpulayer_job() and rt.is_global_zero:
                # a launched run: the launching process reads these back
                out = rt.output_dir()
                (out / f"{stage}-{call}.json").write_text(json.dumps(metrics))
            self._teardown_model(model, stage)
            self.state.status = "finished"
            if verbose and metrics and rt.is_global_zero:
                width = max(len(k) for k in metrics)
                for k, v in metrics.items():
                    print(f"{k:<{width}}  {v:.6g}")
            return _Results(metrics)
        finally:
            self.state.stage = None
            rt.teardown()

    def predict(
        self,
        model: Module | None = None,
        dataloaders: Any = None,
        datamodule: DataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
    ) -> list[Any] | None:
        """Run ``predict_step`` over ``dataloaders`` (no gradients) and return one output per
        batch, moved to the CPU; with several loaders, one list per loader. With
        ``return_predictions=False`` (for example with a ``BasePredictionWriter``) nothing is
        kept in memory.

        With several devices or machines (or a ``target``), prediction runs on GPUlayer
        workers, each process on its shard of the data, and the outputs come back in rank
        order: all of rank 0's batches, then rank 1's, and so on.
        ``trainer.predict_batch_indices`` holds the dataset indices of each batch."""
        self._eval_calls["predict"] += 1
        call = self._eval_calls["predict"]
        keep = return_predictions is not False
        if self.will_launch:
            preds, indices = self._launched_predictions(call, ckpt_path)
            self.predict_batch_indices = indices
            return preds if keep else None
        rt, model, loaders = self._eval_entry("predict", model, dataloaders, ckpt_path, datamodule)
        assert loaders is not None
        many = len(loaders) > 1
        hooks = self._hook_cache
        writer = any(hasattr(c, "write_on_batch_end") for c in self.callbacks)
        all_preds: list[list[Any]] = []
        all_indices: list[list[Any] | None] = []
        try:
            self._call_hook("on_predict_start")
            model.on_predict_model_eval()
            self._call_hook("on_predict_epoch_start")
            with self._eval_context(rt), rt.autocast():
                for dl_idx, loader in enumerate(loaders):
                    self._predictions = []
                    sampler = getattr(loader, "batch_sampler", None)
                    self._predict_indices = None
                    if (writer or in_gpulayer_job()) and sampler is not None:
                        with contextlib.suppress(TypeError):
                            self._predict_indices = [list(b) for b in sampler]
                    limit = _limit(
                        self.limit_predict_batches, _len(loader), "limit_predict_batches"
                    )
                    if dl_idx == 0:
                        self._num_predict_batches = []
                    self._num_predict_batches.append(
                        limit if limit is not None else (_len(loader) or math.inf)
                    )
                    extra = (dl_idx,) if many else ()
                    for batch_idx, batch in enumerate(
                        self._device_batches(rt, model, loader, self.prefetch, dl_idx)
                    ):
                        if limit is not None and batch_idx >= limit:
                            break
                        for fn in hooks.get("on_predict_batch_start", ()):
                            fn(batch, batch_idx, *extra)
                        out = _to_cpu(model.predict_step(batch, batch_idx, *extra))
                        if keep or hooks.get("on_predict_epoch_end") or in_gpulayer_job():
                            self._predictions.append(out)
                        for fn in hooks.get("on_predict_batch_end", ()):
                            fn(out, batch, batch_idx, *extra)
                    all_preds.append(self._predictions)
                    all_indices.append(self._predict_indices)
                self._predictions = all_preds if many else all_preds[0]
                self._predict_indices = all_indices if many else all_indices[0]
            self._call_hook("on_predict_epoch_end")
            self._call_hook("on_predict_end")
            self._teardown_model(model, "predict")
            self.predict_batch_indices = self._predict_indices
            if in_gpulayer_job():
                # a launched run: every rank writes its part; the launching process joins them
                out_file = rt.output_dir() / f"predict-{call}-rank{rt.global_rank}.pt"
                torch.save(
                    {"predictions": self._predictions, "indices": self._predict_indices}, out_file
                )
            self.state.status = "finished"
            return self._predictions if keep else None
        finally:
            self.state.stage = None
            rt.teardown()

    def _launched_files(self, pattern: str, ckpt_path: str | Path | None) -> list[Path]:
        """Files a launched run wrote for this call. A job launched by an earlier ``fit`` of
        this Trainer already ran the rest of the script, so its outputs are used when they are
        there; otherwise a job is submitted now."""
        if self.launch_result is not None:
            found = sorted(self.launch_result.output_dir.rglob(pattern))
            if found:
                return found
        result = self._launch_job(ckpt_path)
        found = sorted(result.output_dir.rglob(pattern))
        if not found:
            raise RemoteError(f"the launched run wrote no {pattern}", result.job)
        return found

    def _launched_metrics(
        self, stage: str, call: int, ckpt_path: str | Path | None
    ) -> dict[str, float]:
        files = self._launched_files(f"{stage}-{call}.json", ckpt_path)
        return {k: float(v) for k, v in json.loads(files[0].read_text()).items()}

    def _launched_predictions(
        self, call: int, ckpt_path: str | Path | None
    ) -> tuple[list[Any], Any]:
        import re

        files = self._launched_files(f"predict-{call}-rank*.pt", ckpt_path)
        files.sort(key=lambda f: int(re.search(r"rank(\d+)", f.name).group(1)))  # type: ignore[union-attr]
        parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        first = parts[0]["predictions"]
        if first and isinstance(first[0], list):  # several loaders: join per loader
            preds: list[Any] = [
                sum((p["predictions"][i] for p in parts), []) for i in range(len(first))
            ]
            idx = [
                sum(((p["indices"] or [None])[i] or [] for p in parts), [])
                for i in range(len(first))
            ]
            return preds, idx
        preds = sum((p["predictions"] for p in parts), [])
        indices = sum(((p["indices"] or []) for p in parts), [])
        return preds, indices

    def _update_bn(self, model: Module) -> None:
        """Recompute BatchNorm running statistics over the training data (after SWA)."""
        rt = self.runtime
        loader = self.train_dataloader
        if rt is None or loader is None:
            return
        momenta = {}
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
                momenta[m] = m.momentum
                m.momentum = None
        was = model.training
        model.train()
        with torch.no_grad(), rt.autocast():
            for batch in rt.batches(loader, prefetch=False):
                model(batch[0] if isinstance(batch, (list, tuple)) else batch)
        for m, mom in momenta.items():
            m.momentum = mom
        model.train(was)

    # ------------------------------------------------------------- checkpoints

    def _restore(self, rt: Runtime, model: Module, ckpt_path: str | Path) -> tuple[int, int]:
        if as_path(ckpt_path).is_dir() and rt.distributed:
            # A sharded checkpoint: each rank loads its own part in place.
            state = rt.load_sharded(model, self.optimizers, Path(ckpt_path))
            return self._restore_rest(rt, model, state)
        state = self.checkpoint_io.load_checkpoint(ckpt_path)
        model.on_load_checkpoint(state)
        for cb in self.callbacks:
            if overrides(cb, "on_load_checkpoint"):
                cb.on_load_checkpoint(self, model, state)
        optim_states = state.get("optimizer_states") or (
            [state["optimizer"]] if "optimizer" in state else []
        )
        if optim_states:
            rt.load_full_state(model, self.optimizers[0], state["state_dict"], optim_states[0])
            for opt, s in zip(self.optimizers[1:], optim_states[1:], strict=False):
                opt.load_state_dict(s)
        else:
            model.load_state_dict(state["state_dict"], strict=model.strict_loading)
        return self._restore_rest(rt, model, state)

    def _restore_rest(self, rt: Runtime, model: Module, state: dict[str, Any]) -> tuple[int, int]:
        """Everything in a checkpoint besides the weights and optimizer states."""
        if state.get("sharded"):
            model.on_load_checkpoint(state)
            for cb in self.callbacks:
                if overrides(cb, "on_load_checkpoint"):
                    cb.on_load_checkpoint(self, model, state)
        for c, s in zip(self.lr_scheduler_configs, state.get("lr_schedulers") or [], strict=False):
            c.scheduler.load_state_dict(s)
        if rt.scaler is not None and state.get("scaler"):
            rt.scaler.load_state_dict(state["scaler"])
        cb_states = state.get("callbacks") or {}
        for cb in self.callbacks:
            if cb.state_key in cb_states:
                cb.load_state_dict(cb_states[cb.state_key])
        if self.datamodule is not None and "datamodule" in state:
            self.datamodule.load_state_dict(state["datamodule"])
        self.global_step = state.get("global_step", 0)
        self.history = state.get("history", [])
        self._restored_rng = state.get("rng_states")
        self._restored_loaders = state.get("dataloaders")
        if "loops" in state:
            return _lightning_position(state["loops"]["fit_loop"])
        epoch = state["epoch"]
        if state.get("batches_done"):
            return epoch, int(state["batches_done"])  # saved within an epoch
        return state.get("next_epoch", epoch + 1), 0

    def save_checkpoint(
        self,
        filepath: str | Path,
        weights_only: bool = False,
        storage_options: Any = None,
        dirpath: str | Path | None = None,
    ) -> Path | None:
        """Write a checkpoint from rank 0 (other ranks return None) and return its path. A
        plain file name goes to ``checkpoints/`` of the run (or ``dirpath``); a path is used as
        it is. Call it on every rank: with ``strategy="fsdp"`` the full state is gathered from
        all of them."""
        if self._model is None or not self.optimizers:
            raise ValidationFailed("save_checkpoint needs a model fitted by this Trainer")
        rt = self.runtime
        model = self._model
        if self._ds_engine is not None:
            # stage 3 holds 1/N of each parameter per rank: gather them (a collective)
            if getattr(self._strategy, "stage", 0) == 3:
                full = self._ds_engine._zero3_consolidated_16bit_state_dict()
                model_state = {k.removeprefix("module."): v for k, v in (full or {}).items()}
            else:
                model_state = model.state_dict()
            optim_states = []
            weights_only = True  # DeepSpeed's own optimizer state is not in Trainer checkpoints
            if rt is not None and not rt.is_global_zero:
                return None
        elif self._sharded_checkpoints():
            model_state, optim_states = {}, list[Any]()  # each rank writes its part below
        elif (
            rt is not None
            and isinstance(self._strategy, (FSDPStrategy, ModelParallelStrategy))
            and rt.distributed
        ):
            model_state, optim_state = rt.full_state(model, self.optimizers[0])
            optim_states = [optim_state]
        elif rt is not None and not rt.is_global_zero:
            return None
        else:
            model_state = model.state_dict()
            optim_states = [] if weights_only else [o.state_dict() for o in self.optimizers]
        sharded = self._sharded_checkpoints()
        if rt is not None and not rt.is_global_zero and not sharded:
            return None
        if rt is not None and rt.xla:
            # XLA tensors would load only on a TPU: save them as CPU tensors, as xm.save does
            model_state, optim_states = _to_cpu(model_state), _to_cpu(optim_states)
        p = as_path(filepath)
        if p.is_absolute() or len(p.parts) > 1:
            path = p
        else:
            folder = as_path(dirpath) if dirpath is not None else self._checkpoint_folder()
            path = folder / p
        path.parent.mkdir(parents=True, exist_ok=True)
        from gpulayer import __version__

        in_fit = self.state.fn == "fit" and self.state.stage is not None
        checkpoint: dict[str, Any] = {
            "epoch": self.current_epoch,
            "next_epoch": self.current_epoch + 1 if in_fit else self.current_epoch,
            "global_step": self.global_step,
            "gpulayer_version": __version__,
            # What Lightning's load_from_checkpoint reads; see utilities.to_lightning_checkpoint.
            LIGHTNING_VERSION_KEY: LIGHTNING_FORMAT,
            "state_dict": model_state,
            "history": self.history,
            "callback_metrics": dict(self.callback_metrics),
        }
        done = self._batch_counter
        if self.state.stage == "train" and done and done < self.num_training_batches:
            # Saved within an epoch: resuming skips the batches already trained.
            checkpoint["batches_done"] = done
            stateful = [
                d.state_dict() for d in loaders_of(self.train_dataloader) if _is_stateful(d)
            ]
            if stateful:
                checkpoint["dataloaders"] = stateful
        if not weights_only:
            checkpoint["optimizer_states"] = optim_states
            checkpoint["optimizer"] = optim_states[0] if optim_states else {}
            checkpoint["lr_schedulers"] = [
                c.scheduler.state_dict() for c in self.lr_scheduler_configs
            ]
            if rt is not None and rt.scaler is not None:
                checkpoint["scaler"] = rt.scaler.state_dict()
            checkpoint["callbacks"] = {
                cb.state_key: cb.state_dict() for cb in self.callbacks if cb.state_dict()
            }
            if self.datamodule is not None:
                checkpoint["datamodule"] = self.datamodule.state_dict()
            if rt is not None and not rt.distributed and not rt.xla:
                # One process: enough to continue with the same random numbers. Several: the
                # data order is exact (DistributedSampler), each rank's own stream is not.
                checkpoint["rng_states"] = {"now": _rng_states(), "epoch_start": self._epoch_rng}
        if model.hparams:
            checkpoint["hyper_parameters"] = dict(model.hparams)
            checkpoint["hparams_name"] = "kwargs"
        if self.datamodule is not None and self.datamodule.hparams:
            checkpoint["datamodule_hyper_parameters"] = dict(self.datamodule.hparams)
        model.on_save_checkpoint(checkpoint)
        for cb in self.callbacks:
            if overrides(cb, "on_save_checkpoint"):
                cb.on_save_checkpoint(self, model, checkpoint)
        if sharded:
            assert rt is not None
            del checkpoint["state_dict"]
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("optimizer", None)
            checkpoint["sharded"] = True
            if is_remote(path):
                raise ValidationFailed(f"sharded checkpoints need a local folder, not {path}")
            path.mkdir(parents=True, exist_ok=True)
            rt.save_sharded(model, self.optimizers, path, checkpoint)
            if not rt.is_global_zero:
                return None
        else:
            self.checkpoint_io.save_checkpoint(checkpoint, path)
        if os.environ.get("GPULAYER_CHECKPOINT_STREAM") and Path(CHECKPOINT).name == p.name:
            # The worker keeps this file for the next attempt, without making training wait.
            print(CHECKPOINT_MARKER, flush=True)
        for lg in self.loggers:
            if not is_remote(path):  # loggers upload a local file
                lg.after_save_checkpoint(path)
        return path

    def _sharded_checkpoints(self) -> bool:
        """True when each rank writes its own part of the checkpoint (FSDP with
        ``state_dict_type="sharded"``, ModelParallelStrategy with
        ``save_distributed_checkpoint=True``)."""
        rt = self.runtime
        return bool(
            rt is not None
            and rt.distributed
            and getattr(self._strategy, "state_dict_type", "full") == "sharded"
        )

    def _checkpoint_folder(self) -> Any:
        """With a remote ``default_root_dir`` checkpoints go there, also from inside a job,
        whose run folder is a local one on the worker."""
        root = self.default_root_dir
        return root / "checkpoints" if is_remote(root) else (self._out_dir or root) / "checkpoints"

    def _write_metrics(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(self.history, indent=2))
        if self._csv:
            self._write_csv(out_dir / "metrics.csv")

    def _write_csv(self, path: Path) -> None:
        """One row per printed step, per mid-epoch validation and per epoch (rank 0)."""
        rows = [{"kind": "step", **r} for r in self._step_rows]
        rows += [{"kind": "val", **r} for r in self.val_history]
        rows += [{"kind": "epoch", **r} for r in self.history]
        rows.sort(key=lambda r: (r.get("epoch", 0), r.get("step", float("inf"))))
        columns = ["kind", "epoch", "step"]
        for r in rows:
            columns += [k for k in r if k not in columns]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)

    def _val_every(self, n_batches: int | None) -> int | None:
        """Batches between validations within an epoch, or None for end of epoch only."""
        v = self.val_check_interval
        if v is None:
            return None
        if isinstance(v, float):
            if n_batches is None:
                raise ValidationFailed("a fractional val_check_interval needs a sized loader")
            return max(1, int(n_batches * v)) if v < 1.0 else None
        return v


_NOOP = contextlib.nullcontext()


class _StepThrough:
    """The optimizer handed to an overridden ``Module.optimizer_step``: ``step`` goes
    through the gradient scaler when there is one."""

    def __init__(self, optimizer: torch.optim.Optimizer, scaler: Any) -> None:
        self._opt, self._scaler = optimizer, scaler

    def __getattr__(self, name: str) -> Any:
        if name == "_opt":
            raise AttributeError(name)
        return getattr(self._opt, name)

    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        if self._scaler is not None:
            if closure is not None:
                closure()
            out = self._scaler.step(self._opt)
            self._scaler.update()
            return out
        return self._opt.step(closure) if closure is not None else self._opt.step()


def _with_trainer(fn: Callable[..., Any], trainer: Trainer, model: Module) -> Callable[..., Any]:
    return lambda *args: fn(trainer, model, *args)


def _legacy_batch_end(
    fn: Callable[..., Any], trainer: Trainer, model: Module
) -> Callable[..., Any]:
    """Adapter for ``on_train_batch_end(trainer, model, loss, batch_idx)`` callbacks."""

    def call(outputs: Any, batch: Any, batch_idx: int) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        fn(trainer, model, loss, batch_idx)

    return call


class _LookAhead:
    """Iterate ``it`` one item ahead: ``last`` is True while the last item is current."""

    def __init__(self, it: Iterable[Any]) -> None:
        self._it = it
        self.last = False

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self._it)
        try:
            current = next(iterator)
        except StopIteration:
            return
        for following in iterator:
            yield current
            current = following
        self.last = True
        yield current


def _profiled(it: Iterator[Any], prof: Profiler, action: str) -> Iterator[Any]:
    iterator = iter(it)
    while True:
        prof.start(action)
        try:
            batch = next(iterator)
        except StopIteration:
            prof.stop(action)
            return
        prof.stop(action)
        yield batch


def _fsdp_wrapped(module: Any) -> bool:
    from torch.nn.parallel import DistributedDataParallel as DDP

    return not isinstance(module, DDP)


def _wait_for_ranks(ranks: list[Any]) -> Any:
    """Wait until every process ended, or stop the rest when one fails; returns the failed
    one. Sentinels stay readable after a process ends, so each is waited on exactly once, and
    a process ending at any moment cannot leave this waiting on nothing."""
    pending = {r.sentinel: r for r in ranks}
    failed = None
    while pending and failed is None:
        for sentinel in multiprocessing.connection.wait(list(pending)):
            rank = pending.pop(sentinel)
            rank.join()
            if rank.exitcode != 0:
                failed = rank
    for rank in pending.values():
        rank.terminate()
        rank.join()
    return failed


def _notebook_rank(
    rank: int, world: int, port: int, out: Path, trainer: Trainer, model: Module, args: Any
) -> None:
    """Entry point of a process forked by ``Trainer._fit_notebook``."""
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world),
        GPULAYER_NODE_RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        GPULAYER_OUTPUT_DIR=str(out),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    trainer.fit(model, *args)


def _xla_spawned_fit(index: int, trainer: Trainer, model: Module, *args: Any) -> None:
    """Entry point of each TPU process started by ``Trainer._fit_xla_spawn``: take this
    process's rank from torch_xla (global across the hosts of a TPU pod), then run the
    ordinary ``fit``; xm.optimizer_step averages the gradients and the Runtime's reductions
    use torch_xla's collectives."""
    import torch_xla.runtime as xr

    world = xr.world_size()
    # Processes on this host; on a TPU pod, world spans every host.
    local = int(getattr(xr, "local_process_count", xr.world_size)())
    if not trainer._all_devices and trainer.devices != local:
        raise ValidationFailed(
            f"devices={trainer.devices}, but this TPU VM has {local} TPU devices: torch_xla "
            f"starts one process per device, so pass devices='auto' (or {local})"
        )
    os.environ.update(
        RANK=str(xr.global_ordinal()),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(xr.local_ordinal()),
        LOCAL_WORLD_SIZE=str(local),
        GPULAYER_NODE_RANK=str(getattr(xr, "host_index", lambda: 0)()),
    )
    trainer.fit(model, *args)


def _average_gradients(model: torch.nn.Module) -> None:
    """Average gradients across ranks after a manual backward: one small all-reduce to agree
    on which parameters have gradients (so every rank sends the same buffer), then one
    coalesced all-reduce per dtype. Averaging again after a second manual_backward is
    harmless: the gradients are already equal on every rank."""
    import torch.distributed as dist

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return
    world = dist.get_world_size()
    dev = params[0].device
    has = torch.tensor([p.grad is not None for p in params], dtype=torch.float32, device=dev)
    dist.all_reduce(has, op=dist.ReduceOp.MAX)
    groups: dict[torch.dtype, list[torch.nn.Parameter]] = {}
    for p, flag in zip(params, has.tolist(), strict=True):
        if flag:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            groups.setdefault(p.grad.dtype, []).append(p)
    for group in groups.values():
        flat = torch.cat([p.grad.reshape(-1) for p in group])  # type: ignore[union-attr]
        dist.all_reduce(flat)
        flat /= world
        offset = 0
        for p in group:
            n = p.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p))  # type: ignore[union-attr]
            offset += n


def _convert_sync_batchnorm(model: Module) -> Module:
    """Replace BatchNorm layers with SyncBatchNorm in place (statistics over all processes)."""
    for name, child in list(model.named_children()):
        converted = torch.nn.SyncBatchNorm.convert_sync_batchnorm(child)
        if converted is not child:
            setattr(model, name, converted)
    return model


def _without_shuffle(loader: DataLoader) -> DataLoader:
    if isinstance(loader.sampler, SequentialSampler) or loader.batch_size is None:
        return loader
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


@dataclass
class _Counts:
    ready: int | float = 0
    started: int | float = 0
    processed: int | float = 0
    completed: int | float = 0


@dataclass
class _Progress:
    current: _Counts
    total: _Counts


def _progress(current: int | float, total: int | float) -> _Progress:
    return _Progress(_Counts(*[current] * 4), _Counts(*[total] * 4))


class _EvaluationLoop:
    def __init__(self, batches: list[int | float]) -> None:
        self.batch_progress = _progress(sum(batches), sum(batches))


class _TrainingEpochLoop:
    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    @property
    def batch_idx(self) -> int:
        return self.trainer._last_batch_idx

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    @property
    def max_steps(self) -> int:
        return self.trainer.max_steps

    @property
    def batch_progress(self) -> _Progress:
        t = self.trainer
        return _progress(t._last_batch_idx + 1, t._total_batches)

    @property
    def val_loop(self) -> _EvaluationLoop:
        return self.trainer.validate_loop


class _FitLoop:
    """A view of the Trainer's state under Lightning's names. GPUlayer's loop is one function,
    not a tree of loop objects, so only what code usually reads is here."""

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.epoch_loop = _TrainingEpochLoop(trainer)

    @property
    def max_epochs(self) -> int:
        return self.trainer.max_epochs

    @max_epochs.setter
    def max_epochs(self, value: int) -> None:
        self.trainer.max_epochs = value

    @property
    def min_epochs(self) -> int | None:
        return self.trainer.min_epochs

    @property
    def max_steps(self) -> int:
        return self.trainer.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self.trainer.max_steps = value

    @property
    def epoch_progress(self) -> _Progress:
        return _progress(self.trainer.current_epoch, self.trainer.current_epoch)

    @property
    def batch_idx(self) -> int:
        return self.trainer._last_batch_idx


class _SkipFirst:
    """A batch sampler without its first ``n`` batches: same order, no data read for them."""

    def __init__(self, sampler: Any, n: int) -> None:
        self.sampler, self.n = sampler, n

    def __iter__(self) -> Iterator[Any]:
        return itertools.islice(iter(self.sampler), self.n, None)

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.n)


def _skipping_loader(loader: Any, n: int) -> DataLoader | None:
    """``loader`` from batch ``n`` on, with its settings, or None when it cannot skip: an
    iterable dataset has no sampler, and a ``DataLoader`` subclass may do anything."""
    if (
        type(loader) is not DataLoader
        or loader.batch_sampler is None
        or isinstance(loader.dataset, IterableDataset)
    ):
        return None
    extra: dict[str, Any] = (
        {"prefetch_factor": loader.prefetch_factor} if loader.num_workers else {}
    )
    return DataLoader(
        loader.dataset,
        batch_sampler=_SkipFirst(loader.batch_sampler, n),
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        generator=loader.generator,
        worker_init_fn=loader.worker_init_fn,
        timeout=loader.timeout,
        multiprocessing_context=loader.multiprocessing_context,
        pin_memory_device=getattr(loader, "pin_memory_device", ""),
        **extra,
    )


def _after_first(batches: Iterable[Any], then: Callable[[], None]) -> Iterator[Any]:
    it = iter(batches)
    first = next(it, _NOOP)
    then()
    if first is not _NOOP:
        yield first
        yield from it


def _is_stateful(loader: Any) -> bool:
    return callable(getattr(loader, "state_dict", None)) and callable(
        getattr(loader, "load_state_dict", None)
    )


def _rng_states() -> dict[str, Any]:
    """Every random number generator training uses, as plain values, so a checkpoint still
    loads with ``weights_only=True``."""
    states: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        name, keys, pos, has_gauss, cached = numpy.random.get_state()
        states["numpy"] = (name, keys.tolist(), pos, has_gauss, cached)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        states["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        states["mps"] = torch.mps.get_rng_state()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore ``_rng_states``. A generator this machine lacks is left alone."""
    torch.set_rng_state(states["torch"])
    random.setstate(states["python"])
    numpy = sys.modules.get("numpy")
    if numpy is not None and "numpy" in states:
        name, keys, pos, has_gauss, cached = states["numpy"]
        numpy.random.set_state(
            (name, numpy.array(keys, dtype=numpy.uint32), pos, has_gauss, cached)
        )
    if "cuda" in states and len(states["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(states["cuda"])
    if "mps" in states and torch.backends.mps.is_available():
        torch.mps.set_rng_state(states["mps"])


def _materialize(loader: Any) -> Any:
    """A loader given as a function (to build it only where training runs) is called here."""
    if callable(loader) and not hasattr(loader, "__iter__"):
        return loader()
    return loader


_LOG_FORBIDDEN = frozenset(
    {
        "on_fit_start",
        "on_fit_end",
        "on_sanity_check_start",
        "on_sanity_check_end",
        "on_train_end",
        "on_validation_end",
        "on_test_end",
    }
)


@contextlib.contextmanager
def _no_log(model: Module | None, hook: str) -> Iterator[None]:
    """``self.log`` is an error during ``hook``, as in Lightning: nothing would collect the
    value. The check costs nothing where logging is allowed."""
    if model is None or (hook in _HOOKS_WITH_LOG and hook not in _LOG_FORBIDDEN):
        yield
        return

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise ValidationFailed(f"self.log cannot be called in {hook}")

    model.__dict__["log"] = forbidden
    try:
        yield
    finally:
        model.__dict__.pop("log", None)


_HOOKS_WITH_LOG = frozenset(
    {
        "on_train_start",
        "on_train_epoch_start",
        "on_validation_start",
        "on_validation_epoch_start",
        "on_test_start",
        "on_test_epoch_start",
        "on_predict_start",
        "on_predict_epoch_start",
        "on_predict_end",
        "on_predict_epoch_end",
    }
)


def _take_dataloader_iter(model: Module) -> None:
    """Support ``training_step(self, dataloader_iter)``: the step pulls its own batches. Until
    the end of ``fit`` the Module's ``training_step`` takes what the loop passes."""
    step = model.training_step
    if "dataloader_iter" not in inspect.signature(step).parameters:
        return

    def from_iterator(dataloader_iter: Any, batch_idx: int) -> Any:
        try:
            return step(dataloader_iter)  # type: ignore[call-arg]
        except StopIteration:  # the step asked for more batches than were left
            return None

    model.__dict__["training_step"] = from_iterator


class _DataloaderIter:
    """What ``training_step(self, dataloader_iter)`` receives: ``next`` gives ``(batch,
    batch_idx, dataloader_idx)``."""

    def __init__(self, batches: Iterable[Any]) -> None:
        self._it = iter(batches)
        self._peeked: list[Any] = []
        self._index = -1

    def __iter__(self) -> _DataloaderIter:
        return self

    def __next__(self) -> tuple[Any, int, int]:
        batch = self._peeked.pop() if self._peeked else next(self._it)
        self._index += 1
        return batch, self._index, 0

    def has_next(self) -> bool:
        if not self._peeked:
            try:
                self._peeked.append(next(self._it))
            except StopIteration:
                return False
        return True


def _IterSteps(batches: Iterable[Any]) -> Iterator[_DataloaderIter]:  # noqa: N802
    """One training step for as long as the loader has batches."""
    feed = _DataloaderIter(batches)
    while feed.has_next():
        yield feed


def _lightning_position(fit_loop: dict[str, Any]) -> tuple[int, int]:
    """The epoch and batch to continue from, read from a Lightning checkpoint's loop state:
    within an epoch it has started but not processed one."""
    epochs = fit_loop["epoch_progress"]["current"]
    batches = fit_loop["epoch_loop.batch_progress"]
    if epochs["ready"] <= epochs["processed"]:
        return epochs["processed"], 0
    if batches.get("is_last_batch"):
        return epochs["ready"], 0
    return epochs["processed"], batches["current"]["processed"]


def _kept_checkpoint() -> Path | None:
    """In a later attempt of a job with a resume policy, the checkpoint the control plane
    kept from the attempt before. It is newer than any ``ckpt_path`` the script names."""
    attempt = int(os.environ.get("GPULAYER_ATTEMPT", "1"))
    folder = os.environ.get("GPULAYER_INPUT_DIR")
    if attempt < 2 or not folder or "GPULAYER_CHECKPOINT_STREAM" not in os.environ:
        return None
    kept = Path(folder) / RESUME_INPUT
    return kept if kept.is_file() else None


def _resume_path(ckpt_path: str | Path | None) -> str | Path | None:
    """Inside a job launched with ckpt_path, the checkpoint arrived as a job input."""
    if ckpt_path is not None and "GPULAYER_INPUT_DIR" in os.environ:
        shipped = Path(os.environ["GPULAYER_INPUT_DIR"]) / RESUME_INPUT
        if shipped.is_file():
            return shipped
        folder = shipped.with_name(RESUME_INPUT + ".d")
        if folder.is_dir():
            return folder
    return ckpt_path


def _suffixed(name: str, kind: str) -> str:
    """``<name>_step`` or ``<name>_epoch``, before any ``/dataloader_idx_N`` (Lightning)."""
    base, sep, rest = name.partition("/dataloader_idx_")
    return f"{base}_{kind}{sep}{rest}"


def _metrics_of(history: list[dict[str, Any]]) -> dict[str, float]:
    """``callback_metrics`` of a run known only by its history."""
    last = history[-1] if history else {}
    return {k: v for k, v in last.items() if k not in ("epoch", "epoch_seconds")}


def _compute_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Compute and reset the ``torchmetrics`` objects logged in a loop, and forget them. Each
    syncs across processes in ``compute()``; every rank logs the same names, so the
    collectives line up."""
    values: dict[str, float] = {}
    for name, metric in metrics.items():
        value = metric.compute()
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            raise ValidationFailed(
                f"self.log({name!r}, metric): compute() returned a tensor of shape "
                f"{tuple(value.shape)}; log a metric that computes one value"
            )
        values[name] = float(value)
        metric.reset()
    metrics.clear()
    return values


def _total(values: list[Any]) -> float:
    """Sum of logged values: one stack and sum for tensors, a plain sum for numbers."""
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    total = float(sum(v for v in values if not isinstance(v, torch.Tensor)))
    if tensors:
        total += float(torch.stack([t.reshape(()) for t in tensors]).sum())
    return total


def _weighted_total(values: list[Any], weights: list[float]) -> float:
    """Sum of value x weight (one stack and multiply on the values' device)."""
    tensors = [
        v.reshape(()) if isinstance(v, torch.Tensor) else torch.tensor(float(v)) for v in values
    ]
    dev = tensors[0].device
    w = torch.tensor(weights, device=dev, dtype=torch.float32)
    return float((torch.stack([t.to(dev).float() for t in tensors]) * w).sum())


def _batch_size(batch: Any) -> float:
    """The batch size Lightning infers when ``self.log`` gets no ``batch_size``: the first
    dimension of the first tensor found in the batch (1 when there is none). Reads only
    shapes, so it does not synchronise with the device."""
    stack = [batch]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Tensor):
            return float(item.shape[0]) if item.dim() else 1.0
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return 1.0


def _extreme(values: list[Any], fx: str) -> float:
    tensors = [
        v.reshape(()) if isinstance(v, torch.Tensor) else torch.tensor(float(v)) for v in values
    ]
    dev = tensors[0].device
    stacked = torch.stack([t.to(dev).float() for t in tensors])
    return float(stacked.max() if fx == "max" else stacked.min())


_HINTED: set[str] = set()


def _performance_hints(rt: Runtime, loaders: list[Any], local_processes: int) -> None:
    """Lightning's two speed hints, once per process, as UserWarnings (silence them with
    ``gpulayer.pytorch.utilities.disable_possible_user_warnings``)."""
    import warnings

    if (
        "tf32" not in _HINTED
        and rt.device.type == "cuda"
        and torch.cuda.get_device_capability(rt.device)[0] >= 8
        and torch.get_float32_matmul_precision() == "highest"
    ):
        _HINTED.add("tf32")
        warnings.warn(
            f"{torch.cuda.get_device_name(rt.device)} has Tensor Cores: "
            "torch.set_float32_matmul_precision('high') makes float32 matmuls several times "
            "faster at slightly lower precision",
            UserWarning,
            stacklevel=3,
        )
    from gpulayer.pytorch.utilities import suggested_max_num_workers

    bound = suggested_max_num_workers(max(1, local_processes))
    for loader in loaders:
        workers = getattr(loader, "num_workers", None)
        if "workers" not in _HINTED and workers is not None and workers < 2 and bound > 1:
            _HINTED.add("workers")
            warnings.warn(
                f"the training DataLoader has num_workers={workers}; loading may limit speed. "
                f"This machine can give each process up to num_workers={bound}",
                UserWarning,
                stacklevel=3,
            )


def _checkpoints_last(callbacks: list[Callback]) -> list[Callback]:
    """Checkpoint callbacks after the others, as in Lightning, so they save the final state."""
    return [c for c in callbacks if not isinstance(c, Checkpoint)] + [
        c for c in callbacks if isinstance(c, Checkpoint)
    ]


def _post_local_sgd(optimizers: list[Any], strategy: Any, rt: Runtime) -> list[Any]:
    """Post-local SGD (``DDPStrategy(model_averaging_period=N)``): wrap each optimizer so the
    models are averaged every N steps after the warm-up, as Lightning does."""
    from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState
    from torch.distributed.algorithms.model_averaging.averagers import PeriodicModelAverager
    from torch.distributed.optim import PostLocalSGDOptimizer

    state = strategy.ddp_comm_state
    if not isinstance(state, PostLocalSGDState):
        raise ValidationFailed(
            "model_averaging_period needs ddp_comm_state=PostLocalSGDState(...) and "
            "ddp_comm_hook=post_localSGD_hook"
        )
    if state.subgroup is None:
        # PyTorch's default sizes the per-machine subgroup from the CUDA device count, which
        # fails on CPU processes: use the processes on each machine, the same group on GPUs.
        import torch.distributed as dist

        state.subgroup, _ = dist.new_subgroups(group_size=rt.local_world_size)
    period = strategy.model_averaging_period
    return [
        PostLocalSGDOptimizer(
            optim=o,
            averager=PeriodicModelAverager(period=period, warmup_steps=state.start_localSGD_iter),
        )
        for o in optimizers
    ]


class _EmptyInit(torch.overrides.TorchFunctionMode):
    """Skip ``torch.nn.init`` calls while a model is built (``init_module(empty_init=True)``),
    as Lightning does: the weights are about to be replaced by a checkpoint."""

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            return kwargs["tensor"] if "tensor" in kwargs else args[0]
        return func(*args, **kwargs)


class SIGTERMException(SystemExit):
    """Raised by ``fit`` after a SIGTERM (cloud preemption, a cancelled job) once the
    checkpoint is saved; a ``SystemExit``, so the process ends with an error status."""

    def __init__(self, message: str = "SIGTERM received") -> None:
        super().__init__(143)
        self.message = message

    def __str__(self) -> str:
        return self.message


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _run_name() -> str:
    """Folder for in-process runs, named after the script and its directory,
    e.g. gpulayer_runs/my_project-train for my_project/train.py."""
    if not sys.argv or not sys.argv[0] or sys.argv[0] in ("-c", "-m"):
        return "local"
    script = Path(sys.argv[0]).resolve()
    return f"{script.parent.name}-{script.stem}"
