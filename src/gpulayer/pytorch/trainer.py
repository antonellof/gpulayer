"""A small Trainer that runs the training loop and also decides WHERE training runs.

    trainer = Trainer(max_epochs=5, accelerator="auto")                    # this process
    trainer = Trainer(max_epochs=5, accelerator="cpu", devices=2)          # 2-process DDP
    trainer = Trainer(max_epochs=5, accelerator="cuda", devices=4, num_nodes=2,
                      target="remote")                                     # 8 GPUs elsewhere
    trainer.fit(model, train_loader, val_loader)

When the run cannot happen in this process (``target`` given, or more than one process),
``fit`` re-executes this script on GPUlayer workers through ``launch``; on the workers the
same ``fit`` call runs the training loop. Afterwards the final weights are loaded into the
local ``model`` and ``callback_metrics`` / ``history`` hold the remote metrics.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.callbacks import Callback, _call
from gpulayer.pytorch.launch import LaunchResult, is_launcher_process, launch
from gpulayer.pytorch.module import Module, _TrainingStep
from gpulayer.pytorch.runtime import GRADIENT_SYNC, Runtime, resolve_accelerator
from gpulayer.sdk.remote import RemoteError, default_client, default_target, explain_failure

CHECKPOINT = "checkpoints/last.ckpt"


class Trainer:
    def __init__(
        self,
        *,
        max_epochs: int = 1,
        max_steps: int | None = None,
        check_val_every_n_epoch: int = 1,
        val_check_interval: int | float | None = None,
        accelerator: str = "auto",
        devices: int = 1,
        num_nodes: int = 1,
        strategy: str = "ddp",
        precision: str = "32-true",
        gradient_sync: str = "auto",
        target: str | None = None,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        accumulate_grad_batches: int = 1,
        log_every_n_steps: int = 50,
        limit_train_batches: int | None = None,
        limit_val_batches: int | None = None,
        num_sanity_val_steps: int = 2,
        fast_dev_run: bool | int = False,
        max_time: float | None = None,
        deterministic: bool = False,
        benchmark: bool | None = None,
        prefetch: bool = False,
        enable_checkpointing: bool = True,
        default_root_dir: str | Path = "gpulayer_runs",
        launch_options: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        if (
            min(devices, num_nodes, max_epochs, accumulate_grad_batches, check_val_every_n_epoch)
            < 1
        ):
            raise ValidationFailed(
                "devices, num_nodes, max_epochs, accumulate_grad_batches and "
                "check_val_every_n_epoch must be >= 1"
            )
        if max_steps is not None and max_steps < 1:
            raise ValidationFailed("max_steps must be >= 1 (or None for no limit)")
        if gradient_clip_algorithm not in ("norm", "value"):
            raise ValidationFailed(
                "gradient_clip_algorithm must be 'norm' or 'value', "
                f"got {gradient_clip_algorithm!r}"
            )
        if num_sanity_val_steps < 0 or (max_time is not None and max_time <= 0):
            raise ValidationFailed("num_sanity_val_steps must be >= 0 and max_time > 0")
        if fast_dev_run:
            # A quick end-to-end check: n training and n validation batches, one epoch, no
            # checkpoints. True means 1.
            n = 1 if fast_dev_run is True else int(fast_dev_run)
            if n < 1:
                raise ValidationFailed("fast_dev_run must be True or a number of batches >= 1")
            max_epochs, max_steps, enable_checkpointing = 1, None, False
            limit_train_batches, limit_val_batches, num_sanity_val_steps = n, n, 0
            log_every_n_steps = 1
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.num_sanity_val_steps = num_sanity_val_steps
        self.fast_dev_run = bool(fast_dev_run)
        self.max_time = max_time
        self.deterministic = deterministic
        if benchmark and deterministic:
            raise ValidationFailed(
                "benchmark=True picks kernels by speed; it conflicts with deterministic=True"
            )
        self.benchmark = benchmark
        self.prefetch = prefetch
        if gradient_sync not in GRADIENT_SYNC:
            raise ValidationFailed(
                f"gradient_sync must be one of {GRADIENT_SYNC}, got {gradient_sync!r}"
            )
        self.gradient_sync = gradient_sync
        if strategy not in ("ddp", "fsdp"):
            raise ValidationFailed(f"strategy must be 'ddp' or 'fsdp', got {strategy!r}")
        if strategy == "fsdp" and precision == "16-mixed":
            raise ValidationFailed(
                "strategy='fsdp' supports precision '32-true' and 'bf16-mixed' (16-mixed needs a "
                "sharded gradient scaler that has not been validated yet)"
            )
        self.strategy = strategy
        self.max_epochs = max_epochs
        self.max_steps = max_steps
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
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.enable_checkpointing = enable_checkpointing
        self.default_root_dir = Path(default_root_dir)
        self.launch_options = launch_options or {}
        self.callback_metrics: dict[str, float] = {}
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
        self.callbacks: list[Callback] = list(callbacks or [])
        self.should_stop = False  # set by a callback (e.g. EarlyStopping) to end training
        self._out_dir: Path | None = None
        self._model: Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None

    # --------------------------------------------------------------- placement

    @property
    def will_launch(self) -> bool:
        """True when ``fit`` will run on GPUlayer workers instead of this process."""
        return is_launcher_process() and (
            self.target is not None or self.devices * self.num_nodes > 1
        )

    @property
    def is_global_zero(self) -> bool:
        return self.runtime.is_global_zero if self.runtime else True

    def _launch(self, model: Module, ckpt_path: str | Path | None = None) -> None:
        options = dict(self.launch_options)
        if ckpt_path is not None:
            # The checkpoint is on this machine: ship it with the job as an input; the copy of
            # this script on the workers finds it there (see _resume_path).
            local = Path(ckpt_path)
            if not local.is_file():
                raise ValidationFailed(f"checkpoint not found: {local}")
            options["inputs"] = {**options.get("inputs", {}), RESUME_INPUT: local}
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
            output_dir=self.default_root_dir,
            exit=False,
            **options,
        )
        assert result is not None
        self.launch_result = result
        if not result.succeeded:
            raise RemoteError(
                f"remote training {result.status.state.value}: {explain_failure(result.status)}",
                result.job,
            )
        ckpts = sorted(result.output_dir.rglob("last.ckpt"))
        if ckpts:
            self.checkpoint_path = ckpts[0]  # node0/... sorts first
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            self.history = state.get("history", [])
            self.callback_metrics = dict(self.history[-1]) if self.history else {}
            self.global_step = state.get("global_step", 0)
            self.current_epoch = state.get("epoch", 0)
        else:
            # Without checkpoints, the metrics still come back (weights do not).
            found = sorted(result.output_dir.rglob("metrics.json"))
            if found:
                self.history = json.loads(found[0].read_text())
                self.callback_metrics = dict(self.history[-1]) if self.history else {}
                if self.history:
                    self.current_epoch = int(self.history[-1].get("epoch", 0))

    # --------------------------------------------------------------- fitting

    def fit(
        self,
        model: Module,
        train_dataloaders: DataLoader,
        val_dataloaders: DataLoader | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        if not isinstance(model, Module):
            raise ValidationFailed("Trainer.fit expects a gpulayer.pytorch.Module")
        if self.will_launch:
            # Training runs on workers (the callbacks run there); here only on_fit_end runs,
            # so callbacks can pick up what came back (e.g. ModelCheckpoint's best file).
            self._launch(model, ckpt_path)
            _call(self.callbacks, "on_fit_end", self, model)
            return
        if self.deterministic:
            # Same results on every run (with seed_everything), at some speed cost. cuBLAS
            # reads its setting when CUDA starts, so set it first.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        rt = Runtime(self.accelerator, self.precision, self.gradient_sync).setup()
        self.runtime = rt
        if rt.device.type == "cuda" and not self.deterministic:
            # cuDNN times its algorithms for each new input shape and keeps the fastest. Faster
            # for fixed shapes (the usual case); varying shapes pay the timing again each time.
            torch.backends.cudnn.benchmark = self.benchmark is not False
        try:
            self._fit_loop(rt, model, train_dataloaders, val_dataloaders, _resume_path(ckpt_path))
        finally:
            rt.teardown()
        if is_launcher_process():
            # Same result as a submitted run: the trained model comes back on the CPU.
            model.cpu()

    def _optimizers(self, model: Module) -> tuple[torch.optim.Optimizer, Any, str]:
        cfg = model.configure_optimizers()
        if isinstance(cfg, torch.optim.Optimizer):
            return cfg, None, "epoch"
        if isinstance(cfg, (tuple, list)) and len(cfg) == 2:
            return cfg[0], cfg[1], "epoch"
        if isinstance(cfg, dict) and "optimizer" in cfg:
            return cfg["optimizer"], cfg.get("lr_scheduler"), cfg.get("interval", "epoch")
        raise ValidationFailed(
            "configure_optimizers must return an optimizer, "
            "(optimizer, scheduler) or a dict with 'optimizer'"
        )

    def _say(self, msg: str) -> None:
        if self.is_global_zero:
            print(msg, flush=True)

    def _fit_loop(
        self,
        rt: Runtime,
        model: Module,
        train: DataLoader,
        val: DataLoader | None,
        ckpt_path: str | Path | None,
    ) -> None:
        model.trainer = self
        model.to(rt.device)
        fsdp = self.strategy == "fsdp" and rt.distributed
        sharded: torch.nn.Module | None = None
        if fsdp:
            # Shard before the optimizer, which must see the sharded parameters. The wrapper
            # (whose forward calls training_step) is the FSDP root, called every step.
            if next(model.parameters(recurse=False), None) is not None:
                raise ValidationFailed(
                    "strategy='fsdp' shards submodules; move parameters the Module owns "
                    "directly (nn.Parameter attributes) into a submodule such as nn.Linear"
                )
            sharded = rt.shard(_TrainingStep(model))
        optimizer, scheduler, interval = self._optimizers(model)
        self._model, self._optimizer, self.should_stop = model, optimizer, False
        start_epoch = 0
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            rt.load_full_state(model, optimizer, state["state_dict"], state["optimizer"])
            start_epoch = state["epoch"] + 1
            self.global_step = state.get("global_step", 0)
            self.history = state.get("history", [])
        # DDP only syncs gradients for calls to its forward, so wrap training_step for DDP;
        # FSDP hooks into each sharded submodule instead, and a single process needs neither
        # (one module call fewer per step).
        step_fn: Any
        if sharded is not None:
            ddp, step_fn = sharded, sharded
        elif rt.distributed:
            ddp = step_fn = rt.module(_TrainingStep(model))
        else:
            ddp, step_fn = model, model.training_step
        if not rt.distributed:
            model.to(rt.device)
        train = rt.dataloader(train)
        val = rt.dataloader(val, shuffle=False) if val is not None else None
        out_dir = rt.output_dir(self.default_root_dir / _run_name())
        self._out_dir = out_dir
        where = f"{rt.accelerator}" + (
            f", rank {rt.global_rank}/{rt.world_size}" if rt.distributed else ""
        )
        if sharded is None and rt.uses_two_level_sync():
            where += f", two-level gradient sync ({rt.local_world_size} per machine)"
        self._say(
            f"[gpulayer] training on {where}, precision {rt.precision}, {self.max_epochs} epoch(s)"
        )

        _call(self.callbacks, "on_fit_start", self, model)
        fit_t0 = time.perf_counter()
        if val is not None and self.num_sanity_val_steps and not ckpt_path:
            # Run a few validation batches first, so a broken validation_step fails now and
            # not after the first training epoch. Results are discarded.
            limit, self.limit_val_batches = (
                self.limit_val_batches,
                min(self.limit_val_batches or self.num_sanity_val_steps, self.num_sanity_val_steps),
            )
            try:
                self._evaluate(rt, model, val)
            finally:
                self.limit_val_batches = limit
        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch
            _call(self.callbacks, "on_train_epoch_start", self, model)
            if isinstance(train.sampler, DistributedSampler):
                train.sampler.set_epoch(epoch)
            model.train()
            # Logged values stay on the device (see Module.log) and are added up once per
            # epoch, not with one tensor operation per step.
            logged: dict[str, list[Any]] = defaultdict(list)
            count = 0
            t0 = time.perf_counter()
            accumulate = self.accumulate_grad_batches
            n_batches = len(train) if hasattr(train, "__len__") else None
            if self.limit_train_batches is not None:
                n_batches = min(n_batches or self.limit_train_batches, self.limit_train_batches)
            val_every = self._val_every(n_batches) if val is not None else None
            for batch_idx, batch in enumerate(rt.batches(train, prefetch=self.prefetch)):
                if self.limit_train_batches is not None and batch_idx >= self.limit_train_batches:
                    break
                model._gl_step_logs = {}
                # Apply the optimizer every `accumulate` batches (and on the last one). On the
                # other batches DDP skips its gradient all-reduce.
                apply = (batch_idx + 1) % accumulate == 0 or batch_idx + 1 == n_batches
                with rt.no_sync(ddp, enabled=not apply):
                    with rt.autocast():
                        loss = step_fn(batch, batch_idx)
                    if not isinstance(loss, torch.Tensor):
                        raise ValidationFailed("training_step must return the loss tensor")
                    rt.backward(loss / accumulate if accumulate > 1 else loss)
                if apply:
                    rt.step(
                        optimizer,
                        model.parameters(),
                        self.gradient_clip_val,
                        self.gradient_clip_algorithm,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and interval == "step":
                        scheduler.step()
                    self.global_step += 1
                    if self.max_steps is not None and self.global_step >= self.max_steps:
                        self.should_stop = True
                    if (
                        self.max_time is not None
                        and not rt.distributed  # ranks agree on stopping at the epoch end
                        and time.perf_counter() - fit_t0 >= self.max_time
                    ):
                        self.should_stop = True
                        break
                logs = model._gl_step_logs or {"train_loss": loss.detach().float()}
                for k, v in logs.items():
                    logged[k].append(v)
                count += 1
                if self.callbacks:
                    _call(
                        self.callbacks, "on_train_batch_end", self, model, loss.detach(), batch_idx
                    )
                if (
                    apply
                    and self.log_every_n_steps
                    and self.global_step % self.log_every_n_steps == 0
                ):
                    values = {k: float(v) for k, v in logs.items()}
                    shown = " ".join(f"{k}={v:.4f}" for k, v in values.items())
                    self._say(f"epoch {epoch} step {self.global_step} {shown}")
                    self._step_rows.append({"epoch": epoch, "step": self.global_step, **values})
                if (
                    val_every
                    and (batch_idx + 1) % val_every == 0
                    and (n_batches is None or batch_idx + 1 < n_batches)
                ):
                    # Same batch index on every rank, so the collectives inside line up.
                    assert val is not None
                    mid = self._evaluate(rt, model, val)
                    self.val_history.append({"epoch": epoch, "step": self.global_step, **mid})
                    self.callback_metrics.update(mid)
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break  # same step count on every rank, so no extra synchronisation
            if scheduler is not None and interval == "epoch":
                scheduler.step()
            metrics = {k: rt.mean(_total(v) / max(count, 1)) for k, v in logged.items()}
            last = epoch + 1 == self.max_epochs or (
                self.max_steps is not None and self.global_step >= self.max_steps
            )
            if val is not None and ((epoch + 1) % self.check_val_every_n_epoch == 0 or last):
                metrics.update(self._evaluate(rt, model, val))
            metrics["epoch"] = epoch
            metrics["epoch_seconds"] = round(time.perf_counter() - t0, 3)
            self.history.append(metrics)
            self.callback_metrics = dict(metrics)
            shown = " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k not in ("epoch", "epoch_seconds")
            )
            self._say(f"epoch {epoch} done in {metrics['epoch_seconds']}s: {shown}")
            _call(self.callbacks, "on_train_epoch_end", self, model)
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
            # Every rank must leave the loop together, or the next collective would hang.
            self.should_stop = rt.any(self.should_stop)
            rt.barrier()
            if self.should_stop:
                break
        _call(self.callbacks, "on_fit_end", self, model)

    def _evaluate(
        self, rt: Runtime, model: Module, loader: DataLoader, step: str = "validation_step"
    ) -> dict[str, float]:
        model.eval()
        step_fn = getattr(model, step)
        sums: dict[str, Any] = defaultdict(float)
        count = 0
        with torch.no_grad(), rt.autocast():
            for batch_idx, batch in enumerate(rt.batches(loader, prefetch=self.prefetch)):
                if self.limit_val_batches is not None and batch_idx >= self.limit_val_batches:
                    break
                model._gl_step_logs = {}
                step_fn(batch, batch_idx)
                for k, v in model._gl_step_logs.items():
                    sums[k] = sums[k] + v
                count += 1
        model.train()
        return {k: rt.mean(float(v) / max(count, 1)) for k, v in sums.items()}

    def validate(self, model: Module, dataloaders: DataLoader) -> dict[str, float]:
        """Run ``validation_step`` over ``dataloaders`` in this process."""
        return self._run_eval(model, dataloaders, "validation_step")

    def test(self, model: Module, dataloaders: DataLoader) -> dict[str, float]:
        """Run ``test_step`` over ``dataloaders`` in this process and return the averaged logs,
        e.g. ``{"test_acc": 0.93}``."""
        return self._run_eval(model, dataloaders, "test_step")

    def predict(self, model: Module, dataloaders: DataLoader) -> list[Any]:
        """Run ``predict_step`` over ``dataloaders`` in this process (no gradients) and return
        one output per batch, moved to the CPU."""
        rt = Runtime(self.accelerator, self.precision, self.gradient_sync).setup()
        self.runtime = rt
        out: list[Any] = []
        try:
            model.to(rt.device)
            model.eval()
            with torch.no_grad(), rt.autocast():
                loader = rt.dataloader(dataloaders, shuffle=False)
                for batch_idx, batch in enumerate(rt.batches(loader, prefetch=self.prefetch)):
                    out.append(_to_cpu(model.predict_step(batch, batch_idx)))
            return out
        finally:
            rt.teardown()

    def _run_eval(self, model: Module, dataloaders: DataLoader, step: str) -> dict[str, float]:
        rt = Runtime(self.accelerator, self.precision, self.gradient_sync).setup()
        self.runtime = rt
        try:
            model.to(rt.device)
            metrics = self._evaluate(rt, model, rt.dataloader(dataloaders, shuffle=False), step)
            self.callback_metrics.update(metrics)
            return metrics
        finally:
            rt.teardown()

    def save_checkpoint(self, filename: str) -> Path | None:
        """Write ``checkpoints/<filename>`` for the current epoch from rank 0 (other ranks
        return None). Usable from callbacks during ``fit``. Call it on every rank: with
        ``strategy="fsdp"`` the full state is gathered from all of them."""
        if self._out_dir is None or self._model is None or self._optimizer is None:
            raise ValidationFailed("save_checkpoint can only be called during fit")
        rt = self.runtime
        if rt is not None and self.strategy == "fsdp" and rt.distributed:
            model_state, optim_state = rt.full_state(self._model, self._optimizer)
        elif rt is not None and not rt.is_global_zero:
            return None
        else:
            model_state, optim_state = self._model.state_dict(), self._optimizer.state_dict()
        if rt is not None and not rt.is_global_zero:
            return None
        path = self._out_dir / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "state_dict": model_state,
                "optimizer": optim_state,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "history": self.history,
            },
            tmp,
        )
        tmp.replace(path)
        return path

    def _write_metrics(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(self.history, indent=2))
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


RESUME_INPUT = "gpulayer-resume.ckpt"


def _resume_path(ckpt_path: str | Path | None) -> str | Path | None:
    """Inside a job launched with ckpt_path, the checkpoint arrived as a job input."""
    if ckpt_path is not None and "GPULAYER_INPUT_DIR" in os.environ:
        shipped = Path(os.environ["GPULAYER_INPUT_DIR"]) / RESUME_INPUT
        if shipped.is_file():
            return shipped
    return ckpt_path


def _total(values: list[Any]) -> float:
    """Sum of logged values: one stack and sum for tensors, a plain sum for numbers."""
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    total = float(sum(v for v in values if not isinstance(v, torch.Tensor)))
    if tensors:
        total += float(torch.stack([t.reshape(()) for t in tensors]).sum())
    return total


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
    import sys

    if not sys.argv or not sys.argv[0] or sys.argv[0] in ("-c", "-m"):
        return "local"
    script = Path(sys.argv[0]).resolve()
    return f"{script.parent.name}-{script.stem}"
