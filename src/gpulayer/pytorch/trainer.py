"""A small Trainer that runs the training loop and also decides WHERE training runs.

    trainer = Trainer(max_epochs=5, accelerator="auto")                    # this process
    trainer = Trainer(max_epochs=5, accelerator="cpu", devices=2)          # 2-process DDP
    trainer = Trainer(max_epochs=5, accelerator="cuda", devices=4, num_nodes=2,
                      target="remote")                                     # 8 GPUs elsewhere
    trainer.fit(model, train_loader, val_loader)

When the run cannot happen in this process (``target`` given, or more than one process),
``fit`` re-executes this script on GpuLayer workers through ``launch``; on the workers the
same ``fit`` call runs the training loop. Afterwards the final weights are loaded into the
local ``model`` and ``callback_metrics`` / ``history`` hold the remote metrics.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.launch import LaunchResult, is_launcher_process, launch
from gpulayer.pytorch.module import Module, _TrainingStep
from gpulayer.pytorch.runtime import Runtime, resolve_accelerator
from gpulayer.sdk.remote import RemoteError, default_client, default_target, explain_failure

CHECKPOINT = "checkpoints/last.ckpt"


class Trainer:
    def __init__(
        self,
        *,
        max_epochs: int = 1,
        accelerator: str = "auto",
        devices: int = 1,
        num_nodes: int = 1,
        precision: str = "32-true",
        target: str | None = None,
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int = 1,
        log_every_n_steps: int = 50,
        limit_train_batches: int | None = None,
        limit_val_batches: int | None = None,
        enable_checkpointing: bool = True,
        default_root_dir: str | Path = "gpulayer_runs",
        launch_options: dict[str, Any] | None = None,
    ) -> None:
        if devices < 1 or num_nodes < 1 or max_epochs < 1 or accumulate_grad_batches < 1:
            raise ValidationFailed(
                "devices, num_nodes, max_epochs and accumulate_grad_batches must be >= 1"
            )
        self.max_epochs = max_epochs
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
        self.global_step = 0
        self.current_epoch = 0
        self.launch_result: LaunchResult | None = None
        self.runtime: Runtime | None = None
        self.checkpoint_path: Path | None = None

    # --------------------------------------------------------------- placement

    @property
    def will_launch(self) -> bool:
        """True when ``fit`` will run on GpuLayer workers instead of this process."""
        return is_launcher_process() and (
            self.target is not None or self.devices * self.num_nodes > 1
        )

    @property
    def is_global_zero(self) -> bool:
        return self.runtime.is_global_zero if self.runtime else True

    def _launch(self, model: Module) -> None:
        options = dict(self.launch_options)
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
            self._launch(model)
            return
        rt = Runtime(self.accelerator, self.precision).setup()
        self.runtime = rt
        try:
            self._fit_loop(rt, model, train_dataloaders, val_dataloaders, ckpt_path)
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
        optimizer, scheduler, interval = self._optimizers(model)
        start_epoch = 0
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=rt.device, weights_only=False)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            self.global_step = state.get("global_step", 0)
            self.history = state.get("history", [])
        step_fn = rt.module(_TrainingStep(model))
        train = rt.dataloader(train)
        val = rt.dataloader(val, shuffle=False) if val is not None else None
        out_dir = rt.output_dir(self.default_root_dir / _run_name())
        where = f"{rt.accelerator}" + (
            f", rank {rt.global_rank}/{rt.world_size}" if rt.distributed else ""
        )
        self._say(
            f"[gpulayer] training on {where}, precision {rt.precision}, {self.max_epochs} epoch(s)"
        )

        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch
            if isinstance(train.sampler, DistributedSampler):
                train.sampler.set_epoch(epoch)
            model.train()
            # Running sums stay on the device; see Module.log.
            sums: dict[str, Any] = defaultdict(float)
            count = 0
            t0 = time.perf_counter()
            accumulate = self.accumulate_grad_batches
            n_batches = len(train) if hasattr(train, "__len__") else None
            if self.limit_train_batches is not None:
                n_batches = min(n_batches or self.limit_train_batches, self.limit_train_batches)
            for batch_idx, batch in enumerate(train):
                if self.limit_train_batches is not None and batch_idx >= self.limit_train_batches:
                    break
                batch = rt.to_device(batch)
                model._gl_step_logs = {}
                # Apply the optimizer every `accumulate` batches (and on the last one). On the
                # other batches DDP skips its gradient all-reduce.
                apply = (batch_idx + 1) % accumulate == 0 or batch_idx + 1 == n_batches
                with rt.no_sync(step_fn, enabled=not apply):
                    with rt.autocast():
                        loss = step_fn(batch, batch_idx)
                    if not isinstance(loss, torch.Tensor):
                        raise ValidationFailed("training_step must return the loss tensor")
                    rt.backward(loss / accumulate if accumulate > 1 else loss)
                if apply:
                    rt.step(optimizer, model.parameters(), self.gradient_clip_val)
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and interval == "step":
                        scheduler.step()
                    self.global_step += 1
                logs = model._gl_step_logs or {"train_loss": loss.detach().float()}
                for k, v in logs.items():
                    sums[k] = sums[k] + v
                count += 1
                if (
                    apply
                    and self.log_every_n_steps
                    and self.global_step % self.log_every_n_steps == 0
                ):
                    shown = " ".join(f"{k}={float(v):.4f}" for k, v in logs.items())
                    self._say(f"epoch {epoch} step {self.global_step} {shown}")
            if scheduler is not None and interval == "epoch":
                scheduler.step()
            metrics = {k: rt.mean(float(v) / max(count, 1)) for k, v in sums.items()}
            if val is not None:
                metrics.update(self._evaluate(rt, model, val))
            metrics["epoch"] = epoch
            metrics["epoch_seconds"] = round(time.perf_counter() - t0, 3)
            self.history.append(metrics)
            self.callback_metrics = dict(metrics)
            shown = " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if k not in ("epoch", "epoch_seconds")
            )
            self._say(f"epoch {epoch} done in {metrics['epoch_seconds']}s: {shown}")
            if self.enable_checkpointing and rt.is_global_zero:
                self._save(out_dir, model, optimizer, epoch)
            rt.barrier()

    def _evaluate(self, rt: Runtime, model: Module, loader: DataLoader) -> dict[str, float]:
        model.eval()
        sums: dict[str, Any] = defaultdict(float)
        count = 0
        with torch.no_grad(), rt.autocast():
            for batch_idx, batch in enumerate(loader):
                if self.limit_val_batches is not None and batch_idx >= self.limit_val_batches:
                    break
                model._gl_step_logs = {}
                model.validation_step(rt.to_device(batch), batch_idx)
                for k, v in model._gl_step_logs.items():
                    sums[k] = sums[k] + v
                count += 1
        model.train()
        return {k: rt.mean(float(v) / max(count, 1)) for k, v in sums.items()}

    def validate(self, model: Module, dataloaders: DataLoader) -> dict[str, float]:
        """Run ``validation_step`` over ``dataloaders`` in this process."""
        rt = Runtime(self.accelerator, self.precision).setup()
        self.runtime = rt
        try:
            model.to(rt.device)
            return self._evaluate(rt, model, rt.dataloader(dataloaders, shuffle=False))
        finally:
            rt.teardown()

    def _save(
        self, out_dir: Path, model: Module, optimizer: torch.optim.Optimizer, epoch: int
    ) -> None:
        path = out_dir / CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "history": self.history,
            },
            tmp,
        )
        tmp.replace(path)
        (out_dir / "metrics.json").write_text(json.dumps(self.history, indent=2))
        self.checkpoint_path = path


def _run_name() -> str:
    """Folder for in-process runs, named after the script and its directory,
    e.g. gpulayer_runs/my_project-train for my_project/train.py."""
    import sys

    if not sys.argv or not sys.argv[0] or sys.argv[0] in ("-c", "-m"):
        return "local"
    script = Path(sys.argv[0]).resolve()
    return f"{script.parent.name}-{script.stem}"
