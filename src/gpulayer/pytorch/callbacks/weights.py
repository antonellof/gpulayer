"""Callbacks that change the weights: averaging (SWA, EMA), fine-tuning schedules, pruning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from torch.optim.swa_utils import SWALR, AveragedModel, get_ema_multi_avg_fn

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.callbacks.base import Callback

if TYPE_CHECKING:
    from gpulayer.pytorch.module import Module
    from gpulayer.pytorch.trainer import Trainer


def _copy_into(dst: nn.Module, src: nn.Module, buffers: bool = True) -> None:
    with torch.no_grad():
        for d, s in zip(dst.parameters(), src.parameters(), strict=True):
            d.data.copy_(s.data.to(d.device))
        if buffers:
            for db, sb in zip(dst.buffers(), src.buffers(), strict=True):
                db.copy_(sb.to(db.device))


def _averaged_state(avg: AveragedModel) -> dict[str, Any]:
    return {k: v for k, v in avg.module.state_dict().items()}


class WeightAveraging(Callback):
    """Keep an average of the weights during training (``torch.optim.swa_utils``), use it for
    validation, and leave it in the model and the checkpoints at the end.

    ``should_update(step_idx=..., epoch_idx=...)`` decides when to average; by default after
    every optimizer step. Extra arguments go to ``AveragedModel`` (``avg_fn`` or
    ``multi_avg_fn``). Checkpoints hold the averaged weights in ``state_dict`` and the
    training weights in ``current_model_state``, so resuming continues both."""

    def __init__(
        self, device: torch.device | str | None = None, use_buffers: bool = True, **kwargs: Any
    ) -> None:
        self._device = device
        self._use_buffers = use_buffers
        self._kwargs = kwargs
        self._average_model: AveragedModel | None = None
        self._backup: nn.Module | None = None
        self._loaded_average: dict[str, Any] | None = None

    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        return step_idx is not None

    def on_fit_start(self, trainer: Trainer, pl_module: Module) -> None:
        device = torch.device(self._device) if self._device is not None else pl_module.device
        self._average_model = AveragedModel(
            pl_module, device=device, use_buffers=self._use_buffers, **self._kwargs
        )
        if self._loaded_average is not None:
            self._average_model.module.load_state_dict(self._loaded_average)
            self._loaded_average = None

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if (
            trainer._stepped
            and self._average_model is not None
            and self.should_update(step_idx=trainer.global_step)
        ):
            self._average_model.update_parameters(pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self._average_model is not None and self.should_update(epoch_idx=trainer.current_epoch):
            self._average_model.update_parameters(pl_module)

    def _swap_in(self, pl_module: Module) -> None:
        if self._average_model is None or self._backup is not None:
            return
        import copy

        self._backup = copy.deepcopy(pl_module)
        _copy_into(pl_module, self._average_model.module, self._use_buffers)

    def _swap_out(self, pl_module: Module) -> None:
        if self._backup is not None:
            _copy_into(pl_module, self._backup)
            self._backup = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        if not trainer.sanity_checking:
            self._swap_in(pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        self._swap_out(pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self._average_model is not None:
            _copy_into(pl_module, self._average_model.module, self._use_buffers)
            trainer._weights_changed_at_end = True

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: Module, checkpoint: dict[str, Any]
    ) -> None:
        if self._average_model is None or self._backup is not None:
            return
        checkpoint["current_model_state"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = _averaged_state(self._average_model)

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: Module, checkpoint: dict[str, Any]
    ) -> None:
        if "current_model_state" in checkpoint:
            self._loaded_average = checkpoint["state_dict"]
            checkpoint["state_dict"] = checkpoint.pop("current_model_state")


class EMAWeightAveraging(WeightAveraging):
    """Exponential moving average of the weights: ``avg = decay * avg + (1 - decay) * w``,
    updated every ``update_every_n_steps`` optimizer steps (from ``update_starting_at_step``
    or ``update_starting_at_epoch``). Uses PyTorch's multi-tensor EMA, one fused update per
    step."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        use_buffers: bool = True,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValidationFailed("decay must be between 0 and 1")
        super().__init__(device, use_buffers, multi_avg_fn=get_ema_multi_avg_fn(decay))
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch
        self._epoch = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._epoch = trainer.current_epoch

    def should_update(self, step_idx: int | None = None, epoch_idx: int | None = None) -> bool:
        if step_idx is None:
            return False
        if self.update_starting_at_step is not None and step_idx < self.update_starting_at_step:
            return False
        if (
            self.update_starting_at_epoch is not None
            and self._epoch < self.update_starting_at_epoch
        ):
            return False
        return step_idx % self.update_every_n_steps == 0


class StochasticWeightAveraging(Callback):
    """Stochastic Weight Averaging: from ``swa_epoch_start`` (an epoch, or a fraction of
    ``max_epochs``), switch the learning rate to ``swa_lrs`` with ``SWALR`` and average the
    weights at the end of each epoch. At the end of training the average replaces the
    weights and BatchNorm statistics are recomputed on the training data."""

    def __init__(
        self,
        swa_lrs: float | list[float],
        swa_epoch_start: int | float = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: Literal["cos", "linear"] = "cos",
        avg_fn: Callable[..., Any] | None = None,
        device: torch.device | str | None = "cpu",
    ) -> None:
        if isinstance(swa_epoch_start, float) and not 0 < swa_epoch_start <= 1:
            raise ValidationFailed("a float swa_epoch_start is a fraction in (0, 1]")
        if annealing_strategy not in ("cos", "linear"):
            raise ValidationFailed("annealing_strategy must be 'cos' or 'linear'")
        self._swa_lrs = swa_lrs
        self._swa_epoch_start = swa_epoch_start
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn
        self._device = device
        self._average_model: AveragedModel | None = None
        self.swa_start = 0
        self.n_averaged = 0

    def on_fit_start(self, trainer: Trainer, pl_module: Module) -> None:
        e = self._swa_epoch_start
        self.swa_start = int(e * (trainer.max_epochs - 1)) if isinstance(e, float) else e
        kwargs: dict[str, Any] = {"device": torch.device(self._device) if self._device else None}
        if self._avg_fn is not None:
            kwargs["avg_fn"] = self._avg_fn
        self._average_model = AveragedModel(pl_module, **kwargs)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        if trainer.current_epoch != self.swa_start:
            return
        from gpulayer.pytorch.trainer import LRSchedulerConfig

        configs = []
        for opt in trainer.optimizers:
            lrs = self._swa_lrs
            sched = SWALR(
                opt,
                swa_lr=lrs,  # type: ignore[arg-type]
                anneal_epochs=self._annealing_epochs,
                anneal_strategy=self._annealing_strategy,
            )
            configs.append(LRSchedulerConfig(sched, interval="epoch"))
        trainer.lr_scheduler_configs = configs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if trainer.current_epoch >= self.swa_start and self._average_model is not None:
            self._average_model.update_parameters(pl_module)
            self.n_averaged += 1

    def on_train_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self._average_model is None or self.n_averaged == 0:
            return
        _copy_into(pl_module, self._average_model.module, buffers=False)
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in pl_module.modules()):
            trainer._update_bn(pl_module)
        trainer._weights_changed_at_end = True


class BaseFinetuning(Callback):
    """Freeze parts of the model before training and unfreeze them on a schedule.
    Implement ``freeze_before_training(pl_module)`` and
    ``finetune_function(pl_module, epoch, optimizer)``. Create the optimizer from the
    trainable parameters only (``filter(lambda p: p.requires_grad, self.parameters())``)."""

    def freeze_before_training(self, pl_module: Module) -> None:
        raise NotImplementedError

    def finetune_function(
        self, pl_module: Module, epoch: int, optimizer: torch.optim.Optimizer
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def flatten_modules(modules: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
        if isinstance(modules, nn.Module):
            modules = [modules]
        out: list[nn.Module] = []
        for m in modules:
            out += [c for c in m.modules() if not list(c.children()) or c._parameters]
        return out

    @staticmethod
    def filter_params(
        modules: nn.Module | Iterable[nn.Module], train_bn: bool = True, requires_grad: bool = True
    ) -> Iterable[nn.Parameter]:
        for m in BaseFinetuning.flatten_modules(modules):
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not train_bn:
                continue
            for p in m.parameters(recurse=False):
                if p.requires_grad == requires_grad:
                    yield p

    @staticmethod
    def make_trainable(modules: nn.Module | Iterable[nn.Module]) -> None:
        for m in BaseFinetuning.flatten_modules(modules):
            for p in m.parameters(recurse=False):
                p.requires_grad = True
            m.train()

    @staticmethod
    def freeze_module(module: nn.Module) -> None:
        for p in module.parameters(recurse=False):
            p.requires_grad = False
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False

    @staticmethod
    def freeze(modules: nn.Module | Iterable[nn.Module], train_bn: bool = True) -> None:
        for m in BaseFinetuning.flatten_modules(modules):
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and train_bn:
                BaseFinetuning.make_trainable(m)
            else:
                BaseFinetuning.freeze_module(m)

    @staticmethod
    def filter_on_optimizer(
        optimizer: torch.optim.Optimizer, params: Iterable[nn.Parameter]
    ) -> list[nn.Parameter]:
        have = {id(p) for g in optimizer.param_groups for p in g["params"]}
        return [p for p in params if id(p) not in have]

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: nn.Module | Iterable[nn.Module],
        optimizer: torch.optim.Optimizer,
        lr: float | None = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ) -> None:
        BaseFinetuning.make_trainable(modules)
        group_lr = optimizer.param_groups[0]["lr"]
        params = BaseFinetuning.filter_on_optimizer(
            optimizer, BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True)
        )
        if params:
            optimizer.add_param_group(
                {"params": params, "lr": lr if lr is not None else group_lr / initial_denom_lr}
            )

    def setup(self, trainer: Trainer, pl_module: Module, stage: str) -> None:
        if stage == "fit":
            self.freeze_before_training(pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        for opt in trainer.optimizers:
            self.finetune_function(pl_module, trainer.current_epoch, opt)


class BackboneFinetuning(BaseFinetuning):
    """Freeze ``pl_module.backbone`` and unfreeze it at ``unfreeze_backbone_at_epoch`` with a
    learning rate of ``backbone_initial_lr`` (or the current rate times
    ``backbone_initial_ratio_lr``), then raise it by ``lambda_func`` each epoch, up to the
    rest of the model's rate when ``should_align``."""

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable[[int], float] = lambda epoch: 1.5,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: float | None = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.lambda_func = lambda_func
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.backbone_initial_lr = backbone_initial_lr
        self.should_align = should_align
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn
        self.verbose = verbose
        self.rounding = rounding
        self.previous_backbone_lr: float | None = None

    def freeze_before_training(self, pl_module: Module) -> None:
        backbone = getattr(pl_module, "backbone", None)
        if not isinstance(backbone, nn.Module):
            raise ValidationFailed("BackboneFinetuning needs a `backbone` submodule on the model")
        self.freeze(backbone)

    def finetune_function(
        self, pl_module: Module, epoch: int, optimizer: torch.optim.Optimizer
    ) -> None:
        backbone = pl_module.backbone
        if epoch == self.unfreeze_backbone_at_epoch:
            current = optimizer.param_groups[0]["lr"]
            lr = self.backbone_initial_lr or current * self.backbone_initial_ratio_lr
            self.previous_backbone_lr = lr
            self.unfreeze_and_add_param_group(
                backbone,
                optimizer,
                lr,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=self.train_bn,
            )
        elif epoch > self.unfreeze_backbone_at_epoch and self.previous_backbone_lr is not None:
            current = optimizer.param_groups[0]["lr"]
            lr = self.previous_backbone_lr * self.lambda_func(epoch + 1)
            if self.should_align:
                lr = min(lr, current)
            optimizer.param_groups[-1]["lr"] = round(lr, self.rounding)
            self.previous_backbone_lr = lr


class ModelPruning(Callback):
    """Prune weights with ``torch.nn.utils.prune`` at the end of training epochs.

    ``pruning_fn``: ``"l1_unstructured"``, ``"random_unstructured"`` or a ``prune`` function.
    ``parameters_to_prune``: ``[(module, "weight"), ...]``; by default every Linear and Conv
    weight. ``amount``: a fraction or count, or a function of the epoch. ``apply_pruning``: a
    bool or a function of the epoch. With ``make_pruning_permanent`` the masks are folded
    into the weights at the end of training, and checkpoints always hold plain weights."""

    _FNS = ("l1_unstructured", "random_unstructured")

    def __init__(
        self,
        pruning_fn: str | Callable[..., Any] = "l1_unstructured",
        parameters_to_prune: list[tuple[nn.Module, str]] | None = None,
        parameter_names: list[str] | None = None,
        use_global_unstructured: bool = True,
        amount: float | int | Callable[[int], float | int] = 0.5,
        apply_pruning: bool | Callable[[int], bool] = True,
        make_pruning_permanent: bool = True,
        prune_on_train_epoch_end: bool = True,
        verbose: int = 0,
    ) -> None:
        from torch.nn.utils import prune

        if isinstance(pruning_fn, str):
            if pruning_fn not in self._FNS:
                raise ValidationFailed(f"pruning_fn must be one of {self._FNS} or a function")
            self._global_method = (
                prune.L1Unstructured
                if pruning_fn == "l1_unstructured"
                else prune.RandomUnstructured
            )
            self._fn = getattr(prune, pruning_fn)
        else:
            self._global_method = None  # type: ignore[assignment]
            self._fn = pruning_fn
        self._params = parameters_to_prune
        self._names = parameter_names or ["weight"]
        self._global = use_global_unstructured and isinstance(pruning_fn, str)
        self.amount = amount
        self.apply_pruning = apply_pruning
        self.make_pruning_permanent = make_pruning_permanent
        self.verbose = verbose

    def _targets(self, pl_module: Module) -> list[tuple[nn.Module, str]]:
        if self._params is not None:
            return self._params
        kinds = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        return [
            (m, n)
            for m in pl_module.modules()
            if isinstance(m, kinds)
            for n in self._names
            if getattr(m, n, None) is not None
        ]

    def apply(self, pl_module: Module, amount: float | int) -> None:
        from torch.nn.utils import prune

        targets = self._targets(pl_module)
        if self._global and self._global_method is not None:
            prune.global_unstructured(targets, pruning_method=self._global_method, amount=amount)
        else:
            for m, n in targets:
                self._fn(m, n, amount=amount)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        epoch = trainer.current_epoch
        apply = self.apply_pruning(epoch) if callable(self.apply_pruning) else self.apply_pruning
        if not apply:
            return
        amount = self.amount(epoch) if callable(self.amount) else self.amount
        if amount:
            self.apply(pl_module, amount)
            if self.verbose:
                zeros = sum(int((getattr(m, n) == 0).sum()) for m, n in self._targets(pl_module))
                total = sum(getattr(m, n).numel() for m, n in self._targets(pl_module))
                trainer._say(f"[gpulayer] pruned to {zeros / max(total, 1):.1%} zeros")

    def make_permanent(self, pl_module: Module) -> None:
        from torch.nn.utils import prune

        for m in pl_module.modules():
            for n in list(getattr(m, "_forward_pre_hooks", {}).values()):
                if isinstance(n, prune.BasePruningMethod) or hasattr(n, "_tensor_name"):
                    prune.remove(m, n._tensor_name)

    def on_train_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self.make_pruning_permanent:
            self.make_permanent(pl_module)
            trainer._weights_changed_at_end = True

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: Module, checkpoint: dict[str, Any]
    ) -> None:
        state = checkpoint["state_dict"]
        masks = [k for k in state if k.endswith("_mask")]
        if not masks or not self.make_pruning_permanent:
            return
        out = {k: v for k, v in state.items() if not k.endswith(("_mask", "_orig"))}
        for k in masks:
            base = k[: -len("_mask")]
            out[base] = state[base + "_orig"] * state[k]
        checkpoint["state_dict"] = out
