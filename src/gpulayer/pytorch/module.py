"""``Module``: an ``nn.Module`` with training hooks used by ``Trainer``; ``DataModule`` for data.

Names and semantics follow PyTorch Lightning's ``LightningModule`` and ``LightningDataModule``,
so a model written for one mostly runs on the other.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.storage import load_file

REDUCE_FX = ("mean", "sum", "max", "min")
_FX_ALIASES = {"amax": "max", "amin": "min"}


class AttributeDict(dict):
    """A dict whose keys are also attributes: ``hparams.lr``."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _init_args(frame: Any) -> dict[str, Any]:
    """The arguments of the ``__init__`` that called ``save_hyperparameters``."""
    info = inspect.getargvalues(frame)
    args = {k: info.locals[k] for k in info.args if k != "self"}
    if info.varargs and info.locals.get(info.varargs):
        args[info.varargs] = info.locals[info.varargs]
    if info.keywords:
        args.update(info.locals.get(info.keywords) or {})
    return args


class HyperparametersMixin:
    """``save_hyperparameters()`` and ``hparams``, shared by Module and DataModule."""

    def save_hyperparameters(
        self, *args: Any, ignore: str | list[str] | None = None, frame: Any = None
    ) -> None:
        """Record the ``__init__`` arguments (all of them, or the names in ``args``) as
        ``self.hparams``. They are saved in checkpoints, and ``load_from_checkpoint`` uses
        them to rebuild the model without passing the arguments again."""
        if frame is None:
            current = inspect.currentframe()
            frame = current.f_back if current is not None else None
        values = _init_args(frame) if frame is not None else {}
        if args and len(args) == 1 and isinstance(args[0], dict):
            values = dict(args[0])
        elif args:
            missing = [a for a in args if a not in values]
            if missing:
                raise ValidationFailed(f"save_hyperparameters: unknown argument(s) {missing}")
            values = {a: values[a] for a in args}
        ignored = [ignore] if isinstance(ignore, str) else list(ignore or [])
        values = {
            k: v
            for k, v in values.items()
            if k not in ignored and not isinstance(v, nn.Module)  # modules are not hparams
        }
        hp = self.__dict__.setdefault("_gl_hparams", AttributeDict())
        hp.update(values)

    @property
    def hparams(self) -> AttributeDict:
        """The arguments recorded by ``save_hyperparameters``, as an ``AttributeDict``."""
        return self.__dict__.setdefault("_gl_hparams", AttributeDict())


class LogOptions(NamedTuple):
    """What ``self.log`` was asked to do with a name. ``step`` is False only for an explicit
    ``on_step=False``; ``on_step`` and ``on_epoch`` keep None for "the default of this hook"."""

    fx: str
    prog_bar: bool
    step: bool
    batch_size: float | None
    on_step: bool | None
    on_epoch: bool | None
    logger: bool


DEFAULT_LOG_OPTIONS = LogOptions("mean", False, True, None, None, None, True)


def _one_value(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.numel() != 1:
        raise ValidationFailed(
            f"self.log({name!r}, ...) got a tensor of shape {tuple(value.shape)}; it must hold "
            "one value, for example its mean"
        )
    return value.float().reshape(())


class Module(HyperparametersMixin, nn.Module):
    """Subclass and implement ``training_step`` and ``configure_optimizers``.

    ``validation_step``, ``test_step`` and ``predict_step`` are optional. Call
    ``self.log(name, value)`` in any step; the Trainer averages logged values per epoch across
    steps and ranks. Override any of the ``on_*`` hooks to run code at that point of the loop.
    """

    _gl_in_step: bool = True  # False after training_step, in the batch-end hooks

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gl_step_logs: dict[str, Any] = {}
        self._gl_log_options: dict[str, LogOptions] = {}
        self._gl_metrics: dict[str, Any] = {}  # torchmetrics objects logged in this loop
        self.trainer: Any = None
        self._gl_automatic_optimization = True
        self.example_input_array: Any = None
        self._strict_loading: bool | None = None
        # Set by ModelParallelStrategy before configure_model (tensor parallelism plans).
        self.device_mesh: Any = None
        self.fabric: Any = None  # Lightning Fabric is not used by GPUlayer

    @property
    def strict_loading(self) -> bool:
        """Whether checkpoints load with ``load_state_dict(strict=True)`` (the default). Set it
        to False when a model leaves frozen weights out of its checkpoints."""
        return self._strict_loading in (None, True)

    @strict_loading.setter
    def strict_loading(self, value: bool) -> None:
        self._strict_loading = value

    def __getstate__(self) -> dict[str, Any]:
        # Copies and pickles leave the Trainer behind (deepcopy for weight averaging, SWA).
        state = dict(super().__getstate__())
        state["trainer"] = None
        state.pop("_gl_manual", None)
        return state

    # ------------------------------------------------------------------ steps

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """Return the loss tensor, or a dict with ``"loss"``. With manual optimization,
        return nothing."""
        raise NotImplementedError("implement training_step(batch, batch_idx) -> loss")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """One validation batch: compute metrics and ``self.log`` them (optional)."""
        return None

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        """One test batch, run by ``Trainer.test``: compute metrics and ``self.log`` them."""
        raise NotImplementedError("implement test_step(batch, batch_idx) to use Trainer.test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Default: ``self(batch)``, or ``self(batch[0])`` for an ``(inputs, ...)`` batch."""
        if isinstance(batch, (list, tuple)):
            return self(batch[0])
        return self(batch)

    def configure_optimizers(self) -> Any:
        """Return an optimizer; a list of optimizers; ``([optimizers], [schedulers])``;
        ``(optimizer, scheduler)``; or ``{"optimizer": ..., "lr_scheduler": ...}``, where the
        scheduler may be a dict ``{"scheduler": ..., "interval": "epoch" | "step",
        "frequency": 1, "monitor": "val_loss"}``. Several optimizers need manual
        optimization (``self.automatic_optimization = False``), as in Lightning."""
        raise NotImplementedError("implement configure_optimizers()")

    def configure_callbacks(self) -> Any:
        """Callbacks this model always needs; they are added to the Trainer's."""
        return []

    def configure_model(self) -> None:
        """Create layers here to build them on the training device (called once per fit)."""

    def configure_sharded_model(self) -> None:
        """Lightning's older name for ``configure_model``; called the same way."""

    # Loaders the model provides itself; the Trainer uses them when fit/validate/test/predict
    # get no loaders and no DataModule. Not overriding one means the model has none.

    def train_dataloader(self) -> Any:
        """The training loader (or a list or dict of loaders)."""
        raise ValidationFailed("train_dataloader is not implemented on this model")

    def val_dataloader(self) -> Any:
        """The validation loader(s)."""
        raise ValidationFailed("val_dataloader is not implemented on this model")

    def test_dataloader(self) -> Any:
        """The test loader(s)."""
        raise ValidationFailed("test_dataloader is not implemented on this model")

    def predict_dataloader(self) -> Any:
        """The prediction loader(s)."""
        raise ValidationFailed("predict_dataloader is not implemented on this model")

    # Mode switches around evaluation, as Lightning's: override to keep, say, dropout on.

    def on_validation_model_zero_grad(self) -> None:
        """Before validation inside ``fit``: free the gradients' memory."""
        self.zero_grad(set_to_none=True)

    def on_validation_model_eval(self) -> None:
        """When validation starts: evaluation mode."""
        self.eval()

    def on_validation_model_train(self) -> None:
        """When validation inside ``fit`` ends: training mode again."""
        self.train()

    def on_test_model_eval(self) -> None:
        """When testing starts: evaluation mode."""
        self.eval()

    def on_test_model_train(self) -> None:
        """When testing ends, if the model was training before."""
        self.train()

    def on_predict_model_eval(self) -> None:
        """When prediction starts: evaluation mode."""
        self.eval()

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients before each optimizer step; by default with the Trainer's values."""
        self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    # ------------------------------------------------------------------ hooks
    # Overridden hooks are called by the Trainer; the ones left as they are cost nothing.

    def setup(self, stage: str) -> None: ...
    def teardown(self, stage: str) -> None: ...
    def prepare_data(self) -> None: ...
    def on_fit_start(self) -> None: ...
    def on_fit_end(self) -> None: ...
    def on_train_start(self) -> None: ...
    def on_train_end(self) -> None: ...
    def on_train_epoch_start(self) -> None: ...
    def on_train_epoch_end(self) -> None: ...
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Any: ...
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None: ...
    def on_validation_start(self) -> None: ...
    def on_validation_end(self) -> None: ...
    def on_validation_epoch_start(self) -> None: ...
    def on_validation_epoch_end(self) -> None: ...
    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None: ...
    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None: ...
    def on_test_start(self) -> None: ...
    def on_test_end(self) -> None: ...
    def on_test_epoch_start(self) -> None: ...
    def on_test_epoch_end(self) -> None: ...
    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None: ...
    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None: ...
    def on_predict_start(self) -> None: ...
    def on_predict_end(self) -> None: ...
    def on_predict_epoch_start(self) -> None: ...
    def on_predict_epoch_end(self) -> None: ...
    def on_predict_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None: ...
    def on_predict_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None: ...
    def on_before_backward(self, loss: torch.Tensor) -> None: ...
    def on_after_backward(self) -> None: ...
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None: ...
    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None: ...
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None: ...
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None: ...
    def on_exception(self, exception: BaseException) -> None: ...

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """Move a batch to ``device``. Override for custom batch types."""
        from gpulayer.pytorch.runtime import move_to_device

        return move_to_device(batch, device)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        """Step a scheduler; ``metric`` is the monitored value for ReduceLROnPlateau."""
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def optimizer_step(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, closure: Any = None
    ) -> None:
        """Run the optimizer step (override to change it, e.g. learning-rate warm-up)."""
        optimizer.step(closure=closure) if closure is not None else optimizer.step()

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Reset gradients after an optimizer step (``set_to_none=True``); override to change it."""
        optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ logging

    def log(
        self,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable[..., Any] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        metric_attribute: str | None = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Record a scalar or a ``torchmetrics.Metric`` for this step. The Trainer aggregates a
        scalar over the epoch with
        ``reduce_fx`` ("mean", "sum", "max" or "min"; a mean is weighted by ``batch_size``,
        or by the size Lightning infers from the batch when it is not given) and combines
        it across processes with the same ``reduce_fx``. ``prog_bar`` shows it in the
        progress bar. ``sync_dist`` is accepted for compatibility: epoch values are always
        combined across processes.

        Where a value goes follows Lightning. ``on_step`` and ``on_epoch`` default to per step
        in ``training_step`` and per epoch in evaluation and ``on_*_epoch_end``; with both, the
        names become ``<name>_step`` and ``<name>_epoch``. ``logger=False`` keeps it from the
        loggers, ``prog_bar=True`` adds it to the progress bar, and
        ``trainer.callback_metrics`` always has it, for callbacks to monitor.

        A ``Metric`` is computed once at the end of the epoch or evaluation and then reset; it
        syncs across processes itself, inside ``compute()``. Call it (or ``update``) in the
        step and pass the object, as in Lightning."""
        if not add_dataloader_idx:
            self.__dict__.setdefault("_gl_no_idx", set()).add(name)
        if isinstance(value, torch.nn.Module):
            value = self._log_metric(name, value)
            if value is None:  # update() only: nothing to show for this step
                return
        if (
            on_step is None
            and on_epoch is None
            and logger is None
            and reduce_fx == "mean"
            and not prog_bar
            and batch_size is None
        ):
            # The common call, kept short: it runs every step.
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if value.ndim or value.dtype != torch.float32:  # a scalar loss needs no op
                    value = _one_value(name, value)
                self._gl_step_logs[name] = value
            else:
                self._gl_step_logs[name] = float(value)
            return
        fx = reduce_fx if isinstance(reduce_fx, str) else getattr(reduce_fx, "__name__", "")
        fx = _FX_ALIASES.get(fx, fx)
        if fx not in REDUCE_FX:
            raise ValidationFailed(f"reduce_fx must be one of {REDUCE_FX}, got {reduce_fx!r}")
        # Tensors stay on the device: reading them (.item()) would make the CPU wait for the
        # GPU on every step. The Trainer reads them only when it prints or ends an epoch.
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.ndim or value.dtype != torch.float32:
                value = _one_value(name, value)
        else:
            value = float(value)
        options = LogOptions(
            fx,
            prog_bar,
            on_step is not False,
            float(batch_size) if batch_size is not None else None,
            on_step,
            on_epoch,
            logger is not False,
        )
        if name in self._gl_step_logs and self._gl_in_step:
            before = self._gl_log_options.get(name, DEFAULT_LOG_OPTIONS)
            if before._replace(batch_size=None) != options._replace(batch_size=None):
                raise ValidationFailed(
                    f"self.log({name!r}, ...) was called twice in this step with different "
                    "options; use one set of options, or another name"
                )
        self._gl_step_logs[name] = value
        self._gl_log_options[name] = options

    def _log_metric(self, name: str, metric: torch.nn.Module) -> torch.Tensor | None:
        """Keep ``metric`` for the end of this loop; return this batch's value, if it has
        one (``metric(preds, target)`` sets it, ``metric.update(...)`` does not)."""
        if not callable(getattr(metric, "compute", None)) or not callable(
            getattr(metric, "reset", None)
        ):
            raise ValidationFailed(
                f"self.log({name!r}, ...) got a {type(metric).__name__}: pass a number, a "
                "tensor or a torchmetrics.Metric"
            )
        self._gl_metrics[name] = metric
        cached = getattr(metric, "_forward_cache", None)
        return cached if isinstance(cached, torch.Tensor) else None

    def log_dict(self, dictionary: dict[str, Any], **kwargs: Any) -> None:
        """``self.log`` for each item of ``dictionary``, with the same options."""
        for k, v in dictionary.items():
            self.log(k, v, **kwargs)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """``print`` on rank 0 only."""
        if self.trainer is None or self.trainer.is_global_zero:
            print(*args, **kwargs)

    # ------------------------------------------------------------- optimization

    @property
    def automatic_optimization(self) -> bool:
        """True (default): the Trainer runs backward and the optimizer step. Set False for manual
        optimization."""
        return self._gl_automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, value: bool) -> None:
        self._gl_automatic_optimization = value

    def _trainer(self) -> Any:
        if self.trainer is None:
            raise ValidationFailed(f"{type(self).__name__} is not attached to a Trainer")
        return self.trainer

    def optimizers(self, use_pl_optimizer: bool = True) -> Any:
        """The optimizer(s) from ``configure_optimizers``: one, or a list."""
        t = self._trainer()
        opts = t._manual_opts if (use_pl_optimizer and t._manual_opts) else t.optimizers
        return opts[0] if len(opts) == 1 else opts

    def lr_schedulers(self) -> Any:
        """The scheduler(s) from ``configure_optimizers``: one, a list, or None."""
        scheds = [c.scheduler for c in self._trainer().lr_scheduler_configs]
        if not scheds:
            return None
        return scheds[0] if len(scheds) == 1 else scheds

    def manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """``loss.backward()`` for manual optimization, with the Trainer's gradient scaling."""
        self._trainer()._manual_backward(loss, *args, **kwargs)

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Run the backward pass; override to pass arguments to ``loss.backward``."""
        loss.backward(*args, **kwargs)

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip the gradients of ``optimizer``'s parameters (unscaled first with 16-mixed)."""
        if not gradient_clip_val:
            return
        self._trainer()._clip(optimizer, gradient_clip_val, gradient_clip_algorithm or "norm")

    def toggle_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Only ``optimizer``'s parameters require gradients until ``untoggle_optimizer``."""
        own = {id(p) for g in optimizer.param_groups for p in g["params"]}
        self._gl_toggled = {id(p): p.requires_grad for p in self.parameters() if id(p) not in own}
        for p in self.parameters():
            if id(p) not in own:
                p.requires_grad = False

    def untoggle_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Restore ``requires_grad`` changed by ``toggle_optimizer``."""
        saved = getattr(self, "_gl_toggled", {})
        for p in self.parameters():
            if id(p) in saved:
                p.requires_grad = saved[id(p)]
        self._gl_toggled = {}

    @contextmanager
    def toggled_optimizer(self, optimizer: torch.optim.Optimizer) -> Iterator[None]:
        """Context manager: ``toggle_optimizer`` on entry, ``untoggle_optimizer`` on exit."""
        self.toggle_optimizer(optimizer)
        try:
            yield
        finally:
            self.untoggle_optimizer(optimizer)

    # ------------------------------------------------------------------ state

    @property
    def current_epoch(self) -> int:
        """The Trainer's current epoch (0 when not attached)."""
        return self.trainer.current_epoch if self.trainer is not None else 0

    @property
    def global_step(self) -> int:
        """Optimizer steps taken so far."""
        return self.trainer.global_step if self.trainer is not None else 0

    @property
    def global_rank(self) -> int:
        """This process's rank among all processes."""
        return self.trainer.global_rank if self.trainer is not None else 0

    @property
    def local_rank(self) -> int:
        """This process's rank on its machine."""
        return self.trainer.local_rank if self.trainer is not None else 0

    @property
    def device(self) -> torch.device:
        """The device of the first parameter (or buffer); CPU for a model without either."""
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the first parameter."""
        for p in self.parameters():
            return p.dtype
        return torch.get_default_dtype()

    @property
    def on_gpu(self) -> bool:
        """True when the parameters are on a CUDA (or ROCm) device."""
        return self.device.type == "cuda"

    @property
    def logger(self) -> Any:
        """The Trainer's first logger, or None."""
        return self.trainer.logger if self.trainer is not None else None

    @property
    def loggers(self) -> list[Any]:
        """The Trainer's loggers."""
        return self.trainer.loggers if self.trainer is not None else []

    def all_gather(self, data: Any, sync_grads: bool = False) -> Any:
        """Gather tensors from every process: each tensor gets a leading world-size
        dimension (unchanged in a single process)."""
        import torch.distributed as dist

        rt = getattr(self.trainer, "runtime", None) if self.trainer is not None else None
        xla = rt is not None and rt.xla and rt.distributed
        if not xla and not (dist.is_available() and dist.is_initialized()):
            return data
        if isinstance(data, dict):
            return {k: self.all_gather(v, sync_grads) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self.all_gather(v, sync_grads) for v in data)
        t = data if isinstance(data, torch.Tensor) else torch.tensor(data, device=self.device)
        if xla:  # TPU processes: torch_xla's collectives, as Lightning's XLAStrategy
            if sync_grads:
                import torch_xla.core.functions as xf

                return xf.all_gather(t.unsqueeze(0), dim=0)
            import torch_xla.core.xla_model as xm

            return xm.all_gather(t.unsqueeze(0), dim=0)
        if sync_grads:
            from torch.distributed.nn.functional import all_gather as ag

            return torch.stack(ag(t))
        out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(out, t.contiguous())
        return torch.stack(out)

    def freeze(self) -> None:
        """No gradients for any parameter, and evaluation mode."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Gradients for every parameter again, and training mode."""
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    # ------------------------------------------------------------------ export

    def to_torchscript(
        self,
        file_path: str | Path | None = None,
        method: str = "script",
        example_inputs: Any = None,
    ) -> Any:
        """TorchScript of this model (``method="script"`` or ``"trace"``), saved when
        ``file_path`` is given."""
        mode = self.training
        self.eval()
        try:
            if method == "script":
                scripted = torch.jit.script(self)
            elif method == "trace":
                inputs = example_inputs if example_inputs is not None else self.example_input_array
                if inputs is None:
                    raise ValidationFailed("method='trace' needs example_inputs")
                scripted = torch.jit.trace(self, inputs)
            else:
                raise ValidationFailed("method must be 'script' or 'trace'")
        finally:
            self.train(mode)
        if file_path is not None:
            torch.jit.save(scripted, str(file_path))
        return scripted

    def to_onnx(self, file_path: str | Path, input_sample: Any = None, **kwargs: Any) -> None:
        """Export to ONNX (needs the ``onnx`` package)."""
        inputs = input_sample if input_sample is not None else self.example_input_array
        if inputs is None:
            raise ValidationFailed("to_onnx needs input_sample or self.example_input_array")
        mode = self.training
        self.eval()
        try:
            torch.onnx.export(self, inputs, str(file_path), **kwargs)
        finally:
            self.train(mode)

    def to_tensorrt(
        self,
        file_path: str | Path | None = None,
        input_sample: Any = None,
        ir: str = "default",
        output_format: str = "exported_program",
        retrace: bool = False,
        default_device: str | torch.device = "cuda",
        **compile_kwargs: Any,
    ) -> Any:
        """Compile with Torch-TensorRT for NVIDIA inference (needs ``torch_tensorrt``) and save
        it when ``file_path`` is given; returns the compiled module. Implemented as in
        Lightning; not run on hardware."""
        try:
            import torch_tensorrt
        except ImportError as e:
            raise ValidationFailed(
                "to_tensorrt needs torch_tensorrt (pip install torch-tensorrt)"
            ) from e
        inputs = input_sample if input_sample is not None else self.example_input_array
        if inputs is None:
            raise ValidationFailed("to_tensorrt needs input_sample or self.example_input_array")
        from gpulayer.pytorch.utilities import move_data_to_device

        inputs = move_data_to_device(inputs, default_device)
        inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        mode = self.training
        self.eval()
        try:
            trt = torch_tensorrt.compile(
                self.to(default_device), ir=ir, inputs=inputs, **compile_kwargs
            )
        finally:
            self.train(mode)
        if file_path is not None:
            torch_tensorrt.save(
                trt, str(file_path), inputs=inputs, output_format=output_format, retrace=retrace
            )
        return trt

    # ------------------------------------------------------------- checkpoints

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *args: Any,
        map_location: Any = "cpu",
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build the model and load the weights saved by the Trainer. Arguments recorded with
        ``save_hyperparameters`` are used unless given again here."""
        from gpulayer.pytorch.plugins import TorchCheckpointIO

        ckpt = TorchCheckpointIO().load_checkpoint(checkpoint_path, map_location=map_location)
        hparams = dict(ckpt.get("hyper_parameters") or {})
        if args:
            model = cls(*args, **kwargs)
        elif "_instantiator" in hparams:
            # Saved by the CLI (load_from_checkpoint_support): rebuild it the CLI's way.
            import importlib

            module_name, _, fn_name = str(hparams.pop("_instantiator")).rpartition(".")
            fn = getattr(importlib.import_module(module_name), fn_name)
            model = fn(cls, {**hparams, **kwargs})
        else:
            hparams.pop("_class_path", None)
            accepted = inspect.signature(cls.__init__).parameters
            if not any(p.kind == p.VAR_KEYWORD for p in accepted.values()):
                hparams = {k: v for k, v in hparams.items() if k in accepted}
            model = cls(**{**hparams, **kwargs})
        model.on_load_checkpoint(ckpt)
        model.load_state_dict(
            ckpt["state_dict"], strict=model.strict_loading if strict is None else strict
        )
        return model


class DataModule(HyperparametersMixin):
    """All the data of a project in one place, as in Lightning's ``LightningDataModule``.

    ``prepare_data`` runs once per machine (download here); ``setup(stage)`` runs in every
    process ("fit", "validate", "test" or "predict"); the ``*_dataloader`` methods return
    loaders. The Trainer calls them only in the processes that train, so a launched job does
    not build its data in the launching process.
    """

    def __init__(self) -> None:
        self.trainer: Any = None

    def prepare_data(self) -> None: ...
    def setup(self, stage: str) -> None: ...
    def teardown(self, stage: str) -> None: ...

    def train_dataloader(self) -> Any:
        """The training loader (needed for ``fit``)."""
        raise NotImplementedError("implement train_dataloader() to use this DataModule for fit")

    def val_dataloader(self) -> Any:
        """The validation loader, or None."""
        return None

    def test_dataloader(self) -> Any:
        """The test loader (needed for ``test``)."""
        raise NotImplementedError("implement test_dataloader() to use Trainer.test")

    def predict_dataloader(self) -> Any:
        """The prediction loader (needed for ``predict``)."""
        raise NotImplementedError("implement predict_dataloader() to use Trainer.predict")

    def state_dict(self) -> dict[str, Any]:
        """State saved in checkpoints (empty by default)."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore what ``state_dict`` returned, when resuming."""

    def on_exception(self, exception: BaseException) -> None:
        """Called when a Trainer run stops with an exception (after the callbacks' and the
        model's ``on_exception``), as in Lightning."""

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: Any = None,
        hparams_file: str | Path | None = None,
        weights_only: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Rebuild this DataModule from the arguments the Trainer saved in a checkpoint
        (``datamodule_hyper_parameters``, recorded with ``save_hyperparameters``), then restore
        its ``state_dict``. Values in ``hparams_file`` (YAML or JSON) and in ``kwargs``
        override the saved ones. ``map_location`` defaults to the CPU; ``weights_only``
        defaults to False because GPUlayer checkpoints hold more than tensors."""
        location = map_location if map_location is not None else "cpu"
        ckpt = load_file(checkpoint_path, map_location=location, weights_only=bool(weights_only))
        hparams = dict(ckpt.get("datamodule_hyper_parameters") or {})
        if hparams_file is not None:
            import json

            path = Path(hparams_file)
            if path.suffix not in (".yaml", ".yml", ".json"):
                raise ValidationFailed(f"hparams_file must be .yaml, .yml or .json: {path.name}")
            text = path.read_text()
            try:
                import yaml

                hparams.update(yaml.safe_load(text) or {})
            except ImportError:  # JSON is valid YAML
                hparams.update(json.loads(text))
        hparams.update(kwargs)
        instantiator = hparams.pop("_instantiator", None)
        if instantiator is not None:  # written by the CLI with load_from_checkpoint_support
            import importlib

            module_name, _, fn_name = str(instantiator).rpartition(".")
            fn = getattr(importlib.import_module(module_name), fn_name)
            dm = fn(cls, hparams)
        else:
            hparams.pop("_class_path", None)
            accepted = inspect.signature(cls.__init__).parameters
            if not any(p.kind == p.VAR_KEYWORD for p in accepted.values()):
                hparams = {k: v for k, v in hparams.items() if k in accepted}
            dm = cls(**hparams)
        if "datamodule" in ckpt:
            dm.load_state_dict(ckpt["datamodule"])
        return dm

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Any = None,
        val_dataset: Any = None,
        test_dataset: Any = None,
        predict_dataset: Any = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **loader_kwargs: Any,
    ) -> DataModule:
        """A DataModule from datasets, with a shuffled training loader."""
        from torch.utils.data import DataLoader

        def make(ds: Any, shuffle: bool) -> Any:
            if ds is None:
                return None
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **loader_kwargs
            )

        dm = cls.__new__(cls)
        DataModule.__init__(dm)
        dm.train_dataloader = lambda: make(train_dataset, True)  # type: ignore[method-assign]
        dm.val_dataloader = lambda: make(val_dataset, False)  # type: ignore[method-assign]
        dm.test_dataloader = lambda: make(test_dataset, False)  # type: ignore[method-assign]
        dm.predict_dataloader = lambda: make(predict_dataset, False)  # type: ignore[method-assign]
        return dm


def overridden(obj: Any, name: str, base: type) -> bool:
    """True when ``obj``'s class overrides ``base.name`` (or the instance replaced it)."""
    if name in getattr(obj, "__dict__", {}):
        return True
    fn = getattr(type(obj), name, None)
    return fn is not None and fn is not getattr(base, name, None)


def deepcopy_module(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


class _TrainingStep(nn.Module):
    """Lets DDP wrap ``training_step`` (DDP only syncs gradients for calls to its forward)."""

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, batch: Any, batch_idx: int) -> Any:
        return self.module.training_step(batch, batch_idx)


def _document_hooks() -> None:
    from gpulayer.pytorch.hooks import apply_hook_docs

    apply_hook_docs(Module)
    apply_hook_docs(DataModule)


_document_hooks()
