"""Trainer plugins, named as in PyTorch Lightning: ``Trainer(plugins=[...])``.

- Precision: ``MixedPrecision("bf16-mixed")``, ``HalfPrecision("16-true")``,
  ``DoublePrecision()``; the same as ``Trainer(precision=...)``.
- Checkpoint I/O: ``TorchCheckpointIO`` (the default), ``AsyncCheckpointIO`` (writes in a
  background thread so training does not wait for the disk), or your own ``CheckpointIO``
  subclass (for example to write to object storage).
- Layer sync: ``TorchSyncBatchNorm()``, the same as ``Trainer(sync_batchnorm=True)``.

Cluster-environment plugins are not needed: GPUlayer starts every process itself and sets the
rank and rendezvous variables.
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any

import torch

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.storage import RemotePath, as_path, load_file, save_file

# ------------------------------------------------------------------ precision


class Precision:
    """Base precision plugin: the Trainer uses its ``precision`` name."""

    precision = "32-true"

    def __init__(self, precision: str | int = "32-true") -> None:
        from gpulayer.pytorch.runtime import normalize_precision

        self.precision = normalize_precision(precision)


class MixedPrecision(Precision):
    """Mixed precision with autocast: ``"16-mixed"`` (with a gradient scaler) or
    ``"bf16-mixed"``. ``device`` is accepted for Lightning compatibility."""

    def __init__(
        self, precision: str | int = "16-mixed", device: str = "cuda", scaler: Any = None
    ) -> None:
        super().__init__(precision)
        if self.precision not in ("16-mixed", "bf16-mixed"):
            raise ValidationFailed("MixedPrecision takes '16-mixed' or 'bf16-mixed'")
        self.device = device
        self.scaler = scaler


class HalfPrecision(Precision):
    """Weights and inputs in half precision: ``"16-true"`` or ``"bf16-true"``."""

    def __init__(self, precision: str = "16-true") -> None:
        super().__init__(precision)
        if self.precision not in ("16-true", "bf16-true"):
            raise ValidationFailed("HalfPrecision takes '16-true' or 'bf16-true'")


class DoublePrecision(Precision):
    """Weights and inputs in float64."""

    def __init__(self) -> None:
        super().__init__("64-true")


# ------------------------------------------------------------------ checkpoint I/O


class CheckpointIO:
    """How checkpoints are written and read. Subclass to store them elsewhere."""

    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: str | Path, storage_options: Any = None
    ) -> None:
        """Write the checkpoint dict to ``path``."""
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
        """Read a checkpoint dict from ``path``."""
        raise NotImplementedError

    def remove_checkpoint(self, path: str | Path) -> None:
        """Delete the checkpoint at ``path`` (used when ModelCheckpoint drops a worse one)."""
        as_path(path).unlink(missing_ok=True)

    def teardown(self) -> None:
        """Called when fit ends; finish pending writes here."""


class TorchCheckpointIO(CheckpointIO):
    """``torch.save`` to a temporary file, then an atomic rename (a crash never leaves a
    half-written checkpoint)."""

    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: str | Path, storage_options: Any = None
    ) -> None:
        target = as_path(path)
        if isinstance(target, RemotePath):
            save_file(checkpoint, target)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(checkpoint, tmp)
        os.replace(tmp, target)

    def load_checkpoint(self, path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
        target = as_path(path)
        if target.is_dir():
            if isinstance(target, RemotePath):
                raise ValidationFailed(f"sharded checkpoints must be local folders: {target}")
            return load_sharded_as_full(target)
        return load_file(target, map_location=map_location, weights_only=False)


def load_sharded_as_full(path: str | Path) -> dict[str, Any]:
    """A sharded checkpoint folder (FSDP ``state_dict_type="sharded"``, ModelParallelStrategy)
    as one full checkpoint in this process, without a process group: the weights under
    ``state_dict``, the rest from ``meta.pt``. Optimizer state is left out (it is keyed by
    parameter names; a sharded run resumes from the folder itself)."""
    import tempfile

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    folder = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "full.pt"
        dcp_to_torch_save(str(folder), str(out))
        full = torch.load(out, map_location="cpu", weights_only=False)
    meta = torch.load(folder / "meta.pt", map_location="cpu", weights_only=False)
    meta["state_dict"] = full["model"]
    return dict(meta)


def _detached_copy(value: Any) -> Any:
    """A copy whose tensors no longer change when training continues (on the CPU)."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _detached_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detached_copy(v) for v in value)
    return value


class AsyncCheckpointIO(CheckpointIO):
    """Write checkpoints in a background thread, so the training loop does not wait for the
    disk. The state is copied to the CPU first (training keeps changing the weights); a
    failed write is raised on the next save or at the end of fit."""

    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        self.checkpoint_io = checkpoint_io or TorchCheckpointIO()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt"
        )
        self._pending: list[concurrent.futures.Future[None]] = []

    def _raise_errors(self) -> None:
        done = [f for f in self._pending if f.done()]
        self._pending = [f for f in self._pending if not f.done()]
        for f in done:
            f.result()

    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: str | Path, storage_options: Any = None
    ) -> None:
        self._raise_errors()
        snapshot = _detached_copy(checkpoint)
        self._pending.append(
            self._executor.submit(self.checkpoint_io.save_checkpoint, snapshot, path)
        )

    def load_checkpoint(self, path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
        self.wait()
        return self.checkpoint_io.load_checkpoint(path, map_location)

    def remove_checkpoint(self, path: str | Path) -> None:
        self.wait()
        self.checkpoint_io.remove_checkpoint(path)

    def wait(self) -> None:
        """Block until every pending write is on disk."""
        for f in self._pending:
            f.result()
        self._pending = []

    def teardown(self) -> None:
        self.wait()


# ------------------------------------------------------------------ layer sync


class LayerSync:
    """Base class: layers that synchronise across processes (``apply`` and ``revert``)."""

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """Make ``model``'s layers synchronise across processes."""
        raise NotImplementedError

    def revert(self, model: torch.nn.Module) -> torch.nn.Module:
        """Undo ``apply``."""
        raise NotImplementedError


class TorchSyncBatchNorm(LayerSync):
    """BatchNorm statistics over all processes (``torch.nn.SyncBatchNorm``)."""

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """BatchNorm layers to ``SyncBatchNorm``."""
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def revert(self, model: torch.nn.Module) -> torch.nn.Module:
        """SyncBatchNorm layers back to BatchNorm (for export or one-process inference)."""
        for name, child in list(model.named_children()):
            if isinstance(child, torch.nn.SyncBatchNorm):
                bn = _BatchNormXd(
                    child.num_features, child.eps, child.momentum, child.affine,
                    child.track_running_stats,
                )  # fmt: skip
                bn.load_state_dict(child.state_dict())
                bn.train(child.training)
                setattr(model, name, bn)
            else:
                self.revert(child)
        return model


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """BatchNorm for any input rank (SyncBatchNorm does not record which one it replaced)."""

    def _check_input_dim(self, input: torch.Tensor) -> None:
        return


# ---------------------------------------------- strategy-specific precision and XLA


class FSDPPrecision(Precision):
    """Precision for ``strategy="fsdp"``: ``"32-true"`` or ``"bf16-mixed"``."""


class DeepSpeedPrecision(Precision):
    """Precision for DeepSpeed (it runs mixed precision itself): ``"16-mixed"``,
    ``"bf16-mixed"``, ``"32-true"``."""


class XLAPrecision(Precision):
    """Precision on TPUs: ``"32-true"``, ``"bf16-true"`` (``XLA_USE_BF16``-style weights) or
    ``"16-true"``."""

    def __init__(self, precision: str | int = "32-true") -> None:
        super().__init__(precision)
        if self.precision not in ("32-true", "bf16-true", "16-true"):
            raise ValidationFailed("XLAPrecision takes '32-true', 'bf16-true' or '16-true'")


class XLACheckpointIO(TorchCheckpointIO):
    """Checkpoints from TPU runs. The Trainer already moves XLA tensors to the CPU before
    saving (as ``xm.save`` does), so this is ``TorchCheckpointIO``; it exists so code written
    for Lightning keeps working."""


# ------------------------------------------------------- layer-replacing precision


def _replace_linear(module: torch.nn.Module, make: Any, ignore: set[str], prefix: str = "") -> int:
    """Replace every ``nn.Linear`` below ``module`` (except names in ``ignore``) with
    ``make(linear)``; returns how many were replaced."""
    n = 0
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if full in ignore or name in ignore:
            continue
        if type(child) is torch.nn.Linear:
            setattr(module, name, make(child))
            n += 1
        else:
            n += _replace_linear(child, make, ignore, full)
    return n


class BitsandbytesPrecision(Precision):
    """Quantized ``nn.Linear`` layers with bitsandbytes (CUDA), as Lightning's plugin:
    ``mode`` is ``"nf4"``, ``"nf4-dq"``, ``"fp4"``, ``"fp4-dq"`` (4-bit; ``-dq`` also quantizes
    the quantization constants), ``"int8"`` (frozen 8-bit weights, for fine-tuning adapters or
    inference) or ``"int8-training"``. ``dtype`` is the compute dtype (bfloat16 by default);
    ``ignore_modules`` names layers to keep, such as the output head. Needs
    ``pip install bitsandbytes``; implemented, not run on hardware."""

    MODES = ("nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training")

    def __init__(
        self, mode: str, dtype: torch.dtype | None = None, ignore_modules: set[str] | None = None
    ) -> None:
        if mode not in self.MODES:
            raise ValidationFailed(f"BitsandbytesPrecision mode must be one of {self.MODES}")
        self.mode = mode
        self.dtype = dtype or torch.bfloat16
        self.ignore_modules = set(ignore_modules or ())
        self.precision = {torch.bfloat16: "bf16-true", torch.float16: "16-true"}.get(
            self.dtype, "32-true"
        )

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Replace ``module``'s ``nn.Linear`` layers with quantized ones, in place."""
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ValidationFailed(
                "BitsandbytesPrecision needs bitsandbytes (pip install bitsandbytes) and a CUDA GPU"
            ) from e
        mode, dtype = self.mode, self.dtype

        def make(lin: torch.nn.Linear) -> torch.nn.Module:
            bias = lin.bias is not None
            if mode.startswith("int8"):
                q = bnb.nn.Linear8bitLt(
                    lin.in_features, lin.out_features, bias=bias,
                    has_fp16_weights=mode == "int8-training", threshold=6.0,
                )  # fmt: skip
                q.weight = bnb.nn.Int8Params(
                    lin.weight.data.to(dtype), requires_grad=mode == "int8-training",
                    has_fp16_weights=mode == "int8-training",
                )  # fmt: skip
            else:
                q = bnb.nn.Linear4bit(
                    lin.in_features, lin.out_features, bias=bias, compute_dtype=dtype,
                    compress_statistics=mode.endswith("-dq"), quant_type=mode.split("-")[0],
                )  # fmt: skip
                q.weight = bnb.nn.Params4bit(
                    lin.weight.data.to(dtype), requires_grad=False,
                    compress_statistics=mode.endswith("-dq"), quant_type=mode.split("-")[0],
                )  # fmt: skip
            if bias:
                q.bias = torch.nn.Parameter(
                    lin.bias.data.to(dtype), requires_grad=lin.bias.requires_grad
                )
            return q  # quantized when moved to the GPU

        if not _replace_linear(module, make, self.ignore_modules):
            raise ValidationFailed("BitsandbytesPrecision found no nn.Linear layer to quantize")
        for p in module.parameters():  # the other weights in the compute dtype
            if p.is_floating_point() and p.dtype == torch.float32:
                p.data = p.data.to(dtype)
        return module


class TransformerEnginePrecision(Precision):
    """FP8 training with NVIDIA Transformer Engine (Hopper, Ada or newer GPUs), as Lightning's
    plugin: ``nn.Linear`` and ``nn.LayerNorm`` become their ``transformer_engine.pytorch``
    versions and the forward pass runs under ``fp8_autocast``. ``weights_dtype`` keeps the
    weights (bfloat16 by default); ``recipe`` is a ``DelayedScaling`` recipe or its keyword
    arguments. Layers whose sizes are not multiples of 16 are kept, as FP8 needs them. Needs
    ``transformer_engine``; implemented, not run on hardware."""

    def __init__(
        self,
        weights_dtype: torch.dtype | None = None,
        recipe: Any = None,
        replace_layers: bool | None = None,
        fallback_compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.weights_dtype = weights_dtype or torch.bfloat16
        self.recipe = recipe
        self.replace_layers = True if replace_layers is None else replace_layers
        self.fallback_compute_dtype = fallback_compute_dtype or self.weights_dtype
        self.precision = {torch.bfloat16: "bf16-true", torch.float16: "16-true"}.get(
            self.weights_dtype, "32-true"
        )

    def _te(self) -> Any:
        try:
            import transformer_engine.pytorch as te
        except ImportError as e:
            raise ValidationFailed(
                "TransformerEnginePrecision needs transformer_engine and an FP8-capable NVIDIA GPU"
            ) from e
        return te

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Replace ``nn.Linear`` and ``nn.LayerNorm`` with Transformer Engine layers, in place."""
        te = self._te()
        if self.replace_layers:
            dtype = self.weights_dtype

            def make(lin: torch.nn.Linear) -> torch.nn.Module:
                if lin.in_features % 16 or lin.out_features % 16:
                    return lin
                new = te.Linear(
                    lin.in_features, lin.out_features, bias=lin.bias is not None, params_dtype=dtype
                )
                with torch.no_grad():
                    new.weight.copy_(lin.weight)
                    if lin.bias is not None:
                        new.bias.copy_(lin.bias)
                return new

            _replace_linear(module, make, set())
            for child in list(module.modules()):
                for sub, layer in list(child.named_children()):
                    if type(layer) is torch.nn.LayerNorm and len(layer.normalized_shape) == 1:
                        ln = te.LayerNorm(
                            layer.normalized_shape[0], eps=layer.eps, params_dtype=dtype
                        )
                        with torch.no_grad():
                            ln.weight.copy_(layer.weight)
                            ln.bias.copy_(layer.bias)
                        setattr(child, sub, ln)
        return module.to(self.weights_dtype)

    def forward_context(self) -> Any:
        """``fp8_autocast`` with the recipe, entered around every forward."""
        te = self._te()
        recipe = self.recipe
        if isinstance(recipe, dict) or recipe is None:
            from transformer_engine.common.recipe import DelayedScaling

            recipe = DelayedScaling(**(recipe or {}))
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)


def apply_plugins(plugins: Any) -> dict[str, Any]:
    """What a ``plugins=`` value sets on the Trainer."""
    out: dict[str, Any] = {}
    items = plugins if isinstance(plugins, (list, tuple)) else [plugins]
    for p in items:
        if isinstance(p, Precision):
            out["precision"] = p.precision
            if hasattr(p, "convert_module") or hasattr(p, "forward_context"):
                out["precision_plugin"] = p
            if isinstance(p, MixedPrecision) and p.scaler is not None:
                out["scaler"] = p.scaler
        elif isinstance(p, CheckpointIO):
            out["checkpoint_io"] = p
        elif isinstance(p, LayerSync):
            out["sync_batchnorm"] = True
        elif isinstance(p, str):
            from gpulayer.pytorch.runtime import normalize_precision

            out["precision"] = normalize_precision(p)
        else:
            raise ValidationFailed(
                f"unsupported plugin {type(p).__name__}: use a Precision, CheckpointIO or "
                "TorchSyncBatchNorm plugin (cluster environments are set by GPUlayer)"
            )
    return out
