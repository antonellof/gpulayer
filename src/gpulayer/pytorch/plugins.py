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
        Path(path).unlink(missing_ok=True)

    def teardown(self) -> None:
        """Called when fit ends; finish pending writes here."""


class TorchCheckpointIO(CheckpointIO):
    """``torch.save`` to a temporary file, then an atomic rename (a crash never leaves a
    half-written checkpoint)."""

    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: str | Path, storage_options: Any = None
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)


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


class TorchSyncBatchNorm:
    """BatchNorm statistics over all processes (``torch.nn.SyncBatchNorm``)."""


def apply_plugins(plugins: Any) -> dict[str, Any]:
    """What a ``plugins=`` value sets on the Trainer."""
    out: dict[str, Any] = {}
    items = plugins if isinstance(plugins, (list, tuple)) else [plugins]
    for p in items:
        if isinstance(p, Precision):
            out["precision"] = p.precision
            if isinstance(p, MixedPrecision) and p.scaler is not None:
                out["scaler"] = p.scaler
        elif isinstance(p, CheckpointIO):
            out["checkpoint_io"] = p
        elif isinstance(p, TorchSyncBatchNorm):
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
