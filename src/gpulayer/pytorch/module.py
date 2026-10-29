"""``Module``: an ``nn.Module`` with training hooks used by ``Trainer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class Module(nn.Module):
    """Subclass and implement ``training_step`` and ``configure_optimizers``.

    ``validation_step`` is optional. Call ``self.log(name, value)`` in either step; the
    Trainer averages logged values per epoch across steps and ranks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._gl_step_logs: dict[str, float] = {}
        self.trainer: Any = None

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError("implement training_step(batch, batch_idx) -> loss")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return None

    def configure_optimizers(self) -> Any:
        """Return an optimizer, ``(optimizer, scheduler)`` or
        ``{"optimizer": ..., "lr_scheduler": ..., "interval": "epoch" | "step"}``."""
        raise NotImplementedError("implement configure_optimizers()")

    def log(self, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        self._gl_step_logs[name] = float(value)

    @classmethod
    def load_from_checkpoint(
        cls, path: str | Path, *args: Any, map_location: Any = "cpu", **kwargs: Any
    ) -> Module:
        """Build ``cls(*args, **kwargs)`` and load the weights saved by the Trainer."""
        model = cls(*args, **kwargs)
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        return model


class _TrainingStep(nn.Module):
    """Lets DDP wrap ``training_step`` (DDP only syncs gradients for calls to its forward)."""

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.module.training_step(batch, batch_idx)
