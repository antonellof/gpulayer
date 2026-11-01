"""Experiment loggers, named as in PyTorch Lightning.

    Trainer(logger=CSVLogger("logs"))
    Trainer(logger=[TensorBoardLogger("tb"), WandbLogger(project="x")])

The Trainer calls ``log_metrics`` on rank 0 only, on logged steps (``log_every_n_steps``) and at
the end of each epoch and validation, with values it already read for its own log line: a
logger adds no device synchronisation. ``log_hyperparams`` receives ``model.hparams``.
TensorBoard, Weights & Biases and MLflow need their packages installed.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from gpulayer.errors import ValidationFailed


class Logger:
    """Base class: override ``log_metrics`` and ``log_hyperparams``."""

    @property
    def name(self) -> str | None:
        """Experiment name, or None."""
        return None

    @property
    def version(self) -> int | str | None:
        """Run version within the experiment, or None."""
        return None

    @property
    def save_dir(self) -> str | None:
        """Root directory of the logs, or None."""
        return None

    @property
    def log_dir(self) -> str | None:
        """Directory of this run's logs (default: ``save_dir``)."""
        return self.save_dir

    @property
    def experiment(self) -> Any:
        """The underlying object (SummaryWriter, wandb run, MlflowClient), created on first use."""
        return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record ``metrics`` at optimizer step ``step`` (rank 0 only)."""

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Record the model's and DataModule's hyperparameters once, at the start of fit."""

    def log_graph(self, model: Any, input_array: Any = None) -> None:
        """Record the model graph, when the logger supports it."""

    def save(self) -> None:
        """Flush what was logged so far."""

    def finalize(self, status: str) -> None:
        """Called once at the end of fit with ``status`` ("success" or "failed")."""
        self.save()

    def after_save_checkpoint(self, checkpoint_path: Path) -> None:
        """Called after each checkpoint is written (for loggers that upload them)."""


def _next_version(root: Path) -> int:
    if not root.is_dir():
        return 0
    found = [
        int(p.name.split("_", 1)[1])
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("version_") and p.name.split("_", 1)[1].isdigit()
    ]
    return max(found) + 1 if found else 0


def _plain(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[k] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
    return out


class CSVLogger(Logger):
    """Write ``<save_dir>/<name>/version_<n>/metrics.csv`` (one row per logged step or epoch)
    and ``hparams.yaml``."""

    def __init__(
        self,
        save_dir: str | Path = "gpulayer_logs",
        name: str | None = "gpulayer_logs",
        version: int | str | None = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        self._root = Path(save_dir)
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._flush_every = flush_logs_every_n_steps
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] = []
        self._written = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int | str:
        if self._version is None:
            self._version = _next_version(self._root / self._name)
        return self._version

    @property
    def save_dir(self) -> str:
        return str(self._root)

    @property
    def log_dir(self) -> str:
        v = self.version
        return str(self._root / self._name / (v if isinstance(v, str) else f"version_{v}"))

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        d = Path(self.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"{k}: {json.dumps(v)}" for k, v in _plain(params).items()]
        (d / "hparams.yaml").write_text("\n".join(lines) + ("\n" if lines else ""))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        row = {"step": step, **{f"{self._prefix}{k}": v for k, v in metrics.items()}}
        self._rows.append(row)
        if len(self._rows) - self._written >= self._flush_every:
            self.save()

    def save(self) -> None:
        if not self._rows:
            return
        cols = list(self._columns)
        for r in self._rows:
            cols += [k for k in r if k not in cols]
        d = Path(self.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self._rows)
        self._columns, self._written = cols, len(self._rows)


class TensorBoardLogger(Logger):
    """TensorBoard event files in ``<save_dir>/<name>/version_<n>`` (needs ``tensorboard``)."""

    def __init__(
        self,
        save_dir: str | Path = "gpulayer_logs",
        name: str | None = "gpulayer_logs",
        version: int | str | None = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        self._root = Path(save_dir)
        self._name = name or ""
        self._version = version
        self._log_graph = log_graph
        self._prefix = prefix
        self._kwargs = kwargs
        self._writer: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int | str:
        if self._version is None:
            self._version = _next_version(self._root / self._name)
        return self._version

    @property
    def save_dir(self) -> str:
        return str(self._root)

    @property
    def log_dir(self) -> str:
        v = self.version
        return str(self._root / self._name / (v if isinstance(v, str) else f"version_{v}"))

    @property
    def experiment(self) -> Any:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ValidationFailed("TensorBoardLogger needs: pip install tensorboard") from e
            self._writer = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._writer

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            self.experiment.add_scalar(f"{self._prefix}{k}", v, step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if params:
            self.experiment.add_text("hparams", json.dumps(_plain(params), indent=2))

    def log_graph(self, model: Any, input_array: Any = None) -> None:
        if self._log_graph and input_array is not None:
            self.experiment.add_graph(model, input_array)

    def save(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def finalize(self, status: str) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class WandbLogger(Logger):
    """Weights & Biases (needs ``wandb`` and a login). Arguments go to ``wandb.init``."""

    def __init__(
        self,
        name: str | None = None,
        save_dir: str | Path = ".",
        version: str | None = None,
        offline: bool = False,
        project: str | None = None,
        log_model: bool = False,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        self._init = {
            "name": name,
            "dir": str(save_dir),
            "id": version,
            "project": project or os.environ.get("WANDB_PROJECT", "gpulayer"),
            "mode": "offline" if offline else None,
            **kwargs,
        }
        self._init = {k: v for k, v in self._init.items() if v is not None}
        self._prefix = prefix
        self._log_model = log_model
        self._run: Any = None

    @property
    def experiment(self) -> Any:
        if self._run is None:
            try:
                import wandb
            except ImportError as e:
                raise ValidationFailed("WandbLogger needs: pip install wandb") from e
            self._run = wandb.run or wandb.init(**self._init)
        return self._run

    @property
    def name(self) -> str | None:
        return self._init.get("name")

    @property
    def version(self) -> str | None:
        return self._run.id if self._run is not None else self._init.get("id")

    @property
    def save_dir(self) -> str:
        return str(self._init.get("dir", "."))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        data = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        if step is not None:
            data["trainer/global_step"] = step
        self.experiment.log(data)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self.experiment.config.update(_plain(params), allow_val_change=True)

    def after_save_checkpoint(self, checkpoint_path: Path) -> None:
        if self._log_model:
            self.experiment.save(str(checkpoint_path))

    def finalize(self, status: str) -> None:
        if self._run is not None:
            self._run.finish()


class MLFlowLogger(Logger):
    """MLflow tracking (needs ``mlflow``)."""

    def __init__(
        self,
        experiment_name: str = "gpulayer_logs",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: bool = False,
        prefix: str = "",
        run_id: str | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri or (f"file:{save_dir}" if save_dir else None)
        self._tags = tags
        self._prefix = prefix
        self._log_model = log_model
        self._run_id = run_id
        self._client: Any = None
        self._save_dir = save_dir

    @property
    def experiment(self) -> Any:
        if self._client is None:
            try:
                from mlflow.tracking import MlflowClient
            except ImportError as e:
                raise ValidationFailed("MLFlowLogger needs: pip install mlflow") from e
            self._client = MlflowClient(self._tracking_uri)
            if self._run_id is None:
                exp = self._client.get_experiment_by_name(self._experiment_name)
                exp_id = (
                    exp.experiment_id
                    if exp is not None
                    else self._client.create_experiment(self._experiment_name)
                )
                tags = dict(self._tags or {})
                if self._run_name:
                    tags["mlflow.runName"] = self._run_name
                self._run_id = self._client.create_run(exp_id, tags=tags).info.run_id
        return self._client

    @property
    def name(self) -> str:
        return self._experiment_name

    @property
    def version(self) -> str | None:
        return self._run_id

    @property
    def save_dir(self) -> str | None:
        return self._save_dir

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        client = self.experiment
        for k, v in metrics.items():
            client.log_metric(self._run_id, f"{self._prefix}{k}".replace("/", "-"), v, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        client = self.experiment
        for k, v in _plain(params).items():
            client.log_param(self._run_id, k, v)

    def after_save_checkpoint(self, checkpoint_path: Path) -> None:
        if self._log_model:
            self.experiment.log_artifact(self._run_id, str(checkpoint_path))

    def finalize(self, status: str) -> None:
        if self._client is not None and self._run_id is not None:
            self._client.set_terminated(
                self._run_id, "FINISHED" if status == "success" else "FAILED"
            )
