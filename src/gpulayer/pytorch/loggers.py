"""Experiment loggers, named as in PyTorch Lightning.

    Trainer(logger=CSVLogger("logs"))
    Trainer(logger=[TensorBoardLogger("tb"), WandbLogger(project="x")])

The Trainer calls ``log_metrics`` on rank 0 only, on logged steps (``log_every_n_steps``) and at
the end of each epoch and validation, with values it already read for its own log line: a
logger adds no device synchronisation. ``log_hyperparams`` receives ``model.hparams``.
TensorBoard, Weights & Biases, MLflow and Comet need their packages installed. Methods a user
may call from a step (``WandbLogger.log_image``, ``watch``, ...) do nothing outside rank 0.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import inspect
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from gpulayer.errors import ValidationFailed

T = TypeVar("T")


def _global_rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def _rank_zero(fn: Callable[..., T]) -> Callable[..., T | None]:
    """Run ``fn`` in the process with global rank 0 only; elsewhere return None."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> T | None:
        return fn(*args, **kwargs) if _global_rank() == 0 else None

    return wrapped


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
    def root_dir(self) -> str:
        """``<save_dir>/<name>``: the parent of every ``version_<n>`` directory."""
        return str(self._root / self._name)

    @property
    def log_dir(self) -> str:
        v = self.version
        return str(Path(self.root_dir) / (v if isinstance(v, str) else f"version_{v}"))

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
    """TensorBoard event files in ``<save_dir>/<name>/version_<n>[/<sub_dir>]`` (needs
    ``tensorboard``). ``sub_dir`` may use ``~`` and environment variables."""

    def __init__(
        self,
        save_dir: str | Path = "gpulayer_logs",
        name: str | None = "gpulayer_logs",
        version: int | str | None = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._root = Path(save_dir)
        self._name = name or ""
        self._version = version
        self._log_graph = log_graph
        self._prefix = prefix
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)
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
    def root_dir(self) -> str:
        """``<save_dir>/<name>``: the parent of every ``version_<n>`` directory."""
        return str(self._root / self._name)

    @property
    def sub_dir(self) -> str | None:
        """Subdirectory of the version directory the event files go to, or None."""
        return self._sub_dir

    @property
    def log_dir(self) -> str:
        v = self.version
        d = os.path.join(self.root_dir, v if isinstance(v, str) else f"version_{v}")
        if self._sub_dir is not None:
            d = os.path.join(d, self._sub_dir)
        return os.path.expanduser(os.path.expandvars(d))

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
    """Weights & Biases (needs ``wandb`` and a login). Extra arguments go to ``wandb.init``.

    ``experiment`` reuses a run you already started. ``log_model=True`` uploads each saved
    checkpoint as a ``model`` artifact named ``checkpoint_name`` (default ``model-<run id>``)
    with the alias ``latest``; ``add_file_policy`` is passed to ``Artifact.add_file``."""

    def __init__(
        self,
        name: str | None = None,
        save_dir: str | Path | None = ".",
        version: str | None = None,
        offline: bool = False,
        dir: str | Path | None = None,
        id: str | None = None,
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: bool = False,
        experiment: Any = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        add_file_policy: str = "mutable",
        **kwargs: Any,
    ) -> None:
        if offline and log_model:
            raise ValidationFailed(
                "WandbLogger: log_model=True cannot upload checkpoints with offline=True"
            )
        if add_file_policy not in ("mutable", "immutable"):
            raise ValidationFailed("WandbLogger: add_file_policy must be 'mutable' or 'immutable'")
        root = save_dir if save_dir is not None else dir
        self._init = {
            "name": name,
            "dir": None if root is None else os.fspath(root),
            "id": version or id,
            "project": project or os.environ.get("WANDB_PROJECT", "gpulayer"),
            "mode": "offline" if offline else None,
            "resume": "allow",
            "anonymous": "allow" if anonymous else None,
            **kwargs,
        }
        self._init = {k: v for k, v in self._init.items() if v is not None}
        self._prefix = prefix
        self._log_model = log_model
        self._checkpoint_name = checkpoint_name
        self.add_file_policy = add_file_policy
        self._run: Any = experiment

    @property
    def experiment(self) -> Any:
        if self._run is None:
            try:
                import wandb
            except ImportError as e:
                raise ValidationFailed("WandbLogger needs: pip install wandb") from e
            if wandb.run is not None:
                self._run = wandb.run
            else:
                self._run = wandb.init(**self._init)
                if getattr(self._run, "define_metric", None) is not None:
                    self._run.define_metric("trainer/global_step")
                    self._run.define_metric("*", step_metric="trainer/global_step", step_sync=True)
        return self._run

    @property
    def name(self) -> str | None:
        return self._init.get("name")

    @property
    def version(self) -> str | None:
        return self._run.id if self._run is not None else self._init.get("id")

    @property
    def save_dir(self) -> str | None:
        return self._init.get("dir")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        data = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        if step is not None:
            data["trainer/global_step"] = step
        self.experiment.log(data)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self.experiment.config.update(_plain(params), allow_val_change=True)

    @_rank_zero
    def watch(
        self,
        model: Any,
        log: str | None = "gradients",
        log_freq: int = 100,
        log_graph: bool = True,
    ) -> None:
        """Have wandb log the model's gradients and/or parameters every ``log_freq`` steps."""
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @_rank_zero
    def log_table(
        self,
        key: str,
        columns: list[str] | None = None,
        data: list[list[Any]] | None = None,
        dataframe: Any = None,
        step: int | None = None,
    ) -> None:
        """Log a ``wandb.Table`` from ``columns`` and ``data`` or from a pandas ``dataframe``."""
        import wandb

        table = wandb.Table(columns=columns, data=data, dataframe=dataframe)
        self.log_metrics({key: table}, step)

    @_rank_zero
    def log_text(
        self,
        key: str,
        columns: list[str] | None = None,
        data: list[list[str]] | None = None,
        dataframe: Any = None,
        step: int | None = None,
    ) -> None:
        """Log text as a table (same arguments as ``log_table``)."""
        self.log_table(key, columns, data, dataframe, step)

    def _log_media(
        self, kind: str, key: str, items: list[Any], step: int | None, kwargs: dict[str, Any]
    ) -> None:
        if not isinstance(items, list):
            raise TypeError(f'Expected a list as "{kind.lower()}s", found {type(items)}')
        n = len(items)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        import wandb

        cls = getattr(wandb, kind)
        media = [cls(item, **{k: kwargs[k][i] for k in kwargs}) for i, item in enumerate(items)]
        data: dict[str, Any] = {key: media}
        self.log_metrics(data, step)

    @_rank_zero
    def log_image(
        self, key: str, images: list[Any], step: int | None = None, **kwargs: Any
    ) -> None:
        """Log ``wandb.Image``s; each keyword (``caption=[...]``) gives one value per image."""
        self._log_media("Image", key, images, step, kwargs)

    @_rank_zero
    def log_audio(
        self, key: str, audios: list[Any], step: int | None = None, **kwargs: Any
    ) -> None:
        """Log ``wandb.Audio`` clips; each keyword (``sample_rate=[...]``) has one per clip."""
        self._log_media("Audio", key, audios, step, kwargs)

    @_rank_zero
    def log_video(
        self, key: str, videos: list[Any], step: int | None = None, **kwargs: Any
    ) -> None:
        """Log ``wandb.Video``s; each keyword (``fps=[...]``) gives one value per video."""
        self._log_media("Video", key, videos, step, kwargs)

    @staticmethod
    @_rank_zero
    def download_artifact(
        artifact: str,
        save_dir: str | Path | None = None,
        artifact_type: str | None = None,
        use_artifact: bool | None = True,
    ) -> str:
        """Download a wandb artifact to ``save_dir`` and return its local path. Inside a run
        (with ``use_artifact``) the run records that it used it."""
        try:
            import wandb
        except ImportError as e:
            raise ValidationFailed("WandbLogger needs: pip install wandb") from e
        if wandb.run is not None and use_artifact:
            art = wandb.run.use_artifact(artifact)
        else:
            art = wandb.Api().artifact(artifact, type=artifact_type)
        return str(art.download(root=None if save_dir is None else os.fspath(save_dir)))

    @_rank_zero
    def use_artifact(self, artifact: str, artifact_type: str | None = None) -> Any:
        """Record that this run uses ``artifact`` and return the ``wandb.Artifact``."""
        return self.experiment.use_artifact(artifact, type=artifact_type)

    def after_save_checkpoint(self, checkpoint_path: Path) -> None:
        if not self._log_model:
            return
        import wandb

        run = self.experiment
        art = wandb.Artifact(
            name=self._checkpoint_name or f"model-{run.id}",
            type="model",
            metadata={"original_filename": Path(checkpoint_path).name},
        )
        art.add_file(str(checkpoint_path), name="model.ckpt", policy=self.add_file_policy)
        run.log_artifact(art, aliases=["latest"])

    def finalize(self, status: str) -> None:
        if self._run is not None:
            self._run.finish()


class MLFlowLogger(Logger):
    """MLflow tracking (needs ``mlflow``). Metrics and parameters go in one ``log_batch`` call
    each. ``artifact_location`` is used when the experiment is created; ``synchronous`` (mlflow
    2.8 or later) is passed to ``log_batch``: False logs without waiting for the server."""

    def __init__(
        self,
        experiment_name: str = "gpulayer_logs",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: bool = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
        synchronous: bool | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri or (f"file:{save_dir}" if save_dir else None)
        self._tags = tags
        self._prefix = prefix
        self._log_model = log_model
        self._artifact_location = artifact_location
        self._run_id = run_id
        self._experiment_id: str | None = None
        self._batch_kwargs = {} if synchronous is None else {"synchronous": synchronous}
        self._client: Any = None
        self._save_dir = save_dir

    @property
    def experiment(self) -> Any:
        if self._client is None:
            try:
                from mlflow.tracking import MlflowClient
            except ImportError as e:
                raise ValidationFailed("MLFlowLogger needs: pip install mlflow") from e
            client = MlflowClient(self._tracking_uri)
            if self._batch_kwargs and (
                "synchronous" not in inspect.signature(client.log_batch).parameters
            ):
                raise ValidationFailed("MLFlowLogger(synchronous=...) needs mlflow>=2.8")
            if self._run_id is not None:
                self._experiment_id = client.get_run(self._run_id).info.experiment_id
            else:
                exp = client.get_experiment_by_name(self._experiment_name)
                if exp is not None and getattr(exp, "lifecycle_stage", "active") != "deleted":
                    self._experiment_id = exp.experiment_id
                else:
                    self._experiment_id = client.create_experiment(
                        self._experiment_name, artifact_location=self._artifact_location
                    )
                tags = dict(self._tags or {})
                if self._run_name:
                    tags["mlflow.runName"] = self._run_name
                self._run_id = client.create_run(self._experiment_id, tags=tags).info.run_id
            self._client = client
        return self._client

    @property
    def run_id(self) -> str | None:
        """The MLflow run id (the run is created on first use)."""
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        """The MLflow experiment id (created on first use when it does not exist)."""
        _ = self.experiment
        return self._experiment_id

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
        from mlflow.entities import Metric

        client = self.experiment
        now = int(time.time() * 1000)
        batch = [
            Metric(
                key=f"{self._prefix}{k}".replace("/", "-"), value=v, timestamp=now, step=step or 0
            )
            for k, v in metrics.items()
            if not isinstance(v, str)
        ]
        client.log_batch(run_id=self._run_id, metrics=batch, **self._batch_kwargs)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        from mlflow.entities import Param

        client = self.experiment
        batch = [Param(key=k, value=str(v)[:250]) for k, v in _plain(params).items()]
        for i in range(0, len(batch), 100):  # MLflow accepts at most 100 params per batch
            client.log_batch(run_id=self._run_id, params=batch[i : i + 100], **self._batch_kwargs)

    def after_save_checkpoint(self, checkpoint_path: Path) -> None:
        if self._log_model:
            self.experiment.log_artifact(self._run_id, str(checkpoint_path))

    def finalize(self, status: str) -> None:
        if self._client is not None and self._run_id is not None:
            self._client.set_terminated(
                self._run_id, "FINISHED" if status == "success" else "FAILED"
            )


class CometLogger(Logger):
    """Comet (needs ``comet_ml``). The experiment is started on first use, on rank 0, with
    ``comet_ml.start``; extra arguments go to ``comet_ml.ExperimentConfig`` (for example
    ``name``, ``tags``, ``offline_directory``). ``online=False`` logs offline. Comet's own
    automatic logging is turned off (``COMET_DISABLE_AUTO_LOGGING=1``), as in Lightning.
    ``finalize`` flushes without ending the experiment, so it can be used after fit."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        workspace: str | None = None,
        project: str | None = None,
        experiment_key: str | None = None,
        mode: str | None = None,
        online: bool | None = None,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        if mode not in (None, "get_or_create", "get", "create"):
            raise ValidationFailed("CometLogger: mode must be 'get_or_create', 'get' or 'create'")
        self._api_key = api_key
        self._workspace = workspace
        self._project = project
        self._experiment_key = experiment_key
        self._mode = mode
        self._online = online
        self._prefix = prefix
        self._kwargs = kwargs
        self._experiment: Any = None

    @property
    def experiment(self) -> Any:
        if self._experiment is None:
            os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
            try:
                import comet_ml
            except ImportError as e:
                raise ValidationFailed("CometLogger needs: pip install comet-ml") from e
            config = dict(self._kwargs)
            if self._online is False:
                config["disabled"] = True
            exp = comet_ml.start(
                api_key=self._api_key,
                workspace=self._workspace,
                project=self._project,
                experiment_key=self._experiment_key,
                mode=self._mode,
                online=self._online,
                experiment_config=comet_ml.ExperimentConfig(**config),
            )
            if exp is None:
                raise ValidationFailed("CometLogger: comet_ml.start returned no experiment")
            self._experiment = exp
            self._experiment_key = exp.get_key()
            self._project = exp.project_name
            exp.log_other("Created from", "gpulayer")
        return self._experiment

    @property
    def name(self) -> str | None:
        return self._project

    @property
    def version(self) -> str | None:
        return self._experiment.get_key() if self._experiment is not None else self._experiment_key

    @property
    def save_dir(self) -> str | None:
        d = self._kwargs.get("offline_directory")
        return None if d is None else os.fspath(d)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        data = dict(metrics)
        epoch = data.pop("epoch", None)
        self.experiment.log_metrics(data, step=step, epoch=epoch, prefix=self._prefix)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self.experiment.log_parameters(dict(params))

    def log_graph(self, model: Any, input_array: Any = None) -> None:
        self.experiment.set_model_graph(model)

    def finalize(self, status: str) -> None:
        if self._experiment is not None:
            self._experiment.flush()


class LitLogger(Logger):
    """Lightning AI's experiment tracker (needs ``pip install litlogger`` and a Lightning AI
    account): metrics, hyperparameters and, with ``log_model=True``, checkpoints go to the
    ``teamspace`` there. Data leaves this machine for Lightning AI's service. The experiment
    starts on first use; the Trainer logs from global rank 0 only."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        name: str | None = None,
        teamspace: str | None = None,
        metadata: dict[str, str] | None = None,
        store_step: bool = True,
        log_model: bool = False,
        save_logs: bool = True,
        checkpoint_name: str | None = None,
    ) -> None:
        self._root_dir = os.fspath(root_dir or "./lightning_logs")
        self._name = name
        self._teamspace = teamspace
        self._metadata = {k: str(v) for k, v in (metadata or {}).items()}
        self._store_step = store_step
        self._log_model = log_model
        self._save_logs = save_logs
        self._checkpoint_name = checkpoint_name
        self._version: str | None = None
        self._experiment: Any = None
        self._step = -1

    def _lib(self) -> Any:
        try:
            import litlogger
        except ImportError as e:
            raise ValidationFailed("LitLogger needs: pip install litlogger") from e
        return litlogger

    @property
    def name(self) -> str:
        """The experiment's name (generated by litlogger when not given)."""
        if self._name is None:
            try:
                from litlogger.generator import _create_name

                self._name = str(_create_name())
            except ImportError:
                self._name = "gpulayer"
        return self._name

    @property
    def version(self) -> str | None:
        """A UTC timestamp, set when the experiment starts."""
        return self._version

    @property
    def root_dir(self) -> str:
        """``root_dir/name``."""
        return os.path.join(self._root_dir, self.name)

    @property
    def save_dir(self) -> str:
        """``root_dir`` as given."""
        return self._root_dir

    @property
    def log_dir(self) -> str:
        """Where local copies of the logs go: ``root_dir/name/version``."""
        return os.path.join(self.root_dir, self._version or "")

    @property
    def experiment(self) -> Any:
        """The ``litlogger.Experiment``, started on first use."""
        if self._experiment is None:
            lib = self._lib()
            from datetime import datetime, timezone

            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            self._version = stamp.replace(":", "-").replace("+00:00", "Z")
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._experiment = lib.Experiment(
                name=self.name, teamspace=self._teamspace, metadata=self._metadata,
                store_step=self._store_step, store_created_at=True, log_dir=self.log_dir,
                save_logs=self._save_logs,
            )  # fmt: skip
            with contextlib.suppress(AttributeError):
                self._experiment.print_url()
        return self._experiment

    @property
    def url(self) -> str:
        """The experiment's page on Lightning AI."""
        return str(self.experiment.url)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append each value to its series at ``step``."""
        exp = self.experiment
        self._step = self._step + 1 if step is None else step
        for key, value in metrics.items():
            v = value.item() if hasattr(value, "item") else value
            exp[key].append(v, step=self._step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Store each hyperparameter as text."""
        exp = self.experiment
        for key, value in dict(params).items():
            exp[key] = str(value)

    def log_metadata(self, params: dict[str, Any]) -> None:
        """Store extra values as text (as ``log_hyperparams``)."""
        self.log_hyperparams(params)

    def log_file(self, path: str) -> None:
        """Upload a file to the teamspace drive, under the experiment's folder."""
        self.experiment[f"{self.name}/{os.path.basename(path)}"] = self._lib().File(path)

    def get_file(self, path: str, verbose: bool = True) -> str:
        """Download a file logged with ``log_file`` to ``path``."""
        return str(self.experiment[f"{self.name}/{os.path.basename(path)}"].save(path))

    def log_model(
        self,
        model: Any,
        staging_dir: str | None = None,
        verbose: bool = False,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upload a model object to Lightning AI's model registry."""
        key = self._checkpoint_name or f"{self.name}-model"
        self.experiment[key] = self._lib().Model(
            model, version=version or self._version, metadata=metadata, staging_dir=staging_dir
        )

    def log_model_artifact(
        self, path: str, verbose: bool = False, version: str | None = None
    ) -> None:
        """Upload a model file or folder to the registry."""
        key = self._checkpoint_name or f"{self.name}-model"
        self.experiment[key] = self._lib().Model(path, version=version or self._version)

    def after_save_checkpoint(self, checkpoint_path: Path) -> None:
        """With ``log_model=True``, upload each saved checkpoint."""
        if self._log_model:
            self.log_model_artifact(str(checkpoint_path), version=f"step-{max(self._step, 0)}")

    def finalize(self, status: str) -> None:
        """End the experiment with ``status``."""
        if self._experiment is not None:
            self._experiment.finalize(status)
