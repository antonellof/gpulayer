"""Lightning parity of the loggers and profilers: CometLogger, the new MLFlowLogger,
TensorBoardLogger, CSVLogger and WandbLogger arguments, PyTorchProfiler options, XLAProfiler and
Profiler.local_rank.

Simulated: comet_ml, mlflow, wandb, tensorboard and torch_xla are replaced by small fake modules
injected into ``sys.modules``; these tests check what GPUlayer calls on them and with which
arguments, not the real services. The PyTorchProfiler table, module ranges and
``group_by_input_shapes`` run the real ``torch.profiler`` on the CPU; ``emit_nvtx`` is checked
with fakes (no CUDA). CPU, single process."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch.loggers import (  # noqa: E402
    CometLogger,
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)
from gpulayer.pytorch.profilers import (  # noqa: E402
    Profiler,
    PyTorchProfiler,
    XLAProfiler,
    resolve_profiler,
)


@pytest.fixture(autouse=True)
def _rank_env(monkeypatch):
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)


def _module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class Lin(glt.Module):
    def __init__(self, lr: float = 0.05):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.net = nn.Linear(4, 1)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def _loader():
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=16)


# ---------------------------------------------------------------------- comet


class FakeCometExperiment:
    def __init__(self, kwargs):
        self.start_kwargs = kwargs
        self.project_name = kwargs["project"] or "general"
        self.calls = []

    def get_key(self):
        return "key-123"

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


@pytest.fixture
def comet(monkeypatch):
    started = []

    class ExperimentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def start(**kwargs):
        exp = FakeCometExperiment(kwargs)
        started.append(exp)
        return exp

    monkeypatch.delenv("COMET_DISABLE_AUTO_LOGGING", raising=False)
    _module(monkeypatch, "comet_ml", start=start, ExperimentConfig=ExperimentConfig)
    return started


def test_comet_logger_starts_experiment_and_logs(comet, tmp_path):
    import os

    lg = CometLogger(
        api_key="k",
        workspace="ws",
        project="proj",
        mode="get_or_create",
        online=False,
        prefix="train",
        name="run1",
        offline_directory=str(tmp_path),
    )
    assert comet == []  # started on first use, not in the constructor
    assert lg.version is None and lg.save_dir == str(tmp_path) and lg.name == "proj"
    lg.log_metrics({"loss": 0.5, "epoch": 2}, step=7)
    (exp,) = comet
    kw = exp.start_kwargs
    assert (kw["api_key"], kw["workspace"], kw["project"], kw["mode"], kw["online"]) == (
        "k",
        "ws",
        "proj",
        "get_or_create",
        False,
    )
    assert kw["experiment_config"].kwargs == {
        "name": "run1",
        "offline_directory": str(tmp_path),
        "disabled": True,
    }
    assert os.environ["COMET_DISABLE_AUTO_LOGGING"] == "1"
    assert ("log_other", ("Created from", "gpulayer"), {}) in exp.calls
    assert (
        "log_metrics",
        ({"loss": 0.5},),
        {"step": 7, "epoch": 2, "prefix": "train"},
    ) in exp.calls
    lg.log_hyperparams({"lr": 0.1})
    model = nn.Linear(2, 2)
    lg.log_graph(model)
    lg.finalize("success")
    names = [c[0] for c in exp.calls]
    assert ("log_parameters", ({"lr": 0.1},), {}) in exp.calls
    assert ("set_model_graph", (model,), {}) in exp.calls
    assert names[-1] == "flush" and "end" not in names
    assert lg.version == "key-123" and lg.experiment is exp and len(comet) == 1


def test_comet_logger_errors(monkeypatch):
    with pytest.raises(ValidationFailed, match="mode"):
        CometLogger(mode="sometimes")
    monkeypatch.setitem(sys.modules, "comet_ml", None)
    with pytest.raises(ValidationFailed, match="pip install comet-ml"):
        _ = CometLogger().experiment


def test_comet_logger_in_a_trainer_fit(comet, tmp_path):
    lg = CometLogger(project="p")
    t = glt.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=1,
        log_every_n_steps=2,
        num_sanity_val_steps=0,
        logger=lg,
    )
    t.fit(Lin(), _loader())
    calls = comet[0].calls
    assert ("log_parameters", ({"lr": 0.05},), {}) in calls
    logged = [c for c in calls if c[0] == "log_metrics"]
    assert logged and all("epoch" not in c[1][0] for c in logged)
    assert any("train_loss" in c[1][0] and c[2]["step"] == 1 for c in logged)  # from 0
    assert calls[-1][0] == "flush"


# --------------------------------------------------------------------- mlflow


class FakeMlflowClient:
    instances: list = []

    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri
        self.calls = []
        FakeMlflowClient.instances.append(self)

    def get_experiment_by_name(self, name):
        self.calls.append(("get_experiment_by_name", name))
        return None

    def create_experiment(self, name, artifact_location=None):
        self.calls.append(("create_experiment", name, artifact_location))
        return "exp-1"

    def create_run(self, experiment_id, tags=None):
        self.calls.append(("create_run", experiment_id, tags))
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-1"))

    def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return types.SimpleNamespace(info=types.SimpleNamespace(experiment_id="exp-9"))

    def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=None):
        self.calls.append(("log_batch", run_id, list(metrics), list(params), synchronous))

    def set_terminated(self, run_id, status=None):
        self.calls.append(("set_terminated", run_id, status))


@pytest.fixture
def mlflow(monkeypatch):
    FakeMlflowClient.instances = []

    class Metric(types.SimpleNamespace):
        pass

    class Param(types.SimpleNamespace):
        pass

    _module(monkeypatch, "mlflow")
    _module(monkeypatch, "mlflow.tracking", MlflowClient=FakeMlflowClient)
    _module(monkeypatch, "mlflow.entities", Metric=Metric, Param=Param)
    return FakeMlflowClient.instances


def test_mlflow_artifact_location_ids_and_synchronous(mlflow, tmp_path):
    lg = MLFlowLogger(
        experiment_name="e",
        run_name="r",
        save_dir=str(tmp_path),
        artifact_location="s3://bucket/a",
        synchronous=False,
        prefix="p/",
    )
    assert lg.experiment_id == "exp-1" and lg.run_id == "run-1"
    (client,) = mlflow
    assert client.tracking_uri == f"file:{tmp_path}"
    assert ("create_experiment", "e", "s3://bucket/a") in client.calls
    assert ("create_run", "exp-1", {"mlflow.runName": "r"}) in client.calls
    lg.log_metrics({"loss": 0.25, "acc": 0.5, "note": "x"}, step=3)
    _, run_id, metrics, params, sync = client.calls[-1]
    assert run_id == "run-1" and params == [] and sync is False
    assert [(m.key, m.value, m.step) for m in metrics] == [("p-loss", 0.25, 3), ("p-acc", 0.5, 3)]
    lg.log_hyperparams({f"k{i}": i for i in range(150)})
    batches = [c for c in client.calls if c[0] == "log_batch" and c[3]]
    assert [len(c[3]) for c in batches] == [100, 50] and batches[0][3][1].value == "1"
    lg.finalize("failed")
    assert client.calls[-1] == ("set_terminated", "run-1", "FAILED")


def test_mlflow_existing_run_id_reads_experiment(mlflow):
    lg = MLFlowLogger(run_id="given")
    assert lg.run_id == "given" and lg.experiment_id == "exp-9"
    assert ("get_run", "given") in mlflow[0].calls
    lg.log_metrics({"a": 1.0})
    assert mlflow[0].calls[-1][4] is None  # synchronous not passed by default


def test_mlflow_synchronous_needs_new_mlflow(mlflow, monkeypatch):
    monkeypatch.setattr(
        FakeMlflowClient, "log_batch", lambda self, run_id, metrics=(), params=(), tags=(): None
    )
    with pytest.raises(ValidationFailed, match="mlflow>=2.8"):
        _ = MLFlowLogger(synchronous=True).experiment


# ------------------------------------------------------- tensorboard and csv


def test_tensorboard_sub_dir_and_root_dir(monkeypatch, tmp_path):
    writers = []

    class SummaryWriter:
        def __init__(self, log_dir, **kwargs):
            self.log_dir, self.kwargs, self.scalars = log_dir, kwargs, []
            writers.append(self)

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    _module(monkeypatch, "torch.utils.tensorboard", SummaryWriter=SummaryWriter)
    monkeypatch.setenv("GL_SUB", "fold1")
    lg = TensorBoardLogger(tmp_path, name="exp", sub_dir="$GL_SUB/train", flush_secs=5)
    assert lg.root_dir == str(tmp_path / "exp") and lg.sub_dir == "$GL_SUB/train"
    assert lg.log_dir == str(tmp_path / "exp" / "version_0" / "fold1" / "train")
    lg.log_metrics({"loss": 1.5}, step=4)
    assert writers[0].log_dir == lg.log_dir and writers[0].kwargs == {"flush_secs": 5}
    assert writers[0].scalars == [("loss", 1.5, 4)]
    plain = TensorBoardLogger(tmp_path, name="", version="v")
    assert plain.root_dir == str(tmp_path) and plain.log_dir == str(tmp_path / "v")


def test_csv_root_dir(tmp_path):
    lg = CSVLogger(tmp_path, name="exp")
    assert lg.root_dir == str(tmp_path / "exp")
    assert lg.log_dir == str(tmp_path / "exp" / "version_0")
    assert CSVLogger(tmp_path, name=None).root_dir == str(tmp_path)


# ---------------------------------------------------------------------- wandb


class FakeRun:
    def __init__(self, **init):
        self.init = init
        self.id = init.get("id", "abc")
        self.calls = []
        self.config = types.SimpleNamespace(update=lambda *a, **k: None)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return f"{name}-result"

        return record


class Media:
    def __init__(self, data, **kwargs):
        self.data, self.kwargs = data, kwargs

    def __eq__(self, other):
        return (type(self), self.data, self.kwargs) == (type(other), other.data, other.kwargs)


@pytest.fixture
def wandb(monkeypatch):
    class Table:
        def __init__(self, columns=None, data=None, dataframe=None):
            self.args = (columns, data, dataframe)

    class Image(Media):
        pass

    class Audio(Media):
        pass

    class Video(Media):
        pass

    class Artifact:
        def __init__(self, name, type, metadata=None):
            self.name, self.type, self.metadata, self.files = name, type, metadata, []

        def add_file(self, path, name=None, policy=None):
            self.files.append((path, name, policy))

    class Downloadable:
        def __init__(self, ref, how):
            self.ref, self.how = ref, how

        def download(self, root=None):
            return f"{self.how}:{self.ref}:{root}"

    class Api:
        def artifact(self, ref, type=None):
            return Downloadable(ref, f"api[{type}]")

    def init(**kwargs):
        run = FakeRun(**kwargs)
        mod.inits.append(run)
        return run

    mod = _module(
        monkeypatch,
        "wandb",
        run=None,
        init=init,
        inits=[],
        Table=Table,
        Image=Image,
        Audio=Audio,
        Video=Video,
        Artifact=Artifact,
        Api=Api,
        Downloadable=Downloadable,
    )
    return mod


def test_wandb_init_arguments(wandb, tmp_path):
    lg = WandbLogger(
        save_dir=None, dir=tmp_path, id="run-7", anonymous=True, project="p", tags=["a"]
    )
    assert lg.save_dir == str(tmp_path) and lg.version == "run-7"
    run = lg.experiment
    assert run.init == {
        "dir": str(tmp_path),
        "id": "run-7",
        "project": "p",
        "resume": "allow",
        "anonymous": "allow",
        "tags": ["a"],
    }
    assert ("define_metric", ("trainer/global_step",), {}) in run.calls
    assert (
        "define_metric",
        ("*",),
        {"step_metric": "trainer/global_step", "step_sync": True},
    ) in run.calls
    # version wins over id, save_dir over dir, as in Lightning
    lg2 = WandbLogger(version="v", id="i", save_dir="s", dir="d")
    assert lg2.version == "v" and lg2.save_dir == "s"


def test_wandb_given_experiment_is_used(wandb):
    run = FakeRun(id="mine")
    lg = WandbLogger(experiment=run)
    lg.log_metrics({"loss": 1.0}, step=3)
    assert wandb.inits == [] and lg.version == "mine"
    assert run.calls == [("log", ({"loss": 1.0, "trainer/global_step": 3},), {})]


def test_wandb_media_tables_and_watch(wandb):
    run = FakeRun()
    lg = WandbLogger(experiment=run, prefix="x/")
    lg.log_table("t", columns=["a"], data=[[1]], step=2)
    (name, (data,), _) = run.calls[-1]
    assert name == "log" and data["trainer/global_step"] == 2
    assert data["x/t"].args == (["a"], [[1]], None)
    lg.log_text("txt", columns=["c"], data=[["hi"]])
    assert run.calls[-1][1][0]["x/txt"].args == (["c"], [["hi"]], None)
    lg.log_image("imgs", ["i1", "i2"], step=5, caption=["one", "two"])
    assert run.calls[-1][1][0]["x/imgs"] == [
        wandb.Image("i1", caption="one"),
        wandb.Image("i2", caption="two"),
    ]
    lg.log_audio("snd", ["a1"], sample_rate=[16000])
    assert run.calls[-1][1][0]["x/snd"] == [wandb.Audio("a1", sample_rate=16000)]
    lg.log_video("vid", ["v1"], fps=[4])
    assert run.calls[-1][1][0]["x/vid"] == [wandb.Video("v1", fps=4)]
    with pytest.raises(TypeError, match="images"):
        lg.log_image("bad", "not-a-list")
    with pytest.raises(ValueError, match="caption"):
        lg.log_image("bad", ["i1", "i2"], caption=["only one"])
    model = nn.Linear(2, 2)
    lg.watch(model, log="all", log_freq=10)
    assert run.calls[-1] == ("watch", (model,), {"log": "all", "log_freq": 10, "log_graph": True})
    assert lg.use_artifact("ent/proj/model:v1", artifact_type="model") == "use_artifact-result"
    assert run.calls[-1] == ("use_artifact", ("ent/proj/model:v1",), {"type": "model"})


def test_wandb_user_methods_do_nothing_off_rank_zero(wandb, monkeypatch):
    run = FakeRun()
    lg = WandbLogger(experiment=run)
    monkeypatch.setenv("RANK", "1")
    lg.log_image("imgs", ["i"])
    lg.watch(nn.Linear(1, 1))
    assert lg.use_artifact("a") is None
    assert WandbLogger.download_artifact("a") is None
    assert run.calls == []


def test_wandb_download_artifact(wandb, tmp_path):
    out = WandbLogger.download_artifact("ent/p/m:v0", save_dir=tmp_path, artifact_type="model")
    assert out == f"api[model]:ent/p/m:v0:{tmp_path}"
    run = FakeRun()
    run.use_artifact = lambda ref: wandb.Downloadable(ref, "run")
    wandb.run = run
    assert WandbLogger.download_artifact("ent/p/m:v0") == "run:ent/p/m:v0:None"
    assert WandbLogger.download_artifact("ent/p/m:v0", use_artifact=False).startswith("api")


def test_wandb_checkpoint_artifact(wandb, tmp_path):
    ckpt = tmp_path / "epoch=0.ckpt"
    ckpt.write_bytes(b"x")
    run = FakeRun(id="r9")
    lg = WandbLogger(experiment=run, log_model=True, add_file_policy="immutable")
    lg.after_save_checkpoint(ckpt)
    name, (art,), kwargs = run.calls[-1]
    assert name == "log_artifact" and kwargs == {"aliases": ["latest"]}
    assert (art.name, art.type) == ("model-r9", "model")
    assert art.files == [(str(ckpt), "model.ckpt", "immutable")]
    lg2 = WandbLogger(experiment=run, log_model=True, checkpoint_name="best-model")
    lg2.after_save_checkpoint(ckpt)
    assert run.calls[-1][1][0].name == "best-model"
    WandbLogger(experiment=run).after_save_checkpoint(ckpt)
    assert len([c for c in run.calls if c[0] == "log_artifact"]) == 2
    with pytest.raises(ValidationFailed, match="offline"):
        WandbLogger(offline=True, log_model=True)
    with pytest.raises(ValidationFailed, match="add_file_policy"):
        WandbLogger(add_file_policy="sometimes")


# ------------------------------------------------------------------ profilers


def test_profiler_local_rank(monkeypatch):
    p = Profiler()
    assert p.local_rank == 0
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert p.local_rank == 3
    p.setup("fit", local_rank=1)
    assert p.local_rank == 1


def test_pytorch_profiler_module_names_shapes_and_table_kwargs():
    prof = PyTorchProfiler(
        group_by_input_shapes=True,
        record_shapes=True,
        row_limit=50,
        table_kwargs={"max_name_column_width": 80},
    )
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    prof.setup("fit")
    with prof.profile("run_training_batch"):
        model(torch.randn(2, 4))
        model(torch.randn(3, 4))
    prof.teardown("fit")
    text = prof.summary()
    assert "[module]torch.nn.modules.linear.Linear" in text
    assert "[module]torch.nn.modules.container.Sequential" in text
    assert "Input Shapes" in text and "[3, 4]" in text and "[2, 4]" in text
    # the module hooks are removed at teardown
    assert prof._hooks == [] and prof._module_ranges == {}


def test_pytorch_profiler_without_module_names():
    prof = PyTorchProfiler(record_module_names=False)
    prof.setup("fit")
    nn.Linear(2, 2)(torch.randn(1, 2))
    prof.teardown("fit")
    text = prof.summary()
    assert "[module]" not in text and "Input Shapes" not in text


def test_pytorch_profiler_table_kwargs_are_checked():
    with pytest.raises(ValidationFailed, match="already an argument"):
        PyTorchProfiler(table_kwargs={"sort_by": "cpu_time"})
    with pytest.raises(ValidationFailed, match="not one of"):
        PyTorchProfiler(table_kwargs={"colour": True})


def test_pytorch_profiler_emit_nvtx(monkeypatch):
    events = []

    class Ctx:
        def __init__(self, label, **kwargs):
            self.label, self.kwargs = label, kwargs

        def __enter__(self):
            events.append(("enter", self.label, self.kwargs))
            return self

        def __exit__(self, *exc):
            events.append(("exit", self.label))

    class EmitNvtx(Ctx):
        def __init__(self, enabled=True, record_shapes=False):
            super().__init__("nvtx", enabled=enabled, record_shapes=record_shapes)

    monkeypatch.setattr(torch.autograd.profiler, "emit_nvtx", EmitNvtx)
    monkeypatch.setattr(torch.cuda.profiler, "profile", lambda: Ctx("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    prof = PyTorchProfiler(emit_nvtx=True, record_shapes=True, export_to_chrome=True)
    prof.setup("fit", log_dir="unused")
    with prof.profile("backward"):
        pass
    prof.teardown("fit")
    assert events == [
        ("enter", "nvtx", {"enabled": True, "record_shapes": True}),
        ("enter", "cuda", {}),
        ("exit", "nvtx"),
        ("exit", "cuda"),
    ]
    assert prof.summary() == ""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValidationFailed, match="CUDA"):
        PyTorchProfiler(emit_nvtx=True).setup("fit")


@pytest.fixture
def xla(monkeypatch):
    log = []

    class Trace:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            log.append(("enter", type(self).__name__, self.name))

        def __exit__(self, *exc):
            log.append(("exit", self.name))

    class StepTrace(Trace):
        def __init__(self, name, step_num):
            super().__init__(name)
            log.append(("step", name, step_num))

    def start_server(port):
        log.append(("server", port))
        return f"server:{port}"

    xp = _module(
        monkeypatch,
        "torch_xla.debug.profiler",
        start_server=start_server,
        Trace=Trace,
        StepTrace=StepTrace,
    )
    debug = _module(monkeypatch, "torch_xla.debug", profiler=xp)
    _module(monkeypatch, "torch_xla", debug=debug)
    return log


def test_xla_profiler_traces_actions(xla):
    prof = XLAProfiler(port=9100)
    prof.setup("fit", local_rank=0)
    with prof.profile("train_dataloader_next"):  # not traced, and does not start the server
        pass
    assert xla == []
    for _ in range(2):
        with prof.profile("run_training_batch"), prof.profile("backward"):
            pass
    with prof.profile("Model.validation_step"):
        pass
    with prof.profile("Model.validation_step"):
        pass
    assert prof.server == "server:9100"
    assert xla[:5] == [
        ("server", 9100),
        ("enter", "Trace", "run_training_batch"),
        ("enter", "Trace", "backward"),
        ("exit", "backward"),
        ("exit", "run_training_batch"),
    ]
    assert xla.count(("server", 9100)) == 1
    steps = [e for e in xla if e[0] == "step"]
    assert steps == [("step", "Model.validation_step", 1), ("step", "Model.validation_step", 2)]
    assert ("enter", "StepTrace", "Model.validation_step") in xla
    assert prof.describe() == ""


def test_xla_profiler_server_only_on_local_rank_zero(xla):
    prof = XLAProfiler()
    prof.setup("fit", local_rank=1)
    with prof.profile("run_training_batch"):
        pass
    assert prof.server is None and ("server", 9012) not in xla
    assert ("enter", "Trace", "run_training_batch") in xla


def test_xla_profiler_string_and_missing_torch_xla(xla, monkeypatch):
    assert isinstance(resolve_profiler("xla"), XLAProfiler)
    monkeypatch.setitem(sys.modules, "torch_xla.debug.profiler", None)
    with pytest.raises(ValidationFailed, match="torch_xla"):
        XLAProfiler()
