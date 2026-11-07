"""Checkpoints and logs on a remote filesystem: fsspec's in-memory store stands in for S3, GCS
and Azure. Simulated: no real object store is involved. CPU, one process."""

from __future__ import annotations

import builtins

import pytest

torch = pytest.importorskip("torch")
fsspec = pytest.importorskip("fsspec")

from test_trainer_api import Lin, loaders  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch.storage import RemotePath, as_path, is_remote  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A clean store; the working directory shows anything written locally by mistake."""
    monkeypatch.chdir(tmp_path)
    memory = fsspec.filesystem("memory")
    memory.store.clear()
    memory.pseudo_dirs[:] = [""]
    yield memory
    assert not list(tmp_path.glob("memory*")), "a URL became a local folder"


def trainer(root="memory://bucket/run", **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("num_sanity_val_steps", 0)
    kw.setdefault("log_every_n_steps", 0)
    return glt.Trainer(default_root_dir=root, **kw)


def names(memory, folder):
    return sorted(p.rsplit("/", 1)[-1] for p in memory.find(folder))


def test_path_operations(store):
    root = as_path("memory://bucket/a")
    assert is_remote(root) and is_remote("s3://b/k") and not is_remote("relative/path")
    assert not is_remote("file:///tmp/x") and not isinstance(as_path("file:///tmp/x"), RemotePath)
    f = root / "sub" / "x.ckpt"
    assert (f.name, f.suffix, str(f.parent)) == ("x.ckpt", ".ckpt", "memory://bucket/a/sub")
    assert not f.exists()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("hello")
    assert f.is_file() and f.exists() and f.read_text() == "hello"
    assert (root / "sub").is_dir() and [p.name for p in (root / "sub").iterdir()] == ["x.ckpt"]
    assert root.rglob("*.ckpt") == [f] and (root / "sub").glob("x*") == [f]
    assert f.stat().st_mtime >= 0
    f.unlink()
    f.unlink(missing_ok=True)
    assert not f.exists()


def test_fit_writes_to_the_store_and_resumes(store):
    t = trainer(max_epochs=1)
    t.fit(Lin(), *loaders())
    assert isinstance(t.checkpoint_path, RemotePath)
    assert str(t.checkpoint_path) == "memory://bucket/run/checkpoints/last.ckpt"
    assert "metrics.json" in names(store, "/bucket/run")

    resumed = trainer(max_epochs=2)
    resumed.fit(Lin(), *loaders(), ckpt_path="last")
    assert resumed.global_step == 2 * t.global_step


def test_load_from_checkpoint_and_explicit_url(store):
    t = trainer(max_epochs=1, enable_checkpointing=False)
    model = Lin()
    t.fit(model, *loaders())
    t.save_checkpoint("memory://bucket/elsewhere/a.ckpt")
    restored = Lin.load_from_checkpoint("memory://bucket/elsewhere/a.ckpt")
    assert torch.equal(restored.net[0].weight.data, model.net[0].weight.data.cpu())


def test_model_checkpoint_keeps_top_k_in_the_store(store):
    best = glt.ModelCheckpoint(
        dirpath="memory://bucket/best", monitor="val_loss", save_top_k=2, filename="{epoch}"
    )
    t = trainer("gpulayer_runs_local", max_epochs=4, callbacks=[best])
    t.fit(Lin(), *loaders())
    assert len(names(store, "/bucket/best")) == 2
    assert isinstance(best.best_model_path, RemotePath) and best.best_model_path.exists()
    t.validate(Lin(), loaders()[1], ckpt_path="best")


def test_csv_logger(store):
    logger = glt.CSVLogger("memory://bucket/logs", name="n")
    t = trainer("local_run", max_epochs=1, logger=logger, log_every_n_steps=1)
    t.fit(Lin(), *loaders())
    logger.save()
    assert "metrics.csv" in names(store, "/bucket/logs/n/version_0")
    assert glt.CSVLogger("memory://bucket/logs", name="n").version == 1


def test_interrupted_upload_leaves_nothing(store, monkeypatch):
    t = trainer(max_epochs=1, enable_checkpointing=False)
    t.fit(Lin(), *loaders())
    calls = []
    monkeypatch.setattr(torch, "save", lambda *a, **k: calls.append(1) or 1 / 0)
    with pytest.raises(ZeroDivisionError):
        t.save_checkpoint("memory://bucket/broken.ckpt")
    assert calls and not as_path("memory://bucket/broken.ckpt").exists()


def test_unknown_scheme_is_an_error(store):
    with pytest.raises(ValidationFailed, match="no filesystem for this scheme"):
        trainer("nosuchstore://bucket/run")
    with pytest.raises(ValidationFailed, match="no filesystem"):
        glt.ModelCheckpoint(dirpath="nosuchstore://bucket/run")


def test_missing_store_package_is_an_error():
    pytest.importorskip("fsspec")
    try:
        import s3fs  # noqa: F401
    except ImportError:
        with pytest.raises(ValidationFailed, match="needs a package"):
            glt.Trainer(default_root_dir="s3://bucket/run")


def test_missing_fsspec_is_an_error(monkeypatch):
    real = builtins.__import__

    def without_fsspec(name, *args, **kw):
        if name.startswith("fsspec"):
            raise ImportError(name)
        return real(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", without_fsspec)
    with pytest.raises(ValidationFailed, match="gpulayer\\[remote\\]"):
        glt.Trainer(default_root_dir="s3://bucket/run")


def test_credentials_in_a_url_are_refused():
    with pytest.raises(ValidationFailed, match="printed in logs") as e:
        glt.Trainer(default_root_dir="s3://KEY:SECRET@bucket/run")
    assert "SECRET" not in str(e.value)


def test_tensorboard_refuses_a_url():
    with pytest.raises(ValidationFailed, match="local event files"):
        glt.TensorBoardLogger("memory://bucket/tb")


def test_launched_run_reads_weights_from_the_store(store, monkeypatch, tmp_path):
    """Workers wrote to the store, so the checkpoint is not among the job's outputs."""
    t = trainer(max_epochs=1)
    model = Lin()
    t.fit(model, *loaders())
    launched = trainer(max_epochs=1)
    fresh = Lin()
    fresh.net[0].weight.data.fill_(0.0)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    launched._load_outputs(outputs, fresh)
    assert torch.equal(fresh.net[0].weight.data, model.net[0].weight.data.cpu())
    assert str(launched.checkpoint_path).startswith("memory://")
