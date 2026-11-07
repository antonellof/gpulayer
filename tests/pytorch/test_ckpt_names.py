"""``ckpt_path="last"`` and ``"best"``, in one process and through the launcher. CPU."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from test_trainer_api import Lin, loaders, trainer  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402


def test_fit_ckpt_path_last_continues_a_finished_run(tmp_path):
    first = trainer(tmp_path, max_epochs=1)
    first.fit(Lin(), *loaders())
    rerun = trainer(tmp_path, max_epochs=2)
    rerun.fit(Lin(), *loaders(), ckpt_path="last")
    assert rerun.global_step == 2 * first.global_step
    assert str(rerun.ckpt_path).endswith("last.ckpt")


def test_fit_ckpt_path_last_finds_model_checkpoint_folder(tmp_path):
    folder = tmp_path / "elsewhere"
    cb = glt.ModelCheckpoint(dirpath=folder, save_last=True)
    first = trainer(tmp_path / "a", max_epochs=1, callbacks=[cb])
    first.fit(Lin(), *loaders())
    rerun = trainer(tmp_path / "b", max_epochs=2, callbacks=[glt.ModelCheckpoint(dirpath=folder)])
    rerun.fit(Lin(), *loaders(), ckpt_path="last")
    assert rerun.global_step == 2 * first.global_step


def test_fit_ckpt_path_last_without_checkpoint_starts_from_scratch(tmp_path, capsys):
    t = trainer(tmp_path, max_epochs=1)
    t.fit(Lin(), *loaders(), ckpt_path="last")
    assert t.current_epoch == 0 or t.global_step > 0
    assert "starting from scratch" in capsys.readouterr().out


def test_fit_ckpt_path_best(tmp_path):
    cb = glt.ModelCheckpoint(monitor="val_loss")
    t = trainer(tmp_path, max_epochs=1, callbacks=[cb])
    t.fit(Lin(), *loaders())
    done = t.global_step
    t.max_epochs = 2
    t.fit(Lin(), *loaders(), ckpt_path="best")
    assert t.global_step == 2 * done
    assert Path(t.ckpt_path).name == Path(cb.best_model_path).name


def test_launched_fit_ships_the_resolved_checkpoint(tmp_path, monkeypatch):
    first = trainer(tmp_path, max_epochs=1)
    first.fit(Lin(), *loaders())
    seen = {}
    t = trainer(tmp_path, max_epochs=2)
    monkeypatch.setattr(glt.Trainer, "will_launch", property(lambda self: True))
    monkeypatch.setattr(glt.Trainer, "_launch", lambda self, model, path: seen.update(path=path))
    t.fit(Lin(), *loaders(), ckpt_path="last")
    assert Path(seen["path"]).name == "last.ckpt" and Path(seen["path"]).is_file()


def test_job_uses_the_shipped_checkpoint_for_last(tmp_path, monkeypatch):
    first = trainer(tmp_path / "a", max_epochs=1)
    first.fit(Lin(), *loaders())
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copy(first.checkpoint_path, inputs / "gpulayer-resume.ckpt")
    monkeypatch.setenv("GPULAYER_INPUT_DIR", str(inputs))
    t = trainer(tmp_path / "b", max_epochs=2)
    t.fit(Lin(), *loaders(), ckpt_path="last")
    assert t.global_step == 2 * first.global_step
