"""Trainer callbacks: hooks, EarlyStopping, ModelCheckpoint, and stopping all DDP ranks."""

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402


class Lin(glt.Module):
    def __init__(self, lr=0.1):
        super().__init__()
        torch.manual_seed(0)
        self.lin, self.lr = nn.Linear(4, 1), lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.lin(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self.lin(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 4, generator=g)
    ds = TensorDataset(x, x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]))
    return DataLoader(ds, batch_size=32), DataLoader(ds, batch_size=64)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


def test_early_stopping_stops_on_a_plateau(tmp_path):
    stop = glt.EarlyStopping(monitor="val_loss", patience=2)
    t = trainer(tmp_path, max_epochs=20, callbacks=[stop])
    t.fit(Lin(lr=0.0), *data())  # lr 0: val_loss never improves after the first epoch
    assert len(t.history) == 3 and stop.stopped_epoch == 2 and t.should_stop


def test_early_stopping_lets_improving_training_run(tmp_path):
    stop = glt.EarlyStopping(monitor="val_loss", patience=2)
    t = trainer(tmp_path, max_epochs=5, callbacks=[stop])
    t.fit(Lin(), *data())
    assert len(t.history) == 5 and stop.stopped_epoch is None


def test_model_checkpoint_keeps_the_best_epoch(tmp_path):
    best = glt.ModelCheckpoint(monitor="val_loss")
    t = trainer(tmp_path, max_epochs=4, callbacks=[best])
    model = Lin()
    t.fit(model, *data())
    assert best.best_model_path and best.best_model_path.name == "best.ckpt"
    saved = torch.load(best.best_model_path, weights_only=False)
    losses = [h["val_loss"] for h in t.history]
    assert saved["epoch"] == losses.index(min(losses))
    restored = Lin.load_from_checkpoint(best.best_model_path)
    assert isinstance(restored, Lin)


def test_hooks_run_in_order(tmp_path):
    calls = []

    class Record(glt.Callback):
        def on_fit_start(self, trainer, model):
            calls.append("fit_start")

        def on_train_epoch_start(self, trainer, model):
            calls.append("epoch_start")

        def on_train_batch_end(self, trainer, model, loss, batch_idx):
            if batch_idx == 0:
                calls.append(f"batch_end:{type(loss).__name__}")

        def on_train_epoch_end(self, trainer, model):
            calls.append(f"epoch_end:{'val_loss' in trainer.callback_metrics}")

        def on_fit_end(self, trainer, model):
            calls.append("fit_end")

    trainer(tmp_path, max_epochs=2, callbacks=[Record()]).fit(Lin(), *data())
    epoch = ["epoch_start", "batch_end:Tensor", "epoch_end:True"]
    assert calls == ["fit_start", *epoch, *epoch, "fit_end"]


def test_monitoring_a_metric_that_was_not_logged_explains(tmp_path):
    t = trainer(tmp_path, max_epochs=2, callbacks=[glt.EarlyStopping(monitor="val_acc")])
    with pytest.raises(ValidationFailed, match="val_acc.*not logged.*val_loss"):
        t.fit(Lin(), *data())


@pytest.mark.distributed
def test_one_rank_asking_to_stop_stops_every_rank(tmp_path):
    script = tmp_path / "ddp_stop.py"
    script.write_text(
        textwrap.dedent("""
        import os, torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import gpulayer.pytorch as glt

        class Lin(glt.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 1)
            def training_step(self, batch, i):
                x, y = batch
                return nn.functional.mse_loss(self.lin(x), y)
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        class StopOnRankOne(glt.Callback):
            def on_train_epoch_end(self, trainer, model):
                if os.environ.get("RANK") == "1":
                    trainer.should_stop = True

        x = torch.randn(64, 4)
        loader = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=8)
        t = glt.Trainer(max_epochs=5, accelerator="cpu", devices=2, log_every_n_steps=0,
                        callbacks=[StopOnRankOne()])
        t.fit(Lin(), loader)
        print("epochs trained:", len(t.history))
    """)
    )
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    p = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "epochs trained: 1" in p.stdout  # both ranks left after epoch 0, no hang


class Scripted(Lin):
    """Logs a fixed val_loss per epoch, to test which checkpoints are kept."""

    VALUES = [5.0, 1.0, 4.0, 2.0, 3.0]

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self.VALUES[self.trainer.current_epoch])


def test_model_checkpoint_save_top_k_keeps_the_k_best(tmp_path):
    best = glt.ModelCheckpoint(monitor="val_loss", save_top_k=2)
    t = trainer(tmp_path, max_epochs=5, callbacks=[best])
    t.fit(Scripted(), *data())
    kept = sorted(p.name for p in best.best_k_models)
    assert kept == ["best-epoch=1.ckpt", "best-epoch=3.ckpt"]  # val_loss 1.0 and 2.0
    on_disk = sorted(p.name for p in best.best_model_path.parent.glob("best-*.ckpt"))
    assert on_disk == kept  # worse ones were deleted
    assert best.best_model_path.name == "best-epoch=1.ckpt"
    with pytest.raises(ValidationFailed, match="save_top_k"):
        glt.ModelCheckpoint(save_top_k=0)


def test_learning_rate_monitor_records_each_epochs_rate(tmp_path):
    class WithScheduler(Lin):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.1)
            return opt, torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    t = trainer(tmp_path, max_epochs=3, callbacks=[glt.LearningRateMonitor()])
    t.fit(WithScheduler(), *data())
    assert [h["lr"] for h in t.history] == pytest.approx([0.1, 0.05, 0.025])
    assert t.callback_metrics["lr"] == pytest.approx(0.025)


def test_model_checkpoint_top_k_finds_the_best_after_a_remote_run(tmp_path):
    """After a run on workers, the kept files come back with the outputs; their scores are read
    from the history stored in each checkpoint."""
    from types import SimpleNamespace

    ckpts = tmp_path / "node0" / "checkpoints"
    ckpts.mkdir(parents=True)
    for epoch, loss in ((1, 0.4), (3, 0.2)):
        history = [{"val_loss": 9.0}] * epoch + [{"val_loss": loss}]
        torch.save({"history": history}, ckpts / f"best-epoch={epoch}.ckpt")
    best = glt.ModelCheckpoint(monitor="val_loss", save_top_k=2)
    remote = SimpleNamespace(launch_result=SimpleNamespace(output_dir=tmp_path))
    best.on_fit_end(remote, None)
    assert best.best_model_path.name == "best-epoch=3.ckpt"
    assert sorted(best.best_k_models.values()) == [0.2, 0.4]
