"""Trainer loop semantics that must match Lightning: gradient accumulation with loaders that
have no length, scheduler stepping, the validation schedule, EarlyStopping timing, and how
epoch metrics are reduced (within a process and across two DDP processes). CPU only."""

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402


def xy(n=64):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 4, generator=g)
    return TensorDataset(x, x.sum(1, keepdim=True))


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


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("num_sanity_val_steps", 0)
    kw.setdefault("enable_checkpointing", False)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


class Unsized(IterableDataset):
    """10 batches of 4, and no __len__."""

    def __iter__(self):
        ds = xy(40)
        for i in range(0, 40, 4):
            yield ds.tensors[0][i : i + 4], ds.tensors[1][i : i + 4]


def test_unsized_loader_applies_the_last_partial_accumulation(tmp_path):
    seen = []

    class Grads(glt.Callback):
        def on_train_epoch_end(self, trainer, model):
            seen.append(all(p.grad is None for p in model.parameters()))

    t = trainer(tmp_path, max_epochs=2, accumulate_grad_batches=4, callbacks=[Grads()])
    t.fit(Lin(), DataLoader(Unsized(), batch_size=None))
    # Steps after batches 4, 8 and the last (10) of each epoch: nothing carried over.
    assert t.global_step == 6
    assert seen == [True, True]


class Recorder(glt.Callback):
    def __init__(self):
        self.lrs = []

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer._stepped:
            self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])


class Scheduled(Lin):
    def __init__(self, frequency):
        super().__init__(lr=1.0)
        self.frequency = frequency

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=1.0)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": self.frequency},
        }


def test_step_scheduler_frequency_counts_optimizer_steps(tmp_path):
    rec = Recorder()
    t = trainer(tmp_path, max_steps=4, callbacks=[rec])
    t.fit(Scheduled(frequency=2), DataLoader(xy(), batch_size=8))
    # lr seen after each step: the scheduler steps after steps 2 and 4, not 1 and 3.
    assert rec.lrs == [1.0, 0.5, 0.5, 0.25]


def test_manual_optimization_does_not_step_schedulers(tmp_path):
    class Manual(Scheduled):
        def __init__(self):
            super().__init__(frequency=1)
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            opt = self.optimizers()
            x, y = batch
            loss = nn.functional.mse_loss(self.lin(x), y)
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            return loss

    t = trainer(tmp_path, max_epochs=2)
    t.fit(Manual(), DataLoader(xy(), batch_size=8))
    # Lightning leaves the scheduler to the model with manual optimization.
    assert t.optimizers[0].param_groups[0]["lr"] == 1.0


def test_step_interval_reduce_on_plateau_is_stepped(tmp_path):
    class Plateau(Lin):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.1)
            self.sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": self.sch,
                    "interval": "step",
                    "monitor": "train_loss",
                },
            }

    model = Plateau()
    t = trainer(tmp_path, max_steps=5)
    t.fit(model, DataLoader(xy(), batch_size=8))
    assert model.sch.last_epoch == 5  # stepped once per optimizer step


class CountValidations(glt.Callback):
    def __init__(self):
        self.at = []

    def on_validation_end(self, trainer, model):
        self.at.append((trainer.current_epoch, trainer.global_step))


def test_integer_val_check_interval_counts_across_epochs(tmp_path):
    count = CountValidations()
    t = trainer(
        tmp_path,
        max_epochs=6,
        val_check_interval=10,
        check_val_every_n_epoch=None,
        callbacks=[count],
    )
    t.fit(Lin(), DataLoader(xy(32), batch_size=8), DataLoader(xy(16), batch_size=8))
    # 4 batches per epoch, 24 in all: after batches 10 and 20, and at the end of the run.
    assert count.at == [(2, 10), (4, 20), (5, 24)]


def test_mid_epoch_validation_respects_check_val_every_n_epoch(tmp_path):
    count = CountValidations()
    t = trainer(
        tmp_path,
        max_epochs=4,
        val_check_interval=0.5,
        check_val_every_n_epoch=2,
        callbacks=[count],
    )
    t.fit(Lin(), DataLoader(xy(32), batch_size=8), DataLoader(xy(16), batch_size=8))
    assert [e for e, _ in count.at] == [1, 1, 3, 3]


def test_early_stopping_counts_validations_not_epochs(tmp_path):
    stop = glt.EarlyStopping(monitor="val_loss", patience=2)
    t = trainer(tmp_path, max_epochs=20, check_val_every_n_epoch=2, callbacks=[stop])
    t.fit(Lin(lr=0.0), DataLoader(xy(), batch_size=8), DataLoader(xy(16), batch_size=8))
    # Validations at epochs 1, 3, 5: the best, then two without improvement.
    assert stop.stopped_epoch == 5 and stop.wait_count == 2


def test_early_stopping_checks_mid_epoch_validations(tmp_path):
    stop = glt.EarlyStopping(monitor="val_loss", patience=2)
    t = trainer(tmp_path, max_epochs=20, val_check_interval=0.25, callbacks=[stop])
    t.fit(Lin(lr=0.0), DataLoader(xy(), batch_size=8), DataLoader(xy(16), batch_size=8))
    assert stop.stopped_epoch == 0  # three validations within the first epoch


def test_epoch_mean_is_weighted_by_inferred_batch_size(tmp_path):
    class Logs(Lin):
        def validation_step(self, batch, batch_idx):
            x, _ = batch
            self.log("first", 1.0 if batch_idx == 0 else 0.0)

    x = torch.randn(100, 4)
    val = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=64)  # 64 + 36
    t = trainer(tmp_path, max_epochs=1)
    t.fit(Logs(), DataLoader(xy(), batch_size=8), val)
    assert t.callback_metrics["first"] == pytest.approx(0.64)  # not 0.5


def test_ddp_epoch_values_follow_reduce_fx_and_tolerate_rank_only_names(tmp_path):
    """Two DDP processes (multi-process, CPU, gloo): a sum adds up across ranks, max takes
    the largest, and a name only one rank logs does not hang the reduction."""
    script = tmp_path / "ddp_reduce.py"
    script.write_text(
        textwrap.dedent("""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import gpulayer.pytorch as glt

        class M(glt.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 1)
            def training_step(self, batch, i):
                x, y = batch
                return nn.functional.mse_loss(self.lin(x), y)
            def validation_step(self, batch, i):
                x, _ = batch
                self.log("n", float(len(x)), reduce_fx="sum")
                self.log("rank", float(self.global_rank), reduce_fx="max")
                if self.global_rank == 1:
                    self.log("only_rank1", 7.0)
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        x = torch.randn(64, 4)
        ds = TensorDataset(x, x.sum(1, keepdim=True))
        t = glt.Trainer(max_epochs=1, accelerator="cpu", devices=2, log_every_n_steps=0,
                        num_sanity_val_steps=0, enable_checkpointing=False)
        t.fit(M(), DataLoader(ds, batch_size=8), DataLoader(ds, batch_size=8))
        m = t.callback_metrics
        print("RESULT", m["n"], m["rank"], m["only_rank1"])
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
    assert "RESULT 64.0 1.0 7.0" in p.stdout, p.stdout
