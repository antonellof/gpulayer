"""``self.log`` with metric objects: torchmetrics, and a stand-in so CI without it covers the
protocol. CPU, one process."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402


class MeanMetric(nn.Module):
    """The part of ``torchmetrics.Metric`` the Trainer uses."""

    def __init__(self) -> None:
        super().__init__()
        self.total, self.count, self.computes, self.resets = 0.0, 0, 0, 0
        self._forward_cache = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.update(value)
        self._forward_cache = value.mean()
        return self._forward_cache

    def update(self, value: torch.Tensor) -> None:
        self.total += float(value.sum())
        self.count += value.numel()

    def compute(self) -> torch.Tensor:
        self.computes += 1
        return torch.tensor(self.total / self.count)

    def reset(self) -> None:
        self.resets += 1
        self.total, self.count = 0.0, 0


def loader(n: int = 8, batch_size: int = 2) -> DataLoader:
    x = torch.arange(n, dtype=torch.float32).reshape(-1, 1)
    return DataLoader(TensorDataset(x, x), batch_size=batch_size)


class Model(glt.Module):
    def __init__(self, metric: nn.Module | None = None, call: bool = True) -> None:
        super().__init__()
        self.net = nn.Linear(1, 1)
        self.metric = metric or MeanMetric()
        self.call = call

    def training_step(self, batch, batch_idx):
        return F.mse_loss(self.net(batch[0]), batch[1])

    def validation_step(self, batch, batch_idx):
        if self.call:
            self.metric(batch[0])
        else:
            self.metric.update(batch[0])
        self.log("metric", self.metric)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4)


def trainer(**kw) -> glt.Trainer:
    return glt.Trainer(accelerator="cpu", enable_checkpointing=False, num_sanity_val_steps=0, **kw)


@pytest.mark.parametrize("call", [True, False], ids=["forward", "update"])
def test_metric_computed_over_all_batches_and_reset(call):
    model = Model(call=call)
    t = trainer(max_epochs=2)
    t.fit(model, loader(), loader())
    assert t.callback_metrics["metric"] == pytest.approx(3.5)  # mean of 0..7, not the last batch
    assert model.metric.computes == 2
    assert model.metric.resets == 2
    assert model.metric.count == 0


def test_training_metric_survives_validation_inside_the_epoch():
    class Both(Model):
        def __init__(self) -> None:
            super().__init__()
            self.train_metric = MeanMetric()

        def training_step(self, batch, batch_idx):
            self.train_metric(batch[0])
            self.log("train_metric", self.train_metric, on_epoch=True)
            return super().training_step(batch, batch_idx)

    model = Both()
    t = trainer(max_epochs=1, val_check_interval=2)
    t.fit(model, loader(), loader())
    assert model.train_metric.computes == 1
    assert t.callback_metrics["train_metric"] == pytest.approx(3.5)


def test_logged_at_epoch_end():
    class AtEnd(Model):
        def validation_step(self, batch, batch_idx):
            self.metric.update(batch[0])

        def on_validation_epoch_end(self):
            self.log("metric", self.metric)

    model = AtEnd()
    t = trainer(max_epochs=1)
    t.fit(model, loader(), loader())
    assert t.callback_metrics["metric"] == pytest.approx(3.5)
    assert model.metric.resets == 1


def test_one_metric_per_dataloader():
    class Two(glt.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(1, 1)
            self.metrics = nn.ModuleList([MeanMetric(), MeanMetric()])

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            self.metrics[dataloader_idx](batch[0])
            self.log("m", self.metrics[dataloader_idx])

    t = trainer()
    result = t.test(Two(), [loader(4), loader(8)])
    assert result["m/dataloader_idx_0"] == pytest.approx(1.5)
    assert result["m/dataloader_idx_1"] == pytest.approx(3.5)


def test_other_modules_are_rejected():
    class Bad(Model):
        def validation_step(self, batch, batch_idx):
            self.log("net", self.net)

    with pytest.raises(ValidationFailed, match="torchmetrics.Metric"):
        trainer(max_epochs=1).fit(Bad(), loader(), loader())


def test_metric_computing_several_values_is_rejected():
    class Vector(MeanMetric):
        def compute(self):
            return torch.ones(3)

    with pytest.raises(ValidationFailed, match="computes one value"):
        trainer(max_epochs=1).fit(Model(metric=Vector()), loader(), loader())


def test_torchmetrics_accuracy():
    torchmetrics = pytest.importorskip("torchmetrics")

    class Accuracy(glt.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(1, 2)
            self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        def training_step(self, batch, batch_idx):
            return F.cross_entropy(self.net(batch[0]), batch[1].long().squeeze(1) % 2)

        def validation_step(self, batch, batch_idx):
            self.acc(self.net(batch[0]), batch[1].long().squeeze(1) % 2)
            self.log("acc", self.acc, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    model = Accuracy()
    t = trainer(max_epochs=2)
    t.fit(model, loader(), loader())
    assert 0.0 <= t.callback_metrics["acc"] <= 1.0
    assert model.acc._update_count == 0  # reset


def test_metric_syncs_across_two_processes(tmp_path):
    """Each rank sees different samples; compute() must combine them (two gloo processes)."""
    pytest.importorskip("torchmetrics")
    from test_trainer_api import run_script

    out = run_script(
        tmp_path,
        """
        import torch, torchmetrics
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import gpulayer.pytorch as glt

        class Model(glt.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(1, 1)
                self.mean = torchmetrics.MeanMetric()
            def training_step(self, batch, i):
                return self.net(batch[0]).mean()
            def validation_step(self, batch, i):
                self.mean(batch[0])
                self.log("mean", self.mean)
            def on_validation_end(self):
                self.print("value", round(self.trainer.callback_metrics["mean"], 4))
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=1e-4)

        x = torch.arange(8.0).reshape(-1, 1)
        loader = DataLoader(TensorDataset(x), batch_size=2)
        t = glt.Trainer(max_epochs=1, accelerator="cpu", devices=2, log_every_n_steps=0,
                        enable_checkpointing=False, num_sanity_val_steps=0)
        t.fit(Model(), loader, loader)
        """,
    )
    assert "value 3.5" in out, out
