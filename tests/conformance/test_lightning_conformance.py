"""The same Lightning-style code on PyTorch Lightning and on the GPUlayer Trainer must behave
the same. Each scenario runs on both, with only the framework swapped, and the results are
compared. CPU, one process; skipped without ``lightning``.

``scripts/compare_api.py`` compares names; this compares behaviour. A difference GPUlayer
keeps on purpose is normalised in the scenario, with a comment saying why. A difference not
fixed yet is a strict ``xfail``, so a fix fails the suite until the mark is removed.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

import lightning as L  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lightning.pytorch import callbacks as lightning_callbacks  # noqa: E402
from lightning.pytorch.loggers import CSVLogger as LightningCSVLogger  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402

FRAMEWORKS = {
    "lightning": SimpleNamespace(
        name="lightning",
        Module=L.LightningModule,
        DataModule=L.LightningDataModule,
        Trainer=L.Trainer,
        Callback=lightning_callbacks.Callback,
        ModelCheckpoint=lightning_callbacks.ModelCheckpoint,
        EarlyStopping=lightning_callbacks.EarlyStopping,
        CSVLogger=LightningCSVLogger,
        seed=L.seed_everything,
    ),
    "gpulayer": SimpleNamespace(
        name="gpulayer",
        Module=glt.Module,
        DataModule=glt.DataModule,
        Trainer=glt.Trainer,
        Callback=glt.Callback,
        ModelCheckpoint=glt.ModelCheckpoint,
        EarlyStopping=glt.EarlyStopping,
        CSVLogger=glt.CSVLogger,
        seed=glt.seed_everything,
    ),
}

QUIET = {
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "enable_checkpointing": False,
}


def data(n: int = 40, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
    x = torch.randn(n, 8, generator=torch.Generator().manual_seed(0))
    return DataLoader(
        TensorDataset(x, (x.sum(1) > 0).long()), batch_size=batch_size, shuffle=shuffle
    )


def model_class(fw: Any) -> type:
    class Model(fw.Module):
        def __init__(self, lr: float = 0.05) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.net = nn.Linear(8, 2)

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = F.cross_entropy(self(x), y)
            self.log("tl", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            self.log("vl", F.cross_entropy(self(x), y))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

    return Model


def rounded(tensor: torch.Tensor, digits: int = 4) -> list[float]:
    return [round(v, digits) for v in tensor.flatten().tolist()]


def outcome(scenario: Any, fw: Any, *args: Any) -> Any:
    """What a scenario returned, or the fact that it raised (compared by "raised or not":
    the frameworks' exception types differ on purpose, ``ValidationFailed`` here)."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            return scenario(fw, *args)
        except Exception as e:  # noqa: BLE001
            return (
                f"raised: {type(e).__name__}: {e}"
                if os.environ.get("CONFORMANCE_DEBUG")
                else "raised"
            )


def same(scenario: Any, *args: Any) -> None:
    logging.disable(logging.WARNING)
    try:
        lightning = outcome(scenario, FRAMEWORKS["lightning"], *args)
        gpulayer = outcome(scenario, FRAMEWORKS["gpulayer"], *args)
    finally:
        logging.disable(logging.NOTSET)
    assert gpulayer == lightning


# ------------------------------------------------------------------ training results


@pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
def test_same_seed_same_weights(shuffle):
    def scenario(fw):
        fw.seed(7)
        model = model_class(fw)()
        fw.Trainer(max_epochs=2, num_sanity_val_steps=0, **QUIET).fit(model, data(shuffle=shuffle))
        return rounded(model.net.weight)

    same(scenario)


@pytest.mark.xfail(strict=True, reason="sanity check draws different random numbers: to trace")
def test_same_seed_same_weights_with_sanity_check():
    def scenario(fw):
        fw.seed(7)
        model = model_class(fw)()
        fw.Trainer(max_epochs=2, **QUIET).fit(model, data(shuffle=True), data())
        return rounded(model.net.weight)

    same(scenario)


def test_accumulation_clipping_and_step_scheduler():
    def scenario(fw):
        rates: list[float] = []

        class Model(model_class(fw)):
            def configure_optimizers(self):
                opt = torch.optim.SGD(self.parameters(), lr=1.0)
                sched = torch.optim.lr_scheduler.StepLR(opt, 1, 0.5)
                return [opt], [{"scheduler": sched, "interval": "step"}]

            def on_train_batch_end(self, *args):
                rates.append(self.optimizers().param_groups[0]["lr"])

        fw.seed(3)
        model = Model()
        t = fw.Trainer(max_epochs=1, accumulate_grad_batches=3, gradient_clip_val=0.1, **QUIET)
        t.fit(model, data())
        return rates, t.global_step, rounded(model.net.bias)

    same(scenario)


@pytest.mark.parametrize(
    "limits",
    [{"max_steps": 15}, {"max_steps": 4}, {"max_steps": 15, "max_epochs": 1}, {"max_epochs": 2}],
    ids=str,
)
def test_training_limits(limits):
    def scenario(fw):
        t = fw.Trainer(**limits, **QUIET)
        t.fit(model_class(fw)(), data())
        return t.global_step

    same(scenario)


def test_several_training_loaders():
    def scenario(fw):
        class Model(model_class(fw)):
            def training_step(self, batch, batch_idx):
                return F.cross_entropy(self(batch["a"][0]), batch["a"][1])

        t = fw.Trainer(max_epochs=1, **QUIET)
        t.fit(Model(), {"a": data(), "b": data(n=12)})
        return t.global_step

    same(scenario)


def test_datamodule_from_datasets():
    def scenario(fw):
        ds = TensorDataset(torch.randn(8, 8), torch.zeros(8).long())
        dm = fw.DataModule.from_datasets(train_dataset=ds, val_dataset=ds, batch_size=2)
        t = fw.Trainer(max_epochs=1, **QUIET)
        t.fit(model_class(fw)(), datamodule=dm)
        return t.global_step

    same(scenario)


# ------------------------------------------------------------------ the training loop


@pytest.mark.xfail(strict=True, reason="item 6: on_before_zero_grad fires after the step")
def test_hook_order():
    def scenario(fw):
        trace: list[str] = []
        names = """setup on_fit_start on_sanity_check_start on_sanity_check_end on_train_start
        on_train_end on_train_epoch_start on_train_epoch_end on_train_batch_start
        on_train_batch_end on_validation_start on_validation_end on_validation_epoch_start
        on_validation_epoch_end on_validation_batch_start on_validation_batch_end
        on_before_backward on_after_backward on_before_optimizer_step on_before_zero_grad
        on_fit_end teardown""".split()
        base = model_class(fw)

        def callback_hook(name):
            return lambda self, *a, **k: trace.append(f"cb:{name}")

        def module_hook(name):
            inherited = getattr(base, name, None)

            def hook(self, *a, **k):
                trace.append(f"m:{name}")
                return inherited(self, *a, **k) if inherited else None

            return hook

        callback = type("Trace", (fw.Callback,), {n: callback_hook(n) for n in names})
        module = type(
            "Traced",
            (base,),
            {n: module_hook(n) for n in names if not n.startswith("on_sanity")},
        )
        t = fw.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            num_sanity_val_steps=1,
            callbacks=[callback()],
            **QUIET,
        )
        t.fit(module(), data(), data())
        return trace

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 6: dataloader_iter form of training_step")
def test_dataloader_iter_step():
    def scenario(fw):
        class Model(model_class(fw)):
            def training_step(self, dataloader_iter):
                batch, _, _ = next(dataloader_iter)
                return F.cross_entropy(self(batch[0]), batch[1])

        t = fw.Trainer(max_epochs=1, limit_train_batches=2, **QUIET)
        t.fit(Model(), data())
        return t.global_step

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 6: None from training_step")
def test_training_step_returning_none():
    def scenario(fw):
        seen: list[Any] = []

        class Model(model_class(fw)):
            def training_step(self, batch, batch_idx):
                if batch_idx == 1:
                    return None
                return {"loss": super().training_step(batch, batch_idx), "extra": 1}

            def on_train_batch_end(self, outputs, batch, batch_idx):
                seen.append(sorted(outputs or {}))

        t = fw.Trainer(max_epochs=1, limit_train_batches=3, **QUIET)
        t.fit(Model(), data())
        return t.global_step, seen

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 6: a second fit trains again")
def test_second_fit_with_the_same_limits():
    def scenario(fw):
        model = model_class(fw)()
        t = fw.Trainer(max_epochs=1, limit_train_batches=2, **QUIET)
        t.fit(model, data())
        t.fit(model, data())
        return t.global_step

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 6: extra validation at the end of the epoch")
@pytest.mark.parametrize("sized", [True, False], ids=["sized", "iterable"])
def test_val_check_interval(sized):
    class Stream(IterableDataset):
        def __iter__(self):
            for _ in range(10):
                yield torch.zeros(8), torch.tensor(0)

    def scenario(fw):
        steps: list[int] = []

        class Model(model_class(fw)):
            def on_validation_end(self):
                steps.append(self.global_step)

        train = data(n=10, batch_size=1) if sized else DataLoader(Stream(), batch_size=1)
        t = fw.Trainer(
            max_steps=10,
            val_check_interval=3,
            num_sanity_val_steps=0,
            limit_val_batches=1,
            **QUIET,
        )
        t.fit(Model(), train, data())
        return steps

    same(scenario)


# ------------------------------------------------------------------ logging


def test_torchmetrics_value():
    torchmetrics = pytest.importorskip("torchmetrics")

    def scenario(fw):
        class Model(model_class(fw)):
            def __init__(self):
                super().__init__()
                self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

            def validation_step(self, batch, batch_idx):
                self.acc(self(batch[0]), batch[1])
                self.log("acc", self.acc, on_epoch=True)

        fw.seed(1)
        model = Model()
        t = fw.Trainer(max_epochs=2, num_sanity_val_steps=0, **QUIET)
        t.fit(model, data(), data())
        return round(float(t.callback_metrics["acc"]), 5), model.acc._update_count

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 3: _step and _epoch names, logger=False")
def test_metric_destinations():
    def scenario(fw):
        class Model(model_class(fw)):
            def training_step(self, batch, batch_idx):
                loss = F.cross_entropy(self(batch[0]), batch[1])
                self.log("both", loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log("bar", 1.0, prog_bar=True, logger=False)
                self.log_dict({"d1": 2.0, "d2": 3.0})
                return loss

        t = fw.Trainer(max_epochs=1, limit_train_batches=3, log_every_n_steps=1, **QUIET)
        t.fit(Model(), data())
        return (
            sorted(t.callback_metrics),
            sorted(t.logged_metrics),
            sorted(t.progress_bar_metrics),
        )

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 3: monitoring the _epoch name")
def test_early_stopping_on_epoch_name():
    def scenario(fw):
        class Model(model_class(fw)):
            def validation_step(self, batch, batch_idx):
                loss = F.cross_entropy(self(batch[0]), batch[1])
                self.log("vl", loss, on_step=True, on_epoch=True)

        t = fw.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            callbacks=[fw.EarlyStopping("vl_epoch")],
            **QUIET,
        )
        t.fit(Model(), data(), data())
        return sorted(t.callback_metrics)

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 3: steps in the logger are 0-based")
def test_csv_logger_rows(tmp_path):
    def scenario(fw):
        quiet = {k: v for k, v in QUIET.items() if k != "logger"}
        logger = fw.CSVLogger(tmp_path / fw.name, name="n")
        t = fw.Trainer(
            max_epochs=2,
            limit_train_batches=5,
            limit_val_batches=1,
            log_every_n_steps=2,
            logger=logger,
            **quiet,
        )
        t.fit(model_class(fw)(), data(), data())
        lines = next((tmp_path / fw.name).rglob("metrics.csv")).read_text().splitlines()
        header = lines[0].split(",")
        rows = [dict(zip(header, line.split(","), strict=False)) for line in lines[1:]]
        return sorted(header), [(r["epoch"], r["step"], bool(r["tl"]), bool(r["vl"])) for r in rows]

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 6: self.log misuse is accepted")
@pytest.mark.parametrize("misuse", ["vector", "in_configure_optimizers", "repeated_key"])
def test_log_misuse_is_an_error(misuse):
    def scenario(fw):
        class Model(model_class(fw)):
            def configure_optimizers(self):
                if misuse == "in_configure_optimizers":
                    self.log("x", 1.0)
                return super().configure_optimizers()

            def training_step(self, batch, batch_idx):
                if misuse == "vector":
                    self.log("v", torch.ones(3))
                if misuse == "repeated_key":
                    self.log("k", 1.0)
                    self.log("k", 2.0, on_epoch=True, reduce_fx="max")
                return super().training_step(batch, batch_idx)

        fw.Trainer(max_epochs=1, limit_train_batches=1, **QUIET).fit(Model(), data())
        return "accepted"

    same(scenario)


def test_missing_monitor_is_an_error():
    def scenario(fw):
        t = fw.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            callbacks=[fw.EarlyStopping("nope")],
            **QUIET,
        )
        t.fit(model_class(fw)(), data(), data())
        return "accepted"

    same(scenario)


# ------------------------------------------------------------------ checkpoints


def test_checkpoint_loads_with_weights_only(tmp_path):
    def scenario(fw):
        t = fw.Trainer(max_epochs=1, limit_train_batches=2, **QUIET)
        t.fit(model_class(fw)(), data())
        path = tmp_path / f"{fw.name}.ckpt"
        t.save_checkpoint(path)
        torch.load(path, weights_only=True)
        return "loaded"

    same(scenario)


def test_ckpt_path_names(tmp_path):
    def scenario(fw):
        folder = tmp_path / fw.name
        quiet = {**QUIET, "enable_checkpointing": True}
        best = fw.ModelCheckpoint(dirpath=folder, monitor="vl", save_last=True)
        limits = {"limit_train_batches": 2, "limit_val_batches": 1}
        t = fw.Trainer(max_epochs=2, callbacks=[best], **limits, **quiet)
        model = model_class(fw)()
        t.fit(model, data(), data())
        t.validate(model, data(), ckpt_path="best", verbose=False)
        t.validate(model, data(), ckpt_path="last", verbose=False)
        again = fw.ModelCheckpoint(dirpath=folder, save_last=True)
        t2 = fw.Trainer(max_epochs=3, callbacks=[again], **limits, **quiet)
        t2.fit(model_class(fw)(), data(), data(), ckpt_path="last")
        return t2.global_step

    same(scenario)


@pytest.mark.xfail(strict=True, reason="item 5: current_epoch after fit")
def test_current_epoch_after_fit():
    def scenario(fw):
        t = fw.Trainer(max_epochs=2, limit_train_batches=2, **QUIET)
        t.fit(model_class(fw)(), data())
        return t.current_epoch, t.global_step

    same(scenario)


def _stop_at(fw: Any, step: int, path: Path) -> Any:
    class Saver(fw.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step == step:
                trainer.save_checkpoint(path)

    return Saver()


@pytest.mark.xfail(strict=True, reason="item 5: Lightning's loop state is not read")
def test_lightning_checkpoint_resumes_mid_epoch_on_gpulayer(tmp_path):
    fw = FRAMEWORKS["lightning"]
    path = tmp_path / "mid.ckpt"
    t = fw.Trainer(max_epochs=1, callbacks=[_stop_at(fw, 7, path)], **QUIET)
    t.fit(model_class(fw)(), data())

    g = FRAMEWORKS["gpulayer"]
    resumed = g.Trainer(max_epochs=2, **QUIET)
    resumed.fit(model_class(g)(), data(), ckpt_path=path)
    assert resumed.global_step == 20  # the 3 batches left in epoch 0, then 10


@pytest.mark.xfail(strict=True, reason="item 5: no way back to Lightning yet")
def test_gpulayer_checkpoint_loads_on_lightning(tmp_path):
    g = FRAMEWORKS["gpulayer"]
    path = tmp_path / "out.ckpt"
    t = g.Trainer(max_epochs=1, limit_train_batches=2, **QUIET)
    t.fit(model_class(g)(), data())
    t.save_checkpoint(path)

    fw = FRAMEWORKS["lightning"]
    restored = model_class(fw).load_from_checkpoint(path)
    assert restored.hparams.lr == 0.05


@pytest.mark.xfail(strict=True, reason="item 4: remote paths are written locally")
def test_object_store_paths(tmp_path, monkeypatch):
    fsspec = pytest.importorskip("fsspec")
    monkeypatch.chdir(tmp_path)

    def scenario(fw):
        url = f"memory://conformance/{fw.name}"
        t = fw.Trainer(max_epochs=1, limit_train_batches=1, default_root_dir=url, **QUIET)
        t.fit(model_class(fw)(), data())
        t.save_checkpoint(f"{url}/a.ckpt")
        return fsspec.filesystem("memory").exists(f"/conformance/{fw.name}/a.ckpt")

    same(scenario)


# ------------------------------------------------------------------ the public surface


@pytest.mark.xfail(strict=True, reason="item 7: trainer.strategy is a configuration object")
@pytest.mark.parametrize(
    "expression",
    [
        "t.strategy.root_device.type",
        "t.strategy.global_rank",
        "t.strategy.world_size",
        "t.strategy.is_global_zero",
        "t.strategy.reduce(torch.tensor(2.0)).item()",
        "t.strategy.broadcast(3)",
        "t.strategy.barrier()",
        "t.strategy.all_gather(torch.tensor(1.0)).shape",
        "t.val_check_batch",
        "t.lightning_module is t.model",
        "t.max_steps",
        "t.limit_val_batches",
        "t.num_val_batches",
        "type(t.val_dataloaders).__name__",
        "t.fit_loop.max_epochs",
        "t.fit_loop.epoch_loop.global_step",
        "t.fit_loop.epoch_progress.current.completed",
        "dict(m.hparams_initial)",
        "m.prepare_data_per_node",
    ],
)
def test_public_attribute(expression):
    def scenario(fw):
        t = fw.Trainer(max_epochs=1, limit_train_batches=2, **QUIET)
        m = model_class(fw)()
        t.fit(m, data(), data())
        return eval(expression)  # noqa: S307

    same(scenario)
