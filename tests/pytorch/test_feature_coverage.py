"""Behaviour tests for the reference API features no other test exercised by name (found by
mapping every public name of the reference trainer to GPUlayer's tests). CPU, one process."""

import types
import warnings

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch import utilities as U  # noqa: E402


def ds(n=32):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    return TensorDataset(x, x.sum(1, keepdim=True))


def dl(n=32, batch=8):
    return DataLoader(ds(n), batch_size=batch)


class Lin(glt.Module):
    def __init__(self, lr: float = 0.01, batch_size: int = 8):
        super().__init__()
        self.lr, self.batch_size = lr, batch_size
        torch.manual_seed(0)
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.log("test_loss", nn.functional.mse_loss(self(x), y))

    def train_dataloader(self):
        return DataLoader(ds(64), batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("num_sanity_val_steps", 0)
    kw.setdefault("enable_checkpointing", False)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


# ---------------------------------------------------------------- callbacks


def test_base_finetuning_freezes_then_unfreezes(tmp_path):
    states = []

    class Unfreeze(glt.BaseFinetuning):
        def freeze_before_training(self, pl_module):
            self.freeze(pl_module.backbone)

        def finetune_function(self, pl_module, epoch, optimizer):
            states.append(all(p.requires_grad for p in pl_module.backbone.parameters()))
            if epoch == 1:
                self.unfreeze_and_add_param_group(pl_module.backbone, optimizer)

    class TrainableOnly(Lin):
        def configure_optimizers(self):  # the documented pattern: trainable parameters only
            return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.01)

    t = trainer(tmp_path, max_epochs=3, callbacks=[Unfreeze()])
    t.fit(TrainableOnly(), dl())
    assert states[0] is False and states[-1] is True
    assert len(t.optimizers[0].param_groups) == 2


def test_lr_and_batch_size_finders_run_when_fit_starts(tmp_path):
    # the suggestion skips the first 10 points, so the search needs more steps than that
    lr_finder = glt.LearningRateFinder(num_training_steps=40, max_lr=0.1)
    m = Lin(lr=1e-6)
    trainer(tmp_path, max_epochs=1, callbacks=[lr_finder]).fit(m)
    assert lr_finder.optimal_lr is not None and m.lr != 1e-6
    bs_finder = glt.BatchSizeFinder(init_val=2, max_trials=3, steps_per_trial=1)
    m2 = Lin(batch_size=2)
    trainer(tmp_path, max_epochs=1, callbacks=[bs_finder]).fit(m2)
    assert bs_finder.optimal_batch_size is not None and m2.batch_size >= 2


def test_tqdm_progress_bar_is_the_progress_bar():
    assert glt.TQDMProgressBar is glt.ProgressBar


def test_weight_averaging_leaves_the_average_in_the_model(tmp_path):
    m = Lin()
    t = trainer(tmp_path, max_epochs=2, callbacks=[glt.WeightAveraging()])
    t.fit(m, dl())
    trained = Lin()
    trainer(tmp_path, max_epochs=2).fit(trained, dl())
    # the averaged weights differ from the last training weights
    assert not torch.allclose(m.head.weight, trained.head.weight)


# ---------------------------------------------------------------- profilers


def test_simple_advanced_and_passthrough_profilers(tmp_path, capsys):
    for name, profiler in (("simple", "simple"), ("advanced", "advanced")):
        trainer(tmp_path, max_epochs=1, profiler=profiler).fit(Lin(), dl())
        out = capsys.readouterr().out
        assert "run_training_batch" in out or "Profile stats" in out, name
    assert isinstance(glt.Trainer(profiler="simple").profiler, glt.SimpleProfiler)
    assert isinstance(glt.Trainer(profiler="advanced").profiler, glt.AdvancedProfiler)
    p = glt.PassThroughProfiler()
    with p.profile("anything"):
        pass
    assert p.describe() in ("", None)


# ---------------------------------------------------------------- plugins


def test_precision_plugins_set_the_trainer_precision(tmp_path):
    assert glt.Trainer(plugins=[glt.Precision("bf16-mixed")]).precision == "bf16-mixed"
    assert (
        glt.Trainer(plugins=[glt.MixedPrecision("bf16-mixed", device="cpu")]).precision
        == "bf16-mixed"
    )
    assert glt.Trainer(plugins=[glt.DeepSpeedPrecision("16-mixed")]).precision == "16-mixed"
    seen = []

    class Check(Lin):
        def training_step(self, batch, batch_idx):
            seen.append(self.head.weight.dtype)
            return super().training_step(batch, batch_idx)

    trainer(tmp_path, max_epochs=1, plugins=[glt.DoublePrecision()]).fit(Check(), dl())
    assert seen and all(d == torch.float64 for d in seen)


def test_custom_checkpoint_io_and_xla_checkpoint_io(tmp_path):
    saved = []

    class Recording(glt.CheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            saved.append(path)
            torch.save(checkpoint, path)

        def load_checkpoint(self, path, map_location="cpu"):
            return torch.load(path, map_location=map_location, weights_only=False)

        def remove_checkpoint(self, path):
            pass

    t = trainer(tmp_path, max_epochs=2, enable_checkpointing=True, plugins=[Recording()])
    t.fit(Lin(), dl())
    assert len(saved) >= 2
    io = glt.XLACheckpointIO()
    io.save_checkpoint({"w": torch.ones(2)}, tmp_path / "x.ckpt")
    assert torch.equal(io.load_checkpoint(tmp_path / "x.ckpt")["w"], torch.ones(2))


# ---------------------------------------------------------------- strategies, accelerators


def test_single_device_strategy_and_strategy_base(tmp_path):
    s = glt.SingleDeviceStrategy("cpu")
    assert s.device == torch.device("cpu") and isinstance(s, glt.Strategy) and s.ddp_kwargs() == {}
    t = trainer(tmp_path, max_epochs=1, strategy=s)
    t.fit(Lin(), dl())
    assert t.global_step == 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_mps_accelerator_object_trains_on_the_apple_gpu(tmp_path):
    seen = []

    class Check(Lin):
        def training_step(self, batch, batch_idx):
            seen.append(batch[0].device.type)
            return super().training_step(batch, batch_idx)

    t = trainer(tmp_path, accelerator=glt.MPSAccelerator(), max_epochs=1)
    t.fit(Check(), dl())
    assert set(seen) == {"mps"} and glt.MPSAccelerator.is_available()


def test_mps_accelerator_reports_availability():
    assert glt.MPSAccelerator.name() == "mps"
    assert glt.MPSAccelerator.is_available() == torch.backends.mps.is_available()


# ---------------------------------------------------------------- utilities


def test_attribute_dict_and_case_insensitive_enum():
    d = U.AttributeDict(lr=0.1)
    d.batch = 4
    assert d.lr == 0.1 and d["batch"] == 4

    class Color(U.CaseInsensitiveEnum):
        RED = "red"

    assert Color.RED == "RED" and Color.from_str("Red") is Color.RED and Color.from_str("x") is None


def test_rank_zero_logging_only_on_rank_zero(monkeypatch, capsys):
    for rank, expected in (("0", True), ("1", False)):
        monkeypatch.setenv("RANK", rank)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            U.rank_zero_info("hello")
            U.rank_zero_warn("careful")
            U.rank_zero_deprecation("old")
        printed = "hello" in capsys.readouterr().out
        kinds = {x.category for x in w}
        assert printed is expected
        assert (UserWarning in kinds and DeprecationWarning in kinds) is expected


# ---------------------------------------------------------------- Trainer arguments


def test_min_steps_keeps_training_after_a_stop_request(tmp_path):
    class StopAtOnce(glt.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            trainer.should_stop = True

    t = trainer(tmp_path, max_epochs=10, min_steps=6, callbacks=[StopAtOnce()])
    t.fit(Lin(), dl())
    assert t.min_steps == 6 and t.global_step >= 6


def test_limit_val_test_and_predict_batches(tmp_path):
    counts = {"val": 0, "test": 0}

    class Count(Lin):
        def validation_step(self, batch, batch_idx):
            counts["val"] += 1

        def test_step(self, batch, batch_idx):
            counts["test"] += 1

    t = trainer(tmp_path, limit_val_batches=2, limit_test_batches=3, limit_predict_batches=1)
    t.validate(Count(), dl(64))
    t.test(Count(), dl(64))
    out = t.predict(Count(), dl(64))
    assert counts == {"val": 2, "test": 3} and len(out) == 1


def test_use_distributed_sampler_false_leaves_the_loader_alone(tmp_path):
    loader = dl()
    rt = types.SimpleNamespace(
        distributed=True, dataloader=lambda *_a, **_k: pytest.fail("sharded")
    )
    t = trainer(tmp_path, use_distributed_sampler=False)
    assert t._prepare_one(rt, loader) is loader


def test_detect_anomaly_and_suggest_integrations(tmp_path):
    seen = []

    class Check(Lin):
        def training_step(self, batch, batch_idx):
            seen.append(torch.is_anomaly_enabled())
            return super().training_step(batch, batch_idx)

    trainer(tmp_path, max_epochs=1, detect_anomaly=True, suggest_integrations=False).fit(
        Check(), dl()
    )
    assert seen and all(seen) and not torch.is_anomaly_enabled()


# ---------------------------------------------------------------- Trainer members


def test_trainer_state_members_during_and_after_runs(tmp_path):
    flags = []

    class Watch(glt.Callback):
        def on_sanity_check_start(self, trainer, pl_module):
            flags.append(("sanity", trainer.sanity_checking))

        def on_test_start(self, trainer, pl_module):
            flags.append(("test", trainer.testing))

        def on_predict_start(self, trainer, pl_module):
            flags.append(("predict", trainer.predicting))

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.is_last_batch:
                flags.append(("last", batch_idx))

    t = trainer(tmp_path, max_epochs=1, num_sanity_val_steps=1, enable_checkpointing=True,
                callbacks=[Watch(), glt.ModelCheckpoint()])  # fmt: skip
    m = Lin()
    t.fit(m, dl(32), dl(16))
    t.test(m, dl(16))
    t.predict(m, dl(16))
    assert ("sanity", True) in flags and ("test", True) in flags and ("predict", True) in flags
    assert ("last", 3) in flags
    assert t.is_global_zero and t.num_devices == 1 and t.estimated_stepping_batches == 4
    assert isinstance(t.checkpoint_callback, glt.ModelCheckpoint)
    assert t.checkpoint_callbacks == [t.checkpoint_callback]
    assert len(t.lr_scheduler_configs) == 0 and t.val_dataloaders is not None
    assert t.test_dataloaders is not None and t.predict_dataloaders is not None
    assert isinstance(t.progress_bar_metrics, dict)


# ---------------------------------------------------------------- Module


def test_module_optimizer_hooks_are_called(tmp_path):
    calls = []

    class Hooks(Lin):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.01)
            return [opt], [torch.optim.lr_scheduler.StepLR(opt, 1)]

        def configure_gradient_clipping(
            self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
        ):
            calls.append("clip")
            self.clip_gradients(optimizer, gradient_clip_val=0.5)

        def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
            calls.append("zero")
            optimizer.zero_grad(set_to_none=True)

        def lr_scheduler_step(self, scheduler, metric):
            calls.append("sched")
            scheduler.step()

        def on_train_epoch_end(self):
            calls.append(("lr_schedulers", type(self.lr_schedulers()).__name__))

    trainer(tmp_path, max_epochs=1, gradient_clip_val=1.0).fit(Hooks(), dl())
    assert "clip" in calls and "zero" in calls and "sched" in calls
    assert ("lr_schedulers", "StepLR") in calls


def test_toggle_optimizers_and_log_dict(tmp_path):
    m = Lin()
    a = torch.optim.SGD(m.backbone.parameters(), lr=0.1)
    b = torch.optim.SGD(m.head.parameters(), lr=0.1)
    m.trainer = types.SimpleNamespace(optimizers=[a, b])
    m.toggle_optimizer(a)
    assert all(p.requires_grad for p in m.backbone.parameters())
    assert not any(p.requires_grad for p in m.head.parameters())
    m.untoggle_optimizer(a)
    assert all(p.requires_grad for p in m.parameters())
    with m.toggled_optimizer(b):
        assert not any(p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.parameters())

    class Logs(Lin):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            self.log_dict({"a": loss * 2, "b": loss * 3})
            return loss

    t = trainer(tmp_path, max_epochs=1)
    t.fit(Logs(), dl())
    assert {"a", "b"} <= set(t.callback_metrics)


def test_module_properties_and_exports(tmp_path):
    m = Lin()
    m.example_input_array = torch.randn(2, 4)
    assert m.on_gpu is False and m.fabric is None
    with pytest.raises(ValidationFailed, match="torch_tensorrt"):
        m.to_tensorrt()
    pytest.importorskip("onnx")
    try:
        m.to_onnx(tmp_path / "m.onnx")
    except ImportError as e:  # recent torch exports through onnxscript
        pytest.skip(f"ONNX export needs: {e}")
    assert (tmp_path / "m.onnx").stat().st_size > 0


# ---------------------------------------------------------------- hooks


def test_evaluation_and_prediction_hooks_in_order(tmp_path):
    calls = []
    names = [
        "on_validation_start", "on_validation_epoch_start", "on_validation_batch_start",
        "on_validation_batch_end", "on_validation_epoch_end", "on_validation_end",
        "on_test_start", "on_test_epoch_start", "on_test_batch_start", "on_test_batch_end",
        "on_test_epoch_end", "on_test_end",
        "on_predict_start", "on_predict_epoch_start", "on_predict_batch_start",
        "on_predict_batch_end", "on_predict_epoch_end", "on_predict_end",
        "on_validation_model_eval", "on_validation_model_train", "on_test_model_eval",
        "on_test_model_train", "on_predict_model_eval",
    ]  # fmt: skip

    class Rec(Lin):
        pass

    for name in names:
        base = getattr(glt.Module, name)

        def hook(self, *a, _n=name, _base=base, **k):
            calls.append(_n)
            return _base(self, *a, **k)

        setattr(Rec, name, hook)
    m = Rec()
    m.train()
    t = trainer(tmp_path)
    t.validate(m, dl(16, 16))
    t.test(m, dl(16, 16))
    t.predict(m, dl(16, 16))
    for stage in ("validation", "test", "predict"):
        seq = [
            c
            for c in calls
            if f"_{stage}_" in c or c.endswith(f"_{stage}_start") or c.endswith(f"_{stage}_end")
        ]
        assert seq[0] in (f"on_{stage}_start", f"on_{stage}_model_eval"), (stage, seq)
        assert f"on_{stage}_batch_start" in seq and f"on_{stage}_end" in seq
    assert "on_validation_model_train" in calls and "on_test_model_train" in calls


def test_batch_transfer_hooks(tmp_path):
    calls = []

    class Transfer(Lin):
        def on_before_batch_transfer(self, batch, dataloader_idx):
            calls.append("before")
            return batch

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            calls.append("transfer")
            return [b.to(device) for b in batch]

        def on_after_batch_transfer(self, batch, dataloader_idx):
            calls.append("after")
            return batch

    trainer(tmp_path, max_epochs=1).fit(Transfer(), dl(16))
    assert calls[:3] == ["before", "transfer", "after"] and calls.count("transfer") == 2


def test_checkpoint_hooks_round_trip(tmp_path):
    class Extra(Lin):
        def on_save_checkpoint(self, checkpoint):
            checkpoint["note"] = "kept"

        def on_load_checkpoint(self, checkpoint):
            self.loaded_note = checkpoint.get("note")

    t = trainer(tmp_path, max_epochs=1, enable_checkpointing=True)
    t.fit(Extra(), dl())
    back = Extra.load_from_checkpoint(t.checkpoint_path)
    assert back.loaded_note == "kept"


def test_predict_dataloader_on_the_model_and_datamodule_from_datasets(tmp_path):
    class WithLoader(Lin):
        def predict_dataloader(self):
            return dl(16, 8)

    out = trainer(tmp_path).predict(WithLoader())
    assert len(out) == 2
    dm = glt.DataModule.from_datasets(train_dataset=ds(32), val_dataset=ds(16), batch_size=8)
    t = trainer(tmp_path, max_epochs=1)
    t.fit(Lin(), datamodule=dm)
    assert t.global_step == 4 and "val_loss" in t.callback_metrics
