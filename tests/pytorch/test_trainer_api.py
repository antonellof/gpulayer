"""Lightning-compatible features of the GPUlayer Trainer (ADR-0024): hooks, manual
optimization, DataModule, schedulers, precision, callbacks, loggers, profilers and tuner.

CPU, single process unless marked; the DDP cases start two real processes (gloo)."""

import math
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


def dataset(n=256):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 4, generator=g)
    return TensorDataset(x, x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]))


def loaders(bs=32):
    ds = dataset()
    return DataLoader(ds, batch_size=bs), DataLoader(ds, batch_size=64)


class Lin(glt.Module):
    def __init__(self, lr: float = 0.05, hidden: int = 8):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)

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

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("num_sanity_val_steps", 0)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


def run_script(tmp_path, body, timeout=240):
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(body))
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    p = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p.stdout


# ------------------------------------------------------------------ Module


def test_hyperparameters_are_saved_and_restore_the_model(tmp_path):
    model = Lin(lr=0.01, hidden=16)
    assert model.hparams == {"lr": 0.01, "hidden": 16} and model.hparams.lr == 0.01
    t = trainer(tmp_path, max_epochs=1)
    t.fit(model, *loaders())
    ckpt = torch.load(t.checkpoint_path, weights_only=False)
    assert ckpt["hyper_parameters"] == {"lr": 0.01, "hidden": 16}
    restored = Lin.load_from_checkpoint(t.checkpoint_path)  # no arguments needed
    assert restored.net[0].out_features == 16
    assert torch.equal(restored.net[0].weight, model.net[0].weight)


def test_hooks_follow_the_reference_order(tmp_path):
    calls = []

    class Hooked(Lin):
        def setup(self, stage):
            calls.append(f"setup:{stage}")

        def on_fit_start(self):
            calls.append("fit_start")

        def on_train_start(self):
            calls.append("train_start")

        def on_train_epoch_start(self):
            calls.append("epoch_start")

        def on_train_batch_start(self, batch, batch_idx):
            if batch_idx == 0:
                calls.append("batch_start")

        def on_before_backward(self, loss):
            if not any(c == "before_backward" for c in calls):
                calls.append("before_backward")

        def on_after_backward(self):
            if not any(c == "after_backward" for c in calls):
                calls.append("after_backward")

        def on_before_optimizer_step(self, optimizer):
            if not any(c == "before_opt" for c in calls):
                calls.append("before_opt")

        def on_before_zero_grad(self, optimizer):
            if not any(c == "zero_grad" for c in calls):
                calls.append("zero_grad")

        def on_validation_epoch_start(self):
            calls.append("val_start")

        def on_validation_epoch_end(self):
            calls.append("val_end")
            self.log("from_val_end", 7.0)

        def on_train_epoch_end(self):
            calls.append("epoch_end")

        def on_train_end(self):
            calls.append("train_end")

        def on_fit_end(self):
            calls.append("fit_end")

        def teardown(self, stage):
            calls.append(f"teardown:{stage}")

    t = trainer(tmp_path, max_epochs=1)
    t.fit(Hooked(), *loaders())
    assert calls == [
        "setup:fit", "fit_start", "train_start", "epoch_start", "batch_start",
        "before_backward", "after_backward", "before_opt", "zero_grad",
        "val_start", "val_end", "epoch_end", "train_end", "fit_end", "teardown:fit",
    ]  # fmt: skip
    assert t.callback_metrics["from_val_end"] == 7.0  # logged in an epoch-end hook


def test_callback_signature_and_outputs(tmp_path):
    seen = {}

    class CB(glt.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            seen.setdefault("outputs", outputs)
            seen.setdefault("batch", batch)

    trainer(tmp_path, max_epochs=1, limit_train_batches=2, callbacks=[CB()]).fit(
        Lin(), loaders()[0]
    )
    assert set(seen["outputs"]) == {"loss"} and isinstance(seen["batch"], list)


def test_returning_none_skips_and_minus_one_ends_the_epoch(tmp_path):
    class Skips(Lin):
        def training_step(self, batch, batch_idx):
            return None if batch_idx % 2 else super().training_step(batch, batch_idx)

        def on_train_batch_start(self, batch, batch_idx):
            return -1 if batch_idx == 4 else None

    t = trainer(tmp_path, max_epochs=1)
    t.fit(Skips(), loaders()[0])
    assert t.global_step == 2  # batches 0 and 2 stepped, 1 and 3 skipped, 4 ended the epoch


def test_manual_optimization_with_two_optimizers(tmp_path):
    class TwoOpt(glt.Module):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False
            self.a, self.b = nn.Linear(4, 1), nn.Linear(4, 1)

        def training_step(self, batch, batch_idx):
            opt_a, opt_b = self.optimizers()
            x, y = batch
            for opt, layer in ((opt_a, self.a), (opt_b, self.b)):
                loss = nn.functional.mse_loss(layer(x), y)
                opt.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(opt, gradient_clip_val=1.0)
                opt.step()
            self.log("loss", loss)

        def configure_optimizers(self):
            return [
                torch.optim.SGD(self.a.parameters(), 0.05),
                torch.optim.SGD(self.b.parameters(), 0.05),
            ]

    t = trainer(tmp_path, max_epochs=3)
    t.fit(TwoOpt(), loaders()[0])
    assert t.global_step == 3 * 8 * 2  # every opt.step() counts, as in Lightning
    assert t.history[-1]["loss"] < t.history[0]["loss"]


def test_clipping_a_manual_optimizer_unscales_the_real_one(tmp_path):
    """Regression (found on 2x T4 over NCCL): clip_gradients on the wrapper returned by
    self.optimizers() must unscale the wrapped optimizer, or GradScaler.step unscales twice."""
    from gpulayer.pytorch.trainer import _ManualOptimizer

    unscaled = []

    class Scaler:
        def unscale_(self, opt):
            unscaled.append(opt)

    model = Lin()
    t = trainer(tmp_path)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    t.runtime = type("RT", (), {"scaler": Scaler()})()
    t._clip(_ManualOptimizer(opt, t), 1.0, "norm")
    t._clip(_ManualOptimizer(opt, t), 1.0, "norm")
    assert unscaled == [opt]  # the real optimizer, once per step


def test_several_optimizers_need_manual_optimization(tmp_path):
    class Two(Lin):
        def configure_optimizers(self):
            return [
                torch.optim.SGD(self.parameters(), 0.1),
                torch.optim.SGD(self.parameters(), 0.1),
            ]

    with pytest.raises(ValidationFailed, match="manual optimization"):
        trainer(tmp_path).fit(Two(), loaders()[0])


# ------------------------------------------------------------------ data


class Data(glt.DataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.calls = []

    def prepare_data(self):
        self.calls.append("prepare")

    def setup(self, stage):
        self.calls.append(f"setup:{stage}")
        self.ds = dataset()

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.ds, batch_size=64)


def test_datamodule_fit_test_and_results(tmp_path):
    dm = Data()
    t = trainer(tmp_path, max_epochs=2)
    t.fit(Lin(), datamodule=dm)
    assert dm.calls[:2] == ["prepare", "setup:fit"] and "val_loss" in t.callback_metrics
    out = t.test(datamodule=dm, ckpt_path="last", verbose=False)
    assert out["test_loss"] == out[0]["test_loss"]  # a dict, and Lightning's list form
    assert "setup:test" in dm.calls


def test_model_dataloader_hooks(tmp_path):
    class Own(Lin):
        def train_dataloader(self):
            return loaders()[0]

    t = trainer(tmp_path, max_epochs=1)
    t.fit(Own())
    assert t.global_step == 8


def test_limits_overfit_and_reload(tmp_path):
    built = []

    def make():
        built.append(1)
        return loaders()[0]

    t = trainer(
        tmp_path, max_epochs=4, limit_train_batches=0.25, reload_dataloaders_every_n_epochs=2
    )
    t.fit(Lin(), make)
    assert t.global_step == 4 * 2 and len(built) == 2  # 25% of 8 batches; rebuilt at epoch 2
    t = trainer(tmp_path, max_epochs=2, overfit_batches=3)
    t.fit(Lin(), *loaders())
    assert t.global_step == 6


def test_min_epochs_hold_off_early_stopping(tmp_path):
    stop = glt.EarlyStopping(monitor="val_loss", patience=1)
    t = trainer(tmp_path, max_epochs=10, min_epochs=5, callbacks=[stop])
    t.fit(Lin(lr=0.0), *loaders())
    assert len(t.history) == 5


# ------------------------------------------------------------- optimizers


def test_scheduler_dict_and_reduce_on_plateau(tmp_path):
    class Plateau(Lin):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.1)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=0.5)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", 1.0)  # never improves

    t = trainer(tmp_path, max_epochs=4)
    t.fit(Plateau(), *loaders())
    assert t.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**3)


def test_step_scheduler_with_frequency(tmp_path):
    class Stepped(Lin):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=1.0)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
            return [opt], [{"scheduler": sched, "interval": "step", "frequency": 4}]

    t = trainer(tmp_path, max_epochs=1)
    t.fit(Stepped(), loaders()[0])
    assert t.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.25)  # 8 steps / 4


def test_log_reduce_fx_and_batch_size(tmp_path):
    class Reduce(Lin):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            self.log("idx_max", float(batch_idx), reduce_fx="max")
            self.log("idx_sum", 1.0, reduce_fx="sum")
            self.log("weighted", float(batch_idx), batch_size=batch_idx + 1)
            return loss

    t = trainer(tmp_path, max_epochs=1)
    t.fit(Reduce(), loaders()[0])
    m = t.history[0]
    assert m["idx_max"] == 7 and m["idx_sum"] == 8
    assert m["weighted"] == pytest.approx(sum(i * (i + 1) for i in range(8)) / sum(range(1, 9)))


# --------------------------------------------------------------- precision


def test_true_precision_and_aliases(tmp_path):
    t = trainer(tmp_path, max_epochs=1, precision="bf16-true")
    model = Lin()
    t.fit(model, loaders()[0])
    assert t.runtime.param_dtype == torch.bfloat16 and math.isfinite(t.history[0]["train_loss"])
    assert glt.Trainer(precision=32).precision == "32-true"
    assert glt.Trainer(precision="bf16").precision == "bf16-mixed"
    with pytest.raises(ValidationFailed, match="precision"):
        glt.Trainer(precision="8-bit")


def test_inference_mode_for_test_and_no_grad_inside_fit(tmp_path):
    seen = {}

    class Probe(Lin):
        def validation_step(self, batch, batch_idx):
            seen["val"] = torch.is_inference_mode_enabled()
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            seen["test"] = torch.is_inference_mode_enabled()
            return super().test_step(batch, batch_idx)

    model = Probe()
    t = trainer(tmp_path, max_epochs=1)
    t.fit(model, *loaders())
    t.test(model, loaders()[1], verbose=False)
    assert seen == {"val": False, "test": True}
    trainer(tmp_path, inference_mode=False).test(model, loaders()[1], verbose=False)
    assert seen["test"] is False


def test_strategy_names():
    assert glt.Trainer(strategy="ddp_find_unused_parameters_true").strategy.find_unused_parameters
    s = glt.DDPStrategy(static_graph=True, gradient_as_bucket_view=True, bucket_cap_mb=50)
    assert s.ddp_kwargs() == {
        "static_graph": True,
        "gradient_as_bucket_view": True,
        "bucket_cap_mb": 50,
    }
    with pytest.raises(ValidationFailed, match="spawn"):
        glt.Trainer(strategy="ddp_spawn")


@pytest.mark.distributed
def test_ddp_options_and_manual_optimization_two_processes(tmp_path):
    out = run_script(
        tmp_path,
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import gpulayer.pytorch as glt

        class BN(glt.Module):
            def __init__(self, manual):
                super().__init__()
                self.automatic_optimization = not manual
                torch.manual_seed(0)
                self.net = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
            def training_step(self, batch, i):
                x, y = batch
                loss = nn.functional.mse_loss(self.net(x), y)
                if self.automatic_optimization:
                    return loss
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
            def on_fit_end(self):
                w = self.net[0].weight.detach().flatten()
                g = self.all_gather(w)
                self.print("ranks agree:", bool(torch.allclose(g[0], g[1])))
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.05)

        torch.manual_seed(1)
        x = torch.randn(64, 4)
        loader = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=8, shuffle=True)
        for manual in (False, True):
            t = glt.Trainer(max_epochs=2, accelerator="cpu", devices=2, log_every_n_steps=0,
                            enable_checkpointing=False,
                            strategy=glt.DDPStrategy(gradient_as_bucket_view=True,
                                                     static_graph=not manual))
            t.fit(BN(manual), loader)
            print("trained", manual, t.global_step if t.launch_result is None else "launched")
        """,
    )
    # Both runs (automatic, then manual optimization) end with the same weights on each rank.
    assert "ranks agree: False" not in out and "ranks agree: True" in out, out
    assert "[rank0] trained False 8" in out and "[rank0] trained True 8" in out, out


def test_sync_batchnorm_conversion():
    from gpulayer.pytorch.trainer import _convert_sync_batchnorm

    m = _convert_sync_batchnorm(Lin())
    m.net.append(nn.BatchNorm1d(1))
    m = _convert_sync_batchnorm(m)
    assert isinstance(m.net[2], nn.SyncBatchNorm) and isinstance(m, Lin)


# ------------------------------------------------------------- checkpoints


def test_checkpoint_template_step_interval_save_last_and_mid_epoch_resume(tmp_path):
    ck = glt.ModelCheckpoint(
        filename="{epoch}-{step}-{train_loss:.3f}",
        monitor="train_loss",
        save_top_k=2,
        every_n_train_steps=3,
        save_last=True,
    )
    t = trainer(tmp_path, max_epochs=2, callbacks=[ck])
    t.fit(Lin(), loaders()[0])
    names = sorted(p.name for p in ck.best_k_models)
    assert len(names) == 2 and all(n.startswith("epoch=") and "step=" in n for n in names)
    assert ck.last_model_path.name == "last.ckpt"
    mid = next(iter(ck.best_k_models))
    state = torch.load(mid, weights_only=False)
    assert "callbacks" in state and "optimizer_states" in state
    # A checkpoint saved at step 3 of epoch 0 resumes within that epoch.
    early = glt.ModelCheckpoint(
        monitor=None, every_n_train_steps=3, filename="early", save_top_k=-1
    )
    t = trainer(tmp_path / "b", max_epochs=1, callbacks=[early])
    t.fit(Lin(), loaders()[0])
    first = min(early.best_k_models, key=early.best_k_models.get)  # score = step
    assert torch.load(first, weights_only=False)["batches_done"] == 3
    t2 = trainer(tmp_path / "c", max_epochs=1)
    t2.fit(Lin(), loaders()[0], ckpt_path=first)
    assert t2.global_step == 8  # 3 restored + 5 trained


def test_on_exception_checkpoint_and_keyboard_interrupt(tmp_path):
    class Boom(Lin):
        def training_step(self, batch, batch_idx):
            if self.global_step == 3:
                raise KeyboardInterrupt
            return super().training_step(batch, batch_idx)

    ck = glt.OnExceptionCheckpoint(dirpath=tmp_path / "exc")
    t = trainer(tmp_path, max_epochs=1, callbacks=[ck])
    t.fit(Boom(), loaders()[0])  # Ctrl+C ends fit gracefully, as in Lightning
    assert t.interrupted and ck.ckpt_path.is_file()
    assert torch.load(ck.ckpt_path, weights_only=False)["global_step"] == 3


def test_early_stopping_thresholds(tmp_path):
    stop = glt.EarlyStopping(monitor="val_loss", stopping_threshold=1e9)
    t = trainer(tmp_path, max_epochs=5, callbacks=[stop])
    t.fit(Lin(), *loaders())
    assert len(t.history) == 1 and stop.stopped_epoch == 0


# ---------------------------------------------------------------- callbacks


def test_utility_callbacks(tmp_path, capsys):
    events = []
    timer = glt.Timer(duration={"days": 1})
    tm = glt.ThroughputMonitor(batch_size_fn=lambda b: b[0].shape[0])
    ds = glt.DeviceStatsMonitor()
    acc = glt.GradientAccumulationScheduler({0: 1, 1: 4})
    lr = glt.LearningRateMonitor(logging_interval="step")
    cbs = [
        timer, tm, ds, acc, lr, glt.ModelSummary(),
        glt.LambdaCallback(on_train_epoch_end=lambda t, m: events.append(t.current_epoch)),
    ]  # fmt: skip
    t = trainer(tmp_path, max_epochs=2, log_every_n_steps=2, callbacks=cbs)
    t.fit(Lin(), loaders()[0])
    assert events == [0, 1]
    assert t.global_step == 8 + 2  # epoch 1 accumulates 4 batches per step
    assert timer.time_elapsed() > 0 and timer.time_remaining() > 0
    assert "train/samples_per_sec" in t.logged_metrics and "lr" in t.logged_metrics
    assert "trainable params" in capsys.readouterr().out


def test_spike_detection_raises_on_a_jump(tmp_path):
    class Spiky(Lin):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            return loss * (1000.0 if batch_idx == 5 else 1.0)

    with pytest.raises(glt.TrainingSpikeException):
        trainer(tmp_path, callbacks=[glt.SpikeDetection(warmup=2)]).fit(Spiky(lr=0.0), loaders()[0])


def test_ema_swa_finetuning_and_pruning(tmp_path):
    model = Lin()
    ema = glt.EMAWeightAveraging(decay=0.5)
    t = trainer(tmp_path, max_epochs=2, callbacks=[ema])
    t.fit(model, *loaders())
    saved = torch.load(t.checkpoint_path, weights_only=False)
    assert "current_model_state" in saved  # averaged weights in state_dict, training ones kept
    assert torch.equal(saved["state_dict"]["net.0.weight"], model.net[0].weight)

    swa = glt.StochasticWeightAveraging(swa_lrs=0.01, swa_epoch_start=1)
    t = trainer(tmp_path, max_epochs=3, callbacks=[swa])
    t.fit(Lin(), loaders()[0])
    assert swa.n_averaged == 2

    class WithBackbone(Lin):
        def __init__(self):
            super().__init__()
            self.backbone = self.net[0]

        def configure_optimizers(self):
            return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.05)

    m = WithBackbone()
    t = trainer(
        tmp_path, max_epochs=2, callbacks=[glt.BackboneFinetuning(unfreeze_backbone_at_epoch=1)]
    )
    t.fit(m, loaders()[0])
    assert len(t.optimizers[0].param_groups) == 2 and m.backbone.weight.requires_grad

    m = Lin(hidden=32)
    t = trainer(tmp_path, max_epochs=1, callbacks=[glt.ModelPruning(amount=0.5)])
    t.fit(m, loaders()[0])
    w = m.net[0].weight
    assert float((w == 0).float().mean()) > 0.3 and not hasattr(m.net[0], "weight_mask")
    Lin.load_from_checkpoint(t.checkpoint_path)  # plain weights in the checkpoint


def test_prediction_writer(tmp_path):
    written = []

    class Writer(glt.BasePredictionWriter):
        def write_on_batch_end(
            self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
        ):
            written.append((batch_idx, batch_indices[:2], prediction.shape[0]))

    t = trainer(tmp_path, callbacks=[Writer("batch")])
    out = t.predict(Lin(), loaders()[1], return_predictions=False)
    assert out is None and written[0] == (0, [0, 1], 64) and len(written) == 4


# ---------------------------------------------------- loggers and profilers


def test_csv_logger_and_profilers(tmp_path, capsys):
    lg = glt.CSVLogger(tmp_path / "logs", name="exp")
    t = trainer(tmp_path, max_epochs=2, log_every_n_steps=4, logger=lg, profiler="simple")
    t.fit(Lin(), *loaders())
    text = (tmp_path / "logs" / "exp" / "version_0" / "metrics.csv").read_text()
    assert "train_loss" in text and "val_loss" in text
    assert "lr: 0.05" in (tmp_path / "logs" / "exp" / "version_0" / "hparams.yaml").read_text()
    report = capsys.readouterr().out
    assert "run_training_batch" in report and "train_dataloader_next" in report
    t = trainer(tmp_path, max_epochs=1, profiler="advanced", logger=False)
    t.fit(Lin(), loaders()[0])
    assert "Profile stats for: run_training_batch" in capsys.readouterr().out
    t = trainer(
        tmp_path, max_epochs=1, limit_train_batches=3, profiler=glt.PyTorchProfiler(row_limit=5)
    )
    t.fit(Lin(), loaders()[0])
    assert "Self CPU" in capsys.readouterr().out


def test_barebones_turns_everything_off(tmp_path):
    t = trainer(tmp_path, max_epochs=1, barebones=True)
    t.fit(Lin(), loaders()[0])
    assert t.checkpoint_path is None and not (t._out_dir / "metrics.csv").exists()


# -------------------------------------------------------------------- tuner


def test_lr_find_and_scale_batch_size(tmp_path):
    model = Lin()
    t = trainer(tmp_path)
    before = model.net[0].weight.clone()
    finder = glt.Tuner(t).lr_find(model, loaders()[0], min_lr=1e-4, max_lr=1.0, num_training=40)
    assert torch.equal(model.net[0].weight, before)  # weights restored
    assert finder.suggestion() == model.hparams.lr and 1e-4 <= model.hparams.lr <= 1.0

    dm = Data(batch_size=2)
    size = glt.Tuner(t).scale_batch_size(Lin(), datamodule=dm, max_trials=4, steps_per_trial=1)
    assert size == 16 and dm.batch_size == 16  # 2, 4, 8, 16 all fit on the CPU


# --------------------------------------------------------------- utilities


def test_module_utilities(tmp_path):
    m = Lin()
    m.freeze()
    assert not any(p.requires_grad for p in m.parameters()) and not m.training
    m.unfreeze()
    scripted = m.to_torchscript(tmp_path / "m.pt", method="trace", example_inputs=torch.randn(2, 4))
    assert torch.allclose(scripted(torch.ones(1, 4)), m(torch.ones(1, 4)))
    assert m.device.type == "cpu" and m.dtype == torch.float32
    assert not hasattr(glt, "LightningModule") and glt.seed_everything(7) == 7


def test_compiled_model_trains_with_its_compile_options(tmp_path):
    m = Lin()
    t = trainer(tmp_path, max_epochs=1)
    t.fit(torch.compile(m, backend="eager"), loaders()[0])
    assert m._compiled_call_impl is not None and t.global_step == 8


def test_configure_callbacks_replaces_same_type(tmp_path):
    mine = glt.EarlyStopping(monitor="val_loss", patience=1)

    class WithCb(Lin):
        def configure_callbacks(self):
            return [mine]

    t = trainer(
        tmp_path, max_epochs=3, callbacks=[glt.EarlyStopping(monitor="val_loss", patience=9)]
    )
    t.fit(WithCb(lr=0.0), *loaders())
    assert t.early_stopping_callback is mine and len(t.early_stopping_callbacks) == 1
