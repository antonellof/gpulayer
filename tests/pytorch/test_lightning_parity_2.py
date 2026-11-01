"""Lightning parity, part 2 (ADR-0024 amendment): several dataloaders, distributed predict and
test, the CLI, plugins, and the DeepSpeed and TPU entry points. CPU; the distributed cases start
real processes through a launched job."""

import json
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


def ds(n, offset=0.0):
    x = torch.arange(n, dtype=torch.float32).reshape(-1, 1).repeat(1, 4) + offset
    return TensorDataset(x, x.sum(1, keepdim=True))


class Lin(glt.Module):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):  # several training loaders
            batch = batch["a"]
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))
        self.log("seen", float(dataloader_idx), add_dataloader_idx=False, reduce_fx="max")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        self.log("test_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("num_sanity_val_steps", 0)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


# ------------------------------------------------------------- several loaders


def test_combined_loader_modes():
    a, b = [1, 2, 3], ["x", "y"]
    assert list(glt.CombinedLoader([a, b], "max_size_cycle")) == [[1, "x"], [2, "y"], [3, "x"]]
    assert list(glt.CombinedLoader([a, b], "min_size")) == [[1, "x"], [2, "y"]]
    assert list(glt.CombinedLoader({"a": a, "b": b}, "max_size")) == [
        {"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": None},
    ]  # fmt: skip
    assert list(glt.CombinedLoader([a, b], "sequential")) == [
        (1, 0, 0), (2, 1, 0), (3, 2, 0), ("x", 0, 1), ("y", 1, 1),
    ]  # fmt: skip
    assert len(glt.CombinedLoader([a, b], "min_size")) == 2


def test_fit_on_a_dict_of_loaders_and_two_val_loaders(tmp_path):
    train = {"a": DataLoader(ds(64), batch_size=8), "b": DataLoader(ds(16), batch_size=8)}
    vals = [DataLoader(ds(32), batch_size=16), DataLoader(ds(32, 1.0), batch_size=16)]
    t = trainer(tmp_path, max_epochs=1)
    t.fit(Lin(), train, vals)
    m = t.history[0]
    assert t.global_step == 8  # max_size_cycle: the longest loader sets the length
    assert {"val_loss/dataloader_idx_0", "val_loss/dataloader_idx_1"} <= set(m)
    assert m["seen"] == 1.0 and "seen/dataloader_idx_1" not in m  # add_dataloader_idx=False
    assert t.num_val_batches == [2, 2]


def test_predict_with_two_loaders(tmp_path):
    out = trainer(tmp_path).predict(
        Lin(), [DataLoader(ds(8), batch_size=4), DataLoader(ds(4), batch_size=4)]
    )
    assert len(out) == 2 and len(out[0]) == 2 and len(out[1]) == 1


def run_script(tmp_path, body, timeout=300):
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(body))
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    p = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=timeout)  # fmt: skip
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p.stdout


@pytest.mark.distributed
def test_distributed_predict_and_test_come_back_from_the_job(tmp_path):
    out = run_script(
        tmp_path,
        """
        import torch
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
            def test_step(self, batch, i):
                x, y = batch
                self.log("test_loss", nn.functional.mse_loss(self.lin(x), y))
            def predict_step(self, batch, i):
                return self.lin(batch[0])
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        x = torch.randn(40, 4)
        data = TensorDataset(x, x.sum(1, keepdim=True))
        t = glt.Trainer(max_epochs=1, accelerator="cpu", devices=2, log_every_n_steps=0,
                        enable_checkpointing=False)
        t.fit(Lin(), DataLoader(data, batch_size=5))
        res = t.test(dataloaders=DataLoader(data, batch_size=5), verbose=False)
        preds = t.predict(dataloaders=DataLoader(data, batch_size=5))
        if glt.is_launcher_process():
            seen = sorted(i for b in t.predict_batch_indices for i in b)
            print("RESULT", round(res["test_loss"], 6) >= 0, len(preds),
                  sum(p.shape[0] for p in preds), seen == list(range(40)))
        """,
    )
    # 40 samples, 2 ranks x 4 batches of 5; every sample predicted once, in rank order
    assert "RESULT True 8 40 True" in out, out


# ------------------------------------------------------------------- CLI


CLI_SCRIPT = """
import sys, torch
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt
sys.path.insert(0, {here!r})
from test_lightning_parity_2 import Lin, ds

class Data(glt.DataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
    def train_dataloader(self):
        return DataLoader(ds(32), batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(ds(32), batch_size=self.batch_size)

cli = glt.LightningCLI(Lin, Data, trainer_defaults={{"accelerator": "cpu", "log_every_n_steps": 0}})
print("LR", cli.model.hparams.lr, "BS", cli.datamodule.batch_size, "EPOCHS", cli.trainer.max_epochs,
      "STEPS", cli.trainer.global_step, "CB", [type(c).__name__ for c in cli.trainer.callbacks])
"""


def run_cli(tmp_path, *args):
    script = tmp_path / "cli.py"
    script.write_text(CLI_SCRIPT.format(here=os.path.dirname(__file__)))
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    p = subprocess.run([sys.executable, str(script), *args], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)  # fmt: skip
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


def test_cli_arguments_config_file_and_class_path(tmp_path):
    out = run_cli(
        tmp_path,
        "fit",
        "--trainer.max_epochs",
        "2",
        "--model.lr",
        "0.05",
        "--data.batch_size",
        "16",
    )
    assert "LR 0.05 BS 16 EPOCHS 2 STEPS 4" in out
    cfg = tmp_path / "exp.json"
    cfg.write_text(json.dumps({
        "trainer": {"max_epochs": 3, "callbacks": [
            {"class_path": "gpulayer.pytorch.LearningRateMonitor", "init_args": {}}]},
        "model": {"lr": 0.2},
    }))  # fmt: skip
    out = run_cli(tmp_path, "fit", "--config", str(cfg), "--trainer.max_epochs", "1")
    assert "LR 0.2 BS 8 EPOCHS 1 STEPS 4" in out and "LearningRateMonitor" in out
    printed = run_cli(tmp_path, "fit", "--config", str(cfg), "--print_config")
    assert "max_epochs" in printed and "class_path" in printed


# ---------------------------------------------------------------- plugins


def test_plugins_precision_sync_bn_and_checkpoint_io(tmp_path):
    t = glt.Trainer(plugins=[glt.HalfPrecision("bf16-true"), glt.TorchSyncBatchNorm()])
    assert t.precision == "bf16-true" and t.sync_batchnorm
    with pytest.raises(ValidationFailed, match="unsupported plugin"):
        glt.Trainer(plugins=[object()])

    saved = []

    class Recording(glt.TorchCheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            saved.append(os.path.basename(path))
            super().save_checkpoint(checkpoint, path)

    t = trainer(tmp_path, max_epochs=2, plugins=[Recording()])
    t.fit(Lin(), DataLoader(ds(16), batch_size=8))
    assert saved == ["last.ckpt", "last.ckpt"]

    io = glt.AsyncCheckpointIO()
    t = trainer(tmp_path / "a", max_epochs=2, plugins=[io])
    model = Lin()
    t.fit(model, DataLoader(ds(16), batch_size=8))
    state = torch.load(t.checkpoint_path, weights_only=False)  # fit waited for the writes
    assert torch.equal(state["state_dict"]["lin.weight"], model.lin.weight.detach())


# --------------------------------------------------------- DeepSpeed and TPU


def test_deepspeed_config_and_missing_package(tmp_path):
    s = glt.DeepSpeedStrategy(stage=3, offload_optimizer=True)
    cfg = s.deepspeed_config("bf16-mixed", 4, 1.0, 32)
    assert cfg["zero_optimization"]["stage"] == 3 and cfg["bf16"] == {"enabled": True}
    assert cfg["gradient_accumulation_steps"] == 4 and cfg["gradient_clipping"] == 1.0
    assert glt.Trainer(strategy="deepspeed_stage_2_offload").strategy.offload_optimizer
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        with pytest.raises(ValidationFailed, match="pip install deepspeed"):
            trainer(tmp_path, strategy="deepspeed").fit(Lin(), DataLoader(ds(8), batch_size=4))


def test_tpu_needs_torch_xla():
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        with pytest.raises(ValidationFailed, match="torch_xla"):
            glt.resolve_accelerator("tpu")
        with pytest.raises(ValidationFailed, match="torch_xla"):
            glt.resolve_accelerator("xla")
