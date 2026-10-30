"""gpulayer.pytorch: runtime helpers, in-process Trainer, and the launch path (real processes)."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch.trainer import _run_name  # noqa: E402

EXAMPLE = Path(__file__).parents[2] / "examples" / "trainer_regression" / "train.py"


class Lin(glt.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 4, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=32, shuffle=True), DataLoader(ds, batch_size=64)


def test_runtime_validation(monkeypatch):
    with pytest.raises(ValidationFailed, match="precision"):
        glt.Runtime("cpu", precision="8-bit")
    with pytest.raises(ValidationFailed, match="CUDA"):
        glt.Runtime("cpu", precision="16-mixed")
    monkeypatch.setenv("GPULAYER_ACCELERATOR", "cpu")
    assert glt.resolve_accelerator("auto") == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(ValidationFailed, match="cuda"):
            glt.resolve_accelerator("cuda")


def test_dataloader_sharding(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    rt = glt.Runtime("cpu")
    train, val = loaders()
    sharded = rt.dataloader(train)
    assert isinstance(sharded.sampler, DistributedSampler)
    assert sharded.sampler.rank == 1 and sharded.sampler.shuffle
    assert len(list(sharded)) == 8  # 256 samples per rank / batch 32
    assert rt.dataloader(val, shuffle=False).sampler.shuffle is False


def test_mps_multi_process_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("GPULAYER_ACCELERATOR", "mps")
    with pytest.raises(ValidationFailed, match="MPS"):
        glt.Runtime("auto")


def test_in_process_fit_checkpoint_and_resume(tmp_path):
    glt.seed_everything(0)
    model = Lin()
    trainer = glt.Trainer(
        max_epochs=3, accelerator="cpu", default_root_dir=tmp_path, log_every_n_steps=0
    )
    assert not trainer.will_launch
    train, val = loaders()
    trainer.fit(model, train, val)
    first, last = trainer.history[0], trainer.history[-1]
    assert last["train_loss"] < first["train_loss"] and "val_loss" in last
    ckpt = tmp_path / _run_name() / "checkpoints" / "last.ckpt"
    assert trainer.checkpoint_path == ckpt and ckpt.exists()
    restored = Lin.load_from_checkpoint(ckpt)
    assert torch.equal(restored.lin.weight, model.lin.weight.cpu())

    resumed = glt.Trainer(
        max_epochs=4, accelerator="cpu", default_root_dir=tmp_path, log_every_n_steps=0
    )
    resumed.fit(Lin(), train, val, ckpt_path=ckpt)
    assert [h["epoch"] for h in resumed.history] == [0, 1, 2, 3]
    assert resumed.validate(restored, val)["val_loss"] >= 0


def test_will_launch_rules(monkeypatch):
    monkeypatch.delenv("GPULAYER_JOB_ID", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    assert glt.Trainer(devices=2).will_launch
    assert glt.Trainer(target="remote").will_launch
    assert not glt.Trainer().will_launch
    monkeypatch.setenv("GPULAYER_JOB_ID", "job-x")
    assert not glt.Trainer(devices=2).will_launch  # inside the job: run the loop


def _run_example(tmp_path, *args):
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.distributed
def test_launch_ddp_two_nodes_loads_weights_locally(tmp_path):
    p = _run_example(
        tmp_path, "--accelerator", "cpu", "--nodes", "2", "--devices", "1", "--epochs", "2"
    )
    assert p.returncode == 0, p.stdout + p.stderr
    # printed once, by the launching process, from the downloaded checkpoint
    assert p.stdout.count("final metrics:") == 1
    assert "node0/checkpoints/last.ckpt" in p.stdout
    assert list((tmp_path / "gpulayer_runs").glob("job-*/node0/metrics.json"))


@pytest.mark.integration
def test_launch_rejects_impossible_placement(tmp_path):
    p = _run_example(tmp_path, "--accelerator", "cuda", "--target", "local", "--epochs", "1")
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    assert p.returncode != 0
    assert "needs 1 usable cuda device(s)" in p.stderr


class _Linear(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.lin(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_gradient_accumulation_matches_one_large_batch(tmp_path):

    torch.manual_seed(1)
    data = TensorDataset(torch.randn(8, 4), torch.randn(8, 1))
    small, large = _Linear(), _Linear()
    common = dict(accelerator="cpu", enable_checkpointing=False, default_root_dir=tmp_path)
    t_small = glt.Trainer(max_epochs=1, accumulate_grad_batches=2, **common)
    t_small.fit(small, DataLoader(data, batch_size=4))
    t_large = glt.Trainer(max_epochs=1, **common)
    t_large.fit(large, DataLoader(data, batch_size=8))
    assert t_small.global_step == t_large.global_step == 1
    for a, b in zip(small.parameters(), large.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)
    assert isinstance(t_small.callback_metrics["train_loss"], float)


def test_max_steps_stops_mid_epoch(tmp_path):
    train, val = loaders()  # 16 batches per epoch
    trainer = glt.Trainer(accelerator="cpu", max_epochs=10, max_steps=20, default_root_dir=tmp_path)
    trainer.fit(Lin(), train, val)
    assert trainer.global_step == 20 and trainer.current_epoch == 1
    assert "val_loss" in trainer.history[-1]  # the last epoch is always validated
    with pytest.raises(ValidationFailed, match="max_steps"):
        glt.Trainer(max_steps=0)


def test_check_val_every_n_epoch(tmp_path):
    train, val = loaders()
    trainer = glt.Trainer(
        accelerator="cpu", max_epochs=5, check_val_every_n_epoch=2, default_root_dir=tmp_path
    )
    trainer.fit(Lin(), train, val)
    validated = [h["epoch"] for h in trainer.history if "val_loss" in h]
    assert validated == [1, 3, 4]  # every 2nd epoch, plus the last


class LinWithTest(Lin):
    def test_step(self, batch, batch_idx):
        x, y = batch
        self.log("test_mse", nn.functional.mse_loss(self(x), y))


def test_test_and_predict_after_fit(tmp_path):
    train, val = loaders()
    trainer = glt.Trainer(accelerator="cpu", max_epochs=5, default_root_dir=tmp_path)
    model = LinWithTest()
    trainer.fit(model, train, val)
    metrics = trainer.test(model, val)
    assert set(metrics) == {"test_mse"} and metrics["test_mse"] < 0.1
    assert trainer.callback_metrics["test_mse"] == metrics["test_mse"]
    preds = trainer.predict(model, val)  # default predict_step: self(batch[0])
    assert len(preds) == 8 and preds[0].shape == (64, 1) and preds[0].device.type == "cpu"
    with pytest.raises(NotImplementedError, match="test_step"):
        trainer.test(Lin(), val)


def test_val_check_interval_validates_within_epochs(tmp_path):
    train, val = loaders()  # 16 batches per epoch
    t = glt.Trainer(
        accelerator="cpu", max_epochs=2, val_check_interval=4, default_root_dir=tmp_path
    )
    t.fit(Lin(), train, val)
    # Batches 4, 8 and 12 of each epoch; batch 16 is the end-of-epoch validation.
    assert [(v["epoch"], v["step"]) for v in t.val_history] == [
        (0, 4),
        (0, 8),
        (0, 12),
        (1, 20),
        (1, 24),
        (1, 28),
    ]
    t2 = glt.Trainer(
        accelerator="cpu", max_epochs=1, val_check_interval=0.5, default_root_dir=tmp_path
    )
    t2.fit(Lin(), train, val)
    assert [v["step"] for v in t2.val_history] == [8]
    for bad in (0, 1.5, -2):
        with pytest.raises(ValidationFailed, match="val_check_interval"):
            glt.Trainer(val_check_interval=bad)


def test_metrics_csv_has_step_val_and_epoch_rows(tmp_path):
    import csv

    train, val = loaders()
    t = glt.Trainer(
        accelerator="cpu",
        max_epochs=2,
        log_every_n_steps=8,
        val_check_interval=8,
        default_root_dir=tmp_path,
    )
    t.fit(Lin(), train, val)
    rows = list(csv.DictReader((t.checkpoint_path.parents[1] / "metrics.csv").open()))
    kinds = [(r["kind"], r["epoch"], r["step"]) for r in rows]
    assert ("step", "0", "8") in kinds and ("val", "0", "8") in kinds
    assert [k for k in kinds if k[0] == "epoch"] == [("epoch", "0", ""), ("epoch", "1", "")]
    assert {"train_loss", "val_loss"} <= set(rows[0])


SCRIPT_NO_CKPT = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

class Lin(glt.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.lin(x), y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

x = torch.randn(256, 4)
t = glt.Trainer(accelerator="cpu", devices=2, max_epochs=2, enable_checkpointing=False,
                launch_options={"quiet": True})
t.fit(Lin(), DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=32))
if glt.is_launcher_process():
    print("METRICS", sorted(t.callback_metrics), len(t.history))
"""


@pytest.mark.distributed
def test_launched_run_without_checkpoints_returns_metrics(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(SCRIPT_NO_CKPT)
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    p = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "METRICS ['epoch', 'epoch_seconds', 'train_loss'] 2" in p.stdout, p.stdout


SCRIPT_FSDP = """
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

class Net(glt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        # Called after sharding, before any forward: each rank holds only its shard.
        local = sum(p.to_local().numel() if hasattr(p, "to_local") else p.numel()
                    for p in self.parameters())
        total = sum(p.numel() for p in self.parameters())
        if self.trainer.runtime.distributed:
            print(f"SHARD rank={self.trainer.runtime.global_rank} local={local} total={total}",
                  flush=True)
        return torch.optim.Adam(self.parameters(), lr=0.01)

torch.manual_seed(0)
x = torch.randn(512, 8)
y = x @ torch.arange(1.0, 9.0).unsqueeze(1)
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
epochs = int(sys.argv[1])
resume = sys.argv[2] if len(sys.argv) > 2 else None
t = glt.Trainer(accelerator="cpu", devices=2, strategy="fsdp", max_epochs=epochs,
                launch_options={"quiet": False})
model = Net()
t.fit(model, loader, ckpt_path=resume)
if glt.is_launcher_process():
    print("LOSS", t.history[0]["train_loss"], t.history[-1]["train_loss"], len(t.history))
    print("CKPT", t.checkpoint_path)
    restored = Net.load_from_checkpoint(t.checkpoint_path)
    with torch.no_grad():
        print("RESTORED", nn.functional.mse_loss(restored.net(x), y).item())
"""


@pytest.mark.distributed
def test_fsdp_strategy_shards_trains_saves_and_resumes(tmp_path):
    script = tmp_path / "fsdp.py"
    script.write_text(SCRIPT_FSDP)
    scripts = [script]
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)

    def run(*args):
        p = subprocess.run(
            [sys.executable, str(scripts[-1]), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        return p.stdout

    def value(out, key):
        return next(line for line in out.splitlines() if line.startswith(key)).split()[1:]

    out = run("3")
    shards = sorted(line for line in out.splitlines() if "SHARD" in line)
    assert len(shards) == 2, out[-2000:]
    total = int(shards[0].split("total=")[1])
    locals_ = [int(s.split("local=")[1].split()[0]) for s in shards]
    assert all(n < total for n in locals_) and sum(locals_) == total  # really sharded
    # FSDP must train exactly like DDP: same losses, same restored model.
    ddp_script = tmp_path / "ddp.py"
    ddp_script.write_text(SCRIPT_FSDP.replace('strategy="fsdp"', 'strategy="ddp"'))
    scripts.append(ddp_script)
    ddp_out = run("3")
    scripts.pop()
    assert value(out, "LOSS") == value(ddp_out, "LOSS")
    assert value(out, "RESTORED") == pytest.approx(value(ddp_out, "RESTORED"))
    first, last, epochs = value(out, "LOSS")
    assert float(last) < float(first) and epochs == "3"
    ckpt = next(line for line in out.splitlines() if line.startswith("CKPT")).split()[1]
    out2 = run("5", ckpt)  # resume a sharded run from the full checkpoint
    second = next(line for line in out2.splitlines() if line.startswith("LOSS")).split()[1:]
    assert second[2] == "5" and float(second[1]) <= float(last) * 1.5


def test_fast_dev_run_and_sanity_check(tmp_path):
    train, val = loaders()
    t = glt.Trainer(accelerator="cpu", fast_dev_run=2, max_epochs=9, default_root_dir=tmp_path)
    t.fit(Lin(), train, val)
    assert t.global_step == 2 and len(t.history) == 1 and t.checkpoint_path is None

    class BrokenVal(Lin):
        def validation_step(self, batch, batch_idx):
            raise RuntimeError("bug in validation_step")

    t = glt.Trainer(accelerator="cpu", max_epochs=3, default_root_dir=tmp_path)
    with pytest.raises(RuntimeError, match="bug in validation_step"):
        t.fit(BrokenVal(), train, val)
    assert t.global_step == 0  # failed before the first training step
    t = glt.Trainer(
        accelerator="cpu", max_epochs=1, num_sanity_val_steps=0, default_root_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="bug in validation_step"):
        t.fit(BrokenVal(), train, val)
    assert t.global_step == 16  # without the check, a whole epoch ran first


def test_max_time_clip_value_and_deterministic(tmp_path):
    train, val = loaders()
    t = glt.Trainer(
        accelerator="cpu",
        max_epochs=1000,
        max_time=0.3,
        default_root_dir=tmp_path,
        enable_checkpointing=False,
    )
    t0 = time.perf_counter()
    t.fit(Lin(), train)
    assert time.perf_counter() - t0 < 5 and len(t.history) < 1000

    class Spy(Lin):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)  # gradients stay visible

    m = Spy()
    t = glt.Trainer(
        accelerator="cpu",
        max_steps=1,
        gradient_clip_val=0.01,
        gradient_clip_algorithm="value",
        enable_checkpointing=False,
        default_root_dir=tmp_path,
    )
    t.fit(m, train)
    assert all(p.grad is None or p.grad.abs().max() <= 0.01 + 1e-7 for p in m.parameters())
    with pytest.raises(ValidationFailed, match="gradient_clip_algorithm"):
        glt.Trainer(gradient_clip_algorithm="l1")

    def run():
        glt.seed_everything(7)
        tr, _ = loaders()
        mod = Lin()
        glt.Trainer(
            accelerator="cpu",
            max_epochs=2,
            deterministic=True,
            enable_checkpointing=False,
            default_root_dir=tmp_path,
        ).fit(mod, tr)
        return torch.cat([p.detach().flatten() for p in mod.parameters()])

    try:
        assert torch.equal(run(), run())
    finally:
        torch.use_deterministic_algorithms(False)
