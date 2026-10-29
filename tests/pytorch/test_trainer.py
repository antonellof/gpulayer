"""gpulayer.pytorch: runtime helpers, in-process Trainer, and the launch path (real processes)."""

import os
import subprocess
import sys
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
