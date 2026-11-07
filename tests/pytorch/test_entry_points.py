"""Starting the Trainer from SLURM and from a notebook. SLURM is SIMULATED: its variables are
set by hand and two processes run on this machine; no cluster is involved. The notebook tests
fake the kernel, except the last, which runs a real Jupyter kernel when one is installed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch.launch import (  # noqa: E402
    adopt_slurm,
    first_host,
    in_slurm_task,
    is_launcher_process,
)

launch_module = sys.modules["gpulayer.pytorch.launch"]

SLURM = {
    "SLURM_PROCID": "1",
    "SLURM_LOCALID": "1",
    "SLURM_NODEID": "0",
    "SLURM_NTASKS": "2",
    "SLURM_NNODES": "1",
    "SLURM_NTASKS_PER_NODE": "2",
    "SLURM_NODELIST": "node[07-09]",
    "SLURM_JOB_ID": "4242",
    "SLURM_JOB_NAME": "train",
    "SLURM_STEP_ID": "0",
}
RANKS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GPULAYER_NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@pytest.fixture
def slurm(monkeypatch):
    for name in RANKS:
        monkeypatch.delenv(name, raising=False)
    for name, value in SLURM.items():
        monkeypatch.setenv(name, value)
    yield monkeypatch
    for name in RANKS:  # adopt_slurm wrote these outside monkeypatch
        os.environ.pop(name, None)


@pytest.mark.parametrize(
    ("nodes", "first"),
    [
        ("node07", "node07"),
        ("node[07-09]", "node07"),
        ("gpu[07,12],cpu01", "gpu07"),
        ("a1,a2", "a1"),
        ("rack2-[03-05]", "rack2-03"),
    ],
)
def test_first_host(nodes, first):
    assert first_host(nodes) == first


def test_slurm_task_trains_in_place(slurm):
    assert in_slurm_task() and not is_launcher_process()
    assert os.environ["RANK"] == "1" and os.environ["WORLD_SIZE"] == "2"
    assert os.environ["LOCAL_RANK"] == "1" and os.environ["LOCAL_WORLD_SIZE"] == "2"
    assert os.environ["MASTER_ADDR"] == "node07"
    assert os.environ["MASTER_PORT"] == str(15000 + 4242)


@pytest.mark.parametrize(
    "change",
    [
        {"SLURM_STEP_ID": "batch"},
        {"SLURM_STEP_ID": None},
        {"SLURM_JOB_NAME": "bash"},
        {"SLURM_JOB_NAME": "interactive"},
    ],
    ids=str,
)
def test_not_a_task(slurm, change):
    """An sbatch script runs once: it must submit or train, not act as one rank of many."""
    for name, value in change.items():
        if value is None:
            slurm.delenv(name)
        else:
            slurm.setenv(name, value)
    adopt_slurm()
    assert not in_slurm_task() and "RANK" not in os.environ


def test_tasks_must_match_devices(slurm, tmp_path):
    t = glt.Trainer(accelerator="cpu", devices=4, default_root_dir=tmp_path)
    with pytest.raises(ValidationFailed, match="srun started 2 tasks"):
        t.fit(glt.Module())


def start_task(tmp_path, rank: int):
    script = tmp_path / "train.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
            import gpulayer.pytorch as glt

            class Model(glt.Module):
                def __init__(self):
                    super().__init__()
                    torch.manual_seed(0)
                    self.net = nn.Linear(1, 1)
                    self.seen = 0
                def training_step(self, batch, batch_idx):
                    self.seen += len(batch[0])
                    return self.net(batch[0]).mean()
                def configure_optimizers(self):
                    return torch.optim.SGD(self.parameters(), lr=1e-3)

            x = torch.arange(16.0).reshape(-1, 1)
            m = Model()
            t = glt.Trainer(accelerator="cpu", devices=2, max_epochs=1, log_every_n_steps=0)
            t.fit(m, DataLoader(TensorDataset(x), batch_size=2))
            print("RESULT", json.dumps({"rank": t.global_rank, "world": t.world_size,
                                        "seen": m.seen, "launched": t.launch_result is not None,
                                        "w": m.net.weight.item()}), flush=True)
            """
        )
    )
    env = {
        **os.environ,
        **SLURM,
        "SLURM_PROCID": str(rank),
        "SLURM_LOCALID": str(rank),
        "SLURM_NODELIST": "127.0.0.1",
        "SLURM_JOB_ID": str(30000 + os.getpid() % 5000),
        "GPULAYER_HOME": str(tmp_path / "home"),
    }
    for name in (*RANKS, "GPULAYER_URL", "GPULAYER_JOB_ID"):
        env.pop(name, None)
    return subprocess.Popen(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_two_tasks_share_the_data_and_submit_nothing(tmp_path):
    tasks = [start_task(tmp_path, r) for r in (0, 1)]
    results = []
    for task in tasks:
        out, err = task.communicate(timeout=120)
        assert task.returncode == 0, out[-2000:] + err[-2000:]
        results.append(json.loads(out.split("RESULT ", 1)[1].splitlines()[0]))
    assert [r["rank"] for r in results] == [0, 1]
    assert all(r["world"] == 2 and r["seen"] == 8 and not r["launched"] for r in results)
    assert results[0]["w"] == pytest.approx(results[1]["w"])


# ------------------------------------------------------------------ notebooks


def test_launch_refuses_a_notebook(monkeypatch):
    monkeypatch.setattr(launch_module, "in_notebook", lambda: True)
    with pytest.raises(ValidationFailed, match="no script to submit"):
        launch_module.launch(devices=2)


def test_ddp_notebook_is_ddp_outside_a_notebook():
    assert isinstance(glt.Trainer(strategy="ddp_notebook", devices=2).strategy, glt.DDPStrategy)
    assert isinstance(glt.Trainer(strategy="ddp_fork").strategy, glt.DDPStrategy)
    with pytest.raises(ValidationFailed, match="spawn"):
        glt.Trainer(strategy="ddp_spawn")


CELL = """
import json, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

class Model(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Linear(1, 1)
    def training_step(self, batch, batch_idx):
        return (self.net(batch[0]) - 2 * batch[0]).pow(2).mean()
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

x = torch.arange(16.0).reshape(-1, 1)
model = Model()
before = model.net.weight.item()
t = glt.Trainer(accelerator="cpu", devices=2, max_epochs=2, log_every_n_steps=0,
                default_root_dir=ROOT)
t.fit(model, DataLoader(TensorDataset(x), batch_size=2))
print("RESULT", json.dumps({"changed": model.net.weight.item() != before,
                            "step": t.global_step, "epoch": t.current_epoch}))
"""


@pytest.mark.parametrize("exits", [(0, 0), (0, 3), (3, 0)], ids=str)
def test_waiting_for_ranks_never_waits_on_nothing(exits):
    """Processes that end at once, again and again: a wait built from those still alive could
    end up with an empty list and block for ever, after a run that had succeeded."""
    import multiprocessing

    from gpulayer.pytorch.trainer import _wait_for_ranks

    context = multiprocessing.get_context("fork" if sys.platform != "darwin" else "spawn")
    for _ in range(100 if sys.platform != "darwin" else 2):
        ranks = [context.Process(target=sys.exit, args=(code,)) for code in exits]
        for rank in ranks:
            rank.start()
        failed = _wait_for_ranks(ranks)
        assert (failed is None) == (exits == (0, 0))
        assert all(not r.is_alive() for r in ranks)


linux_only = pytest.mark.skipif(sys.platform == "darwin", reason="forks crash on macOS")


@linux_only
def test_forked_ranks_return_their_weights_to_the_notebook(tmp_path):
    """The kernel is faked; the forks, the group and the weights coming back are real."""
    script = tmp_path / "nb.py"
    script.write_text(
        "import sys, gpulayer.pytorch\n"
        "sys.modules['gpulayer.pytorch.trainer'].in_notebook = lambda: True\n"
        + CELL.replace("ROOT", repr(str(tmp_path)))
    )
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for name in (*RANKS, "GPULAYER_URL", "GPULAYER_JOB_ID"):
        env.pop(name, None)
    p = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
        timeout=120,
    )  # fmt: skip
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    result = json.loads(p.stdout.split("RESULT ", 1)[1].splitlines()[0])
    assert result == {"changed": True, "step": 8, "epoch": 2}


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_refuses_to_fork(tmp_path):
    t = glt.Trainer(accelerator="cpu", devices=2, default_root_dir=tmp_path)
    with pytest.raises(ValidationFailed, match="macOS crashes"):
        t._fit_notebook(glt.Module(), None, None, None, None)


@linux_only
def test_real_jupyter_kernel(tmp_path):
    client = pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    manager, kernel = client.manager.start_new_kernel(kernel_name="python3", cwd=str(tmp_path))
    output: list[str] = []

    def collect(message):
        if message["msg_type"] == "stream":
            output.append(message["content"]["text"])
        elif message["msg_type"] == "error":
            output.append("\n".join(message["content"]["traceback"]))

    try:
        kernel.execute_interactive(
            CELL.replace("ROOT", repr(str(tmp_path))), timeout=120, output_hook=collect
        )
    finally:
        kernel.stop_channels()
        manager.shutdown_kernel(now=True)
    text = "".join(output)
    assert "RESULT {" in text, text[-3000:]
    result = json.loads(text.split("RESULT ", 1)[1].splitlines()[0])
    assert result == {"changed": True, "step": 8, "epoch": 2}
