"""Several TPU devices through xmp.spawn, as Lightning's XLAStrategy (SIMULATED).

torch_xla is replaced by a fake package that implements the calls the Trainer makes
(xmp.spawn, xr ranks, xm.xla_device, xm.optimizer_step with its gradient all-reduce,
xm.mark_step) on CPU processes over gloo. This tests the Trainer's plumbing: one process per
device, ranks, sharded data, gradients averaged by optimizer_step, rank 0's checkpoint and
metrics loaded in the launching process. It is not evidence that training runs on a TPU."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")

FAKE = {
    "torch_xla/__init__.py": "",
    "torch_xla/core/__init__.py": "",
    "torch_xla/distributed/__init__.py": "",
    "torch_xla/runtime.py": """
        import os
        def world_size(): return int(os.environ["FAKE_XLA_WORLD"])
        def global_ordinal(): return int(os.environ["FAKE_XLA_RANK"])
        def local_ordinal(): return int(os.environ["FAKE_XLA_RANK"])
    """,
    "torch_xla/core/xla_model.py": """
        import json, os
        import torch
        import torch.distributed as dist
        def xla_device(): return torch.device("cpu")
        def mark_step(): pass
        def optimizer_step(optimizer, barrier=False, optimizer_args={}):
            world = dist.get_world_size() if dist.is_initialized() else 1
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None and world > 1:
                        dist.all_reduce(p.grad)
                        p.grad /= world
            log = os.environ.get("FAKE_XLA_LOG")
            if log:
                with open(f"{log}.{os.environ.get('FAKE_XLA_RANK', 'main')}", "a") as f:
                    f.write("step\\n")
            return optimizer.step(**optimizer_args)
    """,
    "torch_xla/distributed/xla_multiprocessing.py": """
        import os
        import torch.multiprocessing as mp
        def _run(index, fn, args):
            os.environ["FAKE_XLA_RANK"] = str(index)
            fn(index, *args)
        def spawn(fn, args=(), nprocs=None, start_method="fork"):
            assert nprocs is None and start_method == "fork"
            mp.start_processes(_run, args=(fn, args), nprocs=int(os.environ["FAKE_XLA_WORLD"]),
                               start_method=start_method)
    """,
}

SCRIPT = """
import json, os, sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

class Lin(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 1)
        self.seen = 0
    def training_step(self, batch, batch_idx):
        x, y = batch
        self.seen += len(x)
        loss = nn.functional.mse_loss(self.lin(x), y)
        self.log("train_loss", loss)
        return loss
    def on_train_end(self):
        w = self.lin.weight.detach().flatten().tolist()
        with open(f"rank{self.global_rank}.json", "w") as f:
            json.dump({"seen": self.seen, "w": w, "world": self.trainer.world_size}, f)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

x = torch.randn(32, 4)
data = TensorDataset(x, x.sum(1, keepdim=True))
model = Lin()
devices = sys.argv[1] if sys.argv[1] == "auto" else int(sys.argv[1])
trainer = glt.Trainer(accelerator="tpu", devices=devices,
                      max_epochs=2, default_root_dir="runs", enable_progress_bar=False)
trainer.fit(model, DataLoader(data, batch_size=4, shuffle=True))
print("RESULT", json.dumps({"metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
      "ckpt": str(trainer.checkpoint_path), "w": model.lin.weight.detach().flatten().tolist()}))
"""


def run(tmp_path, devices, world=2):
    for name, body in FAKE.items():
        f = tmp_path / "fake" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(body))
    (tmp_path / "train.py").write_text(SCRIPT)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPULAYER_", "RANK", "LOCAL_"))}
    env.update(
        PYTHONPATH=os.pathsep.join([str(tmp_path / "fake"), env.get("PYTHONPATH", "")]),
        PJRT_DEVICE="TPU", FAKE_XLA_WORLD=str(world), FAKE_XLA_LOG=str(tmp_path / "steps"),
    )  # fmt: skip
    return subprocess.run(
        [sys.executable, "train.py", str(devices)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=180,
    )  # fmt: skip


def test_several_tpu_devices_train_in_spawned_processes(tmp_path):
    p = run(tmp_path, "auto")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    ranks = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
    assert [r["world"] for r in ranks] == [2, 2]
    # each process trained on its half of the data: 2 epochs x 16 samples
    assert [r["seen"] for r in ranks] == [32, 32]
    # optimizer_step averaged the gradients, so both processes hold the same weights
    assert ranks[0]["w"] == pytest.approx(ranks[1]["w"])
    # 2 epochs x 4 batches per process, every step through xm.optimizer_step
    for r in (0, 1):
        assert len(Path(f"{tmp_path / 'steps'}.{r}").read_text().split()) == 8
    out = json.loads(p.stdout.split("RESULT ", 1)[1])
    assert "train_loss" in out["metrics"]
    assert Path(tmp_path / out["ckpt"]).is_file()
    # the launching process got rank 0's trained weights
    assert out["w"] == pytest.approx(ranks[0]["w"])


def test_devices_must_match_the_tpu_vm(tmp_path):
    p = run(tmp_path, 3)
    assert p.returncode != 0
    assert "this TPU VM has 2 TPU devices" in p.stderr


def test_tpu_pods_and_targets_are_rejected():
    import gpulayer.pytorch as glt
    from gpulayer.errors import ValidationFailed

    for kwargs in ({"num_nodes": 2}, {"target": "remote"}):
        t = glt.Trainer(accelerator="tpu", devices=4, **kwargs)
        with pytest.raises(ValidationFailed, match="TPU pods"):
            t._xla_spawn_needed()
