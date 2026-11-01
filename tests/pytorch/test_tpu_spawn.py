"""Several TPU devices and TPU pods, as Lightning's XLAStrategy (SIMULATED).

torch_xla is replaced by a fake package that implements the calls the Trainer makes
(xmp.spawn, the xr ranks, xm.xla_device, xm.optimizer_step with its gradient all-reduce,
xm.all_reduce, xm.all_gather, xm.rendezvous, xm.mark_step, MpDeviceLoader) on CPU processes
over gloo. This tests the Trainer's plumbing: one process per device, ranks across pod hosts,
sharded data, gradients averaged by optimizer_step, metrics reduced with the XLA collectives,
batches fed through MpDeviceLoader, rank 0's checkpoint loaded in the launching process. It
is not evidence that training runs on a TPU."""

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
        def _local(): return int(os.environ["FAKE_XLA_LOCAL"])
        def _host(): return int(os.environ.get("FAKE_XLA_HOST", "0"))
        def world_size(): return int(os.environ["FAKE_XLA_HOSTS"]) * _local()
        def local_process_count(): return _local()
        def host_index(): return _host()
        def local_ordinal(): return int(os.environ["FAKE_XLA_RANK"])
        def global_ordinal(): return _host() * _local() + local_ordinal()
    """,
    "torch_xla/core/xla_model.py": """
        import os
        import torch
        import torch.distributed as dist
        from torch_xla import runtime as xr
        REDUCE_SUM, REDUCE_MAX = "sum", "max"
        def _group():
            if not dist.is_initialized():
                url = "tcp://127.0.0.1:" + os.environ["FAKE_XLA_PORT"]
                dist.init_process_group("gloo", init_method=url,
                                        rank=xr.global_ordinal(), world_size=xr.world_size())
        def _log(what):
            with open(os.environ["FAKE_XLA_LOG"] + "." + str(xr.global_ordinal()), "a") as f:
                f.write(what + "\\n")
        def xla_device(): return torch.device("cpu")
        def mark_step(): pass
        def rendezvous(tag):
            _group(); dist.barrier()
        def all_reduce(kind, t):
            _group(); _log("all_reduce")
            t = t.clone()
            dist.all_reduce(t, op=dist.ReduceOp.MAX if kind == "max" else dist.ReduceOp.SUM)
            return t
        def all_gather(t, dim=0):
            _group()
            out = [torch.empty_like(t) for _ in range(xr.world_size())]
            dist.all_gather(out, t.contiguous())
            return torch.cat(out, dim)
        def optimizer_step(optimizer, barrier=False, optimizer_args={}):
            _group(); _log("step")
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        dist.all_reduce(p.grad); p.grad /= xr.world_size()
            return optimizer.step(**optimizer_args)
    """,
    "torch_xla/core/functions.py": "from torch_xla.core.xla_model import all_gather\n",
    "torch_xla/distributed/parallel_loader.py": """
        from torch_xla.core.xla_model import _log
        class MpDeviceLoader:
            def __init__(self, loader, device): self.loader = loader
            def __len__(self): return len(self.loader)
            def __iter__(self):
                for batch in self.loader:
                    _log("mp_batch"); yield batch
    """,
    "torch_xla/distributed/xla_multiprocessing.py": """
        import os
        import torch.multiprocessing as mp
        def _run(index, fn, args):
            os.environ["FAKE_XLA_RANK"] = str(index)
            fn(index, *args)
        def spawn(fn, args=(), nprocs=None, start_method="fork"):
            assert nprocs is None and start_method == "fork"
            mp.start_processes(_run, args=(fn, args), nprocs=int(os.environ["FAKE_XLA_LOCAL"]),
                               start_method=start_method)
    """,
}

SCRIPT = """
import json, sys
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
        gathered = self.all_gather(torch.tensor([float(self.global_rank)]))
        with open(f"rank{self.global_rank}.json", "w") as f:
            json.dump({"seen": self.seen, "w": self.lin.weight.detach().flatten().tolist(),
                       "world": self.trainer.world_size,
                       "gathered": gathered.flatten().tolist()}, f)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

torch.manual_seed(1)
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


def start(tmp_path, devices, *, local=2, hosts=1, host=0, port=None):
    for name, body in FAKE.items():
        f = tmp_path / "fake" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(body))
    (tmp_path / "train.py").write_text(SCRIPT)
    from gpulayer.runtime.distributed import free_port

    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPULAYER_", "RANK", "LOCAL_"))}
    env.update(
        PYTHONPATH=os.pathsep.join([str(tmp_path / "fake"), env.get("PYTHONPATH", "")]),
        PJRT_DEVICE="TPU", FAKE_XLA_LOCAL=str(local), FAKE_XLA_HOSTS=str(hosts),
        FAKE_XLA_HOST=str(host), FAKE_XLA_PORT=str(port or free_port("127.0.0.1")),
        FAKE_XLA_LOG=str(tmp_path / "log"),
    )  # fmt: skip
    cwd = tmp_path / f"host{host}"
    cwd.mkdir(exist_ok=True)
    return subprocess.Popen(
        [sys.executable, str(tmp_path / "train.py"), str(devices)], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip


def finish(p):
    out, err = p.communicate(timeout=180)
    return p.returncode, out, err


def log(tmp_path, rank):
    return Path(f"{tmp_path / 'log'}.{rank}").read_text().split()


def test_several_tpu_devices_train_in_spawned_processes(tmp_path):
    code, out, err = finish(start(tmp_path, "auto"))
    assert code == 0, out[-3000:] + err[-3000:]
    ranks = [json.loads((tmp_path / "host0" / f"rank{r}.json").read_text()) for r in (0, 1)]
    assert [r["world"] for r in ranks] == [2, 2]
    assert [r["seen"] for r in ranks] == [32, 32]  # each process: its half, 2 epochs
    assert ranks[0]["w"] == pytest.approx(ranks[1]["w"])  # gradients averaged by optimizer_step
    assert ranks[0]["gathered"] == [0.0, 1.0]  # all_gather through xm
    for r in (0, 1):
        events = log(tmp_path, r)
        assert events.count("step") == 8  # 2 epochs x 4 batches, all via xm.optimizer_step
        assert events.count("mp_batch") == 8  # every batch came through MpDeviceLoader
        assert "all_reduce" in events  # epoch metrics reduced with XLA collectives
    result = json.loads(out.split("RESULT ", 1)[1])
    assert "train_loss" in result["metrics"]
    assert (tmp_path / "host0" / result["ckpt"]).is_file()
    assert result["w"] == pytest.approx(ranks[0]["w"])  # rank 0's weights came back


def test_a_tpu_pod_is_one_run_across_hosts(tmp_path):
    """Two hosts of two devices each, started separately as on a pod (--worker=all)."""
    from gpulayer.runtime.distributed import free_port

    port = free_port("127.0.0.1")
    procs = [start(tmp_path, "auto", hosts=2, host=h, port=port) for h in (0, 1)]
    results = [finish(p) for p in procs]
    for code, out, err in results:
        assert code == 0, out[-3000:] + err[-3000:]
    ranks = {}
    for h in (0, 1):
        for f in (tmp_path / f"host{h}").glob("rank*.json"):
            ranks[int(f.stem[4:])] = json.loads(f.read_text())
    assert sorted(ranks) == [0, 1, 2, 3]
    assert all(r["world"] == 4 for r in ranks.values())
    assert [ranks[r]["seen"] for r in range(4)] == [16] * 4  # a quarter each, 2 epochs
    assert all(r["w"] == pytest.approx(ranks[0]["w"]) for r in ranks.values())
    assert ranks[3]["gathered"] == [0.0, 1.0, 2.0, 3.0]
    # global rank 0 lives on host 0, so its launching process gets the checkpoint
    assert json.loads(results[0][1].split("RESULT ", 1)[1])["ckpt"] != "None"


def test_devices_must_match_the_tpu_vm(tmp_path):
    code, _, err = finish(start(tmp_path, 3))
    assert code != 0
    assert "this TPU VM has 2 TPU devices" in err


def test_num_nodes_and_targets_are_not_used_with_tpus():
    import gpulayer.pytorch as glt
    from gpulayer.errors import ValidationFailed

    for kwargs in ({"num_nodes": 2}, {"target": "remote"}):
        t = glt.Trainer(accelerator="tpu", devices=4, **kwargs)
        with pytest.raises(ValidationFailed, match="--worker=all"):
            t._xla_spawn_needed()


def test_hpu_is_rejected():
    import gpulayer.pytorch as glt
    from gpulayer.errors import ValidationFailed

    with pytest.raises(ValidationFailed, match="removed it"):
        glt.resolve_accelerator("hpu")
