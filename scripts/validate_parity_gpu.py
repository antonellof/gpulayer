"""Validate on NVIDIA GPUs the Trainer features that 0.10 tested on CPU only:

    python scripts/validate_parity_gpu.py        # 1 GPU runs checks 1 and 3; 2 or more run all

Prints `CHECK name: PASS|FAIL detail` lines and exits non-zero on a failure.

1. exact_resume     one GPU: a run stopped mid-epoch and resumed gives bit-identical weights to
                    one that never stopped (dropout draws from the CUDA generator).
2. ddp_resume       N GPUs over NCCL: a resumed run trains on exactly the samples it had not
                    trained on yet, across every rank.
3. notebook_fork    a real Jupyter kernel: `devices=N` forks one process per GPU and the trained
                    weights come back; once the kernel has used CUDA, the Trainer refuses with
                    its own error instead of crashing.
4. metric_sync      N GPUs over NCCL: a torchmetrics `Accuracy` logged with `self.log` equals
                    the accuracy of the final weights over the whole validation set, and a value
                    logged in `on_validation_epoch_end` is the mean over ranks.

Checks 3 and 4 need `pip install -e '.[conformance]'` (Jupyter and torchmetrics).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import gpulayer.pytorch as glt

failed = 0
GPUS = torch.cuda.device_count()
N = min(GPUS, 4)


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    failed += not ok
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)


def guarded(fn):
    def run() -> None:
        try:
            fn()
        except Exception:
            check(fn.__name__, False, traceback.format_exc()[-1500:])

    return run


def run_script(root: Path, body: str, timeout: int = 600) -> str:
    script = root / "script.py"
    script.write_text(textwrap.dedent(body))
    env = {**os.environ, "GPULAYER_HOME": str(root / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    p = subprocess.run(
        [sys.executable, str(script)], cwd=root, env=env, capture_output=True, text=True,
        timeout=timeout,
    )  # fmt: skip
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p.stdout


# ------------------------------------------------------------------ 1. exact resume, one GPU


class Counted(Dataset):
    def __init__(self, n: int = 40) -> None:
        self.x = torch.randn(n, 4, generator=torch.Generator().manual_seed(0))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], torch.tensor([float(i)])


class Noisy(glt.Module):
    """Dropout on the GPU and Python's generator: a wrong random number changes the weights."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Dropout(0.5), nn.Linear(64, 1))

    def training_step(self, batch, batch_idx):
        return (self.net(batch[0]) * (1.0 + random.random())).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class SaveAt(glt.Callback):
    def __init__(self, step: int, path: Path) -> None:
        self.step, self.path = step, path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == self.step:
            trainer.save_checkpoint(self.path)


def _trainer(root: Path, **kw) -> glt.Trainer:
    return glt.Trainer(
        accelerator="cuda", devices=1, max_epochs=3, default_root_dir=root, log_every_n_steps=0,
        num_sanity_val_steps=0, **kw,
    )  # fmt: skip


@guarded
def exact_resume() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        glt.seed_everything(1)
        full = Noisy()
        t = _trainer(root, callbacks=[SaveAt(17, root / "mid.ckpt")])
        t.fit(full, DataLoader(Counted(), batch_size=4, shuffle=True))
        glt.seed_everything(99)  # a rerun of a script need not start from the same state
        resumed = Noisy()
        t2 = _trainer(root)
        data = DataLoader(Counted(), batch_size=4, shuffle=True)
        t2.fit(resumed, data, ckpt_path=root / "mid.ckpt")
        pairs = zip(full.state_dict().values(), resumed.state_dict().values(), strict=True)
        same = all(torch.equal(a, b) for a, b in pairs)
        device = next(resumed.parameters()).device
        check(
            "exact_resume", same and t2.global_step == 30 and device.type == "cuda",
            f"step {t2.global_step}, device {device}, identical weights {same}",
        )  # fmt: skip
    torch.use_deterministic_algorithms(False)


# ------------------------------------------------------------------ 2. resume over NCCL

DDP_RESUME = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

class Stop(glt.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 3:
            trainer.save_checkpoint("ROOT/mid.ckpt")

class Model(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Linear(1, 1)
        self.seen = []
    def training_step(self, batch, batch_idx):
        self.seen += batch[0].flatten().tolist()
        return self.net(batch[0]).mean()
    def on_train_end(self):
        g = self.all_gather(torch.tensor(self.seen, device=self.device))
        self.print("SAMPLES", json.dumps(sorted(int(v) for v in g.flatten().tolist())),
                   "BACKEND", torch.distributed.get_backend(), "DEVICE", self.device)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4)

import json
x = torch.arange(64.0).reshape(-1, 1)
data = DataLoader(TensorDataset(x), batch_size=2, shuffle=True)
kw = dict(max_epochs=1, accelerator="cuda", devices=NPROC, log_every_n_steps=0,
          enable_checkpointing=False)
glt.seed_everything(5)
glt.Trainer(callbacks=[Stop()], **kw).fit(Model(), data)
glt.seed_everything(5)
glt.Trainer(**kw).fit(Model(), data, ckpt_path="ROOT/mid.ckpt")
"""


@guarded
def ddp_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = run_script(Path(tmp), DDP_RESUME.replace("ROOT", tmp).replace("NPROC", str(N)))
        lines = [ln for ln in out.splitlines() if "SAMPLES" in ln][-2:]
        first, second = (json.loads(ln.split("SAMPLES ")[1].split(" BACKEND")[0]) for ln in lines)
        tail = lines[-1].split(" BACKEND ", 1)[1]
        skipped = 3 * 2 * N  # three steps of batch 2 on every rank
        ok = (
            first == list(range(64)) and len(second) == 64 - skipped
            and len(set(second)) == len(second) and tail.startswith("nccl")
        )  # fmt: skip
        check("ddp_resume", ok, f"{N} GPUs, {len(first)} then {len(second)} samples, {tail}")


# ------------------------------------------------------------------ 3. notebook fork

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
        if batch_idx == 0:
            name = torch.cuda.get_device_name(self.device)
            print("RANKDEVICE", self.global_rank, self.device, name, flush=True)
        return (self.net(batch[0]) - 2 * batch[0]).pow(2).mean()
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

x = torch.arange(32.0).reshape(-1, 1)
model = Model()
before = model.net.weight.item()
t = glt.Trainer(accelerator="cuda", devices=NPROC, max_epochs=2, log_every_n_steps=0,
                default_root_dir=ROOT)
t.fit(model, DataLoader(TensorDataset(x), batch_size=2))
print("RESULT", json.dumps({"changed": model.net.weight.item() != before,
                            "step": t.global_step, "epoch": t.current_epoch,
                            "cuda_in_kernel": torch.cuda.is_initialized()}))
"""

AFTER_CUDA = """
torch.zeros(1, device="cuda")
try:
    glt.Trainer(accelerator="cuda", devices=NPROC, max_epochs=1, default_root_dir=ROOT).fit(
        Model(), DataLoader(TensorDataset(x), batch_size=2))
    print("REFUSED none")
except Exception as e:
    print("REFUSED", type(e).__name__, str(e)[:200])
"""


@guarded
def notebook_fork() -> None:
    from jupyter_client.manager import start_new_kernel

    with tempfile.TemporaryDirectory() as tmp:
        manager, kernel = start_new_kernel(kernel_name="python3", cwd=tmp)
        output: list[str] = []

        def collect(message):
            if message["msg_type"] == "stream":
                output.append(message["content"]["text"])
            elif message["msg_type"] == "error":
                output.append("\n".join(message["content"]["traceback"]))

        try:
            for cell in (CELL, AFTER_CUDA):
                kernel.execute_interactive(
                    cell.replace("ROOT", repr(tmp)).replace("NPROC", str(N)), timeout=600,
                    output_hook=collect,
                )  # fmt: skip
        finally:
            kernel.stop_channels()
            manager.shutdown_kernel(now=True)
        text = "".join(output)
        if "RESULT {" not in text:
            check("notebook_fork", False, text[-3000:])
            return
        result = json.loads(text.split("RESULT ", 1)[1].splitlines()[0])
        steps = 2 * (16 // N)  # 16 batches split across the ranks, two epochs
        ranks = sorted(ln for ln in text.splitlines() if ln.startswith("RANKDEVICE"))
        devices = {ln.split()[2] for ln in ranks}
        ok = (
            result == {"changed": True, "step": steps, "epoch": 2, "cuda_in_kernel": False}
            and len(ranks) >= N and len(devices) == N
        )  # fmt: skip
        check("notebook_fork", ok, f"{N} GPUs, {result}, ranks on {sorted(devices)}")
        refused = next((ln for ln in text.splitlines() if ln.startswith("REFUSED")), "REFUSED ?")
        check(
            "notebook_refuses_after_cuda", "ValidationFailed" in refused and "CUDA" in refused,
            refused,
        )  # fmt: skip


# ------------------------------------------------------------------ 4. metric objects over NCCL

METRICS = """
import json, torch, torchmetrics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

g = torch.Generator().manual_seed(0)
x = torch.randn(512, 16, generator=g)
y = (x[:, 0] + 0.5 * torch.randn(512, generator=g) > 0).long()

class Model(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Linear(16, 2)
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=2, average="micro")
    def training_step(self, batch, batch_idx):
        return nn.functional.cross_entropy(self.net(batch[0]), batch[1])
    def validation_step(self, batch, batch_idx):
        self.acc(self.net(batch[0]).argmax(1), batch[1])
        self.log("val_acc", self.acc)
    def on_validation_epoch_end(self):
        rank = torch.tensor(float(self.global_rank), device=self.device)
        self.log("rank_mean", rank, sync_dist=True)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

model = Model()
t = glt.Trainer(accelerator="cuda", devices=NPROC, max_epochs=2, log_every_n_steps=0,
                num_sanity_val_steps=0, enable_checkpointing=False, default_root_dir="ROOT")
loader = lambda shuffle: DataLoader(TensorDataset(x, y), batch_size=32, shuffle=shuffle)
t.fit(model, loader(True), loader(False))
with torch.no_grad():
    truth = (model.net.cpu()(x).argmax(1) == y).float().mean().item()
m = {k: float(v) for k, v in t.callback_metrics.items()}
print("METRICS", json.dumps({"logged": m.get("val_acc"), "truth": truth,
                             "rank_mean": m.get("rank_mean")}))
"""


@guarded
def metric_sync() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = run_script(Path(tmp), METRICS.replace("ROOT", tmp).replace("NPROC", str(N)))
        m = json.loads(out.split("METRICS ", 1)[1].splitlines()[0])
        ok = (
            m["logged"] is not None and abs(m["logged"] - m["truth"]) < 1e-6
            and m["rank_mean"] is not None and abs(m["rank_mean"] - (N - 1) / 2) < 1e-6
        )  # fmt: skip
        check("metric_sync", ok, f"{N} GPUs, {m}")


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA GPU")
        return 1
    name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {GPUS} x {name}")
    exact_resume()
    notebook_fork()
    if GPUS >= 2:
        ddp_resume()
        metric_sync()
    print(f"SUMMARY {'all passed' if not failed else f'{failed} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
