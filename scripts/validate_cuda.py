"""CUDA checks for a real NVIDIA machine. Run by scripts/validate_gpu.sh; needs torch with CUDA.

    python scripts/validate_cuda.py

Prints one "CHECK <name> PASS|FAIL <details>" line per check and exits non-zero if any fail.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer as gl
import gpulayer.pytorch as glt
from gpulayer.contracts import Accelerator, JobState
from gpulayer.gpu.discovery import detect_capabilities

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            RESULTS.append((name, True, f"{detail} ({time.perf_counter() - t0:.1f}s)"))
        except Exception as e:  # noqa: BLE001 - report every failure and carry on
            traceback.print_exc()
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"CHECK {name} {'PASS' if RESULTS[-1][1] else 'FAIL'} {RESULTS[-1][2]}", flush=True)
        return fn

    return wrap


class Net(glt.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 1))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.seen_device = x.device.type
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16384, 256, generator=g)
    w = torch.randn(256, 1, generator=g) / 16
    y = x @ w + 0.01 * torch.randn(16384, 1, generator=g)
    return (
        DataLoader(TensorDataset(x[:14336], y[:14336]), batch_size=256, shuffle=True),
        DataLoader(TensorDataset(x[14336:], y[14336:]), batch_size=1024),
    )


@check("detect-cuda")
def _detect():
    caps = detect_capabilities()
    devs = caps.devices(Accelerator.CUDA)
    assert devs, f"no CUDA devices detected; notes: {caps.notes}"
    assert all(d.execution_supported for d in devs), devs
    assert "torch.cuda" in caps.frameworks and "nccl" in caps.distributed_backends, caps
    return ", ".join(f"{d.name} {d.memory_gb} GB cc{d.compute_capability}" for d in devs)


def fit(precision: str, compiled: bool) -> str:
    model = Net()
    if compiled:
        model.net = torch.compile(model.net)
    train, val = loaders()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = glt.Trainer(
            accelerator="cuda", precision=precision, max_epochs=3, default_root_dir=tmp
        )
        t0 = time.perf_counter()
        trainer.fit(model, train, val)
        torch.cuda.synchronize()
        took = time.perf_counter() - t0
    val_loss = float(trainer.callback_metrics["val_loss"])
    assert val_loss < 0.05, f"val_loss {val_loss:.4f} did not converge"
    # fit() returns the model on the CPU (like a submitted run); check where it trained.
    assert model.seen_device == "cuda", model.seen_device
    return f"val_loss {val_loss:.4f}, 3 epochs {took:.2f}s"


for _precision in ("32-true", "bf16-mixed", "16-mixed"):
    check(f"trainer-in-process-{_precision}")(lambda p=_precision: fit(p, compiled=False))
check("trainer-torch-compile-bf16")(lambda: fit("bf16-mixed", compiled=True))


@check("cuda-job-gets-its-reserved-gpu")
def _job():
    with tempfile.TemporaryDirectory() as tmp, gl.Client(home=Path(tmp)) as c:
        code = (
            "import os, torch; print('CVD', os.environ['CUDA_VISIBLE_DEVICES'], "
            "torch.cuda.device_count(), torch.cuda.get_device_name(0))"
        )
        job = c.submit(gl.command(["python", "-c", code], accelerator="cuda", gpus=1))
        st = job.wait(300)
        logs = job.logs_text()
        assert st.state == JobState.SUCCEEDED, (st.error, logs)
        line = next(x for x in logs.splitlines() if "CVD" in x)
        assert line.split()[2] == "1", line
        return line.strip()


@check("one-gpu-worker-runs-gpu-jobs-one-at-a-time")
def _serial():
    # Two 1-GPU jobs on a 1-GPU worker must not overlap: the second waits for the GPU.
    code = "import time; print('T', time.time()); time.sleep(3); print('T', time.time())"
    with tempfile.TemporaryDirectory() as tmp, gl.Client(home=Path(tmp)) as c:
        jobs = [
            c.submit(gl.command(["python", "-c", code], accelerator="cuda", gpus=1))
            for _ in range(2)
        ]
        spans = []
        for j in jobs:
            st = j.wait(300)
            assert st.state == JobState.SUCCEEDED, (st.error, j.logs_text())
            ts = [float(x.split()[1]) for x in j.logs_text().splitlines() if x.startswith("T ")]
            spans.append((ts[0], ts[-1]))
        spans.sort()
        assert spans[1][0] >= spans[0][1], f"jobs overlapped: {spans}"
        return f"gap {spans[1][0] - spans[0][1]:.2f}s"


@check("trainer-as-gpulayer-job-cuda")
def _remote_trainer():
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(__file__).resolve().parents[1] / "examples/trainer_regression/train.py"
        env = {**os.environ, "GPULAYER_HOME": str(Path(tmp, "home"))}
        import subprocess

        p = subprocess.run(
            [sys.executable, str(script), "--accelerator", "cuda", "--target", "local"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        return "succeeded"


failed = [n for n, ok, _ in RESULTS if not ok]
print(f"SUMMARY {len(RESULTS) - len(failed)}/{len(RESULTS)} passed", failed or "")
sys.exit(1 if failed else 0)
