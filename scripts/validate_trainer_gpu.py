"""Validate the GPUlayer Trainer features that need one NVIDIA GPU (ADR-0024's open items):

    python scripts/validate_trainer_gpu.py

Runs in this process on cuda:0 and prints `CHECK name: PASS|FAIL` lines:

- torch.compile with Inductor: a compiled Module trains and matches the eager loss closely;
- 16-mixed with manual optimization: GradScaler through manual_backward and opt.step();
- "-true" precisions: 16-true (and bf16-true when the GPU supports bf16);
- PyTorchProfiler: the table has CUDA time columns;
- EMAWeightAveraging with 16-mixed; DeviceStatsMonitor reports CUDA memory.
"""

from __future__ import annotations

import math
import sys
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt

failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    failed += not ok
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)


def loader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2048, 128, generator=g)
    y = (x[:, :10].sum(1) > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=128, shuffle=False)


class MLP(glt.Module):
    def __init__(self, manual: bool = False) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 2))
        self.automatic_optimization = not manual

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        if self.automatic_optimization:
            return loss
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def fit(model: glt.Module, **kw) -> glt.Trainer:
    kw.setdefault("max_epochs", 3)
    kw.setdefault("log_every_n_steps", 0)
    t = glt.Trainer(
        accelerator="cuda",
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=False,
        **kw,
    )
    t.fit(model, loader())
    return t


def run(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:  # report and continue with the other checks
        traceback.print_exc()
        check(name, False, f"({type(e).__name__}: {e})")


def compiled() -> None:
    eager = fit(MLP()).history[-1]["train_loss"]
    t = fit(torch.compile(MLP()))
    loss = t.history[-1]["train_loss"]
    check(
        "torch.compile (Inductor) trains",
        math.isfinite(loss) and abs(loss - eager) < 0.05,
        f"eager {eager:.4f} compiled {loss:.4f}",
    )


def manual_amp() -> None:
    t = fit(MLP(manual=True), precision="16-mixed")
    h = t.history
    check(
        "16-mixed manual optimization",
        t.global_step == 3 * 16 and h[-1]["train_loss"] < h[0]["train_loss"],
        f"steps {t.global_step} loss {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f}",
    )


def true_precisions() -> None:
    kinds = ["16-true"] + (["bf16-true"] if torch.cuda.is_bf16_supported() else [])
    for p in kinds:
        m = MLP()
        t = fit(m, precision=p)
        loss = t.history[-1]["train_loss"]
        dtype = next(m.parameters()).dtype
        check(
            f"precision {p}",
            math.isfinite(loss) and dtype != torch.float32,
            f"{dtype} loss {loss:.4f}",
        )


def profiler(capsys_free: bool = True) -> None:
    prof = glt.PyTorchProfiler(row_limit=8)
    fit(MLP(), max_epochs=1, profiler=prof)
    table = prof.summary()
    check("PyTorchProfiler CUDA columns", "CUDA" in table)


def ema_and_stats() -> None:
    ema = glt.EMAWeightAveraging(decay=0.9)
    stats = glt.DeviceStatsMonitor()
    t = fit(MLP(), precision="16-mixed", callbacks=[ema, stats], log_every_n_steps=10)
    check("EMA with 16-mixed", math.isfinite(t.history[-1]["train_loss"]))
    check(
        "DeviceStatsMonitor CUDA memory",
        "device/memory_allocated_mb" in t.logged_metrics,
        str({k: round(v, 1) for k, v in t.logged_metrics.items() if k.startswith("device/")}),
    )


if not torch.cuda.is_available():
    sys.exit("needs a CUDA GPU")
print("GPU", torch.cuda.get_device_name(), "torch", torch.__version__, "cuda", torch.version.cuda)
for name, fn in [
    ("torch.compile (Inductor) trains", compiled),
    ("16-mixed manual optimization", manual_amp),
    ("true precisions", true_precisions),
    ("PyTorchProfiler CUDA columns", profiler),
    ("EMA and device stats", ema_and_stats),
]:
    run(name, fn)
print("ALL CHECKS PASSED" if not failed else f"{failed} CHECK(S) FAILED")
sys.exit(1 if failed else 0)
