"""ROCm checks for a real AMD machine. Run by scripts/validate_rocm.sh; needs a ROCm build of torch.

    python scripts/validate_rocm.py

Prints one "CHECK <name> PASS|FAIL <details>" line per check and exits non-zero if any fail. Every
check is here because the behaviour has never run on AMD hardware; see docs/guides/amd-rocm.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt
from gpulayer.contracts import Accelerator
from gpulayer.gpu.discovery import detect_capabilities

RESULTS: list[tuple[str, bool, str]] = []
SKIP = "SKIP"


def check(name: str):
    def wrap(fn):
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            if detail == SKIP:
                print(f"CHECK {name} SKIP needs two GPUs", flush=True)
                return fn
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
        self.net = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss = nn.functional.mse_loss(self(batch[0]), batch[1])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def loader() -> DataLoader:
    x = torch.randn(512, 64)
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=32)


def train(root: str, **kw) -> glt.Trainer:
    kw.setdefault("accelerator", "rocm")
    t = glt.Trainer(max_epochs=2, log_every_n_steps=0, default_root_dir=root, **kw)
    t.fit(Net(), loader())
    return t


COUNT = torch.cuda.device_count()


@check("is_a_rocm_build")
def _build():
    assert torch.version.hip, "torch.version.hip is empty: not a ROCm build"
    assert torch.cuda.is_available(), "torch cannot see an AMD GPU; check /dev/kfd and groups"
    return f"hip {torch.version.hip}, {COUNT} GPU(s), {torch.cuda.get_device_name(0)}"


@check("discovery_reports_rocm")
def _discovery():
    caps = detect_capabilities()
    devices = caps.devices(Accelerator.ROCM)
    assert len(devices) == COUNT, f"found {len(devices)}, torch sees {COUNT}"
    assert all(d.execution_supported and d.memory_gb for d in devices), devices
    assert "torch.rocm" in caps.frameworks and "torch.cuda" not in caps.frameworks
    return f"{devices[0].name}, {devices[0].memory_gb} GB"


@check("gpu_alias_means_rocm")
def _alias():
    assert glt.Trainer(accelerator="gpu").accelerator == "rocm"


for precision in ("32-true", "16-mixed", "bf16-mixed", "bf16-true"):

    def _precision(precision=precision):
        with tempfile.TemporaryDirectory() as root:
            t = train(root, precision=precision)
            loss = t.callback_metrics["train_loss"]
            assert loss == loss, "loss is NaN"
            return f"loss {loss:.4f}"

    check(f"train_{precision}")(_precision)


@check("checkpoint_and_exact_resume")
def _resume():
    with tempfile.TemporaryDirectory() as root:
        first = train(root)
        again = glt.Trainer(
            accelerator="rocm", max_epochs=3, log_every_n_steps=0, default_root_dir=root
        )
        again.fit(Net(), loader(), ckpt_path=first.checkpoint_path)
        assert again.global_step == first.global_step * 3 // 2


@check("hip_visible_devices_limits_a_job")
def _visible():
    code = "import torch; print(torch.cuda.device_count())"
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "1", out.stdout + out.stderr


DDP = """
    import sys, torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import gpulayer.pytorch as glt

    class Net(glt.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = nn.Linear(16, 1)
        def training_step(self, batch, batch_idx):
            return nn.functional.mse_loss(self.net(batch[0]), batch[1])
        def on_fit_end(self):
            g = self.all_gather(self.net.weight.detach().flatten())
            self.print("AGREE", bool(torch.allclose(g[0], g[1])))
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    if __name__ == "__main__":
        x = torch.randn(256, 16)
        strategy = glt.DDPStrategy(process_group_backend=sys.argv[1])
        t = glt.Trainer(accelerator="rocm", devices=2, max_epochs=1, log_every_n_steps=0,
                        strategy=strategy, enable_checkpointing=False)
        t.fit(Net(), DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=16))
"""

for backend in ("nccl", "gloo"):

    def _ddp(backend=backend):
        """nccl is RCCL under torch's name: AMD's pages do not say this works."""
        if COUNT < 2:
            return SKIP
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "ddp.py"
            script.write_text(textwrap.dedent(DDP))
            p = subprocess.run(
                [sys.executable, str(script), backend],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert p.returncode == 0, (p.stdout + p.stderr)[-1500:]
            assert "AGREE True" in p.stdout + p.stderr, (p.stdout + p.stderr)[-1500:]

    check(f"ddp_two_gpus_{backend}")(_ddp)


failed = [name for name, ok, _ in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} passed", flush=True)
if failed:
    print("failed:", ", ".join(failed), flush=True)
sys.exit(1 if failed else 0)
