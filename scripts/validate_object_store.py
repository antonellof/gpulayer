"""Validate checkpoints and logs in a real object store, through fsspec:

    python scripts/validate_object_store.py s3://bucket/prefix        # or gs://, az://, hf://...

Needs `gpulayer[remote]` and the store's fsspec package (s3fs, gcsfs, adlfs, huggingface_hub),
with credentials from the environment. Writes under ``<url>/<run id>/`` and removes it at the
end (``--keep`` keeps it). CPU only: what is checked is the store, not the GPU. Prints `CHECK`
lines:

1. fit           `default_root_dir`, `ModelCheckpoint(dirpath=...)` and `CSVLogger(save_dir=...)`
                 in the store: the best two and `last.ckpt` are there, and the CSV log.
2. resume        `fit(ckpt_path=<url>/last.ckpt)` continues from its step; so does `"last"`.
3. load          `Module.load_from_checkpoint(<url>)` gives the trained weights.
4. save          `trainer.save_checkpoint(<url>)` writes a file that loads back.
5. ddp           two processes (gloo, a local job) checkpoint into the store: one file per save.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
import uuid

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.pytorch.storage import as_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url", help="a folder in the store, e.g. s3://bucket/prefix")
ap.add_argument("--keep", action="store_true", help="keep what was written")
args = ap.parse_args()
ROOT = f"{args.url.rstrip('/')}/{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    failed += not ok
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)


def loaders() -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 8, generator=g)
    y = x @ torch.randn(8, 1, generator=g)
    data = TensorDataset(x, y)
    return DataLoader(data, batch_size=32, shuffle=True), DataLoader(data, batch_size=64)


class Net(glt.Module):
    def __init__(self, lr: float = 0.05) -> None:
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.net = nn.Linear(8, 1)

    def training_step(self, batch, batch_idx):
        return nn.functional.mse_loss(self.net(batch[0]), batch[1])

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", nn.functional.mse_loss(self.net(batch[0]), batch[1]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def trainer(epochs: int, **kw) -> glt.Trainer:
    return glt.Trainer(
        accelerator="cpu", max_epochs=epochs, default_root_dir=ROOT, log_every_n_steps=1,
        num_sanity_val_steps=0, enable_progress_bar=False, **kw,
    )  # fmt: skip


def names(folder) -> list[str]:
    return sorted(p.name for p in as_path(folder).iterdir()) if as_path(folder).exists() else []


DDP = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

class Net(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Linear(8, 1)
    def training_step(self, batch, batch_idx):
        return nn.functional.mse_loss(self.net(batch[0]), batch[1])
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)

x = torch.randn(256, 8)
t = glt.Trainer(accelerator="cpu", devices=2, max_epochs=2, log_every_n_steps=0,
                default_root_dir="ROOT/ddp", enable_progress_bar=False,
                callbacks=[glt.ModelCheckpoint(dirpath="ROOT/ddp/ckpts", save_last=True,
                                               every_n_epochs=1, save_top_k=-1)])
t.fit(Net(), DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=32))
print("DDP_STEP", t.global_step, flush=True)
"""


def main() -> int:
    print(f"store root {ROOT}", flush=True)
    ckpts = f"{ROOT}/ckpts"
    try:
        t0 = time.perf_counter()
        mc = glt.ModelCheckpoint(dirpath=ckpts, monitor="val_loss", save_top_k=2, save_last=True)
        t = trainer(3, callbacks=[mc], logger=glt.CSVLogger(save_dir=ROOT, name="csv"))
        model = Net()
        t.fit(model, *loaders())
        took = time.perf_counter() - t0
        files = names(ckpts)
        logs = [p for p in as_path(f"{ROOT}/csv").rglob("metrics.csv")]
        check(
            "fit", "last.ckpt" in files and len([f for f in files if f != "last.ckpt"]) == 2
            and bool(logs) and str(mc.best_model_path).startswith(ROOT),
            f"{files}, csv {[str(p)[len(ROOT):] for p in logs]}, {took:.1f}s",
        )  # fmt: skip

        t2 = trainer(5, enable_checkpointing=False)  # "last" below must find ckpts/last.ckpt
        t2.fit(Net(), *loaders(), ckpt_path=f"{ckpts}/last.ckpt")
        t3 = trainer(4, callbacks=[glt.ModelCheckpoint(dirpath=ckpts, save_last=True)])
        t3.fit(Net(), *loaders(), ckpt_path="last")
        check(
            "resume", t2.global_step == 40 and t3.global_step == 32,
            f"from step {t.global_step}: url to {t2.global_step}, 'last' to {t3.global_step}",
        )  # fmt: skip

        best = Net.load_from_checkpoint(mc.best_model_path)
        with as_path(mc.best_model_path).open("rb") as f:
            raw = torch.load(f, map_location="cpu", weights_only=False)
        same = torch.equal(best.net.weight, raw["state_dict"]["net.weight"])
        check("load", best.hparams.lr == 0.05 and same, str(mc.best_model_path)[len(ROOT) :])

        manual = f"{ROOT}/manual/model.ckpt"
        t.save_checkpoint(manual)
        with as_path(manual).open("rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)
        check(
            "save", torch.equal(state["state_dict"]["net.weight"], model.net.weight),
            f"{as_path(manual).stat().st_size} bytes",
        )  # fmt: skip
    except Exception:
        check("store", False, traceback.format_exc()[-2500:])

    try:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "ddp.py")
            with open(script, "w") as f:
                f.write(DDP.replace("ROOT", ROOT))
            env = {**os.environ, "GPULAYER_HOME": os.path.join(tmp, "home")}
            p = subprocess.run([sys.executable, script], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=600)  # fmt: skip
        files = names(f"{ROOT}/ddp/ckpts")
        ok = p.returncode == 0 and "DDP_STEP 8" in p.stdout and "last.ckpt" in files
        detail = f"{files}" if ok else (p.stdout[-1500:] + p.stderr[-1500:])
        check("ddp", ok and len(files) == 3, detail)
    except Exception:
        check("ddp", False, traceback.format_exc()[-2500:])

    if not args.keep:
        from fsspec.core import url_to_fs

        fs, path = url_to_fs(ROOT)
        fs.rm(path, recursive=True)
        print(f"removed {ROOT}")
    print(f"SUMMARY {'all passed' if not failed else f'{failed} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
