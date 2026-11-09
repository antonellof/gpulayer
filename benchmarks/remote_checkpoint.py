"""How long training waits for checkpoints saved to a remote store, with and without
AsyncCheckpointIO:

    python benchmarks/remote_checkpoint.py s3://bucket/prefix [--repeat 2]

Trains a 50 MB model (two 2048x3072 layers) for --steps steps (60) on the CPU and saves a
checkpoint every steps/3, three per run, to ``<url>/<run>/<mode>``. Modes run alternately: no
checkpoints, saving in the loop (the default), and ``plugins=[AsyncCheckpointIO()]``. Prints
fit seconds per mode, then checks that the async run's checkpoints are all in the store and
load. Removes what it wrote.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
import uuid

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.pytorch.storage import as_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--steps", type=int, default=60, help="a checkpoint every steps/3")
args = ap.parse_args()
ROOT = f"{args.url.rstrip('/')}/ckpt-bench-{uuid.uuid4().hex[:6]}"


class Net(glt.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(2048, 3072), nn.ReLU(), nn.Linear(3072, 2048))

    def training_step(self, batch, batch_idx):
        return self.net(batch[0]).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4)  # no optimizer state to save


def fit(mode: str, run: int) -> float:
    data = DataLoader(TensorDataset(torch.randn(args.steps * 8, 2048)), batch_size=8)
    kw: dict = dict(accelerator="cpu", max_epochs=1, log_every_n_steps=0, logger=False,
                    enable_progress_bar=False, default_root_dir=f"{ROOT}/{mode}-{run}")  # fmt: skip
    if mode == "none":
        kw["enable_checkpointing"] = False
    else:
        ckpt = glt.ModelCheckpoint(
            dirpath=f"{ROOT}/{mode}-{run}/ckpts", every_n_train_steps=args.steps // 3, save_top_k=-1
        )
        kw["callbacks"] = [ckpt]
    if mode == "async":
        kw["plugins"] = [glt.AsyncCheckpointIO()]
    t = glt.Trainer(**kw)
    t0 = time.perf_counter()
    t.fit(Net(), data)  # fit returns after pending async writes are done
    return time.perf_counter() - t0


times: dict[str, list[float]] = {"none": [], "sync": [], "async": []}
for run in range(args.repeat):
    for mode in times:
        times[mode].append(fit(mode, run))
        print(f"run {run} {mode:5} {times[mode][-1]:.1f}s", flush=True)
for mode, values in times.items():
    print(f"RESULT {mode:5} median {statistics.median(values):.1f}s over {len(values)} runs")

files = sorted(p.name for p in as_path(f"{ROOT}/async-0/ckpts").iterdir())
last = as_path(f"{ROOT}/async-0/ckpts") / files[-1]
with last.open("rb") as f:
    state = torch.load(f, map_location="cpu", weights_only=False)
ok = len(files) == 3 and state["global_step"] == args.steps
print(f"CHECK async checkpoints in the store: {'PASS' if ok else 'FAIL'} {files}")

from fsspec.core import url_to_fs  # noqa: E402

fs, path = url_to_fs(ROOT)
fs.rm(path, recursive=True)
print(f"removed {ROOT}")
