"""Trainer on CUDA, run on a T4 VM that GPUlayer created through its Google Cloud adapter."""

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt

ACC = os.environ.get("ACC", "cuda")
print(
    "device",
    torch.cuda.get_device_name(0) if ACC == "cuda" else "cpu",
    "torch",
    torch.__version__,
    flush=True,
)


class Net(glt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self.net(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


torch.manual_seed(0)
x = torch.randn(65536, 64)
y = (x[:, :8].sum(1, keepdim=True)).sin()
train = DataLoader(TensorDataset(x[:57344], y[:57344]), batch_size=512, shuffle=True)
val = DataLoader(TensorDataset(x[57344:], y[57344:]), batch_size=1024)
for precision in ("32-true", "16-mixed") if ACC == "cuda" else ("32-true",):
    t = glt.Trainer(accelerator=ACC, devices=1, max_epochs=5, precision=precision)
    model = Net()
    t0 = time.perf_counter()
    t.fit(model, train, val)
    dt = time.perf_counter() - t0
    first, last = t.history[0], t.history[-1]
    after_first = sum(h["epoch_seconds"] for h in t.history[1:]) / (len(t.history) - 1)
    print(
        f"RESULT precision={precision} epochs={len(t.history)} seconds={dt:.2f} "
        f"train_loss {first['train_loss']:.4f}->{last['train_loss']:.4f} "
        f"val_loss={last['val_loss']:.4f} "
        f"epoch_seconds_after_first={after_first:.3f} ckpt={t.checkpoint_path}",
        flush=True,
    )
    assert (ACC != "cuda" or next(model.parameters()).is_cuda) and last["train_loss"] < first[
        "train_loss"
    ]
print("OK", flush=True)
