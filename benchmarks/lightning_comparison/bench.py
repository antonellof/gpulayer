"""One training run: python bench.py {gl|pl} CASE STEPS. Same model, data, optimizer for both."""

import sys
import time

T0 = time.perf_counter()
fw, case, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

CASES = {
    # name: (model kind, batch, accelerator, devices, precision)
    "mlp-fp32": ("mlp", 256, "cuda", 1, "32"),
    "mlp-fp16": ("mlp", 256, "cuda", 1, "16-mixed"),
    "cnn-fp16": ("cnn", 128, "cuda", 1, "16-mixed"),
    "ddp-cpu-2": ("mlp-small", 64, "cpu", 2, "32"),
}
kind, batch, accel, devices, precision = CASES[case]


def make_net():
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 10)
        )
    if kind == "mlp-small":
        return nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10),
    )


def loader():
    g = torch.Generator().manual_seed(0)
    n = steps * batch * devices
    shape = {"mlp": (512,), "mlp-small": (64,), "cnn": (3, 32, 32)}[kind]
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def step_loss(net, batch_):
    x, y = batch_
    return nn.functional.cross_entropy(net(x), y)


if fw == "gl":
    import gpulayer.pytorch as glt

    class M(glt.Module):
        def __init__(self):
            super().__init__()
            self.net = make_net()

        def training_step(self, b, i):
            loss = step_loss(self.net, b)
            self.log("loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    trainer = glt.Trainer(
        accelerator=accel,
        devices=devices,
        precision={"32": "32-true"}.get(precision, precision),
        max_epochs=1,
        enable_checkpointing=False,
        log_every_n_steps=10_000,
        launch_options={"quiet": True} if devices > 1 else None,
    )
else:
    import lightning as L

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = make_net()

        def training_step(self, b, i):
            loss = step_loss(self.net, b)
            self.log("loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    trainer = L.Trainer(
        accelerator=accel,
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        precision=precision,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

torch.manual_seed(0)
model = M()
trainer.fit(model, loader())
if torch.cuda.is_available():
    torch.cuda.synchronize()
if fw == "pl" and trainer.global_rank != 0:
    sys.exit(0)
print(f"ELAPSED {time.perf_counter() - T0:.4f}", flush=True)
