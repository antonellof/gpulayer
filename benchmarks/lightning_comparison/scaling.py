"""One fixed training job, to see how it speeds up with more machines (strong scaling).

    python scaling.py {gl|pl} NODES DEVICES [EPOCHS]

A small CNN on 32,768 synthetic 32x32 images with learnable labels, global batch 256 split
over NODES x DEVICES processes (DDP, gloo, one CPU thread per process). GPUlayer: run here,
`Trainer(num_nodes=NODES, devices=DEVICES)` submits one job to $GPULAYER_URL. Lightning: run
on every node under torchrun (scaling_run.py does that). Rank 0 prints FIT seconds (inside
fit) and the final training loss.
"""

import os
import sys
import time

T0 = time.perf_counter()
fw, nodes, devices = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 3

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

N, GLOBAL_BATCH = 32768, 256
world = nodes * devices
torch.set_num_threads(1)


def make_net() -> nn.Module:
    def block(i: int, o: int) -> list[nn.Module]:
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]

    return nn.Sequential(
        *block(3, 32),
        *block(32, 32),
        nn.MaxPool2d(2),
        *block(32, 64),
        *block(64, 64),
        nn.MaxPool2d(2),
        *block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10),
    )


def data() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, 3, 32, 32, generator=g)
    # Learnable labels: a fixed random projection of a downsampled image.
    proj = torch.randn(3 * 8 * 8, 10, generator=g)
    y = (nn.functional.avg_pool2d(x, 4).flatten(1) @ proj).argmax(1)
    return DataLoader(TensorDataset(x, y), batch_size=GLOBAL_BATCH // world, shuffle=True)


def step_loss(net: nn.Module, batch) -> torch.Tensor:
    x, y = batch
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
            return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)

    trainer = glt.Trainer(
        accelerator="cpu",
        devices=devices,
        num_nodes=nodes,
        max_epochs=epochs,
        enable_checkpointing=False,
        log_every_n_steps=0,
        launch_options={
            "quiet": True,
            "timeout_seconds": 3600,
            # scaling_run.py pins the job to the machines Lightning runs on, for a fair match.
            **(
                {"worker_ids": os.environ["GL_WORKERS"].split(",")}
                if os.environ.get("GL_WORKERS")
                else {}
            ),
        },
    )
else:
    import lightning as L

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = make_net()

        def training_step(self, b, i):
            loss = step_loss(self.net, b)
            self.log("loss", loss, on_epoch=True, sync_dist=False)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=devices,
        num_nodes=nodes,
        strategy="ddp",
        max_epochs=epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

torch.manual_seed(0)
model = M()
t_fit = time.perf_counter()
trainer.fit(model, data())
fit_s = time.perf_counter() - t_fit
if getattr(trainer, "global_rank", 0) == 0 and getattr(trainer, "is_global_zero", True):
    loss = trainer.callback_metrics.get("loss", trainer.callback_metrics.get("loss_epoch"))
    loss = float("nan") if loss is None else loss
    print(
        f"FIT {fit_s:.2f} LOSS {float(loss):.4f} TOTAL {time.perf_counter() - T0:.2f}", flush=True
    )
