"""One training run: python bench.py {gl|gl-base|gl-noprefetch|gl-nobench|pl} CASE STEPS [NODES].
Same model, data, optimizer. The gl-* variants turn off device prefetch and/or cuDNN autotuning
(gl-base: both off; gl-noprefetch: prefetch=False; gl-prefetch: prefetch=True;
gl-nobench: benchmark=False).

NODES (default 1) is for the multi-node cases, run by multinode.py: GPUlayer submits the job
to the server in $GPULAYER_URL; Lightning is started on every node with torchrun."""

import os
import sys
import time

T0 = time.perf_counter()
fw, case, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 1

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

CASES = {
    # name: (model kind, batch, accelerator, devices, precision)
    "mlp-cpu": ("mlp-small", 64, "cpu", 1, "32"),
    "mlp-fp32": ("mlp", 256, "cuda", 1, "32"),
    "mlp-fp16": ("mlp", 256, "cuda", 1, "16-mixed"),
    "cnn-fp16": ("cnn", 128, "cuda", 1, "16-mixed"),
    "ddp-cpu-2": ("mlp-small", 64, "cpu", 2, "32"),
    "ddp-nodes": ("mlp-small", 64, "cpu", 2, "32"),
    "ddp-nodes-big": ("mlp", 64, "cpu", 2, "32"),
    # Several GPUs in one machine, DDP over NCCL; the batch is per GPU, so each step trains on
    # batch x GPUs samples (weak scaling).
    **{f"cnn-gpu{g}": ("cnn", 128, "cuda", g, "16-mixed") for g in (1, 2, 4, 8)},
    **{f"mlp-gpu{g}": ("mlp", 256, "cuda", g, "32") for g in (1, 2, 4, 8)},
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
    n = steps * batch * devices * nodes
    shape = {"mlp": (512,), "mlp-small": (64,), "cnn": (3, 32, 32)}[kind]
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def step_loss(net, batch_):
    x, y = batch_
    return nn.functional.cross_entropy(net(x), y)


VARIANTS: dict[str, dict] = {
    "gl": {},
    "gl-base": {"prefetch": False, "benchmark": False},
    "gl-noprefetch": {"prefetch": False},
    "gl-prefetch": {"prefetch": True},
    "gl-bench": {"benchmark": True},
    "gl-omp": {},  # OMP_NUM_THREADS = all cores in each rank (GPUlayer sets 1 by default)
    "gl-nobench": {"benchmark": False},
}

launch_options: dict | None = {"quiet": True} if devices * nodes > 1 else None
if fw == "gl-omp" and launch_options is not None:
    launch_options["env"] = {"OMP_NUM_THREADS": str(os.cpu_count())}

if fw in VARIANTS:
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
        num_nodes=nodes,
        precision={"32": "32-true"}.get(precision, precision),
        max_epochs=1,
        enable_checkpointing=False,
        log_every_n_steps=10_000,
        launch_options=launch_options,
        **VARIANTS[fw],
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
        num_nodes=nodes,
        strategy="ddp" if devices * nodes > 1 else "auto",
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
if torch.cuda.is_initialized():  # not in GPUlayer's launcher, which never used the GPU
    torch.cuda.synchronize()
if fw == "pl" and trainer.global_rank != 0:
    sys.exit(0)
print(f"ELAPSED {time.perf_counter() - T0:.4f}", flush=True)
