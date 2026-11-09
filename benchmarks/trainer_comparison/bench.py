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

if os.environ.get("BENCH_PROFILE"):  # profile every process of either trainer, same way
    import atexit
    import cProfile
    import pstats

    _prof = cProfile.Profile()
    _prof.enable()

    def _dump() -> None:
        _prof.disable()
        role = f"rank{os.environ['RANK']}" if "RANK" in os.environ else "main"
        with open(f"/tmp/prof-{sys.argv[1]}-{role}-{os.getpid()}.txt", "w") as f:
            pstats.Stats(_prof, stream=f).sort_stats("tottime").print_stats(18)

    atexit.register(_dump)
fw, case, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 1

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

CASES = {
    # name: (model kind, batch, accelerator, devices, precision)
    "mlp-cpu": ("mlp-small", 64, "cpu", 1, "32"),
    "cnn-cpu": ("cnn", 32, "cpu", 1, "32"),
    "mlp-mps": ("mlp", 256, "mps", 1, "32"),
    "cnn-mps": ("cnn", 128, "mps", 1, "32"),
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
    # Several machines with one GPU each, DDP over NCCL (multinode.py); batch per GPU.
    "cnn-gpu-nodes": ("cnn", 128, "cuda", 1, "16-mixed"),
    "mlp-gpu-nodes": ("mlp", 256, "cuda", 1, "32"),
    # Realistic models, bf16-mixed, AdamW: a GPT-2-size decoder (12 layers, width 768, 32k
    # vocabulary, 512 tokens) and ResNet-50 at 224x224. Batch per GPU. Add "-compile" for
    # torch.compile in both trainers.
    **{f"gpt-gpu{g}": ("gpt", 8, "cuda", g, "bf16-mixed") for g in (1, 2, 4, 8)},
    **{f"resnet50-gpu{g}": ("resnet50", 64, "cuda", g, "bf16-mixed") for g in (1, 2, 4, 8)},
}
COMPILE = case.endswith("-compile")
case = case.removesuffix("-compile")
kind, batch, accel, devices, precision = CASES[case]


class GPT(nn.Module):
    """A GPT-2-size decoder from torch's own layers (no extra package): 12 pre-norm blocks of
    width 768 with 12 heads, causal attention, 32k vocabulary, tied embeddings."""

    def __init__(self, vocab: int = 32768, width: int = 768, layers: int = 12, seq: int = 512):
        super().__init__()
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(seq, width)
        layer = nn.TransformerEncoderLayer(
            width, 12, 4 * width, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True,
        )  # fmt: skip
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(seq))

    def forward(self, x):
        h = self.tok(x) + self.pos.weight[: x.shape[1]]
        h = self.blocks(h, mask=self.mask[: x.shape[1], : x.shape[1]], is_causal=True)
        return self.head(self.norm(h))


def make_net():
    if kind == "gpt":
        net = GPT()
    elif kind == "resnet50":
        import torchvision

        net = torchvision.models.resnet50(num_classes=1000)
    else:
        net = make_small_net()
    return torch.compile(net) if COMPILE else net


def make_small_net():
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


class Bank(torch.utils.data.Dataset):
    """ResNet-50 images: 256 random 224x224 images, read in turn (all of them would not fit)."""

    def __init__(self, n: int) -> None:
        g = torch.Generator().manual_seed(0)
        self.n = n
        self.x = torch.randn(256, 3, 224, 224, generator=g)
        self.y = torch.randint(0, 1000, (256,), generator=g)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.x[i % 256], self.y[i % 256]


def loader():
    g = torch.Generator().manual_seed(0)
    if kind == "gpt":  # next-token prediction on random tokens: 513 per sample
        n = steps * batch * devices * nodes
        tokens = torch.randint(0, 32768, (n, 513), generator=g)
        return DataLoader(TensorDataset(tokens), batch_size=batch, shuffle=False)
    if kind == "resnet50":
        return DataLoader(Bank(steps * batch * devices * nodes), batch_size=batch, shuffle=False)
    n = steps * batch * devices * nodes
    shape = {"mlp": (512,), "mlp-small": (64,), "cnn": (3, 32, 32)}[kind]
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def step_loss(net, batch_):
    if kind == "gpt":
        tokens = batch_[0]
        logits = net(tokens[:, :-1])
        return nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
    x, y = batch_
    return nn.functional.cross_entropy(net(x), y)


def optimizer(params):
    if kind in ("gpt", "resnet50"):
        return torch.optim.AdamW(params, lr=1e-4)
    return torch.optim.SGD(params, lr=0.01, momentum=0.9)


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
    launch_options["env"] = {"OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 1) // devices))}

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
            return optimizer(self.parameters())

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
            return optimizer(self.parameters())

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
# GPUlayer takes the loader as a function, built only in the processes that train; Lightning
# takes the loader itself (its launching process becomes rank 0, so the data is used).
_data = loader if fw.startswith("gl") else loader()
_t_fit = time.perf_counter()
if os.environ.get("BENCH_TORCH_PROFILE"):  # GPU kernels, top by CUDA time (one process)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as _tp:
        trainer.fit(model, _data)
    print(
        _tp.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=14, max_name_column_width=60
        ),
        flush=True,
    )
else:
    trainer.fit(model, _data)
if "RANK" in os.environ or fw == "pl" or devices * nodes == 1:
    # Time inside fit() in a process that trained (not GPUlayer's launcher, which only waits).
    print(f"FIT {os.environ.get('RANK', '0')} {time.perf_counter() - _t_fit:.4f}", flush=True)
if torch.cuda.is_initialized():  # not in GPUlayer's launcher, which never used the GPU
    torch.cuda.synchronize()
if accel == "mps" and torch.backends.mps.is_available():
    torch.mps.synchronize()
if fw == "pl" and trainer.global_rank != 0:
    sys.exit(0)
print(f"ELAPSED {time.perf_counter() - T0:.4f}", flush=True)
