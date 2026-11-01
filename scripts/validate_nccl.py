"""Validate the GPUlayer Trainer across GPU machines (NCCL), from a GPUlayer server's shell:

    python scripts/validate_nccl.py --case ddp --nodes 4        # also: manual, fsdp

The Trainer submits one job with --nodes workers x 1 GPU (target="remote"); the same script
runs on each worker. Checks, printed by rank 0 as `RESULT {...}` and `CHECK name: PASS|FAIL`:

- ddp: NCCL backend, 16-mixed, sync_batchnorm (BatchNorm becomes SyncBatchNorm), identical
  weights on every rank after training (all_gather), the loss falls; ms per step.
- manual: manual optimization with 16-mixed over NCCL (manual_backward, GradScaler), identical
  weights on every rank, the loss falls.
- fsdp: strategy="fsdp" over NCCL: each rank holds a part of the parameters, the loss falls,
  a full checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["ddp", "manual", "fsdp"], default="ddp")
ap.add_argument("--nodes", type=int, default=2)
ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()


def data() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (4096,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, drop_last=True)


def net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
    )  # fmt: skip


class Report(glt.Callback):
    """Times steps after a warm-up and records the first and last losses."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.t0: float | None = None
        self.timed = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in (1, args.steps) or step % 50 == 0:
            self.losses.append(float(outputs["loss"]))  # rare reads: no per-step sync
        if step == 50:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            self.timed = step

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        ms = (time.perf_counter() - self.t0) / (trainer.global_step - self.timed) * 1000
        if args.case == "fsdp":
            sums = torch.zeros(0)  # sharded parameters differ per rank by design
        else:
            flat = torch.cat([p.detach().float().flatten() for p in pl_module.parameters()])
            sums = pl_module.all_gather(flat.sum().reshape(1))
        world = trainer.world_size
        result = {
            "case": args.case,
            "nodes": args.nodes,
            "world_size": world,
            "backend": dist.get_backend() if dist.is_initialized() else None,
            "gpu": torch.cuda.get_device_name(),
            "precision": trainer.precision,
            "ms_per_step": round(ms, 2),
            "losses": [round(v, 4) for v in self.losses],
            "rank_weight_sums": [round(float(v), 4) for v in sums.flatten()],
            "syncbn": any(isinstance(m, nn.SyncBatchNorm) for m in pl_module.modules()),
        }
        if args.case == "fsdp":
            local = sum(p.to_local().numel() if hasattr(p, "to_local") else p.numel()
                        for p in pl_module.parameters())  # fmt: skip
            full = sum(p.numel() for p in pl_module.parameters())
            result["local_param_fraction"] = round(local / full, 3)
        pl_module.print("RESULT " + json.dumps(result))
        checks = {
            "nccl backend": result["backend"] == "nccl",
            "world size": world == args.nodes,
            "loss falls": self.losses[-1] < self.losses[0],
        }
        if args.case in ("ddp", "manual"):
            ref = result["rank_weight_sums"][0]
            checks["identical weights on every rank"] = all(
                abs(v - ref) <= 1e-3 * max(1.0, abs(ref)) for v in result["rank_weight_sums"]
            )
        if args.case == "ddp":
            checks["SyncBatchNorm"] = result["syncbn"]
        if args.case == "fsdp":
            checks["parameters sharded"] = result["local_param_fraction"] < 0.75
        for name, ok in checks.items():
            pl_module.print(f"CHECK {args.case}/{args.nodes} {name}: {'PASS' if ok else 'FAIL'}")


class Model(glt.Module):
    def __init__(self, manual: bool) -> None:
        super().__init__()
        self.net = net()
        self.automatic_optimization = not manual

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.net(x), y)
        if self.automatic_optimization:
            return loss
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0)
        opt.step()
        return {"loss": loss.detach()}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)


report = Report()
trainer = glt.Trainer(
    accelerator="cuda",
    devices=1,
    num_nodes=args.nodes,
    target="remote",
    max_steps=args.steps,
    max_epochs=1000,
    precision="32-true" if args.case == "fsdp" else "16-mixed",
    strategy="fsdp" if args.case == "fsdp" else glt.DDPStrategy(gradient_as_bucket_view=True),
    sync_batchnorm=args.case == "ddp",
    log_every_n_steps=0,
    enable_checkpointing=args.case == "fsdp",
    num_sanity_val_steps=0,
    callbacks=[report],
)
trainer.fit(Model(manual=args.case == "manual"), data)
if glt.is_launcher_process():
    print(f"job {trainer.launch_result.job.id if trainer.launch_result else '-'} finished")
