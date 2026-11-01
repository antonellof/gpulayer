"""Validate FSDP options, tensor parallelism and TensorRT export on GPU machines, from a
GPUlayer server's shell (one GPU per machine, NCCL across machines):

    python scripts/validate_parallel.py --case fsdp --nodes 2 [--ac] [--offload] [--sharding SHARD_GRAD_OP]
    python scripts/validate_parallel.py --case fsdp --nodes 2 --sharded --steps 100
    python scripts/validate_parallel.py --case fsdp --nodes 2 --sharded --resume DIR --steps 200
    python scripts/validate_parallel.py --case tp --nodes 2          # tensor parallel over 2 GPUs
    python scripts/validate_parallel.py --case single                # 1 GPU, the baseline for tp
    python scripts/validate_parallel.py --case tensorrt              # in this process, 1 GPU

Rank 0's numbers come back with the job's outputs; the launching process prints
`RESULT {...}` and `CHECK name: PASS|FAIL`.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["fsdp", "tp", "single", "tensorrt"], default="fsdp")
ap.add_argument("--nodes", type=int, default=2)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--ac", action="store_true", help="fsdp: activation checkpointing per Block")
ap.add_argument("--offload", action="store_true", help="fsdp: cpu_offload=True")
ap.add_argument("--sharding", default="FULL_SHARD")
ap.add_argument("--sharded", action="store_true", help="fsdp: state_dict_type='sharded'")
ap.add_argument("--resume", default=None, help="fsdp: resume from this checkpoint folder")
args = ap.parse_args()

WIDTH, BLOCKS, BATCH = 1024, 8, 512


def data() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16384, WIDTH, generator=g)
    y = (x[:, :32].sum(1) > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True, drop_last=True)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, 4 * WIDTH)
        self.fc2 = nn.Linear(4 * WIDTH, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, x):
        return self.norm(x + self.fc2(nn.functional.gelu(self.fc1(x))))


class Net(glt.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.Sequential(*[Block() for _ in range(BLOCKS)])
        self.head = nn.Linear(WIDTH, 2)

    def forward(self, x):
        return self.head(self.blocks(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class TP(Net):
    def configure_model(self):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        tp = self.device_mesh["tensor_parallel"]
        for block in self.blocks:
            parallelize_module(block, tp, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()})


class Report(glt.Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.t0 = 0.0
        self.t_step = 0
        self.trained = 0  # steps this run trained (fewer after a resume)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        self.trained += 1
        if step % 25 == 0 or self.trained == 1:
            self.losses.append(float(outputs["loss"]))  # rare reads
        if self.trained == 10:  # time after a warm-up
            torch.cuda.synchronize()
            self.t0, self.t_step = time.perf_counter(), step

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        out = {
            "losses": self.losses,
            "ms": (time.perf_counter() - self.t0)
            / max(1, trainer.global_step - self.t_step)
            * 1000,
            "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
            "world": trainer.world_size,
            "step": trainer.global_step,
            "trained": self.trained,
        }
        local = full = 0
        for p in pl_module.parameters():
            full += p.numel()
            local += p.to_local().numel() if hasattr(p, "to_local") else p.numel()
        out["local_param_fraction"] = local / full
        fc1 = pl_module.blocks[0].fc1.weight
        out["fc1_local_shape"] = list(
            fc1.to_local().shape if hasattr(fc1, "to_local") else fc1.shape
        )
        out["activation_checkpointing"] = any(
            "CheckpointWrapper" in type(m).__name__ for m in pl_module.modules()
        )
        out["params_on_cpu"] = any(
            (p.to_local() if hasattr(p, "to_local") else p).device.type == "cpu"
            for p in pl_module.parameters()
        )
        if trainer.is_global_zero:
            (trainer.runtime.output_dir() / "result.json").write_text(json.dumps(out))


def train() -> tuple[glt.Trainer, glt.Module]:
    report = Report()
    common = dict(accelerator="cuda", devices=1, max_steps=args.steps, max_epochs=1000,
                  log_every_n_steps=0, num_sanity_val_steps=0, callbacks=[report])  # fmt: skip
    if args.case == "fsdp":
        strategy = glt.FSDPStrategy(
            sharding_strategy=args.sharding, auto_wrap_policy={Block},
            activation_checkpointing_policy={Block} if args.ac else None,
            cpu_offload=args.offload, state_dict_type="sharded" if args.sharded else "full",
        )  # fmt: skip
        t = glt.Trainer(num_nodes=args.nodes, target="remote", strategy=strategy, **common)
        model: glt.Module = Net()
    elif args.case == "tp":
        strategy = glt.ModelParallelStrategy(
            data_parallel_size=1, tensor_parallel_size=args.nodes, save_distributed_checkpoint=False
        )
        t = glt.Trainer(num_nodes=args.nodes, target="remote", strategy=strategy, **common)
        model = TP()
    else:  # single GPU, same model and data: the baseline for tp
        t = glt.Trainer(num_nodes=1, target="remote", **common)
        model = Net()
    t.fit(model, data, ckpt_path=args.resume)
    return t, model


def check(label: str, checks: dict[str, bool]) -> None:
    for name, ok in checks.items():
        print(f"CHECK {label} {name}: {'PASS' if ok else 'FAIL'}")


def tensorrt() -> None:
    model = Net().cuda().eval()
    x = torch.randn(BATCH, WIDTH, device="cuda")
    with torch.no_grad():
        ref = model(x)
    trt = model.to_tensorrt("net.ep", input_sample=x)
    with torch.no_grad():
        out = trt(x)

    def timed(fn) -> float:
        with torch.no_grad():
            for _ in range(10):
                fn(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                fn(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 100 * 1000

    import torch_tensorrt

    diff = float((out.float() - ref.float()).abs().max())
    result = {"case": "tensorrt", "max_abs_diff": round(diff, 5),
              "ms_eager": round(timed(model), 3), "ms_tensorrt": round(timed(trt), 3),
              "torch_tensorrt": torch_tensorrt.__version__}  # fmt: skip
    print("RESULT " + json.dumps(result))
    import os

    check("tensorrt", {"outputs match eager (max diff < 1e-2)": diff < 1e-2,
                       "file saved": os.path.getsize("net.ep") > 0})  # fmt: skip


if args.case == "tensorrt":
    tensorrt()
else:
    trainer, model = train()
    if glt.is_launcher_process():
        r = json.loads(sorted(trainer.launch_result.output_dir.rglob("result.json"))[0].read_text())
        label = args.case + (f"-{args.sharding}" if args.case == "fsdp" else "")
        label += "-ac" if args.ac else ""
        label += "-offload" if args.offload else ""
        label += "-sharded" if args.sharded else ""
        label += "-resumed" if args.resume else ""
        label += f"/{args.nodes if args.case != 'single' else 1}"
        ckpt = trainer.checkpoint_path
        result = {
            "case": label, "world": r["world"], "step": r["step"],
            "peak_mb_rank0": round(r["peak_mb"]), "ms_per_step": round(r["ms"], 2),
            "losses": [round(v, 4) for v in r["losses"]], "trained_steps": r["trained"],
            "local_param_fraction": round(r["local_param_fraction"], 3),
            "fc1_local_shape": r["fc1_local_shape"], "activation_checkpointing": r["activation_checkpointing"],
            "params_on_cpu": r["params_on_cpu"], "checkpoint": str(ckpt) if ckpt else None,
            "checkpoint_is_folder": bool(ckpt and ckpt.is_dir()),
        }  # fmt: skip
        print("RESULT " + json.dumps(result))
        checks = {"loss falls": r["losses"][-1] < r["losses"][0]}
        fresh = Net()
        if ckpt is not None:
            state = trainer.checkpoint_io.load_checkpoint(ckpt)
            missing, unexpected = fresh.load_state_dict(state["state_dict"], strict=False)
            checks["full weights load from the checkpoint"] = not missing and not unexpected
        checks["trained weights came back"] = not torch.allclose(
            model.head.weight, Net().head.weight
        )
        if args.case == "fsdp" and args.sharding == "FULL_SHARD":
            checks["parameters sharded"] = r["local_param_fraction"] < 0.75
        if args.ac:
            checks["activation checkpointing applied"] = r["activation_checkpointing"]
        if args.offload:
            checks["parameters kept on the CPU"] = r["params_on_cpu"]
        if args.sharded:
            checks["checkpoint is a sharded folder"] = result["checkpoint_is_folder"]
        if args.resume:
            checks["resumed after the saved steps"] = (
                r["trained"] < args.steps and r["step"] == args.steps
            )
        if args.case == "tp":
            checks["fc1 split across GPUs"] = r["fc1_local_shape"] == [
                4 * WIDTH // args.nodes,
                WIDTH,
            ]
        check(label, checks)
