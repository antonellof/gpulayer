"""Validate the GPUlayer Trainer with DeepSpeed on GPU machines, from a GPUlayer server's shell:

    python scripts/validate_deepspeed.py --stage 2 --nodes 2          # also --offload
    python scripts/validate_deepspeed.py --case bnb                   # bitsandbytes, 1 GPU

The Trainer submits one job with --nodes workers x 1 GPU (target="remote"), so DeepSpeed gets
the rank environment GPUlayer sets. Rank 0 prints `RESULT {...}` and `CHECK name: PASS|FAIL`:

- deepspeed: ZeRO stage 1, 2 or 3 (optionally offloading optimizer state, and with stage 3
  parameters, to CPU memory) in 16-mixed: the loss falls; the checkpoint holds full weights
  that load into a fresh model; peak GPU memory and ms per step are reported.
- bnb: BitsandbytesPrecision("nf4") and ("int8") on one GPU: nn.Linear layers become
  bitsandbytes layers, the trainable head learns, and GPU memory falls against 32-true.
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
ap.add_argument("--case", choices=["deepspeed", "bnb"], default="deepspeed")
ap.add_argument("--stage", type=int, default=2)
ap.add_argument("--offload", action="store_true")
ap.add_argument("--nodes", type=int, default=1)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--mode", default="nf4", help="bnb case: nf4, nf4-dq, fp4, int8")
args = ap.parse_args()

WIDTH = 2048  # 4 hidden layers of 2048: about 25M parameters, so ZeRO's savings show


def data() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8192, 512, generator=g)
    y = (x[:, :16].sum(1) > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, drop_last=True)


class Net(glt.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        layers: list[nn.Module] = [nn.Linear(512, WIDTH), nn.GELU()]
        for _ in range(4):
            layers += [nn.Linear(WIDTH, WIDTH), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(WIDTH, 2)

    def forward(self, x):
        return self.head(self.body(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if args.case == "bnb":
            return torch.optim.SGD(params, lr=0.01)  # fp16 head: AdamW's eps would underflow
        if args.offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam  # needed by ZeRO offload

            return DeepSpeedCPUAdam(params, lr=1e-4)
        return torch.optim.AdamW(params, lr=1e-4)


class Report(glt.Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.t0 = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in (1, args.steps) or step % 50 == 0:
            self.losses.append(float(outputs["loss"]))  # rare reads: no per-step sync
        if step == 20:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.t0) / (trainer.global_step - 20) * 1000
        self.peak_mb = torch.cuda.max_memory_allocated() / 2**20


def run_deepspeed() -> None:
    report = Report()
    strategy = glt.DeepSpeedStrategy(
        stage=args.stage,
        offload_optimizer=args.offload,
        offload_parameters=args.offload and args.stage == 3,
    )
    trainer = glt.Trainer(
        accelerator="cuda", devices=1, num_nodes=args.nodes, target="remote",
        strategy=strategy, precision="16-mixed", max_steps=args.steps, max_epochs=1000,
        log_every_n_steps=0, num_sanity_val_steps=0, callbacks=[report],
    )  # fmt: skip
    model = Net()
    trainer.fit(model, data)
    if not glt.is_launcher_process():
        if trainer.is_global_zero:  # rank 0's numbers come back with the job's outputs
            (trainer.runtime.output_dir() / "ds_result.json").write_text(
                json.dumps(
                    {
                        "losses": report.losses,
                        "ms": report.ms,
                        "peak_mb": report.peak_mb,
                        "world": trainer.world_size,
                    }
                )  # fmt: skip
            )
        return
    found = sorted(trainer.launch_result.output_dir.rglob("ds_result.json"))
    r = json.loads(found[0].read_text())
    ckpt = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=False)
    fresh = Net()
    missing, unexpected = fresh.load_state_dict(
        {k: v.float() for k, v in ckpt["state_dict"].items()}, strict=False
    )
    full = sum(p.numel() for p in fresh.parameters())
    in_ckpt = sum(v.numel() for v in ckpt["state_dict"].values())
    label = f"stage{args.stage}{'-offload' if args.offload else ''}/{args.nodes}"
    result = {
        "case": label, "nodes": args.nodes, "world_size": r["world"],
        "gpu_peak_mb_rank0": round(r["peak_mb"]), "ms_per_step": round(r["ms"], 2),
        "losses": [round(v, 4) for v in r["losses"]], "params": full,
        "checkpoint_params": in_ckpt, "gpulayer_job": trainer.launch_result.job.id,
    }  # fmt: skip
    print("RESULT " + json.dumps(result))
    checks = {
        "world size": r["world"] == args.nodes,
        "loss falls": r["losses"][-1] < r["losses"][0],
        "full weights in the checkpoint": in_ckpt == full and not missing and not unexpected,
        "trained weights came back": torch.allclose(
            model.head.weight.float(), fresh.head.weight, atol=1e-2
        ),
    }
    for name, ok in checks.items():
        print(f"CHECK {label} {name}: {'PASS' if ok else 'FAIL'}")


def run_bnb() -> None:
    def fit(plugins):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        report = Report()
        model = Net()
        t = glt.Trainer(accelerator="cuda", plugins=plugins, max_steps=args.steps,
                        max_epochs=1000, log_every_n_steps=0, num_sanity_val_steps=0,
                        enable_checkpointing=False, callbacks=[report])  # fmt: skip
        t.fit(model, data())
        return model, report

    _, base = fit(None)
    plugin = glt.BitsandbytesPrecision(args.mode, dtype=torch.float16, ignore_modules={"head"})
    head_before = Net().head.weight.detach().clone()
    model, q = fit([plugin])
    import bitsandbytes as bnb

    # Quantization fidelity: the same untrained weights, quantized and not, on one batch.
    x = torch.randn(256, 512, generator=torch.Generator().manual_seed(1)).cuda()
    ref = Net().cuda().eval()
    quant = Net()
    plugin.convert_module(quant)
    quant = quant.cuda().eval()
    with torch.no_grad():
        a = ref.body(x).float().flatten()
        b = quant.body(x.half()).float().flatten()
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    kinds = {type(m).__name__ for m in model.body if not isinstance(m, nn.GELU)}
    moved = float((model.head.weight.detach().float().cpu() - head_before).abs().max())
    result = {
        "case": f"bnb-{args.mode}", "layers": sorted(kinds),
        "peak_mb_32true": round(base.peak_mb), "peak_mb_quantized": round(q.peak_mb),
        "ms_per_step_32true": round(base.ms, 2), "ms_per_step_quantized": round(q.ms, 2),
        "body_output_cosine_vs_fp32": round(cosine, 4), "head_max_change": round(moved, 5),
        "bitsandbytes": bnb.__version__,
    }  # fmt: skip
    print("RESULT " + json.dumps(result))
    checks = {
        "linear layers quantized": kinds <= {"Linear4bit", "Linear8bitLt"},
        "outputs match fp32 (cosine > 0.95)": cosine > 0.95,
        "trainable head trained": moved > 0,
        "less GPU memory than 32-true": q.peak_mb < base.peak_mb,
    }
    for name, ok in checks.items():
        print(f"CHECK bnb-{args.mode} {name}: {'PASS' if ok else 'FAIL'}")


if args.case == "deepspeed":
    run_deepspeed()
else:
    run_bnb()
