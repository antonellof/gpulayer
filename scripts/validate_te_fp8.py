"""Validate `TransformerEnginePrecision` (FP8) on an NVIDIA GPU with FP8 (Ada, Hopper or newer):

    python scripts/validate_te_fp8.py

Needs `transformer_engine` (NVIDIA's PyTorch containers include it). Prints `CHECK` lines:

1. layers     `nn.Linear` and `nn.LayerNorm` became Transformer Engine layers in bfloat16, and a
              layer whose sizes are not multiples of 16 was kept.
2. fp8        FP8 was on inside `training_step`'s forward pass.
3. trains     the loss falls, and ends close to the same model trained in bf16-true.
4. checkpoint a checkpoint of the FP8 model loads into a new one and gives the same outputs
              (Transformer Engine 2.x refuses it first: NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt
from gpulayer.pytorch.plugins import TransformerEnginePrecision

failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    failed += not ok
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)


def data() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 256, generator=g)
    y = x[:, :32] @ torch.randn(32, 16, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=128, shuffle=False)


class Net(glt.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 16)
        )
        self.odd = nn.Linear(16, 15)  # 15 is not a multiple of 16: kept as nn.Linear
        self.losses: list[torch.Tensor] = []
        self.fp8_seen: list[bool] = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        out = self(batch[0])
        if batch_idx == 0:
            self.fp8_seen.append(FP8GlobalStateManager.is_fp8_enabled())
        loss = nn.functional.mse_loss(out.float(), batch[1])
        self.losses.append(loss.detach())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA GPU")
        return 1
    import transformer_engine
    import transformer_engine.pytorch as te

    cap = torch.cuda.get_device_capability()
    print(
        f"torch {torch.__version__}, transformer_engine {transformer_engine.__version__}, "
        f"{torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            model = Net()
            t = glt.Trainer(
                accelerator="cuda", devices=1, max_epochs=3, plugins=TransformerEnginePrecision(),
                default_root_dir=root, log_every_n_steps=0, enable_checkpointing=False,
            )  # fmt: skip
            t.fit(model, data())
            lin, ln = model.net[0], model.net[1]
            check(
                "layers",
                isinstance(lin, te.Linear) and isinstance(ln, te.LayerNorm)
                and lin.weight.dtype == torch.bfloat16 and type(model.odd) is nn.Linear,
                f"{type(lin).__name__} {lin.weight.dtype}, {type(ln).__name__}, "
                f"odd {type(model.odd).__name__}",
            )  # fmt: skip
            check("fp8", all(model.fp8_seen) and bool(model.fp8_seen), f"{model.fp8_seen}")

            ref = Net()
            glt.Trainer(
                accelerator="cuda", devices=1, max_epochs=3, precision="bf16-true",
                default_root_dir=root, log_every_n_steps=0, enable_checkpointing=False,
            ).fit(ref, data())  # fmt: skip
            first, last = model.losses[0].item(), model.losses[-1].item()
            ref_last = ref.losses[-1].item()
            check(
                "trains", last < 0.5 * first and abs(last - ref_last) < 0.25 * max(ref_last, 1e-3),
                f"fp8 loss {first:.4f} -> {last:.4f}, bf16-true ends at {ref_last:.4f}",
            )  # fmt: skip

            path = root / "fp8.ckpt"
            t.save_checkpoint(path)
            again = Net()
            TransformerEnginePrecision().convert_module(again)
            state = torch.load(path, map_location="cuda", weights_only=False)["state_dict"]
            # Transformer Engine 2.x refuses its pickled FP8 scaling state unless told the
            # checkpoint is trusted; this one was written above.
            try:
                again.load_state_dict(state)
                check("checkpoint_refused", True, "this Transformer Engine loads it as is")
            except RuntimeError as e:
                check("checkpoint_refused", "UNSAFE_PICKLE" in str(e), str(e)[:120])
                os.environ["NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE"] = "1"
                again.load_state_dict(state)
            again.cuda().eval()
            model.cuda().eval()  # fit moves the weights back to the CPU, as Lightning
            x = next(iter(data()))[0][:32].cuda().to(torch.bfloat16)
            with torch.no_grad():
                same = torch.equal(model(x), again(x))
            check("checkpoint", same, f"{len(state)} tensors, same outputs {same}")
        except Exception:
            check("te_fp8", False, traceback.format_exc()[-2500:])
    print(f"SUMMARY {'all passed' if not failed else f'{failed} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
