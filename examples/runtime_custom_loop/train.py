"""Keep your own training loop; let Runtime handle devices, DDP, sharding and precision.

python train.py                                     # this process
python train.py --devices 2                         # 2 processes on this machine (CPU)
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt

ap = argparse.ArgumentParser()
ap.add_argument("--devices", type=int, default=1)
ap.add_argument("--epochs", type=int, default=5)
args = ap.parse_args()

# Outside a job with more than one process, re-run this script on GpuLayer workers.
if args.devices > 1:
    glt.launch(devices=args.devices, accelerator="cpu")

rt = glt.Runtime(accelerator="auto" if args.devices == 1 else "cpu").setup()
glt.seed_everything(0)

x = torch.randn(4096, 16)
y = (x[:, :8].sum(1, keepdim=True) > 0).float()
loader = rt.dataloader(DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True))

model = rt.module(nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1)))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(args.epochs):
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    total, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = rt.to_device((xb, yb))
        with rt.autocast():
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        rt.backward(loss)
        rt.step(optimizer, model.parameters(), clip_val=1.0)
        optimizer.zero_grad()
        total += loss.item() * len(xb)
        correct += ((logits > 0).float() == yb).sum().item()
        seen += len(xb)
    loss_avg, acc = rt.mean(total / seen), rt.mean(correct / seen)
    if rt.is_global_zero:
        print(
            f"epoch {epoch}: loss={loss_avg:.4f} acc={acc:.3f} "
            f"({rt.accelerator}, world size {rt.world_size})",
            flush=True,
        )

if rt.is_global_zero:
    torch.save(model.state_dict(), rt.output_dir() / "model.pt")
rt.teardown()
