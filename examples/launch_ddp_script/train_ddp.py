"""A plain PyTorch DDP script made portable with two lines.

Everything below the `launch()` call is ordinary torchrun-style DDP code. `launch()`
submits this script to GPUlayer workers and returns immediately when it is already
running inside a job.

    python train_ddp.py                              # 1 node x 2 processes on this machine
    python train_ddp.py --nodes 2 --devices 2        # 2 workers x 2 processes
    python train_ddp.py --accelerator cuda --devices 4 --target remote
"""

import argparse
import os

import gpulayer.pytorch as glt

ap = argparse.ArgumentParser()
ap.add_argument("--nodes", type=int, default=1)
ap.add_argument("--devices", type=int, default=2)
ap.add_argument("--accelerator", default="cpu")
ap.add_argument("--target", default="local")
args = ap.parse_args()
glt.launch(nodes=args.nodes, devices=args.devices, accelerator=args.accelerator, target=args.target)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

backend = os.environ.get("GPULAYER_COMM_BACKEND", "gloo")
dist.init_process_group(backend)
rank, world = dist.get_rank(), dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if backend == "nccl":
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

torch.manual_seed(0)  # identical synthetic dataset on every rank; the sampler shards it
x = torch.randn(8192, 8)
true_w = torch.arange(1.0, 9.0).unsqueeze(1)
y = x @ true_w + 0.1 * torch.randn(8192, 1)
data = TensorDataset(x, y)
sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=0)
loader = DataLoader(data, batch_size=256, sampler=sampler)

model = DDP(nn.Linear(8, 1).to(device), device_ids=[local_rank] if backend == "nccl" else None)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
epochs = int(os.environ.get("EPOCHS", "5"))
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    total = torch.zeros(2, device=device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()  # DDP all-reduces gradients across every rank here
        opt.step()
        total += torch.tensor([loss.item() * len(xb), len(xb)], device=device)
    dist.all_reduce(total)
    if rank == 0:
        print(
            f"epoch {epoch}  global loss {(total[0] / total[1]).item():.4f}  world={world}",
            flush=True,
        )

print(f"rank {rank}/{world} on node {os.environ.get('GPULAYER_NODE_RANK')} done", flush=True)
if rank == 0:
    w = model.module.weight.detach().cpu().squeeze().tolist()
    print("learned weights:", [round(v, 2) for v in w])
    torch.save(
        model.module.state_dict(),
        os.path.join(os.environ.get("GPULAYER_OUTPUT_DIR", "."), "ddp_model.pt"),
    )
dist.barrier()
dist.destroy_process_group()
