"""DistributedDataParallel training launched by GpuLayer through torchrun.

torchrun provides RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT; GpuLayer
provides GPULAYER_COMM_BACKEND (gloo or nccl), GPULAYER_NODE_RANK and GPULAYER_OUTPUT_DIR.
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

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
        print(f"epoch {epoch}  global loss {(total[0] / total[1]).item():.4f}  world={world}",
              flush=True)

print(f"rank {rank}/{world} on node {os.environ.get('GPULAYER_NODE_RANK')} done", flush=True)
if rank == 0:
    w = model.module.weight.detach().cpu().squeeze().tolist()
    print("learned weights:", [round(v, 2) for v in w])
    torch.save(model.module.state_dict(),
               os.path.join(os.environ.get("GPULAYER_OUTPUT_DIR", "."), "ddp_model.pt"))
dist.barrier()
dist.destroy_process_group()
