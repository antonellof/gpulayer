"""Train a small regression MLP on synthetic data, on whatever accelerator GpuLayer granted.

GpuLayer sets GPULAYER_ACCELERATOR (cpu | cuda | mps) from the job's ResourceRequest and
GPULAYER_OUTPUT_DIR for files that should be returned as output artifacts.
"""

import json
import os
import time

import torch
from torch import nn

accel = os.environ.get("GPULAYER_ACCELERATOR", "cpu")
device = torch.device(accel if accel in ("cuda", "mps") else "cpu")
epochs = int(os.environ.get("EPOCHS", "200"))
torch.manual_seed(0)

# y = 3*x0 - 2*x1 + 0.5 + noise
x = torch.randn(4096, 2)
y = (3 * x[:, :1] - 2 * x[:, 1:] + 0.5) + 0.01 * torch.randn(4096, 1)
x, y = x.to(device), y.to(device)

model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
start = time.perf_counter()
for epoch in range(epochs):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:4d}  loss {loss.item():.5f}", flush=True)

elapsed = time.perf_counter() - start
print(f"trained on {device} in {elapsed:.2f}s")
out = os.environ.get("GPULAYER_OUTPUT_DIR", ".")
torch.save(model.state_dict(), os.path.join(out, "model.pt"))
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump({"device": str(device), "final_loss": loss.item(), "seconds": elapsed}, f)
