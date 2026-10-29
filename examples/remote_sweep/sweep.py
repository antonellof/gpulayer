"""Run one PyTorch function as many parallel jobs (a learning-rate sweep).

python examples/remote_sweep/sweep.py                    # CPU
python examples/remote_sweep/sweep.py --accelerator mps  # Apple Silicon
"""

import argparse
import os

import gpulayer as gl


@gl.remote(frameworks={"torch": ">=2.0"})
def train(lr: float, epochs: int = 200) -> dict:
    import torch
    from torch import nn

    device = torch.device(os.environ.get("GPULAYER_ACCELERATOR", "cpu"))
    torch.manual_seed(0)
    x = torch.randn(2048, 2, device=device)
    y = 3 * x[:, :1] - 2 * x[:, 1:] + 0.5
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return {"lr": lr, "loss": round(loss.item(), 5), "device": str(device)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    print("plain local call:", train(0.1, epochs=20))
    sweep = train.options(accelerator=args.accelerator)
    results = sweep.map([0.3, 0.1, 0.03, 0.01])  # four jobs, run in parallel
    for r in results:
        print(r)
    print("best:", min(results, key=lambda r: r["loss"]))
