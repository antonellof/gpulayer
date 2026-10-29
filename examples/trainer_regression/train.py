"""One training file that runs anywhere GpuLayer can place it.

python train.py                                   # this process, best local device
python train.py --accelerator cpu --devices 2     # 2-process DDP on this machine
python train.py --accelerator cpu --nodes 2 --devices 2         # 2 local workers x 2
python train.py --accelerator cuda --devices 4 --target remote  # a remote GPU box
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt


class Regressor(glt.Module):
    def __init__(self, lr: float = 1e-2) -> None:
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def make_loaders() -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8192, 8, generator=g)
    y = x @ torch.arange(1.0, 9.0).unsqueeze(1) + 0.1 * torch.randn(8192, 1, generator=g)
    train = TensorDataset(x[:7168], y[:7168])
    val = TensorDataset(x[7168:], y[7168:])
    return DataLoader(train, batch_size=128, shuffle=True), DataLoader(val, batch_size=256)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--target", default=None, choices=[None, "local", "remote", "cloud"])
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    glt.seed_everything(0)
    model = Regressor()
    trainer = glt.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        target=args.target,
        log_every_n_steps=0,
    )
    train_loader, val_loader = make_loaders()
    trainer.fit(model, train_loader, val_loader)

    # Code after fit() also runs on the workers; print only in the launching process.
    if glt.is_launcher_process():
        print("final metrics:", {k: round(v, 4) for k, v in trainer.callback_metrics.items()})
        print("checkpoint:", trainer.checkpoint_path)
        w = model.net[0].weight.shape
        print("model is trained in this process too; first layer", tuple(w))
