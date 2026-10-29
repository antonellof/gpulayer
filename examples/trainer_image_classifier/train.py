"""Image classification with a small CNN, written once and run on any target.

The dataset is generated on the fly (circles, squares and crosses on noisy 32x32 canvases),
so the example needs nothing beyond torch.

    python train.py                                          # this process, best device
    python train.py --accelerator cpu --devices 2            # DDP, 2 processes
    python train.py --accelerator cuda --target remote       # a remote GPU worker
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpulayer.pytorch as glt

CLASSES = ("circle", "square", "cross")


def make_shapes(n: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    images = torch.empty(n, 1, 32, 32)
    labels = torch.randint(0, 3, (n,), generator=g)
    for i in range(n):
        cx, cy = (torch.rand(2, generator=g) * 16 + 8).tolist()
        r = float(torch.rand(1, generator=g) * 4 + 4)
        if labels[i] == 0:
            img = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).float()
        elif labels[i] == 1:
            img = (((xx - cx).abs() <= r) & ((yy - cy).abs() <= r)).float()
        else:
            img = (
                (((xx - cx).abs() <= 1.5) & ((yy - cy).abs() <= r))
                | (((yy - cy).abs() <= 1.5) & ((xx - cx).abs() <= r))
            ).float()
        images[i, 0] = img + 0.3 * torch.randn(32, 32, generator=g)
    return TensorDataset(images, labels)


class ShapeNet(glt.Module):
    def __init__(self, lr: float = 2e-3) -> None:
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", (logits.argmax(1) == y).float().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", nn.functional.cross_entropy(logits, y))
        self.log("val_acc", (logits.argmax(1) == y).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--target", default=None, choices=[None, "local", "remote", "cloud"])
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    glt.seed_everything(0)
    train_loader = DataLoader(make_shapes(3000, seed=1), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_shapes(600, seed=2), batch_size=200)

    model = ShapeNet()
    trainer = glt.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        target=args.target,
        log_every_n_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)

    if glt.is_launcher_process():
        print(f"val_acc={trainer.callback_metrics['val_acc']:.3f}")
        model.eval()
        x, y = val_loader.dataset[:6]
        with torch.no_grad():
            pred = model(x).argmax(1)
        for p, t in zip(pred.tolist(), y.tolist(), strict=True):
            print(f"predicted {CLASSES[p]:7} actual {CLASSES[t]}")
