"""A small character-level transformer language model.

Shows a learning-rate schedule, gradient clipping, mixed precision, resuming from a
checkpoint, and using the trained model in the launching process afterwards.

    python train.py                                     # this process
    python train.py --accelerator cpu --devices 2       # DDP
    python train.py --precision bf16-mixed --accelerator cuda   # mixed precision on NVIDIA
    python train.py --resume gpulayer_runs/trainer_char_lm-train/checkpoints/last.ckpt --epochs 6
"""

import argparse
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import gpulayer.pytorch as glt

TEXT = (
    """GpuLayer runs PyTorch code on whatever compute you can reach. You write one training
file. GpuLayer decides where it can run, starts it there, and brings back the logs, metrics,
checkpoints and results. The same file trains in your process, as a multi-process job on your
machine, or across a group of remote workers. """
    * 20
)

VOCAB = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(VOCAB)}


class Chars(Dataset):
    def __init__(self, text: str, block: int) -> None:
        self.data = torch.tensor([STOI[c] for c in text])
        self.block = block

    def __len__(self) -> int:
        return len(self.data) - self.block - 1

    def __getitem__(self, i: int):
        chunk = self.data[i : i + self.block + 1]
        return chunk[:-1], chunk[1:]


class TinyGPT(glt.Module):
    def __init__(self, block: int = 64, dim: int = 128, layers: int = 2, lr: float = 3e-3):
        super().__init__()
        self.block, self.lr = block, lr
        self.tok = nn.Embedding(len(VOCAB), dim)
        self.pos = nn.Embedding(block, dim)
        layer = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=4 * dim, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(dim, len(VOCAB))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        return self.head(self.blocks(h, mask=mask, is_causal=True))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x).flatten(0, 1), y.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x).flatten(0, 1), y.flatten())
        self.log("val_loss", loss)
        self.log("val_ppl", math.exp(loss.item()))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
        return {"optimizer": opt, "lr_scheduler": sched, "interval": "step"}

    @torch.no_grad()
    def generate(self, prompt: str, n: int = 120) -> str:
        self.eval()
        device = next(self.parameters()).device
        idx = torch.tensor([[STOI[c] for c in prompt]], device=device)
        for _ in range(n):
            logits = self(idx[:, -self.block :])[:, -1]
            nxt = torch.multinomial(torch.softmax(logits / 0.8, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return "".join(VOCAB[i] for i in idx[0].tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--target", default=None, choices=[None, "local", "remote", "cloud"])
    ap.add_argument("--precision", default="32-true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    glt.seed_everything(0)
    split = int(len(TEXT) * 0.9)
    train_loader = DataLoader(Chars(TEXT[:split], 64), batch_size=32, shuffle=True)
    val_loader = DataLoader(Chars(TEXT[split:], 64), batch_size=64)

    model = TinyGPT()
    trainer = glt.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        target=args.target,
        precision=args.precision,
        gradient_clip_val=1.0,
        limit_train_batches=120,
        log_every_n_steps=40,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    if glt.is_launcher_process():
        print(f"val_ppl={trainer.callback_metrics['val_ppl']:.2f}")
        print("sample:", model.cpu().generate("GpuLayer "))
