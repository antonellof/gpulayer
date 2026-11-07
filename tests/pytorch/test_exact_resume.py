"""A run stopped and resumed gives the same weights as one that never stopped: same data
order, same random numbers. Lightning restores neither with a plain DataLoader, so this has
no conformance test. CPU, one process; the last test uses two gloo processes."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from test_trainer_api import run_script  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402


class Counted(Dataset):
    """Knows which samples were read."""

    def __init__(self, n: int = 40) -> None:
        self.x = torch.randn(n, 4, generator=torch.Generator().manual_seed(0))
        self.reads: list[int] = []

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.reads.append(i)
        return self.x[i], torch.tensor([float(i)])


class Noisy(glt.Module):
    """Dropout and Python's generator, so a wrong random number changes the weights."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 1))
        self.seen: list[int] = []

    def training_step(self, batch, batch_idx):
        self.seen += [int(v) for v in batch[1].flatten()]
        scale = 1.0 + random.random()
        return (self.net(batch[0]) * scale).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class SaveAt(glt.Callback):
    def __init__(self, step, path):
        self.step, self.path = step, path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == self.step:
            trainer.save_checkpoint(self.path)


def trainer(root, **kw):
    kw.setdefault("max_epochs", 3)
    return glt.Trainer(
        accelerator="cpu", default_root_dir=root, log_every_n_steps=0, num_sanity_val_steps=0, **kw
    )


def loader(**kw) -> DataLoader:
    kw.setdefault("shuffle", True)
    return DataLoader(Counted(), batch_size=4, **kw)


def full_run(root):
    glt.seed_everything(1)
    model = Noisy()
    t = trainer(root, callbacks=[SaveAt(17, root / "mid.ckpt")])
    t.fit(model, loader())
    return model, t


def resumed(root, ckpt):
    glt.seed_everything(99)  # a rerun of a script need not start from the same state
    model = Noisy()
    t = trainer(root)
    t.fit(model, loader(), ckpt_path=ckpt)
    return model, t


def test_resume_at_an_epoch_boundary_gives_the_same_weights(tmp_path):
    full, _ = full_run(tmp_path)
    glt.seed_everything(1)
    t = trainer(tmp_path, max_epochs=2)
    t.fit(Noisy(), loader())
    model, t2 = resumed(tmp_path, t.checkpoint_path)
    assert t2.global_step == 30
    assert torch.equal(model.net[0].weight, full.net[0].weight)


def test_resume_within_an_epoch_gives_the_same_weights(tmp_path):
    full, _ = full_run(tmp_path)
    model, t = resumed(tmp_path, tmp_path / "mid.ckpt")
    assert t.global_step == 30
    assert torch.equal(model.net[0].weight, full.net[0].weight)
    # The rest of epoch 1: 3 batches of 4, and epoch 2 in full.
    assert sorted(full.seen[-len(model.seen) :]) == sorted(model.seen)


def test_trained_batches_are_not_read_again(tmp_path):
    full_run(tmp_path)
    data = loader()
    glt.seed_everything(99)
    t = trainer(tmp_path, max_epochs=2)
    t.fit(Noisy(), data, ckpt_path=tmp_path / "mid.ckpt")
    assert len(data.dataset.reads) == 3 * 4


def test_skipping_keeps_the_loader_settings(tmp_path):
    from gpulayer.pytorch.trainer import _skipping_loader

    def collate(batch):
        return batch

    data = DataLoader(Counted(), batch_size=4, collate_fn=collate, drop_last=True)
    skipping = _skipping_loader(data, 2)
    assert skipping.collate_fn is collate
    assert len(skipping) == len(data) - 2
    assert [b[0][1].item() for b in skipping][0] == pytest.approx(8.0)


def test_loaders_that_cannot_skip_are_read_and_thrown_away(tmp_path):
    class Stream(IterableDataset):
        def __iter__(self):
            for i in range(20):
                yield torch.zeros(4), torch.tensor([float(i)])

    def stream():
        return DataLoader(Stream(), batch_size=2)

    t = trainer(tmp_path, max_epochs=1, callbacks=[SaveAt(4, tmp_path / "s.ckpt")])
    t.fit(Noisy(), stream())
    model = Noisy()
    t2 = trainer(tmp_path, max_epochs=1)
    t2.fit(model, stream(), ckpt_path=tmp_path / "s.ckpt")
    assert model.seen == list(range(8, 20))


def test_checkpoint_without_random_state_still_resumes(tmp_path):
    full_run(tmp_path)
    state = torch.load(tmp_path / "mid.ckpt", weights_only=False)
    del state["rng_states"]
    torch.save(state, tmp_path / "old.ckpt")
    model, t = resumed(tmp_path, tmp_path / "old.ckpt")
    assert t.global_step == 30 and len(model.seen) == 13 * 4


def test_numpy_generator_is_restored(tmp_path):
    numpy = pytest.importorskip("numpy")
    from gpulayer.pytorch.trainer import _rng_states, set_rng_states

    numpy.random.seed(3)
    states = _rng_states()
    expected = numpy.random.random(3).tolist()
    set_rng_states(states)
    assert numpy.random.random(3).tolist() == expected


def test_stateful_dataloader_continues_where_it_stopped(tmp_path):
    stateful = pytest.importorskip("torchdata.stateful_dataloader")

    def data():
        x = torch.zeros(40, 4)
        ids = torch.arange(40.0).reshape(-1, 1)
        return stateful.StatefulDataLoader(TensorDataset(x, ids), batch_size=4, shuffle=False)

    t = trainer(tmp_path, max_epochs=1, callbacks=[SaveAt(4, tmp_path / "s.ckpt")])
    t.fit(Noisy(), data())
    state = torch.load(tmp_path / "s.ckpt", weights_only=False)
    assert state["dataloaders"]
    model = Noisy()
    t2 = trainer(tmp_path, max_epochs=1)
    t2.fit(model, data(), ckpt_path=tmp_path / "s.ckpt")
    assert model.seen == list(range(16, 40))
    assert t2.global_step == 10


def test_two_processes_continue_with_the_rest_of_their_samples(tmp_path):
    out = run_script(
        tmp_path,
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import gpulayer.pytorch as glt

        class Stop(glt.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if trainer.global_step == 3:
                    trainer.save_checkpoint("/CKPT/mid.ckpt")

        class Model(glt.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.net = nn.Linear(1, 1)
                self.seen = []
            def training_step(self, batch, batch_idx):
                self.seen += batch[0].flatten().tolist()
                return self.net(batch[0]).mean()
            def on_train_end(self):
                g = self.all_gather(torch.tensor(self.seen))
                self.print("SAMPLES", sorted(int(v) for v in g.flatten().tolist()))
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=1e-4)

        x = torch.arange(32.0).reshape(-1, 1)
        data = DataLoader(TensorDataset(x), batch_size=2, shuffle=True)
        kw = dict(max_epochs=1, accelerator="cpu", devices=2, log_every_n_steps=0,
                  enable_checkpointing=False)
        glt.seed_everything(5)
        glt.Trainer(callbacks=[Stop()], **kw).fit(Model(), data)
        glt.seed_everything(5)
        glt.Trainer(**kw).fit(Model(), data, ckpt_path="/CKPT/mid.ckpt")
        """.replace("/CKPT", str(tmp_path)),
    )
    first, second = [line for line in out.splitlines() if "SAMPLES" in line][-2:]
    assert len(eval(first.split("SAMPLES ")[1])) == 32  # noqa: S307
    assert len(eval(second.split("SAMPLES ")[1])) == 32 - 3 * 2 * 2  # noqa: S307
