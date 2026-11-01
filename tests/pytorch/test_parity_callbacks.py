"""Lightning parity of the callbacks: ``Checkpoint``, ``ModelCheckpoint(save_on_exception)``,
``LearningRateMonitor(log_key_prefix)``, ``DeviceStatsMonitor(filter_keys)``, the
``ModelPruning`` options (lottery ticket, structured pruning) and the rich callbacks.

All tests run on the CPU, in process, with small models. ``rich`` is an optional package:
the ``RichProgressBar`` and ``RichModelSummary`` tests that check calls use a fake ``rich``
module put in ``sys.modules`` (they test our use of the API, not rich's rendering); the
test marked ``real rich`` runs only when ``rich`` is installed.
"""

import io
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.nn.utils import prune  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch.callbacks as cbs  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch import Module, Trainer  # noqa: E402


class Net(Module):
    def __init__(self, lr=0.1, fail_at=None):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.lr, self.fail_at = lr, fail_at

    def training_step(self, batch, batch_idx):
        if self.fail_at == (self.current_epoch, batch_idx):
            raise RuntimeError("boom")
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self.net(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 4, generator=g)
    ds = TensorDataset(x, x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]))
    return DataLoader(ds, batch_size=32), DataLoader(ds, batch_size=64)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("enable_model_summary", False)
    return Trainer(default_root_dir=tmp_path, **kw)


def zeros(t):
    return float((t == 0).float().mean())


# ---------------------------------------------------------------- Checkpoint


def test_checkpoint_callbacks_share_the_checkpoint_base_class():
    assert issubclass(cbs.Checkpoint, cbs.Callback)
    assert isinstance(cbs.ModelCheckpoint(), cbs.Checkpoint)
    assert isinstance(cbs.OnExceptionCheckpoint(), cbs.Checkpoint)


def test_model_checkpoint_saves_on_exception(tmp_path):
    ckpt = cbs.ModelCheckpoint(monitor="val_loss", save_on_exception=True, save_last=True)
    t = trainer(tmp_path, max_epochs=3, callbacks=[ckpt])
    with pytest.raises(RuntimeError, match="boom"):
        t.fit(Net(fail_at=(1, 2)), *data())
    folder = ckpt.best_model_path.parent
    # The best file (from epoch 0) is kept as it was; the exception one is separate.
    best = torch.load(folder / "best.ckpt", weights_only=False)
    assert best["global_step"] == 8
    saved = torch.load(folder / "best-on_exception.ckpt", weights_only=False)
    assert saved["global_step"] == 10 and saved["epoch"] == 1
    assert torch.load(folder / "last.ckpt", weights_only=False)["global_step"] == 10


def test_model_checkpoint_on_exception_uses_the_filename_template(tmp_path):
    ckpt = cbs.ModelCheckpoint(monitor=None, filename="{epoch}-{step}", save_on_exception=True)
    t = trainer(tmp_path, max_epochs=3, callbacks=[ckpt])
    with pytest.raises(RuntimeError):
        t.fit(Net(fail_at=(1, 2)), *data())
    assert (ckpt.best_model_path.parent / "epoch=1-step=10.ckpt").exists()


def test_model_checkpoint_does_not_save_on_exception_by_default(tmp_path):
    ckpt = cbs.ModelCheckpoint(monitor="val_loss")
    t = trainer(tmp_path, max_epochs=3, callbacks=[ckpt])
    with pytest.raises(RuntimeError):
        t.fit(Net(fail_at=(1, 2)), *data())
    assert not list(ckpt.best_model_path.parent.glob("*on_exception*"))


# ---------------------------------------------------------------- monitors


def test_learning_rate_monitor_prefixes_every_name(tmp_path):
    lr = cbs.LearningRateMonitor(log_key_prefix="optim/", log_momentum=True)
    t = trainer(tmp_path, max_epochs=1, callbacks=[lr])
    t.fit(Net(), *data())
    assert t.callback_metrics["optim/lr"] == pytest.approx(0.1)
    assert "optim/momentum" in t.history[-1] and "lr" not in t.callback_metrics


def test_device_stats_monitor_keeps_only_the_filtered_keys(tmp_path, monkeypatch):
    stats = {
        "device/memory_allocated_mb": 1.0,
        "device/memory_reserved_mb": 2.0,
        "cpu/percent": 3.0,
    }
    monkeypatch.setattr(cbs.DeviceStatsMonitor, "_stats", lambda self, trainer: dict(stats))
    mon = cbs.DeviceStatsMonitor(filter_keys={"memory_allocated", "cpu/percent", "nope"})
    t = trainer(tmp_path, max_epochs=1, log_every_n_steps=4, callbacks=[mon])
    logged = []
    monkeypatch.setattr(t, "log_metrics", lambda m, step=None: logged.append(m))
    with pytest.warns(UserWarning, match="nope"):
        t.fit(Net(), *data())
    assert logged and all(set(m) == {"device/memory_allocated_mb", "cpu/percent"} for m in logged)


# ---------------------------------------------------------------- pruning


def test_structured_pruning_removes_whole_rows(tmp_path):
    m = Net()
    p = cbs.ModelPruning(
        "ln_structured",
        parameters_to_prune=[(m.net[0], "weight")],
        use_global_unstructured=False,
        pruning_dim=0,
        pruning_norm=2,
        amount=0.5,
    )
    trainer(tmp_path, max_epochs=1, callbacks=[p]).fit(m, data()[0])
    rows = (m.net[0].weight == 0).all(dim=1)
    assert int(rows.sum()) == 4 and not hasattr(m.net[0], "weight_mask")


def test_random_structured_pruning_needs_only_the_dim(tmp_path):
    m = Net()
    p = cbs.ModelPruning(
        "random_structured", use_global_unstructured=False, pruning_dim=1, amount=0.5
    )
    trainer(tmp_path, max_epochs=1, callbacks=[p]).fit(m, data()[0])
    cols = (m.net[0].weight == 0).all(dim=0)
    assert int(cols.sum()) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pruning_fn": "ln_structured", "pruning_dim": 0, "pruning_norm": 2},  # global
        {"pruning_fn": "ln_structured", "use_global_unstructured": False, "pruning_norm": 2},
        {"pruning_fn": "ln_structured", "use_global_unstructured": False, "pruning_dim": 0},
        {"pruning_fn": "nope"},
        {"pruning_fn": prune.L1Unstructured, "use_global_unstructured": False},
        {"pruning_fn": prune.LnStructured},
    ],
)
def test_pruning_rejects_incomplete_settings(kwargs):
    with pytest.raises(ValidationFailed):
        cbs.ModelPruning(**kwargs)


def test_pruning_accepts_a_pruning_method_class(tmp_path):
    m = Net()
    p = cbs.ModelPruning(prune.L1Unstructured, amount=0.5)
    trainer(tmp_path, max_epochs=1, callbacks=[p]).fit(m, data()[0])
    assert zeros(torch.cat([m.net[0].weight.flatten(), m.net[2].weight.flatten()])) == 0.5


def test_lottery_ticket_resets_remaining_weights_to_their_initial_values(tmp_path):
    m = Net()
    initial = m.net[0].weight.detach().clone()
    p = cbs.ModelPruning(
        amount=0.5, use_lottery_ticket_hypothesis=True, make_pruning_permanent=False
    )
    trainer(tmp_path, max_epochs=1, callbacks=[p]).fit(m, data()[0])
    layer = m.net[0]
    assert torch.equal(layer.weight_orig, initial)
    assert zeros(layer.weight_mask) > 0
    assert torch.equal(layer.weight, initial * layer.weight_mask)


def test_lottery_ticket_as_a_function_of_the_epoch(tmp_path):
    m = Net()
    initial = m.net[0].weight.detach().clone()
    snaps = []
    p = cbs.ModelPruning(
        amount=0.2,
        use_lottery_ticket_hypothesis=lambda epoch: epoch == 1,
        make_pruning_permanent=False,
    )
    record = cbs.LambdaCallback(
        on_train_epoch_end=lambda t, mod: snaps.append(mod.net[0].weight_orig.detach().clone())
    )
    trainer(tmp_path, max_epochs=2, callbacks=[p, record]).fit(m, data()[0])
    assert not torch.equal(snaps[0], initial)  # trained weights kept after epoch 0
    assert torch.equal(snaps[1], initial)  # reset after epoch 1


def test_lottery_ticket_resample_draws_new_weights(tmp_path):
    m = Net()
    initial = m.net[0].weight.detach().clone()
    p = cbs.ModelPruning(
        amount=0.5,
        use_lottery_ticket_hypothesis=True,
        resample_parameters=True,
        make_pruning_permanent=False,
    )
    trainer(tmp_path, max_epochs=1, callbacks=[p]).fit(m, data()[0])
    assert not torch.equal(m.net[0].weight_orig, initial)
    assert torch.equal(p._original_layers[id(m.net[0])].weight, initial)  # original kept
    assert zeros(m.net[0].weight) >= 0.5


def test_pruning_after_validation(tmp_path):
    m = Net()
    p = cbs.ModelPruning(amount=0.5, prune_on_train_epoch_end=False, make_pruning_permanent=False)
    t = trainer(tmp_path, max_epochs=1, num_sanity_val_steps=2, callbacks=[p])
    t.fit(m, *data())
    weights = torch.cat([m.net[0].weight.flatten(), m.net[2].weight.flatten()])
    assert zeros(weights) == 0.5  # once: not during the sanity check
    t.validate(m, data()[1])  # not when only validating
    weights = torch.cat([m.net[0].weight.flatten(), m.net[2].weight.flatten()])
    assert zeros(weights) == 0.5


# ---------------------------------------------------------------- rich (fakes)


class _Recorder:
    def __init__(self):
        self.progress = []
        self.tables = []
        self.printed = []


def fake_rich(monkeypatch):
    """A fake ``rich`` with the parts the callbacks use; returns what they recorded."""
    rec = _Recorder()

    class Column:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

    class Progress:
        def __init__(self, *columns, **kwargs):
            self.columns, self.kwargs = columns, kwargs
            self.updates, self.refreshes, self.started, self.stopped = [], 0, False, False
            self.tasks = []
            rec.progress.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def add_task(self, description, total=None, **fields):
            self.tasks.append((description, total, fields))
            return len(self.tasks) - 1

        def update(self, task, **kwargs):
            self.updates.append(kwargs)

        def refresh(self):
            self.refreshes += 1

    class Table:
        def __init__(self, **kwargs):
            self.kwargs, self.columns, self.rows = kwargs, [], []
            rec.tables.append(self)

        @classmethod
        def grid(cls, **kwargs):
            return cls(grid=True, **kwargs)

        def add_column(self, name="", **kwargs):
            self.columns.append(name)

        def add_row(self, *cells):
            self.rows.append(cells)

    class Console:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def print(self, obj):
            rec.printed.append(obj)

    console = Console()
    rich = types.ModuleType("rich")
    rich.get_console = lambda: console
    mods = {
        "rich": rich,
        "rich.console": types.SimpleNamespace(Console=Console),
        "rich.progress": types.SimpleNamespace(
            Progress=Progress, BarColumn=Column, TextColumn=Column, TimeElapsedColumn=Column
        ),
        "rich.table": types.SimpleNamespace(Table=Table),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return rec


def on_a_terminal(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)


def test_rich_callbacks_need_rich(monkeypatch):
    monkeypatch.setitem(sys.modules, "rich", None)
    with pytest.raises(ValidationFailed, match="pip install rich"):
        cbs.RichProgressBar()
    with pytest.raises(ValidationFailed, match="pip install rich"):
        cbs.RichModelSummary()


def test_rich_progress_bar_draws_one_bar_per_epoch_on_a_terminal(tmp_path, monkeypatch):
    rec = fake_rich(monkeypatch)
    on_a_terminal(monkeypatch)
    bar = cbs.RichProgressBar(refresh_rate=1e9, console_kwargs={"width": 80})
    t = trainer(
        tmp_path, max_epochs=2, log_every_n_steps=1, enable_progress_bar=True, callbacks=[bar]
    )
    t.fit(Net(), data()[0])
    assert [c for c in t.callbacks if isinstance(c, cbs.ProgressBar)] == [bar]
    assert len(rec.progress) == 2
    first = rec.progress[0]
    assert first.started and first.stopped and first.kwargs["transient"] is True
    assert first.kwargs["console"].kwargs == {"width": 80}
    # the Trainer sets num_training_batches before on_train_epoch_start, so epoch 0 knows it
    assert first.tasks[0][:2] == ("Epoch 0", 8) and {"total": 8} not in first.updates
    assert rec.progress[1].tasks[0][:2] == ("Epoch 1", 8)
    assert first.refreshes == 9  # every batch (no rate limit here) and at the epoch end
    assert first.updates[-1]["completed"] == 8 and first.updates[-1]["batches"] == "8/8"
    assert "train_loss: " in first.updates[-1]["metrics"]


def test_rich_progress_bar_respects_the_refresh_rate(tmp_path, monkeypatch):
    rec = fake_rich(monkeypatch)
    on_a_terminal(monkeypatch)
    bar = cbs.RichProgressBar(refresh_rate=1e-6, leave=True)
    trainer(tmp_path, max_epochs=1, callbacks=[bar]).fit(Net(), data()[0])
    (progress,) = rec.progress
    assert progress.refreshes == 1 and progress.kwargs["transient"] is False  # epoch end only


def test_rich_progress_bar_draws_nothing_off_a_terminal_or_when_off(tmp_path, monkeypatch):
    rec = fake_rich(monkeypatch)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    trainer(tmp_path, max_epochs=1, callbacks=[cbs.RichProgressBar()]).fit(Net(), data()[0])
    on_a_terminal(monkeypatch)
    trainer(tmp_path, max_epochs=1, callbacks=[cbs.RichProgressBar(refresh_rate=0)]).fit(
        Net(), data()[0]
    )
    disabled = cbs.RichProgressBar()
    disabled.disable()
    trainer(tmp_path, max_epochs=1, callbacks=[disabled]).fit(Net(), data()[0])
    assert rec.progress == []


def test_rich_progress_bar_stops_on_an_exception(tmp_path, monkeypatch):
    rec = fake_rich(monkeypatch)
    on_a_terminal(monkeypatch)
    t = trainer(tmp_path, max_epochs=1, callbacks=[cbs.RichProgressBar()])
    with pytest.raises(RuntimeError):
        t.fit(Net(fail_at=(0, 3)), data()[0])
    assert rec.progress[0].stopped


def test_rich_model_summary_prints_a_table(tmp_path, monkeypatch):
    rec = fake_rich(monkeypatch)
    summary = cbs.RichModelSummary(max_depth=-1, header_style="bold")
    trainer(tmp_path, max_epochs=1, callbacks=[summary]).fit(Net(), data()[0])
    table, grid = rec.tables
    assert table.kwargs["header_style"] == "bold"
    assert table.columns == [" ", "Name", "Type", "Params", "Mode"]
    assert [r[1:3] for r in table.rows] == [
        ("net", "Sequential"),
        ("net.0", "Linear"),
        ("net.1", "ReLU"),
        ("net.2", "Linear"),
    ]
    assert table.rows[1][3] == "40" and rec.printed == [table, grid]
    assert any("Total params[/]: 49" in r[0] for r in grid.rows)


def test_plain_model_summary_output_is_unchanged(tmp_path, capsys):
    trainer(tmp_path, max_epochs=1, callbacks=[cbs.ModelSummary()]).fit(Net(), data()[0])
    out = capsys.readouterr().out
    assert "  | Name | Type       | Params | Mode" in out
    assert "0 | net | Sequential |     49 | train" in out
    assert "49 trainable params, 0 non-trainable" in out


# ---------------------------------------------------------------- real rich


def test_real_rich_progress_bar_and_summary(tmp_path, monkeypatch):
    """Real rich: runs only when rich is installed."""
    pytest.importorskip("rich")
    on_a_terminal(monkeypatch)
    out = io.StringIO()
    bar = cbs.RichProgressBar(leave=True, console_kwargs={"file": out, "force_terminal": True})
    t = trainer(tmp_path, max_epochs=1, callbacks=[cbs.RichModelSummary(), bar])
    t.fit(Net(), data()[0])
    assert "Epoch 0" in out.getvalue() and "8/8" in out.getvalue()
