"""Lightning parity of the CLI (constructor arguments, overridable methods and their order,
SaveConfigCallback, --optimizer/--lr_scheduler, subclass mode, links), DataModule.on_exception
and DataModule.load_from_checkpoint, and CombinedLoader (sampler, batch_sampler, limits,
reset). Real behaviour, CPU only, in one process; nothing is mocked."""

import json

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch.cli import (  # noqa: E402
    GPULayerArgumentParser,
    GPULayerCLI,
    ReduceLROnPlateau,
    SaveConfigCallback,
)
from gpulayer.pytorch.combined import CombinedLoader  # noqa: E402


def ds(n):
    x = torch.arange(n, dtype=torch.float32).reshape(-1, 1).repeat(1, 4)
    return TensorDataset(x, x.sum(1, keepdim=True))


class NoOpt(glt.Module):
    """A model without configure_optimizers: the CLI's --optimizer provides it."""

    def __init__(self, hidden: int = 4, batch_size: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.log("test_loss", nn.functional.mse_loss(self(x), y))


class WithOpt(NoOpt):
    def __init__(self, hidden: int = 4, batch_size: int = 0, lr: float = 0.1):
        super().__init__(hidden, batch_size)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


class Data(glt.DataModule):
    def __init__(self, batch_size: int = 8, size: int = 16):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.size = size
        self.num_features = 4
        self.restored = None

    def train_dataloader(self):
        return DataLoader(ds(self.size), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(ds(self.size), batch_size=self.batch_size)

    def state_dict(self):
        return {"seen": self.size}

    def load_state_dict(self, state_dict):
        self.restored = state_dict


def defaults(tmp_path, **kw):
    out = {
        "accelerator": "cpu",
        "log_every_n_steps": 0,
        "num_sanity_val_steps": 0,
        "default_root_dir": str(tmp_path),
        "max_epochs": 1,
    }
    out.update(kw)
    return out


# ---------------------------------------------------------------- CLI order


class RecordingCLI(GPULayerCLI):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def init_parser(self, **kwargs):
        self.calls.append("init_parser")
        return super().init_parser(**kwargs)

    def setup_parser(self, *args):
        self.calls.append("setup_parser")
        super().setup_parser(*args)

    def add_default_arguments_to_parser(self, parser):
        self.calls.append("add_default_arguments_to_parser")
        super().add_default_arguments_to_parser(parser)

    def add_core_arguments_to_parser(self, parser):
        self.calls.append("add_core_arguments_to_parser")
        super().add_core_arguments_to_parser(parser)

    def add_arguments_to_parser(self, parser):
        self.calls.append("add_arguments_to_parser")
        parser.add_argument("--notes", default="none")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_features", "model.hidden", apply_on="instantiate")

    @staticmethod
    def link_optimizers_and_lr_schedulers(parser):
        parser._gl_linked = True
        GPULayerCLI.link_optimizers_and_lr_schedulers(parser)

    def parse_arguments(self, parser, args):
        self.calls.append("parse_arguments")
        super().parse_arguments(parser, args)

    def before_instantiate_classes(self):
        self.calls.append("before_instantiate_classes")

    def instantiate_classes(self):
        self.calls.append("instantiate_classes")
        super().instantiate_classes()

    def instantiate_trainer(self, **kwargs):
        self.calls.append("instantiate_trainer")
        return super().instantiate_trainer(**kwargs)

    def after_instantiate_classes(self):
        self.calls.append("after_instantiate_classes")

    def before_fit(self):
        self.calls.append("before_fit")

    def after_fit(self):
        self.calls.append("after_fit")


def test_cli_calls_overridable_methods_in_order(tmp_path):
    cli = RecordingCLI(
        WithOpt,
        Data,
        trainer_defaults=defaults(tmp_path),
        args=["fit", "--data.batch_size", "4", "--notes", "hello"],
        parser_kwargs={"description": "main", "fit": {"description": "the fit parser"}},
    )
    per_parser = [
        "init_parser",
        "add_default_arguments_to_parser",
        "add_core_arguments_to_parser",
        "add_arguments_to_parser",
    ]
    assert cli.calls == [
        "setup_parser",
        "init_parser",  # the main parser
        *per_parser * 4,  # fit, validate, test, predict
        "parse_arguments",
        "before_instantiate_classes",
        "instantiate_classes",
        "instantiate_trainer",
        "after_instantiate_classes",
        "before_fit",
        "after_fit",
    ]
    assert all(getattr(p, "_gl_linked", False) for p in cli._subcommand_parsers.values())
    assert cli.parser.description == "main"
    assert cli._subcommand_parsers["fit"].description == "the fit parser"
    assert cli.subcommand == "fit" and cli.config["notes"] == "hello"
    # parse link: data.batch_size -> model.batch_size; instantiate link: the DataModule's
    # num_features attribute -> model.hidden
    assert cli.model.hparams.batch_size == 4 and cli.model.hparams.hidden == 4
    assert cli.datamodule.batch_size == 4
    assert cli.trainer.global_step == 4  # 16 samples, batch 4, one epoch
    assert cli.config_init["model"] is cli.model and cli.config_init["data"] is cli.datamodule


def test_cli_save_config_callback_writes_resolved_config(tmp_path):
    cli = GPULayerCLI(
        WithOpt,
        Data,
        trainer_defaults=defaults(tmp_path),
        args=["fit", "--model.lr", "0.05", "--trainer.max_epochs", "1"],
    )
    saved = [c for c in cli.trainer.callbacks if isinstance(c, SaveConfigCallback)]
    assert len(saved) == 1 and saved[0].already_saved
    path = tmp_path / next(p.name for p in tmp_path.iterdir() if p.is_dir()) / "config.yaml"
    assert path.is_file()
    text = path.read_text()
    try:
        import yaml

        config = yaml.safe_load(text)
    except ImportError:
        config = json.loads(text)
    assert config["model"]["lr"] == 0.05 and config["trainer"]["max_epochs"] == 1
    assert config["trainer"]["accelerator"] == "cpu" and config["optimizer"] is None

    # overwrite=False refuses an existing file, as in Lightning
    with pytest.raises(RuntimeError, match="NOT exist"):
        GPULayerCLI(
            WithOpt,
            Data,
            trainer_defaults=defaults(tmp_path),
            save_config_kwargs={"overwrite": False},
            args=["fit"],
        )

    cli = GPULayerCLI(
        WithOpt, Data, trainer_defaults=defaults(tmp_path), save_config_callback=None, args=["fit"]
    )
    assert not any(isinstance(c, SaveConfigCallback) for c in cli.trainer.callbacks)

    with pytest.raises(ValidationFailed, match="save_config"):
        SaveConfigCallback(None, {}, save_to_log_dir=False)


def test_cli_optimizer_and_scheduler_define_configure_optimizers(tmp_path):
    cli = GPULayerCLI(
        NoOpt,
        Data,
        trainer_defaults=defaults(tmp_path, max_epochs=2),
        args=[
            "fit",
            "--optimizer",
            "SGD",
            "--optimizer.lr",
            "0.5",
            "--optimizer.momentum=0.9",
            "--lr_scheduler",
            "StepLR",
            "--lr_scheduler.step_size",
            "1",
            "--lr_scheduler.gamma",
            "0.1",
        ],
    )
    (opt,) = cli.trainer.optimizers
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["initial_lr"] == 0.5 and opt.param_groups[0]["momentum"] == 0.9
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * 0.1**2)  # two epochs of StepLR
    assert {p for g in opt.param_groups for p in g["params"]} == set(cli.model.parameters())
    assert cli.config["optimizer"] == {
        "class_path": "torch.optim.sgd.SGD",
        "init_args": {"lr": 0.5, "momentum": 0.9},
    }

    # ReduceLROnPlateau with a monitor gives Lightning's dict form
    model = NoOpt()
    sched = ReduceLROnPlateau(torch.optim.SGD(model.parameters(), lr=0.1), monitor="val_loss")
    out = GPULayerCLI.configure_optimizers(model, sched.optimizer, sched)
    assert out["lr_scheduler"] == {"scheduler": sched, "monitor": "val_loss"}

    # a model with its own configure_optimizers is overridden with a warning, as in Lightning
    with pytest.warns(UserWarning, match="will be overridden"):
        cli = GPULayerCLI(
            WithOpt,
            Data,
            trainer_defaults=defaults(tmp_path),
            args=["fit", "--optimizer", "torch.optim.Adam", "--optimizer.lr", "0.01"],
        )
    assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)

    # without auto_configure_optimizers there is no --optimizer
    with pytest.raises(SystemExit):
        GPULayerCLI(
            WithOpt,
            Data,
            trainer_defaults=defaults(tmp_path),
            auto_configure_optimizers=False,
            args=["fit", "--optimizer", "SGD"],
        )


def test_cli_subclass_mode_print_config_and_run_false(tmp_path, capsys):
    cli = GPULayerCLI(
        trainer_defaults=defaults(tmp_path),
        args=[
            "fit",
            "--model",
            "WithOpt",
            "--model.lr",
            "0.2",
            "--data",
            f"{Data.__module__}.Data",
        ],
    )
    assert type(cli.model) is WithOpt and cli.model.hparams.lr == 0.2
    assert type(cli.datamodule) is Data
    assert cli.config["model"]["class_path"].endswith("WithOpt")
    # load_from_checkpoint_support records how to rebuild the model in its hparams
    assert cli.model.hparams["_instantiator"] == "gpulayer.pytorch.cli.instantiate_module"
    assert cli.model.hparams["_class_path"] == cli.config["model"]["class_path"]

    with pytest.raises(ValidationFailed, match="not a subclass"):
        GPULayerCLI(
            WithOpt,
            Data,
            subclass_mode_model=True,
            trainer_defaults=defaults(tmp_path),
            args=["fit", "--model", f"{NoOpt.__module__}.NoOpt"],
        )

    with pytest.raises(SystemExit) as e:
        GPULayerCLI(WithOpt, Data, args=["fit", "--model.lr", "0.3", "--print_config"])
    assert e.value.code == 0
    printed = capsys.readouterr().out
    assert '"lr": 0.3' in printed or "lr: 0.3" in printed

    cli = GPULayerCLI(
        WithOpt,
        Data,
        trainer_defaults=defaults(tmp_path),
        run=False,
        args=["--model.lr", "0.4", "--seed_everything", "7"],
    )
    assert cli.subcommand is None and cli.trainer.global_step == 0
    assert cli.model.hparams.lr == 0.4 and cli.config["seed_everything"] == 7

    class Custom(GPULayerArgumentParser):
        pass

    cli = GPULayerCLI(
        WithOpt, Data, parser_class=Custom, trainer_defaults=defaults(tmp_path), run=False, args=[]
    )
    assert isinstance(cli.parser, Custom)
    with pytest.raises(ValidationFailed, match="parser_class"):
        GPULayerCLI(WithOpt, Data, parser_class=object, args=[])


def test_cli_ckpt_path_hparams_and_test_subcommand(tmp_path):
    cli = GPULayerCLI(
        WithOpt, Data, trainer_defaults=defaults(tmp_path), args=["fit", "--model.lr", "0.07"]
    )
    ckpt = cli.trainer.checkpoint_path
    cli = GPULayerCLI(
        WithOpt,
        Data,
        trainer_defaults=defaults(tmp_path),
        args=["test", "--ckpt_path", str(ckpt), "--verbose", "false"],
    )
    assert cli.model.hparams.lr == 0.07  # read back from the checkpoint, as in Lightning
    assert cli.config["verbose"] is False and "test_loss" in cli.result


# ----------------------------------------------------------------- DataModule


def test_datamodule_load_from_checkpoint_and_on_exception(tmp_path):
    t = glt.Trainer(**defaults(tmp_path))
    t.fit(WithOpt(), datamodule=Data(batch_size=4, size=12))
    ckpt = t.checkpoint_path
    saved = torch.load(ckpt, weights_only=False)
    assert saved["datamodule_hyper_parameters"] == {"batch_size": 4, "size": 12}

    dm = Data.load_from_checkpoint(ckpt)
    assert isinstance(dm, Data) and dm.batch_size == 4 and dm.size == 12
    assert dm.restored == {"seen": 12}  # the DataModule's state_dict is restored too
    assert Data.load_from_checkpoint(ckpt, batch_size=2).batch_size == 2

    hp = tmp_path / "hparams.json"
    hp.write_text(json.dumps({"size": 20}))
    assert Data.load_from_checkpoint(ckpt, hparams_file=hp).size == 20
    with pytest.raises(ValidationFailed, match="hparams_file"):
        Data.load_from_checkpoint(ckpt, hparams_file=tmp_path / "x.txt")

    # written by the CLI (load_from_checkpoint_support): rebuilt through instantiate_module
    cli = GPULayerCLI(
        WithOpt,
        Data,
        subclass_mode_data=True,
        trainer_defaults=defaults(tmp_path),
        args=["fit", "--data.batch_size", "8"],
    )
    hparams = torch.load(cli.trainer.checkpoint_path, weights_only=False)[
        "datamodule_hyper_parameters"
    ]
    assert hparams["_instantiator"] == "gpulayer.pytorch.cli.instantiate_module"
    dm = glt.DataModule.load_from_checkpoint(cli.trainer.checkpoint_path)
    assert type(dm) is Data and dm.batch_size == 8

    # the hook exists and does nothing by default, as Lightning's
    assert Data().on_exception(RuntimeError("x")) is None


# -------------------------------------------------------------- CombinedLoader


def test_combined_loader_sampler_limits_and_reset():
    a = DataLoader(ds(8), batch_size=2)  # 4 batches
    b = DataLoader(ds(6), batch_size=2, shuffle=True)  # 3 batches
    cl = CombinedLoader({"a": a, "b": b}, mode="max_size_cycle")
    assert cl.iterables == {"a": a, "b": b}
    assert cl.sampler == {"a": a.sampler, "b": b.sampler}
    assert cl.batch_sampler == {"a": a.batch_sampler, "b": b.batch_sampler}
    assert CombinedLoader([a, b]).sampler == [a.sampler, b.sampler]
    assert CombinedLoader(a).batch_sampler is a.batch_sampler

    assert cl.limits is None and len(cl) == 4
    cl.limits = 2
    assert cl.limits == [2, 2] and len(cl) == 2 and len(list(cl)) == 2
    cl.limits = [3, 1]
    assert len(cl) == 3 and len(list(cl)) == 3
    fixed = CombinedLoader([a, DataLoader(ds(6), batch_size=2)])
    fixed.limits = [3, 1]
    batches = list(fixed)
    assert len(batches) == 3  # the second loader cycles over its first batch only
    assert all(torch.equal(x[1][0], batches[0][1][0]) for x in batches)
    with pytest.raises(ValidationFailed, match="Mismatch"):
        cl.limits = [1, 2, 3]
    cl.limits = [float("inf"), 2]
    assert len(cl) == 4
    cl.limits = None
    assert len(cl) == 4

    seq = CombinedLoader([a, b], mode="sequential")
    seq.limits = [1, 2]
    assert len(seq) == 3
    assert [(i, d) for _, i, d in seq] == [(0, 0), (0, 1), (1, 1)]
    for mode, expected in (("min_size", 1), ("max_size", 2)):
        c = CombinedLoader([a, b], mode=mode)
        c.limits = [2, 1]
        assert len(c) == expected and len(list(c)) == expected

    w = DataLoader(ds(8), batch_size=2, num_workers=1, persistent_workers=True)
    c = CombinedLoader([w])
    assert len(list(c)) == 4 and w._iterator is not None
    c.reset()
    assert w._iterator is None
    assert len(list(c)) == 4  # starts again from the beginning

    c = CombinedLoader({"a": a, "b": b})
    c.flattened = [b, a]
    assert c.iterables == {"a": b, "b": a}


def test_an_ambiguous_bare_class_name_is_an_error():
    """Two DataModules named Data in different modules: the CLI must not pick one silently
    (found when renamed test files changed the import order)."""
    from gpulayer.pytorch.cli import _resolve_class

    ns: dict = {}
    exec("import gpulayer.pytorch as glt\nclass Twin(glt.DataModule):\n    pass", ns)
    ns2: dict = {}
    exec("import gpulayer.pytorch as glt\nclass Twin(glt.DataModule):\n    pass", ns2)
    with pytest.raises(ValidationFailed, match="ambiguous"):
        _resolve_class("Twin", (glt.DataModule,))
