"""A command line for training scripts, like PyTorch Lightning's ``LightningCLI``.

    # train.py
    from gpulayer.pytorch.cli import GPULayerCLI
    GPULayerCLI(MyModel, MyDataModule)

    python train.py fit --trainer.max_epochs 10 --model.lr 0.01 --data.batch_size 64
    python train.py fit --config experiment.yaml --trainer.devices 4
    python train.py fit --print_config > experiment.yaml
    python train.py test --ckpt_path best.ckpt

Arguments come from the ``__init__`` signatures of the Trainer, the model and the DataModule,
converted with their type annotations (int, float, bool, str, lists and dicts as JSON,
``None``/``null``). A config file (JSON, or YAML when PyYAML is installed) has the sections
``trainer``, ``model``, ``data``, ``seed_everything`` and ``ckpt_path``; command-line values
override it. Objects are written as ``{"class_path": "pkg.Class", "init_args": {...}}``, for
example callbacks, loggers, or the model itself when no model class is given. Uses only the
standard library (Lightning's uses ``jsonargparse``).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any

from gpulayer.errors import ValidationFailed

SUBCOMMANDS = ("fit", "validate", "test", "predict")


def instantiate(spec: Any) -> Any:
    """Build ``{"class_path": ..., "init_args": {...}}`` (nested ones too); other values are
    returned as they are."""
    if isinstance(spec, list):
        return [instantiate(v) for v in spec]
    if isinstance(spec, dict) and "class_path" in spec:
        module, _, name = spec["class_path"].rpartition(".")
        if not module:
            raise ValidationFailed(f"class_path must be module.Class, got {spec['class_path']!r}")
        cls = getattr(importlib.import_module(module), name)
        args = {k: instantiate(v) for k, v in (spec.get("init_args") or {}).items()}
        return cls(**args)
    return spec


def _coerce(text: str, annotation: Any) -> Any:
    """A command-line string as the annotated type."""
    low = text.strip().lower()
    if low in ("none", "null"):
        return None
    ann = annotation
    if isinstance(ann, str):  # postponed annotations: read the annotation's text
        names = [n for n in ann.replace(" ", "").split("|") if n not in ("None",)]
        simple = {"bool": bool, "int": int, "float": float, "str": str}
        ann = simple.get(names[0], Any) if len(names) == 1 else Any
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        ann = args[0] if len(args) == 1 else Any
    if ann is bool:
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValidationFailed(f"expected true or false, got {text!r}")
    if ann is int:
        return int(text)
    if ann is float:
        return float(text)
    if ann is str:
        return text
    if low in ("true", "false"):
        return low == "true"
    try:  # numbers, lists, dicts and class_path objects as JSON; anything else as text
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _params(cls: Any) -> dict[str, inspect.Parameter]:
    if cls is None:
        return {}
    sig = inspect.signature(cls.__init__)
    return {
        n: p
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _load_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise ValidationFailed(
                "YAML configs need PyYAML (pip install pyyaml); JSON works without it"
            ) from e
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _dump(config: dict[str, Any]) -> str:
    try:
        import yaml

        return str(yaml.safe_dump(config, sort_keys=False))
    except ImportError:
        return json.dumps(config, indent=2, default=str)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            _merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict) and "class_path" not in v
            else v
        )
    return out


class GPULayerCLI:
    """Parse the command line, build the model, data and Trainer, and run the subcommand.

    After construction: ``cli.config`` (the merged configuration), ``cli.model``,
    ``cli.datamodule``, ``cli.trainer`` and ``cli.subcommand``. With ``run=False`` there are no
    subcommands and nothing is run, so the script can call the Trainer itself."""

    def __init__(
        self,
        model_class: type | None = None,
        datamodule_class: type | None = None,
        trainer_class: type | None = None,
        trainer_defaults: dict[str, Any] | None = None,
        seed_everything_default: int | bool | None = True,
        args: list[str] | dict[str, Any] | None = None,
        run: bool = True,
        description: str | None = None,
    ) -> None:
        from gpulayer.pytorch.trainer import Trainer

        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.trainer_class = trainer_class or Trainer
        self.trainer_defaults = dict(trainer_defaults or {})
        self.seed_default = seed_everything_default
        argv = args if isinstance(args, list) else sys.argv[1:] if args is None else []
        parser = self._parser(run, description)
        ns = parser.parse_args(argv)
        self.subcommand: str | None = getattr(ns, "subcommand", None) if run else None
        config: dict[str, Any] = {
            "seed_everything": self.seed_default,
            "trainer": dict(self.trainer_defaults),
            "model": {},
            "data": {},
            "ckpt_path": None,
        }
        if isinstance(args, dict):
            config = _merge(config, args)
        for path in getattr(ns, "config", None) or []:
            config = _merge(config, _load_config(path))
        config = _merge(config, self._from_args(ns))
        self.config = config
        if getattr(ns, "print_config", False):
            print(_dump(config))
            raise SystemExit(0)
        self._instantiate()
        if run:
            self._run()

    # ---------------------------------------------------------------- parsing
    def _parser(self, run: bool, description: str | None) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        if not run:
            self._add_arguments(parser)
            return parser
        sub = parser.add_subparsers(dest="subcommand", required=True)
        for name in SUBCOMMANDS:
            self._add_arguments(sub.add_parser(name, help=f"Trainer.{name}"))
        return parser

    def _add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", action="append", help="a JSON or YAML config file (repeatable)")
        p.add_argument(
            "--print_config", action="store_true", help="print the merged config and exit"
        )
        p.add_argument("--seed_everything", default=argparse.SUPPRESS)
        p.add_argument("--ckpt_path", default=argparse.SUPPRESS)
        sections = [
            ("trainer", self.trainer_class),
            ("model", self.model_class),
            ("data", self.datamodule_class),
        ]
        for section, cls in sections:
            if cls is None and section != "trainer":
                p.add_argument(
                    f"--{section}",
                    default=argparse.SUPPRESS,
                    help=f"the {section} as JSON with class_path and init_args",
                )
                continue
            group = p.add_argument_group(section)
            for name, param in _params(cls).items():
                default = None if param.default is inspect.Parameter.empty else param.default
                group.add_argument(
                    f"--{section}.{name}",
                    dest=f"{section}.{name}",
                    default=argparse.SUPPRESS,
                    metavar=str(param.annotation).replace("typing.", "")
                    if param.annotation is not inspect.Parameter.empty
                    else "VALUE",
                    help=f"default: {default!r}",
                )

    def _from_args(self, ns: argparse.Namespace) -> dict[str, Any]:
        out: dict[str, Any] = {}
        classes = {
            "trainer": self.trainer_class,
            "model": self.model_class,
            "data": self.datamodule_class,
        }
        for key, value in vars(ns).items():
            if key in ("config", "print_config", "subcommand"):
                continue
            if key == "seed_everything":
                out[key] = _coerce(value, int | bool)
            elif key == "ckpt_path":
                out[key] = None if value.lower() in ("none", "null") else value
            elif key in ("model", "data", "trainer"):
                out[key] = json.loads(value)
            elif "." in key:
                section, name = key.split(".", 1)
                param = _params(classes[section]).get(name)
                ann = param.annotation if param is not None else Any
                out.setdefault(section, {})[name] = _coerce(value, ann)
        return out

    # ---------------------------------------------------------------- building
    def _build(self, cls: type | None, spec: dict[str, Any]) -> Any:
        if "class_path" in spec:
            return instantiate(spec)
        if cls is None:
            return None
        return cls(**{k: instantiate(v) for k, v in spec.items()})

    def _instantiate(self) -> None:
        from gpulayer.pytorch.runtime import seed_everything

        seed = self.config.get("seed_everything")
        if seed is True:
            self.config["seed_everything"] = seed = seed_everything()
        elif isinstance(seed, int) and seed is not False:
            seed_everything(seed)
        self.model = self._build(self.model_class, self.config.get("model") or {})
        if self.model is None:
            raise ValidationFailed("no model: pass model_class or --model with class_path")
        self.datamodule = self._build(self.datamodule_class, self.config.get("data") or {})
        trainer_args = {k: instantiate(v) for k, v in (self.config.get("trainer") or {}).items()}
        self.trainer = self.trainer_class(**trainer_args)

    def _run(self) -> None:
        name = self.subcommand
        assert name is not None
        fn = getattr(self.trainer, name)
        kwargs: dict[str, Any] = (
            {"datamodule": self.datamodule} if self.datamodule is not None else {}
        )
        ckpt = self.config.get("ckpt_path")
        if ckpt:
            kwargs["ckpt_path"] = ckpt
        self.result = fn(self.model, **kwargs)


LightningCLI = GPULayerCLI
