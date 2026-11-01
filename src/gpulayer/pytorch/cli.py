"""A command line for training scripts, like PyTorch Lightning's ``LightningCLI``.

    # train.py
    from gpulayer.pytorch.cli import GPULayerCLI
    GPULayerCLI(MyModel, MyDataModule)

    python train.py fit --trainer.max_epochs 10 --model.lr 0.01 --data.batch_size 64
    python train.py fit --config experiment.yaml --trainer.devices 4
    python train.py fit --print_config > experiment.yaml
    python train.py fit --optimizer Adam --optimizer.lr 0.001 --lr_scheduler StepLR \\
        --lr_scheduler.step_size 10
    python train.py test --ckpt_path best.ckpt

Arguments come from the ``__init__`` signatures of the Trainer, the model and the DataModule,
converted with their type annotations (int, float, bool, str, lists and dicts as JSON,
``None``/``null``). A config file (JSON, or YAML when PyYAML is installed) has the sections
``trainer``, ``model``, ``data``, ``optimizer``, ``lr_scheduler``, ``seed_everything`` and the
subcommand's arguments (``ckpt_path``, ...); command-line values override it. Objects are
written as ``{"class_path": "pkg.Class", "init_args": {...}}``, for example callbacks, loggers,
or the model itself when no model class is given (subclass mode: ``--model MyModel
--model.lr 0.1``).

The class calls the same overridable methods as Lightning's, in the same order:
``setup_parser`` (``init_parser``, then for each parser ``add_default_arguments_to_parser``,
``add_core_arguments_to_parser``, ``add_arguments_to_parser`` and
``link_optimizers_and_lr_schedulers``), ``parse_arguments``, ``before_instantiate_classes``,
``instantiate_classes`` (which calls ``instantiate_trainer``), ``after_instantiate_classes``,
then ``before_<subcommand>``, the subcommand and ``after_<subcommand>``.

Differences from Lightning's: it uses only the standard library (``argparse``, no
``jsonargparse``), so ``cli.config`` is a flat dict (not nested under the subcommand name),
type checking is limited to the conversions above, and ``SaveConfigCallback`` overwrites
``config.yaml`` by default because GPUlayer reuses one run folder per script.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import types
import typing
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.callbacks.base import Callback
from gpulayer.pytorch.module import DataModule, Module, overridden

SUBCOMMANDS = ("fit", "validate", "test", "predict")
_NOT_CONFIG = ("config", "main_config", "print_config", "subcommand", "help")


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """``torch.optim.lr_scheduler.ReduceLROnPlateau`` with the ``monitor`` it reads, so
    ``--lr_scheduler ReduceLROnPlateau --lr_scheduler.monitor val_loss`` works from the CLI."""

    def __init__(self, optimizer: torch.optim.Optimizer, monitor: str, *args: Any, **kwargs: Any):
        super().__init__(optimizer, *args, **kwargs)
        self.monitor = monitor


LRSchedulerTypeTuple = (torch.optim.lr_scheduler.LRScheduler, ReduceLROnPlateau)


# ------------------------------------------------------------------ helpers


def _import_class(path: str) -> Any:
    module, _, name = path.rpartition(".")
    if not module:
        raise ValidationFailed(f"class_path must be module.Class, got {path!r}")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise ValidationFailed(f"cannot import {path!r}: {e}") from e


def _class_path_of(cls: Any) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _subclasses(bases: tuple[type, ...]) -> list[type]:
    seen: list[type] = []
    todo = list(bases)
    while todo:
        c = todo.pop()
        if c not in seen:
            seen.append(c)
            todo.extend(c.__subclasses__())
    return seen


def _resolve_class(name: str, bases: tuple[type, ...]) -> Any:
    """``pkg.mod.Class``, or a bare class name looked up among the subclasses of ``bases``
    (classes defined here first, then ``torch``'s, then any other)."""
    if "." in name:
        return _import_class(name)
    found = [c for c in _subclasses(bases) if c.__name__ == name]
    if not found:
        raise ValidationFailed(f"unknown class {name!r}: give module.Class or import it first")
    found.sort(key=lambda c: (c.__module__ != __name__, not c.__module__.startswith("torch")))
    return found[0]


def instantiate(spec: Any) -> Any:
    """Build ``{"class_path": ..., "init_args": {...}}`` (nested ones too); other values are
    returned as they are."""
    if isinstance(spec, list):
        return [instantiate(v) for v in spec]
    if isinstance(spec, dict) and "class_path" in spec:
        cls = _import_class(spec["class_path"])
        args = {k: instantiate(v) for k, v in (spec.get("init_args") or {}).items()}
        args.update({k: instantiate(v) for k, v in (spec.get("dict_kwargs") or {}).items()})
        return cls(**args)
    return spec


def _has_class_path(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_class_path(v) for v in value)
    return isinstance(value, dict) and "class_path" in value


def _record_instantiator(obj: Any, init: dict[str, Any], class_path: str | None) -> None:
    """Keep what is needed to rebuild ``obj`` in its hparams, as Lightning's CLI does with
    ``load_from_checkpoint_support``: the class path and the ``class_path`` specs of injected
    objects (``save_hyperparameters`` leaves out modules)."""
    hp = getattr(obj, "hparams", None)
    if not hp:  # the class does not call save_hyperparameters
        return
    hp["_instantiator"] = f"{__name__}.instantiate_module"
    if class_path:
        hp["_class_path"] = class_path
    for k, v in init.items():
        if _has_class_path(v):
            hp[k] = v


def instantiate_module(class_type: Any, config: dict[str, Any]) -> Any:
    """Build a Module or DataModule from hyperparameters saved by the CLI (``_class_path``
    and nested ``class_path`` specs). ``load_from_checkpoint`` calls it when the checkpoint's
    hyperparameters name it as ``_instantiator``."""
    args = dict(config)
    args.pop("_instantiator", None)
    class_path = args.pop("_class_path", None)
    cls = _import_class(class_path) if class_path else class_type
    if isinstance(class_type, type) and not (isinstance(cls, type) and issubclass(cls, class_type)):
        raise ValidationFailed(f"{class_path} is not a subclass of {class_type.__name__}")
    accepted = _params(cls, keep_var_keyword=True)
    if not any(p.kind == p.VAR_KEYWORD for p in accepted.values()):
        args = {k: v for k, v in args.items() if k in accepted}
    obj = cls(**{k: instantiate(v) for k, v in args.items()})
    _record_instantiator(obj, args, class_path)
    return obj


def _coerce(text: str, annotation: Any) -> Any:
    """A command-line string as the annotated type."""
    low = text.strip().lower()
    if low in ("none", "null"):
        return None
    ann = annotation
    if isinstance(ann, str):  # postponed annotations: read the annotation's text
        simple = {"bool": bool, "int": int, "float": float, "str": str, "Path": str}
        names = [n for n in ann.replace(" ", "").split("|") if n != "None"]
        kinds = {simple.get(n, Any) for n in names}
        ann = kinds.pop() if len(kinds) == 1 else Any
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
    if ann in (str, Path):
        return text
    if low in ("true", "false"):
        return low == "true"
    try:  # numbers, lists, dicts and class_path objects as JSON; anything else as text
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _params(cls: Any, keep_var_keyword: bool = False) -> dict[str, inspect.Parameter]:
    """The keyword parameters of a class's ``__init__`` (or of a function)."""
    if cls is None:
        return {}
    fn = cls.__init__ if inspect.isclass(cls) else cls
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    return {
        n: p
        for n, p in sig.parameters.items()
        if n != "self"
        and p.kind != p.VAR_POSITIONAL
        and (keep_var_keyword or p.kind != p.VAR_KEYWORD)
    }


def _load_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            try:  # JSON is valid YAML
                return dict(json.loads(text))
            except json.JSONDecodeError:
                pass
            raise ValidationFailed(
                "YAML configs need PyYAML (pip install pyyaml); JSON works without it"
            ) from e
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _plain(value: Any) -> Any:
    """``value`` with objects that YAML or JSON cannot write replaced by their ``repr``."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _dump(config: dict[str, Any]) -> str:
    """The config as YAML (JSON when PyYAML is not installed; JSON is valid YAML)."""
    plain = _plain(config)
    try:
        import yaml

        return str(yaml.safe_dump(plain, sort_keys=False))
    except ImportError:
        return json.dumps(plain, indent=2)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            _merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict) and "class_path" not in v
            else v
        )
    return out


def _split(spec: Any) -> tuple[str | None, dict[str, Any]]:
    """``(class_path, init_args)`` of a section: a class_path spec or a dict of arguments."""
    if not isinstance(spec, dict):
        return None, {}
    init = {k: v for k, v in spec.items() if k not in ("class_path", "init_args", "dict_kwargs")}
    init.update(spec.get("init_args") or {})
    init.update(spec.get("dict_kwargs") or {})
    return spec.get("class_path"), init


def _cfg_get(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if isinstance(node, dict) and "class_path" in node and part not in node:
            node = node.get("init_args") or {}
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _cfg_set(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        if "class_path" in node and part not in node:
            node = node.setdefault("init_args", {})
        child = node.get(part)
        if not isinstance(child, dict):
            child = node[part] = {}
        node = child
    if "class_path" in node and parts[-1] not in ("class_path", "init_args"):
        node = node.setdefault("init_args", {})
    node[parts[-1]] = value


class _Raw(str):
    """A command-line value for a subclass-mode section, converted once its class is known."""


class _ClassSpec:
    """Marker: the argument names a class (``Name``, ``pkg.Class`` or a JSON spec)."""


def _class_spec(text: str) -> Any:
    low = text.strip().lower()
    if low in ("none", "null"):
        return None
    if text.lstrip().startswith("{"):
        return json.loads(text)
    return {"class_path": text}


def _short_doc(fn: Any) -> str | None:
    doc = inspect.getdoc(fn) if fn is not None else None
    return doc.strip().split("\n")[0] if doc else None


def _broadcast(value: Any) -> Any:
    """Rank 0's ``value`` on every rank (the value itself without a process group)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


class _Section:
    def __init__(
        self,
        key: str,
        bases: tuple[Any, ...],
        default_class: Any,
        subclass_mode: bool,
        required: bool,
        skip: set[str],
    ) -> None:
        self.key = key
        self.bases = bases
        self.default_class = default_class
        self.subclass_mode = subclass_mode
        self.required = required
        self.skip = skip


# ------------------------------------------------------------------- parser


class GPULayerArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` with the methods of Lightning's ``LightningArgumentParser``
    that CLI subclasses use: ``add_lightning_class_args``, ``add_class_arguments``,
    ``add_optimizer_args``, ``add_lr_scheduler_args``, ``add_method_arguments``,
    ``link_arguments`` and ``set_defaults`` with a dict of dotted keys."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)  # --model.lr must not match --model.lr_decay
        kwargs.pop("dump_header", None)  # a jsonargparse option
        super().__init__(*args, **kwargs)
        self.callback_keys: list[str] = []
        self._optimizers: dict[str, tuple[Any, str]] = {}
        self._lr_schedulers: dict[str, tuple[Any, str]] = {}
        self._sections: dict[str, _Section] = {}
        self._coerce_types: dict[str, Any] = {}
        self._links: list[tuple[str, str, Callable[[Any], Any] | None, str]] = []
        self._config_defaults: dict[str, Any] = {}
        self._gl_defaults: dict[str, Any] | None = None

    def add_class_arguments(
        self,
        theclass: Any,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
        skip: set[str] | None = None,
        default_class: Any = None,
    ) -> list[str]:
        """Add ``--<nested_key>.<arg>`` for each ``__init__`` argument of ``theclass``. In
        subclass mode, add ``--<nested_key> Name`` (a subclass of ``theclass``, which may be a
        tuple of base classes) and accept ``--<nested_key>.<arg>`` for the chosen class.
        Returns the added keys."""
        skip = set(skip or ())
        bases = theclass if isinstance(theclass, tuple) else (theclass,)
        if not subclass_mode and default_class is None:
            default_class = theclass
        self._sections[nested_key] = _Section(
            nested_key,
            tuple(b for b in bases if isinstance(b, type)),
            default_class,
            subclass_mode,
            required,
            skip,
        )
        group = self.add_argument_group(nested_key)
        if subclass_mode:
            names = ", ".join(b.__name__ for b in bases if isinstance(b, type))
            group.add_argument(
                f"--{nested_key}",
                default=argparse.SUPPRESS,
                metavar="CLASS",
                help=f"a subclass of {names}: Name, module.Class or JSON with class_path and "
                f"init_args; then --{nested_key}.ARG VALUE for its arguments",
            )
            self._coerce_types[nested_key] = _ClassSpec
            return [nested_key]
        added = []
        for name, param in _params(theclass).items():
            if name in skip:
                continue
            default = None if param.default is inspect.Parameter.empty else param.default
            dest = f"{nested_key}.{name}"
            group.add_argument(
                f"--{dest}",
                dest=dest,
                default=argparse.SUPPRESS,
                metavar=str(param.annotation).replace("typing.", "")
                if param.annotation is not inspect.Parameter.empty
                else "VALUE",
                help=f"default: {default!r}",
            )
            self._coerce_types[dest] = param.annotation
            added.append(dest)
        return added

    def add_lightning_class_args(
        self,
        lightning_class: Any,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
    ) -> list[str]:
        """Add the arguments of a Trainer, Module, DataModule or Callback class (or a function
        returning one) under ``nested_key``. Callbacks added this way are built and given to
        the Trainer."""
        if inspect.isclass(lightning_class) and issubclass(lightning_class, Callback):
            self.callback_keys.append(nested_key)
        default = None
        if subclass_mode and lightning_class not in (Module, DataModule):
            default = lightning_class  # the class itself when no subclass is named
        return self.add_class_arguments(
            lightning_class, nested_key, subclass_mode, required, default_class=default
        )

    def add_optimizer_args(
        self,
        optimizer_class: Any = (torch.optim.Optimizer,),
        nested_key: str = "optimizer",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """Add optimizer arguments: one class (``--optimizer.lr``) or, for a tuple of base
        classes, any subclass (``--optimizer Adam --optimizer.lr 0.01``). ``link_to="AUTOMATIC"``
        lets the CLI define ``configure_optimizers``; a dotted key such as ``model.optimizer``
        passes the ``class_path`` spec to that argument instead."""
        classes = optimizer_class if isinstance(optimizer_class, tuple) else (optimizer_class,)
        if not all(inspect.isclass(c) and issubclass(c, torch.optim.Optimizer) for c in classes):
            raise ValidationFailed(f"not an optimizer class: {optimizer_class}")
        self.add_class_arguments(
            optimizer_class,
            nested_key,
            subclass_mode=isinstance(optimizer_class, tuple),
            required=False,
            skip={"params"},
        )
        self._optimizers[nested_key] = (optimizer_class, link_to)

    def add_lr_scheduler_args(
        self,
        lr_scheduler_class: Any = LRSchedulerTypeTuple,
        nested_key: str = "lr_scheduler",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """Add learning-rate scheduler arguments, as ``add_optimizer_args``."""
        classes = (
            lr_scheduler_class if isinstance(lr_scheduler_class, tuple) else (lr_scheduler_class,)
        )
        if not all(inspect.isclass(c) and issubclass(c, LRSchedulerTypeTuple) for c in classes):
            raise ValidationFailed(f"not a learning-rate scheduler class: {lr_scheduler_class}")
        self.add_class_arguments(
            lr_scheduler_class,
            nested_key,
            subclass_mode=isinstance(lr_scheduler_class, tuple),
            required=False,
            skip={"optimizer"},
        )
        self._lr_schedulers[nested_key] = (lr_scheduler_class, link_to)

    def add_method_arguments(
        self, theclass: Any, themethod: str, nested_key: str | None = None, skip: Any = None
    ) -> list[str]:
        """Add ``--<arg>`` (or ``--<nested_key>.<arg>``) for the arguments of a method, such
        as ``Trainer.fit``'s ``ckpt_path``. Returns the argument names."""
        skip = set(skip or ())
        added = []
        for name, param in _params(getattr(theclass, themethod)).items():
            if name in skip:
                continue
            dest = f"{nested_key}.{name}" if nested_key else name
            default = param.default
            simple = default is None or isinstance(default, (bool, int, float, str))
            self.add_argument(
                f"--{dest}",
                dest=dest,
                default=default if simple else argparse.SUPPRESS,
                help=f"{theclass.__name__}.{themethod}({name}=...)",
            )
            self._coerce_types[dest] = param.annotation
            added.append(name)
        return added

    def link_arguments(
        self,
        source: str,
        target: str,
        compute_fn: Callable[[Any], Any] | None = None,
        apply_on: str = "parse",
    ) -> None:
        """Set ``target`` (for example ``model.num_classes``) from ``source``: a config value
        after parsing (``apply_on="parse"``), or an attribute of a built object
        (``apply_on="instantiate"``, e.g. ``data.num_classes`` read from the DataModule).
        ``compute_fn`` transforms the value. Linked values are passed to ``__init__`` as they
        are (a ``class_path`` spec is not built)."""
        if apply_on not in ("parse", "instantiate"):
            raise ValidationFailed("apply_on must be 'parse' or 'instantiate'")
        self._links.append((source, target, compute_fn, apply_on))

    def set_defaults(self, *args: Any, **kwargs: Any) -> None:
        """``argparse``'s ``set_defaults(**kwargs)``, plus Lightning's form: a dict of dotted
        keys, such as ``{"trainer.max_epochs": 5}``."""
        for d in args:
            if not isinstance(d, dict):
                raise ValidationFailed("set_defaults takes dicts of dotted keys")
            self._config_defaults.update(d)
        super().set_defaults(**kwargs)

    def _take_defaults(self) -> dict[str, Any]:
        """The defaults as a nested config; argparse then only reports given values."""
        if self._gl_defaults is None:
            out: dict[str, Any] = {}
            for action in self._actions:
                if action.default is argparse.SUPPRESS or action.dest in _NOT_CONFIG:
                    continue
                if isinstance(action, argparse._SubParsersAction):
                    continue
                _cfg_set(out, action.dest, action.default)
                action.default = argparse.SUPPRESS
            for key, value in self._config_defaults.items():
                _cfg_set(out, key, value)
            for key, value in self._defaults.items():
                if key not in _NOT_CONFIG:
                    _cfg_set(out, key, value)
            self._defaults = {}
            self._gl_defaults = out
        return self._gl_defaults


LightningArgumentParser = GPULayerArgumentParser


# ----------------------------------------------------------- SaveConfigCallback


def _default_log_dir(trainer: Any) -> str:
    """Where the Trainer writes this run when no logger names a folder."""
    from gpulayer.pytorch.trainer import _run_name

    root = getattr(trainer, "default_root_dir", "gpulayer_runs")
    return os.environ.get("GPULAYER_OUTPUT_DIR") or str(Path(root) / _run_name())


class SaveConfigCallback(Callback):
    """Write the CLI's resolved config to ``<log_dir>/config.yaml`` when the run starts
    (``setup``), on global rank 0, as Lightning's. ``config.yaml`` is YAML when PyYAML is
    installed, else JSON (valid YAML). ``overwrite`` defaults to True because GPUlayer reuses
    one run folder per script (Lightning's default is False); with False an existing file
    raises ``RuntimeError``. ``multifile`` is accepted and ignored (one file is written).
    Override ``save_config`` to also save it elsewhere, for example to a logger."""

    def __init__(
        self,
        parser: Any,
        config: dict[str, Any],
        config_filename: str = "config.yaml",
        overwrite: bool = True,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.save_to_log_dir = save_to_log_dir
        self.already_saved = False
        if not save_to_log_dir and not overridden(self, "save_config", SaveConfigCallback):
            raise ValidationFailed(
                "save_to_log_dir=False only makes sense in a subclass that implements save_config"
            )

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Write the config file (rank 0), then call ``save_config``."""
        if self.already_saved:
            return
        if self.save_to_log_dir:
            log_dir = Path(trainer.log_dir or _default_log_dir(trainer))
            path = log_dir / self.config_filename
            if not self.overwrite:
                exists = _broadcast(path.is_file() if trainer.is_global_zero else False)
                if exists:
                    raise RuntimeError(
                        f"{type(self).__name__} expected {path} to NOT exist. Delete it, set "
                        "save_config_callback=None, or save_config_kwargs={'overwrite': True}"
                    )
            if trainer.is_global_zero:
                log_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(_dump(self.config))
        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True
        self.already_saved = bool(_broadcast(self.already_saved))

    def save_config(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Override to save the config somewhere else too (runs on rank 0 only)."""


class _AutoConfigureOptimizers:
    """``configure_optimizers`` set on the model from ``--optimizer``/``--lr_scheduler``. The
    optimizer is built when the Trainer calls it, from the parameters the model has then."""

    def __init__(
        self,
        model: Any,
        fn: Callable[..., Any],
        optimizer: tuple[Any, dict[str, Any]],
        lr_scheduler: tuple[Any, dict[str, Any]] | None,
    ) -> None:
        self.model = model
        self.fn = fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.__name__ = "configure_optimizers"

    def __call__(self) -> Any:
        cls, init = self.optimizer
        optimizer = cls(self.model.parameters(), **{k: instantiate(v) for k, v in init.items()})
        scheduler = None
        if self.lr_scheduler is not None:
            scls, sinit = self.lr_scheduler
            scheduler = scls(optimizer, **{k: instantiate(v) for k, v in sinit.items()})
        return self.fn(self.model, optimizer=optimizer, lr_scheduler=scheduler)


# ---------------------------------------------------------------------- CLI


class GPULayerCLI:
    """Parse the command line, build the model, data and Trainer, and run the subcommand.

    After construction: ``cli.config`` (the merged configuration, a flat dict),
    ``cli.config_init`` (the built objects), ``cli.model``, ``cli.datamodule``,
    ``cli.trainer``, ``cli.subcommand`` and ``cli.parser``. With ``run=False`` there are no
    subcommands and nothing is run, so the script can call the Trainer itself.

    Arguments as in Lightning: ``save_config_callback`` writes the config to the run's folder
    (None disables it), with ``save_config_kwargs``; ``parser_kwargs`` go to the parser (or,
    under a subcommand's name, to that subcommand's parser); ``parser_class`` must subclass
    ``GPULayerArgumentParser``; ``subclass_mode_model``/``subclass_mode_data`` accept any
    subclass by name (``--model MyModel``); ``auto_configure_optimizers`` adds
    ``--optimizer`` and ``--lr_scheduler``, which replace the model's
    ``configure_optimizers`` when given; ``load_from_checkpoint_support`` records the class
    path and injected ``class_path`` specs in the hparams of models and DataModules that call
    ``save_hyperparameters``."""

    def __init__(
        self,
        model_class: Any = None,
        datamodule_class: Any = None,
        save_config_callback: type[SaveConfigCallback] | None = SaveConfigCallback,
        save_config_kwargs: dict[str, Any] | None = None,
        trainer_class: Any = None,
        trainer_defaults: dict[str, Any] | None = None,
        seed_everything_default: int | bool | None = True,
        parser_kwargs: dict[str, Any] | None = None,
        parser_class: type[GPULayerArgumentParser] = GPULayerArgumentParser,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: list[str] | dict[str, Any] | None = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
        load_from_checkpoint_support: bool = True,
        description: str | None = None,
    ) -> None:
        from gpulayer.pytorch.trainer import Trainer

        if not (inspect.isclass(parser_class) and issubclass(parser_class, GPULayerArgumentParser)):
            raise ValidationFailed("parser_class must subclass GPULayerArgumentParser")
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = dict(save_config_kwargs or {})
        self.trainer_class = trainer_class or Trainer
        self.trainer_defaults = dict(trainer_defaults or {})
        self.seed_everything_default = seed_everything_default
        self.seed_default = seed_everything_default
        self.parser_kwargs = dict(parser_kwargs or {})
        self.parser_class = parser_class
        self.auto_configure_optimizers = auto_configure_optimizers
        self.load_from_checkpoint_support = load_from_checkpoint_support

        self.model_class = model_class
        self._model_class = model_class or Module
        self.subclass_mode_model = model_class is None or subclass_mode_model
        self.datamodule_class = datamodule_class
        self._datamodule_class = datamodule_class or DataModule
        self.subclass_mode_data = datamodule_class is None or subclass_mode_data

        self._raw_targets: set[str] = set()
        self._linked_values: dict[str, Any] = {}
        self._subcommand_parsers: dict[str, GPULayerArgumentParser] = {}
        self._subcommand_method_arguments: dict[str, list[str]] = {}

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(self.parser_kwargs)
        if description is not None:
            main_kwargs.setdefault("description", description)
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args)
        self.subcommand: str | None = self.config.get("subcommand") if run else None
        self.config.pop("subcommand", None)
        self._parse_ckpt_path()
        self._set_seed()
        self.before_instantiate_classes()
        self.instantiate_classes()
        self.after_instantiate_classes()
        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    # ---------------------------------------------------------------- parser
    def _setup_parser_kwargs(
        self, parser_kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        names = self.subcommands().keys()
        main = {k: v for k, v in parser_kwargs.items() if k not in names}
        sub = {k: v for k, v in parser_kwargs.items() if k in names}
        return main, sub

    def init_parser(self, **kwargs: Any) -> GPULayerArgumentParser:
        """Create a parser (``parser_class(**kwargs)``) with ``-c/--config`` and
        ``--print_config``."""
        parser = self.parser_class(**kwargs)
        parser.add_argument(
            "-c",
            "--config",
            action="append",
            default=argparse.SUPPRESS,
            help="a JSON or YAML config file (repeatable; later files win)",
        )
        parser.add_argument(
            "--print_config",
            action="store_true",
            default=argparse.SUPPRESS,
            help="print the merged config and exit",
        )
        return parser

    def setup_parser(
        self, add_subcommands: bool, main_kwargs: dict[str, Any], subparser_kwargs: dict[str, Any]
    ) -> None:
        """Create ``self.parser`` and add the subcommands (or, with ``run=False``, the
        arguments)."""
        self.parser = self.init_parser(**main_kwargs)
        if add_subcommands:
            for action in self.parser._actions:  # the subcommand's own --config must not
                if action.dest == "config":  # replace a config given before it
                    action.dest = "main_config"
            self._subcommand_method_arguments = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)
            self.parser.add_argument("--ckpt_path", default=None, help="a checkpoint path")
            self.parser._coerce_types["ckpt_path"] = "str | None"

    def add_default_arguments_to_parser(self, parser: GPULayerArgumentParser) -> None:
        """Add ``--seed_everything``."""
        parser.add_argument(
            "--seed_everything",
            default=self.seed_everything_default,
            help="an int seeds everything before building the classes; true picks a seed; "
            "false does not seed",
        )
        parser._coerce_types["seed_everything"] = int | bool

    def add_core_arguments_to_parser(self, parser: GPULayerArgumentParser) -> None:
        """Add the Trainer's, the model's and the DataModule's arguments."""
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {
            "trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"
        }
        parser.set_defaults(trainer_defaults)
        parser.add_lightning_class_args(
            self._model_class, "model", subclass_mode=self.subclass_mode_model
        )
        parser.add_lightning_class_args(
            self._datamodule_class,
            "data",
            subclass_mode=self.subclass_mode_data,
            required=self.datamodule_class is not None,
        )

    def _add_arguments(self, parser: GPULayerArgumentParser) -> None:
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        if self.auto_configure_optimizers:
            if not parser._optimizers:  # unless add_arguments_to_parser added some
                parser.add_optimizer_args((torch.optim.Optimizer,))
            if not parser._lr_schedulers:
                parser.add_lr_scheduler_args(LRSchedulerTypeTuple)
        self.link_optimizers_and_lr_schedulers(parser)

    def add_arguments_to_parser(self, parser: GPULayerArgumentParser) -> None:
        """Override to add arguments or links (``parser.link_arguments``)."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """The subcommands, each with the Trainer method arguments not offered as flags."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
        }

    def _add_subcommands(self, parser: GPULayerArgumentParser, **kwargs: Any) -> None:
        self._subcommand_parsers = {}
        factory: Any = lambda **kw: self.init_parser(**kw)  # noqa: E731 (init_parser per parser)
        sub = parser.add_subparsers(dest="subcommand", required=True, parser_class=factory)
        for name, skip in self.subcommands().items():
            description = _short_doc(getattr(self.trainer_class, name, None))
            kw = dict(kwargs.get(name, {}))
            kw.setdefault("description", description)
            p = sub.add_parser(name, help=description, **kw)
            self._add_arguments(p)
            added = p.add_method_arguments(self.trainer_class, name, skip=set(skip))
            self._subcommand_method_arguments[name] = added
            self._subcommand_parsers[name] = p

    @staticmethod
    def link_optimizers_and_lr_schedulers(parser: GPULayerArgumentParser) -> None:
        """Link optimizer and scheduler arguments whose ``link_to`` is not ``AUTOMATIC`` to
        that key (for example ``model.optimizer``)."""
        for key, (_, link_to) in {**parser._optimizers, **parser._lr_schedulers}.items():
            if link_to != "AUTOMATIC":
                parser.link_arguments(key, link_to)

    def _parser(self, subcommand: str | None) -> GPULayerArgumentParser:
        return self.parser if subcommand is None else self._subcommand_parsers[subcommand]

    # --------------------------------------------------------------- parsing
    def parse_arguments(self, parser: GPULayerArgumentParser, args: Any) -> None:
        """Parse the command line (``args``: a list of strings, a dict config, or None for
        ``sys.argv``) and store the merged configuration in ``self.config``."""
        given: dict[str, Any] = {}
        if isinstance(args, dict):
            given = dict(args)
            argv = [str(given.pop("subcommand"))] if "subcommand" in given else []
        else:
            argv = list(args) if args is not None else sys.argv[1:]
        parsers = [parser, *self._subcommand_parsers.values()]
        defaults = {id(p): p._take_defaults() for p in parsers}
        ns, extras = parser.parse_known_args(argv)
        sub = getattr(ns, "subcommand", None)
        active = self._subcommand_parsers[sub] if sub else parser

        config: dict[str, Any] = {"seed_everything": self.seed_everything_default}
        for key, section in active._sections.items():
            config[key] = None if section.subclass_mode and not section.default_class else {}
        config = _merge(config, defaults[id(parser)])
        if active is not parser:
            config = _merge(config, defaults[id(active)])
        if given:
            config = _merge(config, given.get(sub, given) if sub else given)
        for path in getattr(ns, "main_config", None) or []:
            loaded = _load_config(path)
            config = _merge(
                config, loaded[sub] if sub and isinstance(loaded.get(sub), dict) else loaded
            )
        for path in getattr(ns, "config", None) or []:
            config = _merge(config, _load_config(path))
        config = _merge(config, self._from_args(active, ns))
        config = _merge(config, self._from_extras(active, extras))
        self._normalize(active, config)
        for source, target, fn, apply_on in active._links:
            if apply_on != "parse":
                continue
            try:
                value = _cfg_get(config, source)
            except KeyError:
                continue
            _cfg_set(config, target, fn(value) if fn is not None else value)
            self._raw_targets.add(target)
        if sub:
            config["subcommand"] = sub
        self.config = config
        if getattr(ns, "print_config", False):
            print(_dump({k: v for k, v in config.items() if k != "subcommand"}))
            raise SystemExit(0)

    def _from_args(self, parser: GPULayerArgumentParser, ns: argparse.Namespace) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in vars(ns).items():
            if key in _NOT_CONFIG:
                continue
            ann = parser._coerce_types.get(key)
            if ann is _ClassSpec:
                value = _class_spec(value)
            elif key in parser._coerce_types and isinstance(value, str):
                value = _coerce(value, ann)
            _cfg_set(out, key, value)
        return out

    def _from_extras(self, parser: GPULayerArgumentParser, extras: list[str]) -> dict[str, Any]:
        """``--<section>.<arg> VALUE`` for subclass-mode sections, whose arguments depend on
        the class named."""
        out: dict[str, Any] = {}
        unknown: list[str] = []
        i = 0
        while i < len(extras):
            token = extras[i]
            i += 1
            key = token[2:] if token.startswith("--") else ""
            section_key, _, rest = key.partition(".")
            section = parser._sections.get(section_key)
            if not rest or section is None or not section.subclass_mode:
                unknown.append(token)
                continue
            if "=" in rest:
                rest, value = rest.split("=", 1)
            elif i < len(extras):
                value = extras[i]
                i += 1
            else:
                parser.error(f"{token} needs a value")
            node = out.setdefault(section_key, {})
            if rest == "class_path":
                node["class_path"] = value
                continue
            for prefix in ("init_args.", "dict_kwargs."):
                rest = rest.removeprefix(prefix)
            if "." in rest:
                unknown.append(token)
                continue
            node.setdefault("init_args", {})[rest] = _Raw(value)
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        return out

    def _normalize(self, parser: GPULayerArgumentParser, config: dict[str, Any]) -> None:
        """Resolve subclass-mode sections to ``{"class_path": full path, "init_args": ...}``
        and convert their command-line values with the class's annotations."""
        for key, section in parser._sections.items():
            spec = config.get(key)
            if not section.subclass_mode or spec is None:
                continue
            class_path, init = _split(spec)
            cls = _resolve_class(class_path, section.bases) if class_path else None
            cls = cls or section.default_class
            if cls is None:
                if init:
                    raise ValidationFailed(f"--{key}: name a class (--{key} ClassName)")
                config[key] = None
                continue
            bases = tuple(b for b in section.bases if b not in (Module, DataModule))
            if inspect.isclass(cls) and bases and not issubclass(cls, bases):
                names = ", ".join(b.__name__ for b in bases)
                raise ValidationFailed(f"--{key}: {cls.__name__} is not a subclass of {names}")
            params = _params(cls)
            for name, value in list(init.items()):
                if isinstance(value, _Raw):
                    param = params.get(name)
                    ann = param.annotation if param is not None else Any
                    init[name] = _coerce(str(value), ann)
            config[key] = {"class_path": _class_path_of(cls), "init_args": init}

    def _parse_ckpt_path(self) -> None:
        """With a checkpoint path, use the model hyperparameters saved in it, as Lightning."""
        if not self.subcommand:
            return
        ckpt_path = self.config.get("ckpt_path")
        if not ckpt_path or not Path(str(ckpt_path)).is_file():
            return
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        hparams = dict(ckpt.get("hyper_parameters") or {}) if isinstance(ckpt, dict) else {}
        hparams.pop("_instantiator", None)
        if not hparams:
            return
        class_path = hparams.pop("_class_path", None)
        old_path, init = _split(self.config.get("model"))
        class_path = class_path or old_path
        cls = _import_class(class_path) if class_path else self.model_class
        accepted = _params(cls, keep_var_keyword=True)
        if accepted and not any(p.kind == p.VAR_KEYWORD for p in accepted.values()):
            hparams = {k: v for k, v in hparams.items() if k in accepted}
        init.update(hparams)
        self.config["model"] = {"class_path": class_path, "init_args": init} if class_path else init

    def _set_seed(self) -> None:
        from gpulayer.pytorch.runtime import seed_everything

        seed = self.config.get("seed_everything")
        if seed is None or seed is False:
            return
        self.config["seed_everything"] = seed_everything(None if seed is True else int(seed))

    # ------------------------------------------------------------- building
    def before_instantiate_classes(self) -> None:
        """Override to run code before the classes are built (``self.config`` is set)."""

    def instantiate_classes(self) -> None:
        """Build the extra sections (callbacks added with ``add_lightning_class_args``), the
        DataModule, the model and the Trainer."""
        parser = self._parser(self.subcommand)
        skip = {"trainer", "model", "data", *parser._optimizers, *parser._lr_schedulers}
        self.config_init: dict[str, Any] = {}
        for key in parser._sections:
            if key not in skip:
                self.config_init[key] = self._build_section(parser, key)
        self.datamodule = self._build_section(parser, "data")
        self.config_init["data"] = self.datamodule
        for source, target, fn, apply_on in parser._links:
            if apply_on != "instantiate":
                continue
            section, _, attr = source.partition(".")
            value: Any = self.config_init.get(section)
            for part in attr.split(".") if attr else []:
                value = getattr(value, part)
            self._linked_values[target] = fn(value) if fn is not None else value
        self.model = self._build_section(parser, "model")
        if self.model is None:
            raise ValidationFailed("no model: pass model_class or --model with class_path")
        self.config_init["model"] = self.model
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    def after_instantiate_classes(self) -> None:
        """Override to run code after the classes are built."""

    def _build_section(self, parser: GPULayerArgumentParser, key: str) -> Any:
        section = parser._sections.get(key)
        spec = self.config.get(key)
        if section is None or spec is None:
            return None
        class_path, init = _split(spec)
        cls = _import_class(class_path) if class_path else section.default_class
        if cls is None:
            return None
        args = {
            k: v if f"{key}.{k}" in self._raw_targets else instantiate(v) for k, v in init.items()
        }
        for target, value in self._linked_values.items():
            sec, _, name = target.partition(".")
            if sec == key and name:
                args[name] = value
        obj = cls(**args)
        if self.load_from_checkpoint_support and key in ("model", "data"):
            _record_instantiator(obj, init, class_path if section.subclass_mode else None)
        return obj

    def instantiate_trainer(self, **kwargs: Any) -> Any:
        """Build the Trainer from the ``trainer`` section; ``kwargs`` override it."""
        parser = self._parser(self.subcommand)
        callbacks = [self.config_init[k] for k in parser.callback_keys if self.config_init.get(k)]
        trainer_config = {k: instantiate(v) for k, v in (self.config.get("trainer") or {}).items()}
        trainer_config.update(kwargs)
        return self._instantiate_trainer(trainer_config, callbacks)

    def _instantiate_trainer(self, config: dict[str, Any], callbacks: list[Any]) -> Any:
        if "callbacks" in _params(self.trainer_class):
            value = config.get("callbacks")
            cbs = [] if value is None else list(value) if isinstance(value, list) else [value]
            cbs.extend(callbacks)
            if "callbacks" in self.trainer_defaults:
                extra = self.trainer_defaults["callbacks"]
                cbs += extra if isinstance(extra, list) else [extra]
            if self.save_config_callback is not None and not config.get("fast_dev_run", False):
                cbs.append(
                    self.save_config_callback(
                        self._parser(self.subcommand), self.config, **self.save_config_kwargs
                    )
                )
            config["callbacks"] = cbs
        else:
            warnings.warn(
                f"{getattr(self.trainer_class, '__qualname__', self.trainer_class)} has no "
                "callbacks argument: CLI callbacks are not included",
                stacklevel=2,
            )
        return self.trainer_class(**config)

    # ------------------------------------------------------------ optimizers
    @staticmethod
    def configure_optimizers(
        lightning_module: Any, optimizer: torch.optim.Optimizer, lr_scheduler: Any = None
    ) -> Any:
        """What the model's ``configure_optimizers`` returns when ``--optimizer`` is given.
        Override to change it."""
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }
        return [optimizer], [lr_scheduler]

    def _section_class(
        self, parser: GPULayerArgumentParser, key: str
    ) -> tuple[Any, dict[str, Any]] | None:
        section = parser._sections[key]
        spec = self.config.get(key)
        if spec is None:
            return None
        class_path, init = _split(spec)
        cls = _import_class(class_path) if class_path else section.default_class
        return None if cls is None else (cls, init)

    def _add_configure_optimizers_method_to_model(self, subcommand: str | None) -> None:
        if not self.auto_configure_optimizers:
            return
        parser = self._parser(subcommand)
        optimizers = [k for k, (_, link) in parser._optimizers.items() if link == "AUTOMATIC"]
        schedulers = [k for k, (_, link) in parser._lr_schedulers.items() if link == "AUTOMATIC"]
        if not optimizers:
            return
        if len(optimizers) > 1 or len(schedulers) > 1:
            raise ValidationFailed(
                "at most one optimizer and one lr_scheduler can be AUTOMATIC, found "
                f"{optimizers + schedulers}: link them and implement configure_optimizers"
            )
        optimizer = self._section_class(parser, optimizers[0])
        if optimizer is None:
            return
        scheduler = self._section_class(parser, schedulers[0]) if schedulers else None
        if overridden(self.model, "configure_optimizers", Module):
            warnings.warn(
                f"{type(self.model).__name__}.configure_optimizers will be overridden by "
                f"{type(self).__name__}.configure_optimizers",
                stacklevel=2,
            )
        self.model.configure_optimizers = _AutoConfigureOptimizers(
            self.model, type(self).configure_optimizers, optimizer, scheduler
        )

    # ---------------------------------------------------------------- running
    def _run_subcommand(self, subcommand: str) -> None:
        before = getattr(self, f"before_{subcommand}", None)
        if callable(before):
            before()
        fn = getattr(self, subcommand, getattr(self.trainer, subcommand))
        self.result = fn(**self._prepare_subcommand_kwargs(subcommand))
        after = getattr(self, f"after_{subcommand}", None)
        if callable(after):
            after()

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        names = self._subcommand_method_arguments.get(subcommand, [])
        kwargs = {k: self.config[k] for k in names if k in self.config}
        if kwargs.get("ckpt_path") is None:
            kwargs.pop("ckpt_path", None)
        kwargs["model"] = self.model
        if self.datamodule is not None:
            kwargs["datamodule"] = self.datamodule
        return kwargs


LightningCLI = GPULayerCLI
