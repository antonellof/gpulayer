"""Write docs/pytorch/api.md, the reference of ``gpulayer.pytorch``, from the code.

    python scripts/make_api_reference.py          # write the page
    python scripts/make_api_reference.py --check  # exit 1 if the page is out of date

Every public class, function, method and property exported by ``gpulayer.pytorch`` is listed
with its signature and docstring. Edit the docstrings, not the page.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "docs" / "pytorch" / "api.md"

# Sections, in page order: (title, intro, exported names).
SECTIONS: list[tuple[str, str, list[str]]] = [
    (
        "Trainer",
        "Runs the loop here or launches it on workers. Guide: [Trainer and Module](trainer.md).",
        ["Trainer", "LRSchedulerConfig"],
    ),
    (
        "Module and DataModule",
        "Your model and data. Guide: [Trainer and Module](trainer.md).",
        ["Module", "DataModule", "CombinedLoader", "AttributeDict"],
    ),
    (
        "Strategies",
        "How the model is replicated or sharded across processes.",
        ["DDPStrategy", "FSDPStrategy", "DeepSpeedStrategy"],
    ),
    (
        "Callbacks",
        "Guide: [Callbacks, loggers and profilers](callbacks.md).",
        [
            "Callback",
            "ModelCheckpoint",
            "EarlyStopping",
            "LearningRateMonitor",
            "Timer",
            "GradientAccumulationScheduler",
            "StochasticWeightAveraging",
            "WeightAveraging",
            "EMAWeightAveraging",
            "BaseFinetuning",
            "BackboneFinetuning",
            "ModelPruning",
            "ThroughputMonitor",
            "DeviceStatsMonitor",
            "SpikeDetection",
            "ModelSummary",
            "ProgressBar",
            "BasePredictionWriter",
            "OnExceptionCheckpoint",
            "LambdaCallback",
            "LearningRateFinder",
            "BatchSizeFinder",
            "TrainingSpikeException",
        ],
    ),
    (
        "Loggers",
        "Called from rank 0 with values the Trainer already read.",
        ["Logger", "CSVLogger", "TensorBoardLogger", "WandbLogger", "MLFlowLogger"],
    ),
    (
        "Profilers",
        'Pass one as `Trainer(profiler=...)`, or `"simple"`, `"advanced"`, `"pytorch"`.',
        [
            "Profiler",
            "SimpleProfiler",
            "AdvancedProfiler",
            "PyTorchProfiler",
            "PassThroughProfiler",
        ],
    ),
    ("Tuner", "Learning-rate and batch-size search.", ["Tuner"]),
    (
        "Plugins",
        "Pass as `Trainer(plugins=[...])`.",
        [
            "Precision",
            "MixedPrecision",
            "HalfPrecision",
            "DoublePrecision",
            "CheckpointIO",
            "TorchCheckpointIO",
            "AsyncCheckpointIO",
            "TorchSyncBatchNorm",
        ],
    ),
    (
        "Command line",
        "Guide: [Command line](cli.md). `glt.LightningCLI` is the same class.",
        ["GPULayerCLI"],
    ),
    (
        "Launch and runtime",
        "Lower layers, usable without the Trainer. Guides: [launch()](launch.md), "
        "[Runtime helpers](runtime.md).",
        [
            "launch",
            "LaunchResult",
            "is_launcher_process",
            "in_gpulayer_job",
            "Runtime",
            "resolve_accelerator",
            "seed_everything",
        ],
    ),
]

# Members not worth a line: nn.Module's own API, dataclass internals, aliases.
SKIP_INHERITED_FROM = ("torch.nn.modules.module",)


def _doc(obj: Any) -> str:
    """The docstring, with reStructuredText ``code`` turned into Markdown `code`."""
    doc = inspect.getdoc(obj) or ""
    return re.sub(r"``([^`]+)``", r"`\1`", doc.strip())


def _params(obj: Any) -> list[inspect.Parameter] | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [p for p in sig.parameters.values() if p.name not in ("self", "cls")]


def _param_text(p: inspect.Parameter, annotate: bool) -> str:
    prefix = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "")
    text = prefix + p.name
    if annotate and p.annotation is not inspect.Parameter.empty:
        ann = (
            p.annotation
            if isinstance(p.annotation, str)
            else inspect.formatannotation(p.annotation)
        )
        text += f": {ann}"
    if p.default is not inspect.Parameter.empty:
        d = p.default
        shown = d.__name__ if callable(d) and hasattr(d, "__name__") else repr(d)
        shown = re.sub(r" at 0x[0-9a-f]+", "", shown)  # no memory addresses in the page
        text += f" = {shown}" if annotate else f"={shown}"
    return text


def _sig(obj: Any, name: str) -> str:
    """Short form for tables: argument names and defaults, no types."""
    params = _params(obj)
    if params is None:
        return f"{name}(...)"
    return f"{name}({', '.join(_param_text(p, False) for p in params)})"


def _sig_block(obj: Any, name: str) -> str:
    """Long form for code blocks: one argument per line when it does not fit on one."""
    params = _params(obj)
    if params is None:
        return f"{name}(...)"
    parts = []
    for p in params:
        if p.kind == p.KEYWORD_ONLY and not any(q.startswith("*") for q in parts):
            parts.append("*")
        parts.append(_param_text(p, True))
    one = f"{name}({', '.join(parts)})"
    if len(one) <= 96:
        return one
    return f"{name}(\n" + "".join(f"    {x},\n" for x in parts) + ")"


def _own_members(cls: type, exported: set[str]) -> list[tuple[str, Any]]:
    """Public methods and properties defined by cls, or by private gpulayer bases (such as
    the hyperparameter mixin); members of exported bases are listed under those bases."""
    out: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        mod = getattr(klass, "__module__", "")
        if not mod.startswith("gpulayer"):
            continue
        if klass is not cls and klass.__name__ in exported:
            continue
        for name, value in vars(klass).items():
            if name.startswith("_"):
                continue
            if any(
                name in vars(base)
                for base in cls.__mro__[1:]
                if base.__name__ in exported and base is not klass
            ):
                continue  # an override: documented on the exported base
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if callable(value) or isinstance(value, property):
                out[name] = value
    return sorted(out.items(), key=lambda kv: kv[0])


def _exported_bases(cls: type, exported: set[str]) -> list[str]:
    return [
        b.__name__
        for b in cls.__mro__[1:]
        if b.__name__ in exported and getattr(b, "__module__", "").startswith("gpulayer")
    ]


def _first_para(doc: str) -> str:
    return doc.split("\n\n", 1)[0].replace("\n", " ") if doc else ""


def render() -> str:
    import gpulayer.pytorch as glt

    lines = [
        "# PyTorch API reference",
        "",
        "Every public name of `gpulayer.pytorch` (`import gpulayer.pytorch as glt`), with its",
        "signature and docstring. This page is generated by `scripts/make_api_reference.py` from",
        "the code; the guides explain how the pieces fit together.",
        "",
        "`glt.LightningModule` and `glt.LightningDataModule` are the same classes as `glt.Module`",
        "and `glt.DataModule`, for code moving from PyTorch Lightning.",
        "",
    ]
    listed: set[str] = set()
    for title, intro, names in SECTIONS:
        lines += [f"## {title}", "", intro, ""]
        for name in names:
            obj = getattr(glt, name)
            listed.add(name)
            lines += _render_object(name, obj)
    missing = sorted(
        n
        for n in glt.__all__
        if n not in listed
        and n not in ("LightningModule", "LightningDataModule", "TQDMProgressBar", "LightningCLI")
    )
    if missing:
        raise SystemExit(f"exported but not placed in a section: {missing}")
    return "\n".join(lines).rstrip() + "\n"


def _render_object(name: str, obj: Any) -> list[str]:
    out = [f"### `{name}`", ""]
    if inspect.isclass(obj):
        init = obj.__init__ if "__init__" in vars(obj) else None
        head = (
            _sig_block(obj, f"glt.{name}")
            if init or not issubclass(obj, BaseException)
            else f"glt.{name}"
        )
        out += ["```python", head, "```", ""]
        doc = _doc(obj)
        if doc:
            out += [doc, ""]
        if issubclass(obj, BaseException):
            return out
        import gpulayer.pytorch as glt

        exported = set(glt.__all__)
        bases = _exported_bases(obj, exported)
        if bases:
            out += [f"Inherits from {', '.join(f'`{b}`' for b in bases)}.", ""]
        members = _own_members(obj, exported)
        props = [(n, v) for n, v in members if isinstance(v, property)]
        methods = [(n, v) for n, v in members if not isinstance(v, property)]
        hooks = [
            (n, v)
            for n, v in methods
            if n.startswith("on_") or n in ("setup", "teardown", "prepare_data")
        ]
        methods = [(n, v) for n, v in methods if (n, v) not in hooks]
        if methods:
            out += ["| Method | |", "|---|---|"]
            for n, v in methods:
                out.append(f"| `{_sig(v, n)}` | {_cell(_first_para(_doc(v)))} |")
            out.append("")
        if props:
            out += ["| Property | |", "|---|---|"]
            for n, v in props:
                out.append(f"| `{n}` | {_cell(_first_para(_doc(v)))} |")
            out.append("")
        if hooks:
            out += ["| Hook | |", "|---|---|"]
            for n, v in hooks:
                out.append(f"| `{_sig(v, n)}` | {_cell(_first_para(_doc(v)))} |")
            out.append("")
    else:
        out += ["```python", _sig_block(obj, f"glt.{name}"), "```", ""]
        doc = _doc(obj)
        if doc:
            out += [doc, ""]
    return out


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = render()
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(
                f"{OUT} is out of date: run python scripts/make_api_reference.py", file=sys.stderr
            )
            raise SystemExit(1)
        return
    OUT.write_text(text)
    print(OUT)


if __name__ == "__main__":
    main()
