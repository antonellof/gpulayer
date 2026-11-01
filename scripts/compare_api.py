"""List PyTorch Lightning public names that gpulayer.pytorch does not have.

    git clone https://github.com/Lightning-AI/pytorch-lightning /tmp/lightning
    python scripts/compare_api.py /tmp/lightning

Compares the ``__all__`` of Lightning's callbacks, loggers, profilers, plugins, strategies,
accelerators and utilities, the Trainer's arguments and members, and the LightningModule's
methods and hooks. Prints what is missing (nothing when GPUlayer covers them) and exits 1 if
anything is. It reads Lightning's source and does not import it.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path


def _all_of(path: Path) -> list[str]:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "__all__" for t in node.targets
        ):
            return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]  # type: ignore[attr-defined]
    return []


def _class(path: Path, name: str) -> ast.ClassDef:
    return next(
        c
        for c in ast.parse(path.read_text()).body
        if isinstance(c, ast.ClassDef) and c.name == name
    )


# Reference names GPUlayer provides under its own name (its code carries no Lightning names).
RENAMED = {
    "LightningModule": "Module",
    "LightningDataModule": "DataModule",
    "LightningCLI": "GPULayerCLI",
    "LightningArgumentParser": "GPULayerArgumentParser",
    "LightningEnum": "CaseInsensitiveEnum",
    "lightning_module": "module",
}


def main(root: str) -> int:
    import torch  # noqa: F401  (gpulayer.pytorch needs it)

    import gpulayer.pytorch as glt
    from gpulayer.pytorch import utilities

    src = Path(root) / "src" / "lightning" / "pytorch"
    missing: list[str] = []
    for sub in (
        "callbacks",
        "loggers",
        "profilers",
        "plugins",
        "strategies",
        "accelerators",
        "utilities",
    ):
        for name in _all_of(src / sub / "__init__.py"):
            name = RENAMED.get(name, name)
            if not hasattr(glt, name) and not hasattr(utilities, name):
                missing.append(f"{sub}.{name}")
    trainer_cls = _class(src / "trainer" / "trainer.py", "Trainer")
    init = next(
        f for f in trainer_cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    ours = inspect.signature(glt.Trainer.__init__).parameters
    missing += [
        f"Trainer({a.arg}=)" for a in init.args.args[1:] + init.args.kwonlyargs if a.arg not in ours
    ]
    trainer = glt.Trainer()
    for f in trainer_cls.body:
        if (
            isinstance(f, ast.FunctionDef)
            and not f.name.startswith("_")
            and not hasattr(trainer, RENAMED.get(f.name, f.name))
        ):
            missing.append(f"Trainer.{f.name}")

    class _M(glt.Module):
        pass

    model = _M()
    for rel in ("core/module.py", "core/hooks.py"):
        for c in ast.parse((src / rel).read_text()).body:
            if isinstance(c, ast.ClassDef):
                for f in c.body:
                    if (
                        isinstance(f, ast.FunctionDef)
                        and not f.name.startswith("_")
                        and not hasattr(model, f.name)
                        and not hasattr(glt.DataModule, f.name)
                    ):
                        missing.append(f"{c.name}.{f.name}")
    for m in sorted(set(missing)):
        print(m)
    print(f"{len(set(missing))} missing", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
