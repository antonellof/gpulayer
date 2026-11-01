"""``CombinedLoader``: several loaders iterated together, as in PyTorch Lightning.

    loader = CombinedLoader({"images": a, "text": b}, mode="max_size_cycle")
    for batch in loader:            # {"images": batch_a, "text": batch_b}
        ...

Modes:

- ``max_size_cycle`` (default for training): as many steps as the longest loader; shorter
  loaders start again from the beginning.
- ``min_size``: stop when the shortest loader runs out.
- ``max_size``: as many steps as the longest loader; exhausted loaders give ``None``.
- ``sequential``: one loader after the other; yields ``(batch, batch_idx, dataloader_idx)``.

``Trainer.fit`` wraps a list or dict of training loaders in ``max_size_cycle`` mode; each
loader is sharded across processes on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from gpulayer.errors import ValidationFailed

MODES = ("max_size_cycle", "min_size", "max_size", "sequential")


class CombinedLoader:
    """Iterate a list or dict of loaders together (see the module docstring for the modes)."""

    def __init__(self, iterables: Any, mode: str = "max_size_cycle") -> None:
        if mode not in MODES:
            raise ValidationFailed(f"mode must be one of {MODES}, got {mode!r}")
        if isinstance(iterables, dict):
            self._keys: list[Any] | None = list(iterables)
            self._items = list(iterables.values())
        elif isinstance(iterables, (list, tuple)):
            self._keys = None
            self._items = list(iterables)
        else:
            self._keys, self._items = None, [iterables]
        if not self._items:
            raise ValidationFailed("CombinedLoader needs at least one loader")
        self.iterables = iterables
        self.mode = mode

    @property
    def flattened(self) -> list[Any]:
        """The loaders, in order."""
        return list(self._items)

    @flattened.setter
    def flattened(self, loaders: list[Any]) -> None:
        if len(loaders) != len(self._items):
            raise ValidationFailed("flattened: the number of loaders cannot change")
        self._items = list(loaders)

    def _lengths(self) -> list[int]:
        try:
            return [len(x) for x in self._items]
        except TypeError as e:
            raise TypeError("CombinedLoader has no length: a loader has none") from e

    def __len__(self) -> int:
        n = self._lengths()
        if self.mode == "sequential":
            return sum(n)
        return min(n) if self.mode == "min_size" else max(n)

    def _pack(self, values: list[Any]) -> Any:
        if self._keys is not None:
            return dict(zip(self._keys, values, strict=True))
        return values if isinstance(self.iterables, list) else tuple(values)

    def __iter__(self) -> Iterator[Any]:
        if self.mode == "sequential":
            for idx, loader in enumerate(self._items):
                for batch_idx, batch in enumerate(loader):
                    yield batch, batch_idx, idx
            return
        its: list[Iterator[Any] | None] = [iter(x) for x in self._items]
        done = [False] * len(its)
        try:
            steps = len(self)
        except TypeError:
            steps = None
        step = 0
        while steps is None or step < steps:
            values = []
            for i, it in enumerate(its):
                try:
                    if it is None:
                        raise StopIteration
                    values.append(next(it))
                except StopIteration:
                    done[i] = True
                    if self.mode == "min_size" or (steps is None and all(done)):
                        return
                    if self.mode == "max_size_cycle":
                        its[i] = iter(self._items[i])
                        values.append(next(its[i]))  # type: ignore[arg-type]
                    else:  # max_size
                        its[i] = None
                        values.append(None)
            yield self._pack(values)
            step += 1


def loaders_of(value: Any) -> list[Any]:
    """The loaders in a single loader, a list, a dict or a CombinedLoader."""
    if isinstance(value, CombinedLoader):
        return value.flattened
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

