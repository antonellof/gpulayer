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

``limits`` caps the number of batches taken from each loader (one number for all, or one per
loader), as in Lightning: ``len`` and iteration then use ``min(len(loader), limit)`` per loader.

``Trainer.fit`` wraps a list or dict of training loaders in ``max_size_cycle`` mode; each
loader is sharded across processes on its own.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Iterator
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
        self._limits: list[int | float] | None = None

    @property
    def flattened(self) -> list[Any]:
        """The loaders, in order."""
        return list(self._items)

    @flattened.setter
    def flattened(self, loaders: list[Any]) -> None:
        if len(loaders) != len(self._items):
            raise ValidationFailed("flattened: the number of loaders cannot change")
        self._items = list(loaders)
        self.iterables = self._map(lambda x: x)

    def _map(self, fn: Callable[[Any], Any]) -> Any:
        """``fn`` of each loader, in the structure of ``iterables``."""
        values = [fn(x) for x in self._items]
        if self._keys is None and not isinstance(self.iterables, (list, tuple)):
            return values[0]
        return self._pack(values)

    @property
    def sampler(self) -> Any:
        """The loaders' ``sampler`` attributes (None for a loader without one), in the same
        structure as ``iterables``: a dict, list or tuple, or a single value."""
        return self._map(lambda x: getattr(x, "sampler", None))

    @property
    def batch_sampler(self) -> Any:
        """The loaders' ``batch_sampler`` attributes (None for a loader without one), in the
        same structure as ``iterables``."""
        return self._map(lambda x: getattr(x, "batch_sampler", None))

    @property
    def limits(self) -> list[int | float] | None:
        """The maximum number of batches taken from each loader, or None for no limit."""
        return self._limits

    @limits.setter
    def limits(self, limits: int | float | list[int | float] | None) -> None:
        if isinstance(limits, (int, float)):
            limits = [limits] * len(self._items)
        elif isinstance(limits, (list, tuple)):
            if len(limits) != len(self._items):
                raise ValidationFailed(
                    f"Mismatch in number of limits ({len(limits)}) and number of iterables "
                    f"({len(self._items)})"
                )
            limits = list(limits)
        elif limits is not None:
            raise ValidationFailed(f"limits must be a number, a list or None, got {limits!r}")
        if limits is not None and any(lim < 0 for lim in limits):
            raise ValidationFailed(f"limits must not be negative, got {limits}")
        self._limits = limits

    def reset(self) -> None:
        """Shut down the worker processes of loaders that keep them between epochs (the next
        iteration starts from the beginning)."""
        for loader in self._items:
            _shutdown_workers(loader)

    def _limit(self, i: int) -> int | None:
        if self._limits is None or math.isinf(self._limits[i]):
            return None
        return int(self._limits[i])

    def _iter(self, i: int) -> Iterator[Any]:
        it = iter(self._items[i])
        limit = self._limit(i)
        return it if limit is None else itertools.islice(it, limit)

    def _lengths(self) -> list[int]:
        try:
            lengths = [len(x) for x in self._items]
        except TypeError as e:
            raise TypeError("CombinedLoader has no length: a loader has none") from e
        out = []
        for i, n in enumerate(lengths):
            limit = self._limit(i)
            out.append(n if limit is None else min(n, limit))
        return out

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
            for idx in range(len(self._items)):
                for batch_idx, batch in enumerate(self._iter(idx)):
                    yield batch, batch_idx, idx
            return
        its: list[Iterator[Any] | None] = [self._iter(i) for i in range(len(self._items))]
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
                        its[i] = self._iter(i)
                        values.append(next(its[i], None))  # type: ignore[arg-type]
                    else:  # max_size
                        its[i] = None
                        values.append(None)
            yield self._pack(values)
            step += 1


def _shutdown_workers(loader: Iterable[Any]) -> None:
    """Drop a DataLoader's persistent worker iterator, as Lightning's ``reset`` does."""
    it = getattr(loader, "_iterator", None)
    if it is None:
        return
    shutdown = getattr(it, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    loader._iterator = None  # type: ignore[attr-defined]


def loaders_of(value: Any) -> list[Any]:
    """The loaders in a single loader, a list, a dict or a CombinedLoader."""
    if isinstance(value, CombinedLoader):
        return value.flattened
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
