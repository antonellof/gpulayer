"""Paths in object stores and other remote filesystems, through ``fsspec`` (optional).

``as_path`` is the one place a user's path becomes an object: a ``pathlib.Path`` for local
paths, a ``RemotePath`` with the same methods for a URL such as ``s3://bucket/run``. A URL
that cannot be served is an error at once: it must never turn into a local folder named
``s3:``, where a checkpoint would vanish with the machine.

Credentials come from ``fsspec``'s usual sources (the environment, the cloud SDK's
configuration). They are refused in URLs, because paths are printed in logs.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from gpulayer.errors import ValidationFailed

_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_PASSWORD = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/@]*:[^/@]*@")


def is_remote(path: Any) -> bool:
    """True for a ``RemotePath`` or a URL other than ``file://``."""
    if isinstance(path, RemotePath):
        return True
    return isinstance(path, str) and bool(_URL.match(path)) and not path.startswith("file://")


def as_path(path: str | os.PathLike[str] | RemotePath) -> Any:
    """``path`` as a ``Path``, or a ``RemotePath`` for a URL. Typed ``Any`` because callers use
    the methods both have."""
    if isinstance(path, RemotePath):
        return path
    if isinstance(path, str) and path.startswith("file://"):
        return Path(path[len("file://") :])
    return RemotePath(path) if is_remote(path) else Path(path)


def _filesystem(url: str) -> tuple[Any, str]:
    if _PASSWORD.match(url):
        raise ValidationFailed(
            "credentials in a path would be printed in logs; use the environment or the "
            f"cloud SDK's configuration (path scheme {url.split('://')[0]}://)"
        )
    try:
        from fsspec.core import url_to_fs
    except ImportError as e:
        raise ValidationFailed(
            f"remote path {url!r} needs fsspec: pip install 'gpulayer[remote]', plus the "
            "store's package (s3fs, gcsfs, adlfs)"
        ) from e
    try:
        fs, path = url_to_fs(url)
    except ImportError as e:
        raise ValidationFailed(f"remote path {url!r} needs a package: {e}") from e
    except ValueError as e:
        raise ValidationFailed(f"remote path {url!r}: no filesystem for this scheme ({e})") from e
    return fs, path


class RemotePath:
    """The ``pathlib.Path`` methods the Trainer uses, on an ``fsspec`` filesystem. It has no
    ``__fspath__``: code that needs a local file fails loudly instead of writing elsewhere."""

    def __init__(self, url: str | os.PathLike[str], _fs: Any = None) -> None:
        self._url = os.fspath(url).rstrip("/")
        if _fs is None:
            self._fs, self._path = _filesystem(self._url)
        else:
            self._fs, self._path = _fs, _fs._strip_protocol(self._url)

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"RemotePath({self._url!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RemotePath) and other._url == self._url

    def __lt__(self, other: RemotePath) -> bool:
        return self._url < other._url

    def __hash__(self) -> int:
        return hash(self._url)

    def __truediv__(self, other: str | os.PathLike[str]) -> RemotePath:
        return RemotePath(f"{self._url}/{os.fspath(other)}", self._fs)

    @property
    def name(self) -> str:
        return self._url.rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix

    @property
    def parent(self) -> RemotePath:
        return RemotePath(self._url.rsplit("/", 1)[0], self._fs)

    def is_absolute(self) -> bool:
        return True

    def resolve(self) -> RemotePath:
        return self

    def exists(self) -> bool:
        return bool(self._fs.exists(self._path))

    def is_dir(self) -> bool:
        return bool(self._fs.isdir(self._path))

    def is_file(self) -> bool:
        return bool(self._fs.isfile(self._path))

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        self._fs.makedirs(self._path, exist_ok=True)  # object stores have no folders

    def unlink(self, missing_ok: bool = False) -> None:
        if missing_ok and not self.exists():
            return
        self._fs.rm_file(self._path)

    def _child(self, found: str) -> RemotePath:
        relative = found[len(self._fs._strip_protocol(self._url)) :].lstrip("/")
        return self / relative

    def iterdir(self) -> list[RemotePath]:
        return [self._child(p) for p in self._fs.ls(self._path, detail=False)]

    def glob(self, pattern: str) -> list[RemotePath]:
        return [self._child(p) for p in self._fs.glob(f"{self._path}/{pattern}")]

    def rglob(self, pattern: str) -> list[RemotePath]:
        return self.glob(f"**/{pattern}")

    def stat(self) -> SimpleNamespace:
        info = self._fs.info(self._path)
        stamp = next(
            (info[k] for k in ("mtime", "LastModified", "last_modified", "updated", "created")
             if info.get(k) is not None),
            0.0,
        )  # fmt: skip
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        when = stamp.timestamp() if isinstance(stamp, datetime) else float(stamp)
        return SimpleNamespace(st_mtime=when, st_size=info.get("size", 0))

    def open(self, mode: str = "r", **kwargs: Any) -> Any:
        return self._fs.open(self._path, mode, **kwargs)

    def read_text(self) -> str:
        with self.open("r") as f:
            return str(f.read())

    def write_text(self, text: str) -> None:
        with self.open("w") as f:
            f.write(text)

    def put(self, local: str | os.PathLike[str]) -> None:
        """Upload a finished local file. A failed write leaves nothing at this path."""
        self._fs.put_file(os.fspath(local), self._path)

    def get(self, local: str | os.PathLike[str]) -> None:
        """Download this file."""
        self._fs.get_file(self._path, os.fspath(local))


def save_file(checkpoint: dict[str, Any], path: RemotePath) -> None:
    """Write to a local file first: the store never holds half a checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "checkpoint"
        torch.save(checkpoint, local)
        path.put(local)


def load_file(path: str | Path | RemotePath, **torch_load: Any) -> Any:
    """``torch.load``, from a local path or a URL (downloaded first: loading seeks a lot)."""
    target = as_path(path)
    if isinstance(target, Path):
        return torch.load(target, **torch_load)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "checkpoint"
        target.get(local)
        return torch.load(local, **torch_load)
