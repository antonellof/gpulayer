"""Content-addressed artifact storage and deterministic source bundles.

Artifacts are identified by SHA-256; the URI ``artifact://sha256/<hex>`` is location
independent, so a worker that does not share a filesystem with the submitter fetches
the bytes through the control plane and verifies the digest before use.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol

from gpulayer.contracts import ArtifactKind, ArtifactReference
from gpulayer.errors import NotFound, ValidationFailed

MAX_ARTIFACT_BYTES = 2 * 1024**3
_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".gpulayer",
    "gpulayer_runs",  # Trainer/launch outputs: results, not source
}


def ref_for(
    data: bytes, name: str = "", kind: ArtifactKind = ArtifactKind.INPUT
) -> ArtifactReference:
    digest = hashlib.sha256(data).hexdigest()
    return ArtifactReference(
        uri=f"artifact://sha256/{digest}",
        sha256=digest,
        size_bytes=len(data),
        name=name,
        kind=kind,
    )


class ArtifactStore(Protocol):
    def put(
        self, data: bytes, name: str = "", kind: ArtifactKind = ArtifactKind.INPUT
    ) -> ArtifactReference: ...
    def get(self, sha256: str) -> bytes: ...
    def exists(self, sha256: str) -> bool: ...
    def list(self) -> Iterator[StoredArtifact]: ...
    def delete(self, sha256: str) -> None: ...


@dataclass(frozen=True)
class StoredArtifact:
    sha256: str
    size_bytes: int
    stored_at: datetime


class LocalArtifactStore:
    """Filesystem implementation. A remote object store implements the same protocol."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValidationFailed("invalid sha256 digest")
        return self.root / sha256[:2] / sha256

    def put(
        self, data: bytes, name: str = "", kind: ArtifactKind = ArtifactKind.INPUT
    ) -> ArtifactReference:
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValidationFailed("artifact exceeds maximum size")
        ref = ref_for(data, name, kind)
        path = self._path(ref.sha256)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic: readers never see partial content
        return ref

    def get(self, sha256: str) -> bytes:
        path = self._path(sha256)
        if not path.exists():
            raise NotFound(f"artifact {sha256} not found")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValidationFailed(f"artifact {sha256} failed integrity check")
        return data

    def exists(self, sha256: str) -> bool:
        return self._path(sha256).exists()

    def list(self) -> Iterator[StoredArtifact]:
        for path in self.root.glob("??/*"):
            if len(path.name) == 64 and path.is_file():
                st = path.stat()
                yield StoredArtifact(
                    path.name, st.st_size, datetime.fromtimestamp(st.st_mtime, timezone.utc)
                )

    def delete(self, sha256: str) -> None:
        self._path(sha256).unlink(missing_ok=True)


def build_source_bundle(directory: str | Path) -> bytes:
    """Deterministic zip of ``directory`` (sorted entries, fixed timestamps)."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValidationFailed(f"source directory not found: {directory}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            # Also skip any GpuLayer home (metadata.db + artifacts/): bundling the artifact
            # store would make every launch upload all previous jobs' data.
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _EXCLUDE_DIRS and not (Path(dirpath) / d / "metadata.db").exists()
            )
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.is_symlink() or not p.is_file() or fn.endswith(".pyc"):
                    continue
                rel = p.relative_to(root).as_posix()
                info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
                mode = 0o755 if os.access(p, os.X_OK) else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, p.read_bytes())
    return buf.getvalue()


def extract_bundle(data: bytes, dest: str | Path) -> None:
    """Extract a bundle, refusing absolute paths, '..' and symlinks (zip-slip)."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValidationFailed(f"unsafe path in bundle: {info.filename!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValidationFailed(f"symlink in bundle: {info.filename!r}")
            target = (dest / Path(*name.parts)).resolve()
            if not target.is_relative_to(dest):
                raise ValidationFailed(f"unsafe path in bundle: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                target.chmod(mode)
