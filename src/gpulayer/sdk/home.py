"""Local state directory ($GPULAYER_HOME, default ~/.gpulayer) and embedded control plane."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gpulayer.artifacts.store import LocalArtifactStore
from gpulayer.control_plane.service import ControlPlane, ControlPlaneConfig
from gpulayer.storage.sqlite import SqliteStore


def gpulayer_home(home: str | Path | None = None) -> Path:
    p = Path(home or os.environ.get("GPULAYER_HOME") or Path.home() / ".gpulayer")
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    return p


def open_control_plane(
    home: str | Path | None = None,
    *,
    config: ControlPlaneConfig | None = None,
    providers: Iterable[Any] = (),
) -> ControlPlane:
    """Embedded control plane over $GPULAYER_HOME/metadata.db and artifacts/."""
    h = gpulayer_home(home)
    return ControlPlane(
        SqliteStore(h / "metadata.db"),
        LocalArtifactStore(h / "artifacts"),
        config=config,
        providers=providers,
    )
