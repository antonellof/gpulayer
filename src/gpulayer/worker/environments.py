"""Per-job Python environments, built once per worker and reused (ADR-0022).

A job's ``environment`` lists packages beyond what the worker's interpreter has. The worker
builds a virtual environment on top of its interpreter (``--system-site-packages``, so PyTorch
and the CUDA libraries are not installed again), keyed by the content of the request and the
interpreter, and reuses it for every later job with the same request. An environment counts
as ready only once a marker is written after a successful build, so a crash never leaves a
half-built one that looks ready; builds are locked, so concurrent jobs wait for one build
instead of running two. (Environments are built in place: a virtualenv's scripts hard-code
its path, so it cannot be built elsewhere and moved.)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gpulayer.errors import GpuLayerError

# Seconds an install may take before the task fails (large CUDA wheels on a slow link).
INSTALL_TIMEOUT = 1800.0


@dataclass(frozen=True)
class PreparedEnvironment:
    python: str
    bin_dir: str
    seconds: float
    cached: bool


class EnvironmentCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._python_ids: dict[str, str] = {}

    def prepare(
        self,
        base_python: str,
        requirements: list[str],
        requirements_text: str | None = None,
        log: Callable[[str], None] = lambda line: None,
    ) -> PreparedEnvironment:
        """The interpreter of an environment with these packages, building it if needed."""
        t0 = time.perf_counter()
        key = self._key(base_python, requirements, requirements_text)
        env_dir = self.root / key
        with self._lock(key), _file_lock(self.root / f"{key}.lock"):
            cached = (env_dir / ".ready").exists()
            if not cached:
                self._build(env_dir, base_python, requirements, requirements_text, log)
            (env_dir / ".ready").touch()  # also marks when it was last used, for cleanup
        bin_dir = env_dir / "bin"
        return PreparedEnvironment(
            python=str(bin_dir / "python"),
            bin_dir=str(bin_dir),
            seconds=time.perf_counter() - t0,
            cached=cached,
        )

    # ------------------------------------------------------------------ internals

    def _key(self, python: str, requirements: list[str], text: str | None) -> str:
        blob = json.dumps(
            {
                "python": self._python_id(python),
                "requirements": sorted(requirements),
                "file": text or "",
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _python_id(self, python: str) -> str:
        # The interpreter's path and version: an environment built for one Python must not be
        # reused by another.
        if python not in self._python_ids:
            out = subprocess.run(
                [python, "-c", "import sys, platform; print(sys.version, platform.platform())"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._python_ids[python] = f"{os.path.realpath(python)} {out.stdout.strip()}"
        return self._python_ids[python]

    def _lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _build(
        self,
        env_dir: Path,
        python: str,
        requirements: list[str],
        text: str | None,
        log: Callable[[str], None],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = env_dir  # see the module docstring: built in place, ready once marked
        shutil.rmtree(tmp, ignore_errors=True)
        req_args = list(requirements)
        if text:
            req_file = env_dir.with_name(env_dir.name + ".requirements.txt")
            req_file.write_text(text)
            req_args += ["-r", str(req_file)]
        env_python = str(tmp / "bin" / "python")
        uv = shutil.which("uv")
        create = (
            [
                uv,
                "venv",
                "--quiet",
                "--seed",
                "--python",
                python,
                "--system-site-packages",
                str(tmp),
            ]
            if uv
            else [python, "-m", "venv", "--system-site-packages", str(tmp)]
        )
        log(f"building environment: {' '.join(requirements) or 'requirements file'}")
        # pip, not uv, installs: it treats packages anywhere on sys.path as installed, so the
        # worker's own packages linked in below (PyTorch, the CUDA libraries) are not installed
        # again, and a requirement that depends on torch keeps the worker's CUDA build.
        install = [
            env_python,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            *req_args,
        ]
        try:
            _run_step(create)
            _link_worker_packages(python, env_python)
            _run_step(install)
        except subprocess.TimeoutExpired as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise GpuLayerError(f"environment build timed out after {INSTALL_TIMEOUT:.0f}s") from e
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise


def _run_step(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=INSTALL_TIMEOUT)
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
        raise GpuLayerError(f"environment build failed ({Path(cmd[0]).name}):\n{tail}")


_SITE_DIRS = (
    "import json, sysconfig; p = sysconfig.get_paths(); "
    "print(json.dumps(sorted({p['purelib'], p['platlib']})))"
)


def _link_worker_packages(worker_python: str, env_python: str) -> None:
    """Make the worker interpreter's packages visible in the new environment, after its own.

    ``--system-site-packages`` only exposes the *base* interpreter's packages; when the worker
    runs in a virtualenv (the usual case), its packages live elsewhere. A .pth line calling
    site.addsitedir adds them, including editable installs (which are .pth files themselves).
    """
    worker_dirs = json.loads(_capture([worker_python, "-c", _SITE_DIRS]))
    env_dirs = json.loads(_capture([env_python, "-c", _SITE_DIRS]))
    lines = "".join(f"import site; site.addsitedir({d!r})\n" for d in worker_dirs)
    Path(env_dirs[0], "_gpulayer_worker_packages.pth").write_text(lines)


def _capture(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise GpuLayerError(f"environment build failed: {p.stderr.strip()[-500:]}")
    return p.stdout


class _file_lock:
    """An exclusive lock on a file, so workers sharing a cache directory build once."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
