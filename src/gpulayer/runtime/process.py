"""Subprocess execution with streaming logs, timeouts and cancellation.

The workload runs in its own session/process group so that cancellation also stops
children (torchrun spawns one process per local rank). This is process management,
NOT a security sandbox: see docs/architecture/security.md.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

LineSink = Callable[[str, str], None]  # (stream, line)


@dataclass
class ProcessOutcome:
    exit_code: int | None
    timed_out: bool = False
    cancelled: bool = False
    error: str | None = None


def _kill_group(proc: subprocess.Popen[bytes], grace: float) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows is not a supported worker platform yet
            proc.terminate()
        proc.wait(timeout=grace)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()


def run_process(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: str,
    on_line: LineSink,
    timeout: float | None = None,
    cancel: threading.Event | None = None,
    kill_grace: float = 10.0,
) -> ProcessOutcome:
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
    except OSError as e:
        return ProcessOutcome(exit_code=None, error=f"could not start process: {e}")

    def pump(stream_name: str, pipe: object) -> None:
        for raw in iter(pipe.readline, b""):  # type: ignore[attr-defined]
            on_line(stream_name, raw.decode("utf-8", "replace").rstrip("\r\n"))

    pumps = [
        threading.Thread(target=pump, args=(n, p), daemon=True)
        for n, p in (("stdout", proc.stdout), ("stderr", proc.stderr))
    ]
    for t in pumps:
        t.start()

    deadline = time.monotonic() + timeout if timeout else None
    outcome: ProcessOutcome | None = None
    while proc.poll() is None:
        if cancel is not None and cancel.wait(0.1):
            _kill_group(proc, kill_grace)
            outcome = ProcessOutcome(exit_code=proc.returncode, cancelled=True)
            break
        if cancel is None:
            time.sleep(0.1)
        if deadline is not None and time.monotonic() > deadline:
            _kill_group(proc, kill_grace)
            outcome = ProcessOutcome(exit_code=proc.returncode, timed_out=True)
            break
    for t in pumps:
        t.join(timeout=5)
    # Reap any orphans left in the group after a normal exit (e.g. stray children).
    if outcome is None and os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    return outcome or ProcessOutcome(exit_code=proc.returncode)
