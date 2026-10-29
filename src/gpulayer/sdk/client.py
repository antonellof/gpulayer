"""The GPUlayer SDK: one client for local, remote and cloud targets.

from gpulayer import Client, command
with Client() as c:                    # embedded control plane in $GPULAYER_HOME
    job = c.submit(command(["python", "-c", "print(42)"]))
    print(job.wait().state, job.logs_text())

c = Client(url="https://cp.example:8470", token=...)   # self-hosted control plane
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gpulayer.api.client import HttpControlPlaneClient, HttpTransport
from gpulayer.artifacts.store import build_source_bundle
from gpulayer.contracts import (
    ArtifactKind,
    ArtifactReference,
    ExecutionEvent,
    ExecutionTarget,
    JobResult,
    JobSpec,
    JobStatus,
    LogLine,
    WorkerOrigin,
    WorkerStatus,
)
from gpulayer.control_plane.service import ControlPlane
from gpulayer.errors import GpuLayerError
from gpulayer.sdk.home import gpulayer_home, open_control_plane


class Job:
    def __init__(self, client: Client, job_id: str) -> None:
        self.client = client
        self.id = job_id

    def __repr__(self) -> str:
        return f"Job({self.id!r})"

    def status(self) -> JobStatus:
        return self.client._api.get_job(self.id)

    def wait(self, timeout: float | None = None, poll: float = 0.2) -> JobStatus:
        """Block until the job is terminal. Long-polls the control plane, so it returns as
        soon as the job finishes; ``poll`` is only the back-off for servers without it."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            left = 10.0 if deadline is None else max(0.0, deadline - time.monotonic())
            asked = time.monotonic()
            st = self.client._api.wait_job(self.id, min(left, 10.0))
            if st.state.terminal:
                return st
            if deadline and time.monotonic() >= deadline:
                raise TimeoutError(f"job {self.id} still {st.state.value} after {timeout}s")
            if time.monotonic() - asked < poll / 2:
                time.sleep(poll)

    def result(self) -> JobResult:
        return self.client._api.job_result(self.id)

    def cancel(self) -> JobStatus:
        return self.client._api.cancel_job(self.id)

    def logs(self, after_seq: int = 0) -> list[LogLine]:
        return self.client._api.job_logs(self.id, after_seq=after_seq)

    def logs_text(self) -> str:
        return "\n".join(line.line for line in self.logs() if line.stream != "system")

    def follow_logs(self, poll: float = 0.3) -> Iterator[LogLine]:
        seq = 0
        while True:
            lines = self.logs(seq)
            yield from lines
            if lines:
                seq = lines[-1].seq
            elif self.status().state.terminal:
                yield from self.logs(seq)
                return
            else:
                # Returns early when the job finishes, so the last lines are not delayed.
                self.client._api.wait_job(self.id, poll)

    def events(self) -> list[ExecutionEvent]:
        return self.client._api.events(job_id=self.id)


class Client:
    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        *,
        home: str | Path | None = None,
        ca_file: str | None = None,
        fingerprint: str | None = None,
        local_worker: bool = True,
        worker_python: str | None = None,
    ) -> None:
        url = url or os.environ.get("GPULAYER_URL")
        self._local_worker_enabled = local_worker
        self._worker_python = (
            worker_python or os.environ.get("GPULAYER_WORKER_PYTHON") or sys.executable
        )
        self._local: list[tuple[Any, threading.Thread]] = []  # (agent, thread)
        self._local_caps: Any = None
        self._lock = threading.Lock()
        self.cp: ControlPlane | None = None
        if url:
            token = token or os.environ.get("GPULAYER_TOKEN")
            if not token:
                raise GpuLayerError("a client token is required for a remote control plane")
            fingerprint = fingerprint or os.environ.get("GPULAYER_FINGERPRINT")
            self._api: Any = HttpControlPlaneClient(
                HttpTransport(url, token, ca_file=ca_file, fingerprint=fingerprint)
            )
            self.url = url
            self.mode = "remote"
        else:
            self.home = gpulayer_home(home)
            self.cp = open_control_plane(self.home)
            self._api = self.cp
            self.mode = "embedded"

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- jobs

    def submit(self, spec: JobSpec) -> Job:
        if self.mode == "embedded" and spec.placement.target == ExecutionTarget.LOCAL:
            # A local distributed job needs one in-process worker per node.
            self.ensure_local_workers(spec.num_nodes)
        return Job(self, self._api.submit(spec).job_id)

    def submit_batch(self, specs: list[JobSpec]) -> list[Job]:
        """Independent jobs, scheduled separately. For one coordinated workload use
        ``gpulayer.distributed(...)`` instead."""
        return [self.submit(s) for s in specs]

    def run(self, spec: JobSpec, timeout: float | None = None) -> JobResult:
        job = self.submit(spec)
        job.wait(timeout)
        return job.result()

    def job(self, job_id: str) -> Job:
        return Job(self, job_id)

    def jobs(self, limit: int = 50) -> list[JobStatus]:
        return list(self._api.list_jobs(limit=limit))

    def workers(self) -> list[WorkerStatus]:
        return list(self._api.list_workers())

    # ----------------------------------------------------------- artifacts

    def upload(
        self, data: bytes | str | Path, name: str = "", kind: ArtifactKind = ArtifactKind.INPUT
    ) -> ArtifactReference:
        if not isinstance(data, bytes):
            path = Path(data)
            name = name or path.name
            data = path.read_bytes()
        return self._api.put_artifact(data, name, kind)

    def package(self, directory: str | Path) -> ArtifactReference:
        """Upload a deterministic source bundle of ``directory``."""
        return self._api.put_artifact(
            build_source_bundle(directory), Path(directory).name, ArtifactKind.SOURCE
        )

    def download(self, ref: ArtifactReference) -> bytes:
        return self._api.get_artifact(ref.sha256)

    # --------------------------------------------------------- local worker

    def ensure_local_worker(self) -> None:
        """Start an in-process local worker (embedded mode) if none is running."""
        self.ensure_local_workers(1)

    def ensure_local_workers(self, n: int) -> None:
        """Run at least ``n`` in-process local workers (embedded mode only).

        Each one is a separate registered worker, so independent jobs run in parallel and
        a local distributed job can span ``n`` nodes on this machine.
        """
        if not self._local_worker_enabled or self.cp is None:
            return
        with self._lock:
            self._local = [(a, t) for a, t in self._local if t.is_alive()]
            if len(self._local) >= n:
                return
            from gpulayer.gpu.discovery import detect_capabilities
            from gpulayer.worker.agent import WorkerAgent
            from gpulayer.worker.channel import EmbeddedChannel

            if self._local_caps is None:
                self._local_caps = detect_capabilities(
                    origin=WorkerOrigin.LOCAL,
                    advertise_address="127.0.0.1",
                    runner=_python_runner(self._worker_python),
                )
            while len(self._local) < n:
                name = f"local-{os.getpid()}-{len(self._local)}"
                channel = EmbeddedChannel.register(self.cp, name, self._local_caps)
                agent = WorkerAgent(
                    channel,
                    self._local_caps,
                    self.home / "workers" / channel.worker_id,
                    python=self._worker_python,
                )
                thread = threading.Thread(target=agent.run, daemon=True, name=name)
                thread.start()
                self._local.append((agent, thread))

    def close(self) -> None:
        with self._lock:
            for agent, _ in self._local:
                agent.stop.set()
            if self.cp is not None:
                self.cp.signal()  # wake workers blocked in a long-poll acquire
            for _, thread in self._local:
                thread.join(timeout=30)
            self._local = []


def _python_runner(python: str) -> Any:
    """Probe torch with the interpreter that will actually run workloads."""
    from gpulayer.gpu import discovery

    def runner(cmd: list[str], timeout: float) -> tuple[int, str]:
        if len(cmd) >= 2 and cmd[1] == "-c" and cmd[0] == sys.executable:
            cmd = [python, *cmd[1:]]
        return discovery.run_command(cmd, timeout)

    return runner
