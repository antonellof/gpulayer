"""How a worker talks to its control plane. Two implementations share one protocol:

* ``EmbeddedChannel`` -- in-process calls (local mode, tests);
* ``gpulayer.api.client.HttpWorkerChannel`` -- authenticated HTTP(S) to a remote plane.

The worker agent is written against ``WorkerChannel`` only, so the same code path runs
local, remote and provider-provisioned workers.
"""

from __future__ import annotations

import threading
from typing import Protocol

from gpulayer.contracts import (
    ArtifactKind,
    ArtifactReference,
    HeartbeatResponse,
    RendezvousInfo,
    TaskAssignment,
    TaskOutcome,
    WorkerCapabilities,
    WorkerMetrics,
)
from gpulayer.control_plane.service import ControlPlane


class WorkerChannel(Protocol):
    worker_id: str
    heartbeat_interval: float

    def heartbeat(
        self, draining: bool = False, metrics: WorkerMetrics | None = None
    ) -> HeartbeatResponse: ...
    def acquire(
        self, wait: float = 0.0, stop: threading.Event | None = None
    ) -> TaskAssignment | None: ...
    def start(self, task_id: str) -> None: ...
    def publish_rendezvous(self, task_id: str, addr: str, port: int) -> RendezvousInfo: ...
    def logs(self, task_id: str, lines: list[tuple[str, str]]) -> None: ...
    def finish(self, task_id: str, outcome: TaskOutcome) -> None: ...
    def put_artifact(self, data: bytes, name: str, kind: ArtifactKind) -> ArtifactReference: ...
    def get_artifact(self, sha256: str) -> bytes: ...
    def deregister(self) -> None: ...


class EmbeddedChannel:
    def __init__(self, cp: ControlPlane, worker_id: str, heartbeat_interval: float) -> None:
        self.cp = cp
        self.worker_id = worker_id
        self.heartbeat_interval = heartbeat_interval

    @classmethod
    def register(
        cls,
        cp: ControlPlane,
        name: str,
        caps: WorkerCapabilities,
        *,
        instance_token: str | None = None,
    ) -> EmbeddedChannel:
        reg = cp.register_worker(name, caps, instance_token=instance_token)
        return cls(cp, reg.worker_id, reg.heartbeat_interval_seconds)

    def heartbeat(
        self, draining: bool = False, metrics: WorkerMetrics | None = None
    ) -> HeartbeatResponse:
        return self.cp.heartbeat(self.worker_id, draining=draining, metrics=metrics)

    def acquire(
        self, wait: float = 0.0, stop: threading.Event | None = None
    ) -> TaskAssignment | None:
        return self.cp.acquire_task(self.worker_id, wait=wait, stop=stop)

    def start(self, task_id: str) -> None:
        self.cp.start_task(self.worker_id, task_id)

    def publish_rendezvous(self, task_id: str, addr: str, port: int) -> RendezvousInfo:
        return self.cp.publish_rendezvous(self.worker_id, task_id, addr, port)

    def logs(self, task_id: str, lines: list[tuple[str, str]]) -> None:
        self.cp.append_logs(self.worker_id, task_id, lines)

    def finish(self, task_id: str, outcome: TaskOutcome) -> None:
        self.cp.finish_task(self.worker_id, task_id, outcome)

    def put_artifact(self, data: bytes, name: str, kind: ArtifactKind) -> ArtifactReference:
        return self.cp.put_artifact(data, name, kind)

    def get_artifact(self, sha256: str) -> bytes:
        return self.cp.get_artifact(sha256)

    def deregister(self) -> None:
        self.cp.deregister_worker(self.worker_id)
