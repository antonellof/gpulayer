"""HTTP clients for the control-plane API (stdlib urllib; no extra dependencies)."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from gpulayer.contracts import (
    PROTOCOL_VERSION,
    ArtifactKind,
    ArtifactReference,
    ExecutionEvent,
    HeartbeatResponse,
    JobResult,
    JobSpec,
    JobState,
    JobStatus,
    LogLine,
    Registration,
    RendezvousInfo,
    TaskAssignment,
    TaskOutcome,
    WorkerCapabilities,
    WorkerStatus,
)
from gpulayer.errors import ERRORS_BY_CODE, GpuLayerError


class HttpTransport:
    def __init__(
        self, url: str, token: str, *, ca_file: str | None = None, timeout: float = 60
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("control plane URL must start with http:// or https://")
        self.url = url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self._ctx = ssl.create_default_context(cafile=ca_file) if url.startswith("https") else None

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
        token: str | None = None,
    ) -> Any:
        hdrs = {
            "Authorization": f"Bearer {token or self._token}",
            "X-GpuLayer-Protocol": PROTOCOL_VERSION,
            **(headers or {}),
        }
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
                content = resp.read()
                if resp.headers.get("Content-Type") == "application/octet-stream":
                    return content
                return json.loads(content) if content else None
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())["error"]
            except Exception:
                raise GpuLayerError(f"HTTP {e.code} from control plane") from None
            raise ERRORS_BY_CODE.get(err["code"], GpuLayerError)(err["message"]) from None


class HttpControlPlaneClient:
    """Client-role operations; method names mirror ``ControlPlane`` for the SDK."""

    def __init__(self, transport: HttpTransport) -> None:
        self.t = transport

    def submit(self, spec: JobSpec) -> JobStatus:
        return JobStatus.model_validate(
            self.t.request("POST", "/v1/jobs", spec.model_dump(mode="json"))
        )

    def get_job(self, job_id: str) -> JobStatus:
        return JobStatus.model_validate(self.t.request("GET", f"/v1/jobs/{job_id}"))

    def list_jobs(self, *, limit: int = 50, state: JobState | None = None) -> list[JobStatus]:
        q = {"limit": limit, **({"state": state.value} if state else {})}
        return [
            JobStatus.model_validate(j) for j in self.t.request("GET", f"/v1/jobs?{urlencode(q)}")
        ]

    def cancel_job(self, job_id: str) -> JobStatus:
        return JobStatus.model_validate(self.t.request("POST", f"/v1/jobs/{job_id}/cancel", {}))

    def job_result(self, job_id: str) -> JobResult:
        return JobResult.model_validate(self.t.request("GET", f"/v1/jobs/{job_id}/result"))

    def job_logs(self, job_id: str, *, after_seq: int = 0) -> list[LogLine]:
        return [
            LogLine.model_validate(x)
            for x in self.t.request("GET", f"/v1/jobs/{job_id}/logs?after={after_seq}")
        ]

    def events(self, *, job_id: str, after_seq: int = 0) -> list[ExecutionEvent]:
        return [
            ExecutionEvent.model_validate(x)
            for x in self.t.request("GET", f"/v1/jobs/{job_id}/events?after={after_seq}")
        ]

    def list_workers(self) -> list[WorkerStatus]:
        return [WorkerStatus.model_validate(w) for w in self.t.request("GET", "/v1/workers")]

    def revoke_worker(self, worker_id: str) -> None:
        self.t.request("POST", f"/v1/workers/{worker_id}/revoke", {})

    def put_artifact(
        self, data: bytes, name: str = "", kind: ArtifactKind = ArtifactKind.INPUT
    ) -> ArtifactReference:
        return ArtifactReference.model_validate(
            self.t.request(
                "PUT",
                "/v1/artifacts",
                raw=data,
                headers={
                    "X-Artifact-Name": name,
                    "X-Artifact-Kind": kind.value,
                    "Content-Type": "application/octet-stream",
                },
            )
        )

    def get_artifact(self, sha256: str) -> bytes:
        data: bytes = self.t.request("GET", f"/v1/artifacts/{sha256}")
        return data


class HttpWorkerChannel:
    """``WorkerChannel`` over HTTP. Holds the per-worker session token, not the join token."""

    def __init__(self, transport: HttpTransport, registration: Registration) -> None:
        self.t = transport
        self.worker_id = registration.worker_id
        self.heartbeat_interval = registration.heartbeat_interval_seconds
        self._session = registration.session_token

    @classmethod
    def register(
        cls,
        url: str,
        join_token: str,
        name: str,
        caps: WorkerCapabilities,
        *,
        ca_file: str | None = None,
    ) -> HttpWorkerChannel:
        t = HttpTransport(url, join_token, ca_file=ca_file)
        reg = Registration.model_validate(
            t.request(
                "POST",
                "/v1/workers/register",
                {"name": name, "capabilities": caps.model_dump(mode="json")},
            )
        )
        return cls(HttpTransport(url, reg.session_token, ca_file=ca_file), reg)

    def heartbeat(self, draining: bool = False) -> HeartbeatResponse:
        return HeartbeatResponse.model_validate(
            self.t.request("POST", "/v1/worker/heartbeat", {"draining": draining})
        )

    def acquire(self) -> TaskAssignment | None:
        data = self.t.request("POST", "/v1/worker/acquire", {})
        return TaskAssignment.model_validate(data) if data else None

    def start(self, task_id: str) -> None:
        self.t.request("POST", f"/v1/worker/tasks/{task_id}/start", {})

    def publish_rendezvous(self, task_id: str, addr: str, port: int) -> RendezvousInfo:
        return RendezvousInfo.model_validate(
            self.t.request(
                "POST", f"/v1/worker/tasks/{task_id}/rendezvous", {"addr": addr, "port": port}
            )
        )

    def logs(self, task_id: str, lines: list[tuple[str, str]]) -> None:
        self.t.request(
            "POST", f"/v1/worker/tasks/{task_id}/logs", {"lines": [list(x) for x in lines]}
        )

    def finish(self, task_id: str, outcome: TaskOutcome) -> None:
        self.t.request(
            "POST", f"/v1/worker/tasks/{task_id}/finish", outcome.model_dump(mode="json")
        )

    def put_artifact(self, data: bytes, name: str, kind: ArtifactKind) -> ArtifactReference:
        return HttpControlPlaneClient(self.t).put_artifact(data, name, kind)

    def get_artifact(self, sha256: str) -> bytes:
        return HttpControlPlaneClient(self.t).get_artifact(sha256)

    def deregister(self) -> None:
        self.t.request("POST", "/v1/worker/deregister", {})
