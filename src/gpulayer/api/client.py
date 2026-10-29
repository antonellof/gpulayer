"""HTTP clients for the control-plane API (stdlib only; no extra dependencies)."""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import threading
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlencode, urlparse

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
    WorkerMetrics,
    WorkerStatus,
)
from gpulayer.control_plane.service import LogEntry
from gpulayer.errors import ERRORS_BY_CODE, GpuLayerError, Unauthorized


class HttpTransport:
    """JSON-over-HTTP(S) with bearer auth.

    TLS is verified against the system CAs (or ``ca_file``). Alternatively ``fingerprint``
    ("sha256:<hex>") pins the server certificate, which is how ``gpulayer up`` works with a
    self-signed certificate: the connection is refused before any token is sent if the
    certificate does not match.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        ca_file: str | None = None,
        fingerprint: str | None = None,
        timeout: float = 60,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("control plane URL must look like https://host:port")
        if fingerprint and parsed.scheme != "https":
            raise ValueError("a certificate fingerprint needs an https:// URL")
        self.url = url.rstrip("/")
        self._parsed = parsed
        self._token = token
        self.timeout = timeout
        self.fingerprint = normalize_fingerprint(fingerprint) if fingerprint else None
        self._ctx: ssl.SSLContext | None = None
        if parsed.scheme == "https":
            if self.fingerprint:
                # Pinning replaces CA validation; the digest check below is mandatory.
                self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                self._ctx.check_hostname = False
                self._ctx.verify_mode = ssl.CERT_NONE
                self._ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            else:
                self._ctx = ssl.create_default_context(cafile=ca_file)

    def _connect(self) -> http.client.HTTPConnection:
        host, port = self._parsed.hostname or "", self._parsed.port
        if self._ctx is None:
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
            conn.connect()
            return conn
        tls = http.client.HTTPSConnection(host, port, timeout=self.timeout, context=self._ctx)
        tls.connect()
        if self.fingerprint:
            der = tls.sock.getpeercert(binary_form=True) if tls.sock else None
            if not der or "sha256:" + hashlib.sha256(der).hexdigest() != self.fingerprint:
                tls.close()
                raise Unauthorized(
                    "server certificate does not match the expected fingerprint; "
                    "refusing to send credentials"
                )
        return tls

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
        conn = self._connect()
        try:
            conn.request(
                method, (self._parsed.path.rstrip("/") or "") + path, body=data, headers=hdrs
            )
            resp = conn.getresponse()
            content = resp.read()
        finally:
            conn.close()
        if resp.status >= 400:
            try:
                err = json.loads(content)["error"]
            except Exception:
                raise GpuLayerError(f"HTTP {resp.status} from control plane") from None
            raise ERRORS_BY_CODE.get(err["code"], GpuLayerError)(err["message"])
        ctype = resp.getheader("Content-Type") or ""
        if ctype == "application/octet-stream":
            return content
        if ctype.startswith("text/plain"):
            return content.decode()
        return json.loads(content) if content else None


def normalize_fingerprint(fp: str) -> str:
    hexpart = fp.lower().removeprefix("sha256:").replace(":", "")
    if len(hexpart) != 64 or any(c not in "0123456789abcdef" for c in hexpart):
        raise ValueError("fingerprint must be 'sha256:' followed by 64 hex characters")
    return "sha256:" + hexpart


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

    def wait_job(self, job_id: str, timeout: float) -> JobStatus:
        return JobStatus.model_validate(
            self.t.request("GET", f"/v1/jobs/{job_id}?wait={min(timeout, 30.0):.3f}")
        )

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

    def metrics_text(self) -> str:
        text: str = self.t.request("GET", "/metrics")
        return text

    def metrics_history(self, minutes: float = 60.0) -> dict[str, Any]:
        history: dict[str, Any] = self.t.request("GET", f"/v1/metrics/history?minutes={minutes}")
        return history

    def whoami(self) -> dict[str, str]:
        who: dict[str, str] = self.t.request("GET", "/v1/whoami")
        return who

    def gc(
        self, older_than_seconds: float, *, delete_jobs: bool = False, dry_run: bool = False
    ) -> dict[str, int]:
        report: dict[str, int] = self.t.request(
            "POST",
            "/v1/admin/gc",
            {
                "older_than_seconds": older_than_seconds,
                "delete_jobs": delete_jobs,
                "dry_run": dry_run,
            },
        )
        return report

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
        fingerprint: str | None = None,
    ) -> HttpWorkerChannel:
        t = HttpTransport(url, join_token, ca_file=ca_file, fingerprint=fingerprint)
        reg = Registration.model_validate(
            t.request(
                "POST",
                "/v1/workers/register",
                {"name": name, "capabilities": caps.model_dump(mode="json")},
            )
        )
        return cls(
            HttpTransport(url, reg.session_token, ca_file=ca_file, fingerprint=fingerprint), reg
        )

    def heartbeat(
        self, draining: bool = False, metrics: WorkerMetrics | None = None
    ) -> HeartbeatResponse:
        body: dict[str, Any] = {"draining": draining}
        if metrics is not None:
            body["metrics"] = metrics.model_dump(mode="json", exclude_none=True)
        return HeartbeatResponse.model_validate(
            self.t.request("POST", "/v1/worker/heartbeat", body)
        )

    def acquire(
        self, wait: float = 0.0, stop: threading.Event | None = None
    ) -> TaskAssignment | None:
        # Long poll: the server holds the request until a task is assigned or `wait` passes.
        # Servers without long-poll support answer at once; the agent then backs off.
        data = self.t.request("POST", "/v1/worker/acquire", {"wait": wait})
        return TaskAssignment.model_validate(data) if data else None

    def start(self, task_id: str) -> None:
        self.t.request("POST", f"/v1/worker/tasks/{task_id}/start", {})

    def publish_rendezvous(self, task_id: str, addr: str, port: int) -> RendezvousInfo:
        return RendezvousInfo.model_validate(
            self.t.request(
                "POST", f"/v1/worker/tasks/{task_id}/rendezvous", {"addr": addr, "port": port}
            )
        )

    def logs(self, task_id: str, lines: Sequence[LogEntry]) -> None:
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
