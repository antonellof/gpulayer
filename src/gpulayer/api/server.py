"""HTTP(S) API for a self-hosted control plane (stdlib only).

Every route except ``GET /v1/health`` requires a bearer token and the
``X-GpuLayer-Protocol`` header. Binding to a non-loopback address without TLS is
refused unless explicitly overridden, because tokens would cross the network in clear.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ValidationError

from gpulayer.api.auth import CLIENT_ROLES, METRICS_ROLES, ServerCredentials
from gpulayer.contracts import (
    API_VERSION,
    PROTOCOL_VERSION,
    ArtifactKind,
    JobSpec,
    JobState,
    TaskOutcome,
    WorkerCapabilities,
    WorkerMetrics,
)
from gpulayer.control_plane.service import ControlPlane
from gpulayer.errors import (
    Forbidden,
    GpuLayerError,
    NotFound,
    ProtocolMismatch,
    RateLimited,
    Unauthorized,
    ValidationFailed,
)
from gpulayer.observability import metrics

log = logging.getLogger("gpulayer.api")

MAX_JSON_BYTES = 16 * 1024**2
MAX_ARTIFACT_BYTES = 2 * 1024**3

Handler = Callable[["_Req"], Any]


class _Req:
    def __init__(self, h: _RequestHandler, params: dict[str, str]) -> None:
        self.h = h
        self.params = params
        self.query = {k: v[-1] for k, v in parse_qs(urlparse(h.path).query).items()}
        self.worker_id: str | None = None
        self.principal = "anonymous"
        self.role: str | None = None
        self._body: bytes | None = None

    def body(self, limit: int = MAX_JSON_BYTES) -> bytes:
        if self._body is None:
            n = int(self.h.headers.get("Content-Length") or 0)
            if n > limit:
                raise ValidationFailed("request body too large")
            self._body = self.h.rfile.read(n) if n else b""
        return self._body

    @property
    def wid(self) -> str:
        """Authenticated worker id; only set on worker-scoped routes."""
        if self.worker_id is None:
            raise Unauthorized("worker credentials required")
        return self.worker_id

    def json(self) -> Any:
        raw = self.body()
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise ValidationFailed(f"invalid JSON: {e}") from None


class ControlPlaneServer:
    def __init__(
        self,
        cp: ControlPlane,
        credentials: ServerCredentials,
        *,
        host: str = "127.0.0.1",
        port: int = 8470,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        allow_insecure_remote: bool = False,
        reconcile_interval: float = 1.0,
    ) -> None:
        self.cp = cp
        self.creds = credentials
        self.limiter = FailureLimiter()
        if not tls_cert and not _is_loopback(host) and not allow_insecure_remote:
            raise ValueError(
                f"refusing to serve plaintext HTTP on non-loopback address {host!r}; "
                "configure --tls-cert/--tls-key or pass --allow-insecure-remote"
            )
        self.httpd = ThreadingHTTPServer((host, port), _RequestHandler)
        self.httpd.daemon_threads = True
        self.httpd.app = self  # type: ignore[attr-defined]
        self.tls = bool(tls_cert)
        if tls_cert:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(tls_cert, tls_key)
            # The handshake runs in each connection's own thread (_RequestHandler.setup), not
            # in the accept loop, so one slow client cannot hold up the others.
            self.httpd.socket = ctx.wrap_socket(
                self.httpd.socket, server_side=True, do_handshake_on_connect=False
            )
        self._stop = threading.Event()
        self._reconcile_interval = reconcile_interval
        self._threads: list[threading.Thread] = []
        self.routes = self._build_routes()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"{'https' if self.tls else 'http'}://{host!s}:{port}"

    def start(self) -> None:
        for target in (self.httpd.serve_forever, self._reconcile_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def serve_forever(self) -> None:
        threading.Thread(target=self._reconcile_loop, daemon=True).start()
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def _reconcile_loop(self) -> None:
        while not self._stop.wait(self._reconcile_interval):
            try:
                self.cp.reconcile()
                self.cp.schedule_pending()
            except Exception:  # keep the loop alive; errors are logged
                log.exception("reconcile failed")

    # --------------------------------------------------------------- routes

    def _build_routes(self) -> list[tuple[str, re.Pattern[str], str, Handler]]:
        cp = self.cp
        r: list[tuple[str, str, str, Handler]] = [
            ("GET", r"/", "none", lambda q: _ui("index.html")),
            ("GET", r"/ui/?(?P<file>[a-z.]*)", "none", lambda q: _ui(q.params["file"])),
            (
                "GET",
                r"/metrics",
                "metrics",
                lambda q: _Raw(metrics.render(cp).encode(), metrics.CONTENT_TYPE),
            ),
            ("GET", r"/v1/whoami", "client", lambda q: {"principal": q.principal, "role": q.role}),
            (
                "GET",
                r"/v1/metrics/history",
                "client",
                lambda q: cp.metrics_history(float(q.query.get("minutes", 60))),
            ),
            (
                "GET",
                r"/v1/health",
                "none",
                lambda q: {
                    "status": "ok",
                    "api_version": API_VERSION,
                    "protocol": PROTOCOL_VERSION,
                },
            ),
            (
                "POST",
                r"/v1/jobs",
                "client",
                lambda q: cp.submit(JobSpec.model_validate(q.json()), submitted_by=q.principal),
            ),
            (
                "GET",
                r"/v1/jobs",
                "client",
                lambda q: cp.list_jobs(
                    limit=min(int(q.query.get("limit", 50)), 500),
                    state=JobState(q.query["state"]) if "state" in q.query else None,
                ),
            ),
            (
                "GET",
                r"/v1/jobs/(?P<id>[\w-]+)",
                "client",
                lambda q: (
                    cp.wait_job(q.params["id"], float(q.query["wait"]))
                    if "wait" in q.query
                    else cp.get_job(q.params["id"])
                ),
            ),
            (
                "POST",
                r"/v1/jobs/(?P<id>[\w-]+)/cancel",
                "client",
                self._cancel,
            ),
            (
                "GET",
                r"/v1/jobs/(?P<id>[\w-]+)/result",
                "client",
                lambda q: cp.job_result(q.params["id"]),
            ),
            (
                "GET",
                r"/v1/jobs/(?P<id>[\w-]+)/logs",
                "client",
                lambda q: cp.job_logs(q.params["id"], after_seq=int(q.query.get("after", 0))),
            ),
            (
                "GET",
                r"/v1/jobs/(?P<id>[\w-]+)/events",
                "client",
                lambda q: cp.events(job_id=q.params["id"], after_seq=int(q.query.get("after", 0))),
            ),
            ("GET", r"/v1/workers", "client", lambda q: cp.list_workers()),
            (
                "POST",
                r"/v1/workers/(?P<id>[\w-]+)/revoke",
                "admin",
                lambda q: cp.revoke_worker(q.params["id"]),
            ),
            (
                "POST",
                r"/v1/admin/gc",
                "admin",
                lambda q: cp.gc(
                    float(q.json()["older_than_seconds"]),
                    delete_jobs=bool(q.json().get("delete_jobs")),
                    dry_run=bool(q.json().get("dry_run")),
                ),
            ),
            ("PUT", r"/v1/artifacts", "client_or_worker", self._put_artifact),
            (
                "GET",
                r"/v1/artifacts/(?P<sha>[0-9a-f]{64})",
                "client_or_worker",
                self._get_artifact,
            ),
            ("POST", r"/v1/workers/register", "join", self._register),
            (
                "POST",
                r"/v1/worker/heartbeat",
                "worker",
                lambda q: cp.heartbeat(
                    q.wid,
                    draining=bool(q.json().get("draining")),
                    metrics=(
                        WorkerMetrics.model_validate(q.json()["metrics"])
                        if q.json().get("metrics") is not None
                        else None
                    ),
                ),
            ),
            (
                "POST",
                r"/v1/worker/acquire",
                "worker",
                lambda q: cp.acquire_task(q.wid, wait=float(q.json().get("wait") or 0)),
            ),
            (
                "POST",
                r"/v1/worker/tasks/(?P<tid>[\w-]+)/start",
                "worker",
                lambda q: cp.start_task(q.wid, q.params["tid"]),
            ),
            (
                "POST",
                r"/v1/worker/tasks/(?P<tid>[\w-]+)/rendezvous",
                "worker",
                lambda q: cp.publish_rendezvous(
                    q.wid, q.params["tid"], str(q.json()["addr"]), int(q.json()["port"])
                ),
            ),
            ("POST", r"/v1/worker/tasks/(?P<tid>[\w-]+)/logs", "worker", self._logs),
            (
                "POST",
                r"/v1/worker/tasks/(?P<tid>[\w-]+)/finish",
                "worker",
                lambda q: cp.finish_task(
                    q.wid, q.params["tid"], TaskOutcome.model_validate(q.json())
                ),
            ),
            ("POST", r"/v1/worker/deregister", "worker", lambda q: cp.deregister_worker(q.wid)),
        ]
        return [(m, re.compile(p + r"/?$"), a, f) for m, p, a, f in r]

    def _put_artifact(self, q: _Req) -> Any:
        name = q.h.headers.get("X-Artifact-Name", "")[:255]
        kind = ArtifactKind(q.h.headers.get("X-Artifact-Kind", "input"))
        return self.cp.put_artifact(q.body(MAX_ARTIFACT_BYTES), name, kind)

    def _register(self, q: _Req) -> Any:
        body = q.json()
        caps = WorkerCapabilities.model_validate(body["capabilities"])
        token = q.h._bearer()
        if caps.origin.value == "provider":
            # Provider workers authenticate with their single-use instance token only.
            return self.cp.register_worker(
                str(body.get("name", "worker"))[:64], caps, instance_token=token
            )
        if not self.creds.is_join(token):
            self.limiter.failure(q.h.client_address[0])
            raise Unauthorized("a valid join token is required to register a worker")
        return self.cp.register_worker(str(body.get("name", "worker"))[:64], caps)

    def _logs(self, q: _Req) -> Any:
        lines = q.json().get("lines", [])
        if not isinstance(lines, list) or len(lines) > 5000:
            raise ValidationFailed("lines must be a list of at most 5000 [stream, line] pairs")
        entries: list[tuple[str, str, int | None]] = []
        for e in lines:
            if not isinstance(e, list) or len(e) not in (2, 3):
                raise ValidationFailed("each log entry is [stream, line] or [stream, line, rank]")
            rank = e[2] if len(e) == 3 else None
            if rank is not None and (not isinstance(rank, int) or not 0 <= rank < 1_000_000):
                raise ValidationFailed("log rank must be a non-negative integer")
            entries.append((str(e[0]), str(e[1]), rank))
        return self.cp.append_logs(q.wid, q.params["tid"], entries)

    def _cancel(self, q: _Req) -> Any:
        job_id = q.params["id"]
        if q.role != "admin" and self.cp.get_job(job_id).submitted_by != q.principal:
            raise Forbidden("only an admin or the job's submitter can cancel it")
        return self.cp.cancel_job(job_id)

    def _get_artifact(self, q: _Req) -> Any:
        sha = q.params["sha"]
        if q.worker_id is not None and not self.cp.worker_can_read_artifact(q.worker_id, sha):
            raise Forbidden("this artifact is not used by any task assigned to this worker")
        return _Raw(self.cp.get_artifact(sha))

    def authorize(self, q: _Req, access: str) -> None:
        if access == "none":
            return
        # Prometheus cannot send our protocol header; /metrics is a plain read-only text format.
        if access != "metrics" and q.h.headers.get("X-GpuLayer-Protocol") != PROTOCOL_VERSION:
            raise ProtocolMismatch(f"this control plane speaks protocol {PROTOCOL_VERSION}")
        client_ip = q.h.client_address[0]
        self.limiter.check(client_ip)
        try:
            self._authorize(q, access)
        except Unauthorized:
            self.limiter.failure(client_ip)
            raise

    def _authorize(self, q: _Req, access: str) -> None:
        token = q.h._bearer()
        if access == "join":
            return  # checked in _register (join token or instance token)
        if access == "metrics":
            entry = self.creds.authenticate(token, METRICS_ROLES)
            if entry is None:
                raise Unauthorized("a metrics, submitter or admin token is required")
            q.principal, q.role = f"client:{entry.name}", entry.role
            return
        if access in ("client", "admin", "client_or_worker"):
            entry = self.creds.authenticate(token, CLIENT_ROLES)
            if entry is not None:
                if access == "admin" and entry.role != "admin":
                    raise Forbidden("this endpoint needs an admin token")
                q.principal, q.role = f"client:{entry.name}", entry.role
                return
        if access in ("worker", "client_or_worker"):
            q.worker_id = self.cp.authenticate_worker(token)
            q.principal = f"worker:{q.worker_id}"
            return
        raise Unauthorized("invalid credentials for this endpoint")


class FailureLimiter:
    """Blocks an address for ``window`` seconds after ``limit`` failed authentications."""

    def __init__(self, limit: int = 20, window: float = 60.0) -> None:
        self.limit, self.window = limit, window
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, ip: str, now: float) -> list[float]:
        recent = [t for t in self._fails.get(ip, []) if now - t < self.window]
        self._fails[ip] = recent
        return recent

    def check(self, ip: str) -> None:
        with self._lock:
            if len(self._recent(ip, time.monotonic())) >= self.limit:
                raise RateLimited("too many failed authentication attempts; try again later")

    def failure(self, ip: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._recent(ip, now).append(now)


class _Raw:
    def __init__(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.data = data
        self.content_type = content_type
        self.headers = headers or {}


UI_DIR = Path(__file__).parent / "ui"
UI_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "icon.svg": "image/svg+xml",
}
UI_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}


def _ui(name: str) -> _Raw:
    name = name or "index.html"
    if name not in UI_FILES:  # fixed allow-list: no path traversal possible
        raise NotFound(f"no such dashboard file: {name}")
    return _Raw((UI_DIR / name).read_bytes(), UI_FILES[name], UI_HEADERS)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _encode(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_encode(v) for v in value]
    return value


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "GPUlayer"
    protocol_version = "HTTP/1.1"
    # Connections stay open between requests (clients reuse them); an idle one is closed after
    # this many seconds. Longer than the longest long poll (MAX_LONG_POLL).
    timeout = 120

    def setup(self) -> None:
        super().setup()
        # Headers and body go out as separate writes; without TCP_NODELAY, Nagle's algorithm
        # holds the body until the client's delayed ACK, about 40 ms per request on Linux.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._handshake_failed = False
        if isinstance(self.connection, ssl.SSLSocket):
            try:
                self.connection.do_handshake()
            except (ssl.SSLError, OSError) as e:  # not TLS, wrong client, closed early
                log.debug("TLS handshake with %s failed: %s", self.client_address[0], e)
                self._handshake_failed = True

    def handle(self) -> None:
        if not self._handshake_failed:
            super().handle()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("%s %s", self.address_string(), format % args)  # never logs headers/tokens

    def _bearer(self) -> str:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or len(auth) < 8:
            raise Unauthorized("missing bearer token")
        return auth[7:].strip()

    def _dispatch(self, method: str) -> None:
        app: ControlPlaneServer = self.server.app  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        status, payload, raw = 200, None, None
        raw_type = "application/octet-stream"
        raw_headers: dict[str, str] = {}
        q: _Req | None = None
        try:
            for m, pattern, access, fn in app.routes:
                match = pattern.match(path)
                if match and m == method:
                    q = _Req(self, match.groupdict())
                    app.authorize(q, access)
                    result = fn(q)
                    if isinstance(result, _Raw):
                        raw, raw_type = result.data, result.content_type
                        raw_headers = result.headers
                    else:
                        payload = _encode(result)
                    break
            else:
                raise NotFound(f"no route for {method} {path}")
        except ValidationError as e:
            status, payload = 400, _err("invalid_request", str(e))
        except GpuLayerError as e:
            status, payload = e.http_status, _err(e.code, str(e))
        except (ValueError, KeyError, TypeError) as e:
            status, payload = 400, _err("invalid_request", f"{type(e).__name__}: {e}")
        except Exception:
            log.exception("unhandled error")
            status, payload = 500, _err("error", "internal error")
        # Drain an unread body so keep-alive connections stay in sync.
        if q is None or q._body is None:
            n = int(self.headers.get("Content-Length") or 0)
            if 0 < n <= MAX_JSON_BYTES:
                self.rfile.read(n)
            elif n:
                self.close_connection = True
        body = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", raw_type if raw is not None else "application/json")
        for k, v in raw_headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")


def _err(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}
