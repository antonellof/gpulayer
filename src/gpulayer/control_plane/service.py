"""The control plane: job lifecycle, worker registry, leases, scheduling, coordination.

One ``ControlPlane`` instance backs both deployment shapes:

* embedded -- the SDK/CLI and local workers call it in-process over a SQLite file;
* served   -- ``gpulayer.api.server`` exposes it over authenticated HTTP(S).

Delivery semantics are at-least-once at best: a lease expiry marks a task ``lost``
but cannot prove the process stopped, so a retried task may duplicate side effects.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from gpulayer.artifacts.store import ArtifactStore
from gpulayer.contracts import (
    ArtifactKind,
    ArtifactReference,
    CostRecord,
    DistributedJob,
    ExecutionEvent,
    HeartbeatResponse,
    JobResult,
    JobSpec,
    JobState,
    JobStatus,
    LogLine,
    Registration,
    RendezvousInfo,
    SchedulingDecision,
    TaskAssignment,
    TaskOutcome,
    TaskState,
    TaskStatus,
    WorkerCapabilities,
    WorkerMetrics,
    WorkerOrigin,
    WorkerState,
    WorkerStatus,
    utcnow,
)
from gpulayer.contracts.models import JOB_TRANSITIONS
from gpulayer.control_plane.scheduler import (
    WorkerView,
    ever_schedulable,
    pick_devices,
    schedule,
)
from gpulayer.errors import Conflict, Forbidden, NotFound, Unauthorized, ValidationFailed
from gpulayer.observability.metrics import observe
from gpulayer.storage.sqlite import SqliteStore, parse_ts, ts

ACTIVE_TASK_STATES = (TaskState.ASSIGNED.value, TaskState.RUNNING.value)
MAX_LOG_LINE = 8192


@dataclass
class ControlPlaneConfig:
    heartbeat_interval_seconds: float = 2.0
    lease_ttl_seconds: float = 20.0
    timeout_grace_seconds: float = 30.0
    public_url: str | None = None  # given to provisioned workers for bootstrap
    provisioning_retry_seconds: float = 30.0  # back-off after a retryable capacity error


MAX_LONG_POLL = 30.0  # seconds a long-polling request may be held open
POLL_SLICE = 0.2  # re-check interval while waiting, for changes made by other processes


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class ControlPlane:
    def __init__(
        self,
        store: SqliteStore,
        artifacts: ArtifactStore,
        *,
        config: ControlPlaneConfig | None = None,
        providers: Iterable[Any] = (),
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        from gpulayer.control_plane.provisioning import ProvisioningManager

        self.store = store
        self.artifacts = artifacts
        self.config = config or ControlPlaneConfig()
        self.now = clock
        self.provisioning = ProvisioningManager(self, {p.name: p for p in providers})
        # Wakes long-polling workers and job waiters when work appears or a job changes
        # state. Only covers this process; waiters also re-check the database periodically
        # so changes made by another process (a CLI submit) are seen within POLL_SLICE.
        self._changed = threading.Condition()
        self._generation = 0

    # ================================================================== jobs

    def submit(self, spec: JobSpec, *, submitted_by: str | None = None) -> JobStatus:
        spec = JobSpec.model_validate(spec.model_dump())  # re-validate at the trust boundary
        for ref in self._referenced_artifacts(spec):
            if not self.artifacts.exists(ref.sha256):
                raise ValidationFailed(f"artifact {ref.uri} has not been uploaded")
        if spec.placement.provisioning and not self.provisioning.has(
            spec.placement.provisioning.provider
        ):
            raise ValidationFailed(
                f"provider {spec.placement.provisioning.provider!r} is not configured on this "
                "control plane"
            )
        job_id = _new_id("job")
        now = ts(self.now())
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO jobs (id, name, state, spec, created_at, updated_at, submitted_by) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    job_id,
                    spec.name,
                    JobState.QUEUED.value,
                    spec.model_dump_json(),
                    now,
                    now,
                    submitted_by,
                ),
            )
            self.store.add_event(
                "job.submitted",
                f"{spec.workload_type.value} job '{spec.name}' submitted, target "
                f"{spec.placement.target.value}",
                job_id=job_id,
            )
            self.store.inc(
                "gpulayer_jobs_submitted_total",
                {"workload_type": spec.workload_type.value, "target": spec.placement.target.value},
            )
        self.schedule_pending()
        return self.get_job(job_id)

    @staticmethod
    def _referenced_artifacts(spec: JobSpec) -> list[ArtifactReference]:
        refs = [i.artifact for i in spec.inputs]
        src = getattr(spec.workload, "source", None)
        if src is not None:
            refs.append(src)
        return refs

    def _job_row(self, job_id: str) -> Any:
        row = self.store.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if row is None:
            raise NotFound(f"job {job_id} not found")
        return row

    def _set_job_state(
        self, job_id: str, new: JobState, *, error: str | None = None, message: str | None = None
    ) -> None:
        self.signal()  # waiters re-read after this transaction commits (they share its lock)
        with self.store.tx() as c:
            row = self._job_row(job_id)
            old = JobState(row["state"])
            if old == new:
                return
            if new not in JOB_TRANSITIONS.get(old, frozenset()):
                raise Conflict(f"illegal job transition {old.value} -> {new.value}")
            c.execute(
                "UPDATE jobs SET state=?, updated_at=?, error=COALESCE(?, error) WHERE id=?",
                (new.value, ts(self.now()), error, job_id),
            )
            self.store.add_event(
                f"job.{new.value}",
                message or f"job {old.value} -> {new.value}",
                job_id=job_id,
                data={"error": error} if error else None,
            )
        if new.terminal:
            self._finalize(job_id)

    def get_job(self, job_id: str) -> JobStatus:
        self.reconcile()
        row = self._job_row(job_id)
        spec = JobSpec.model_validate_json(row["spec"])
        return JobStatus(
            job_id=row["id"],
            name=row["name"],
            state=JobState(row["state"]),
            workload_type=spec.workload_type,
            spec=spec,
            created_at=parse_ts(row["created_at"]),
            updated_at=parse_ts(row["updated_at"]),
            attempts=row["attempts"],
            cancel_requested=bool(row["cancel_requested"]),
            submitted_by=row["submitted_by"],
            error=row["error"],
            scheduling=SchedulingDecision.model_validate_json(row["scheduling"])
            if row["scheduling"]
            else None,
            tasks=self._tasks(job_id),
        )

    def list_jobs(self, *, limit: int = 50, state: JobState | None = None) -> list[JobStatus]:
        sql, args = "SELECT id FROM jobs", cast(tuple[Any, ...], ())
        if state:
            sql, args = sql + " WHERE state=?", (state.value,)
        rows = self.store.query(sql + " ORDER BY created_at DESC LIMIT ?", (*args, limit))
        return [self.get_job(r["id"]) for r in rows]

    def _tasks(self, job_id: str, attempt: int | None = None) -> list[TaskStatus]:
        sql = "SELECT * FROM tasks WHERE job_id=?"
        args: tuple[Any, ...] = (job_id,)
        if attempt is not None:
            sql, args = sql + " AND attempt=?", (job_id, attempt)
        rows = self.store.query(sql + " ORDER BY attempt, node_rank", args)
        return [self._task_status(r) for r in rows]

    @staticmethod
    def _task_status(r: Any) -> TaskStatus:
        return TaskStatus(
            task_id=r["id"],
            job_id=r["job_id"],
            attempt=r["attempt"],
            node_rank=r["node_rank"],
            worker_id=r["worker_id"],
            state=TaskState(r["state"]),
            exit_code=r["exit_code"],
            error=r["error"],
            lease_expires_at=parse_ts(r["lease_expires_at"]),
            started_at=parse_ts(r["started_at"]),
            finished_at=parse_ts(r["finished_at"]),
        )

    def job_result(self, job_id: str) -> JobResult:
        row = self._job_row(job_id)
        if row["result"]:
            return JobResult.model_validate_json(row["result"])
        return JobResult(
            job_id=job_id,
            state=JobState(row["state"]),
            error=row["error"],
            task_results=self._tasks(job_id),
            costs=self._costs(job_id),
        )

    def job_logs(self, job_id: str, *, after_seq: int = 0, limit: int = 5000) -> list[LogLine]:
        self._job_row(job_id)
        rows = self.store.query(
            "SELECT * FROM logs WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?",
            (job_id, after_seq, limit),
        )
        return [
            LogLine(
                seq=r["seq"],
                task_id=r["task_id"],
                node_rank=r["node_rank"],
                stream=r["stream"],
                line=r["line"],
                timestamp=parse_ts(r["ts"]),
            )
            for r in rows
        ]

    def events(self, *, job_id: str | None = None, after_seq: int = 0) -> list[ExecutionEvent]:
        sql, args = "SELECT * FROM events WHERE seq>?", cast(tuple[Any, ...], (after_seq,))
        if job_id:
            sql, args = sql + " AND job_id=?", (after_seq, job_id)
        rows = self.store.query(sql + " ORDER BY seq", args)
        return [
            ExecutionEvent(
                seq=r["seq"],
                job_id=r["job_id"],
                task_id=r["task_id"],
                worker_id=r["worker_id"],
                timestamp=parse_ts(r["ts"]),
                type=r["type"],
                message=r["message"],
                data=json.loads(r["data"]),
            )
            for r in rows
        ]

    def cancel_job(self, job_id: str) -> JobStatus:
        """Idempotent. Unstarted tasks are cancelled at once; running ones via heartbeat."""
        with self.store.tx() as c:
            row = self._job_row(job_id)
            if JobState(row["state"]).terminal:
                return self.get_job(job_id)
            c.execute(
                "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                (ts(self.now()), job_id),
            )
            c.execute(
                "UPDATE tasks SET state=?, finished_at=?, error='cancelled before start' "
                "WHERE job_id=? AND state=?",
                (TaskState.CANCELLED.value, ts(self.now()), job_id, TaskState.ASSIGNED.value),
            )
            c.execute(
                "UPDATE tasks SET cancel_requested=1 WHERE job_id=? AND state=?",
                (job_id, TaskState.RUNNING.value),
            )
            self.store.add_event("job.cancel_requested", "cancellation requested", job_id=job_id)
        self._update_job(job_id)
        return self.get_job(job_id)

    # ============================================================ scheduling

    def _worker_views(self) -> list[WorkerView]:
        running: dict[str, int] = {}
        busy: dict[str, set[int]] = {}
        for r in self.store.query(
            "SELECT worker_id, devices FROM tasks WHERE state IN (?,?)", ACTIVE_TASK_STATES
        ):
            running[r["worker_id"]] = running.get(r["worker_id"], 0) + 1
            if r["devices"]:
                busy.setdefault(r["worker_id"], set()).update(json.loads(r["devices"]))
        views = []
        for r in self.store.query("SELECT * FROM workers WHERE revoked=0"):
            views.append(
                WorkerView(
                    worker_id=r["id"],
                    state=WorkerState(r["state"]),
                    capabilities=WorkerCapabilities.model_validate_json(r["capabilities"]),
                    running_tasks=running.get(r["id"], 0),
                    busy_devices=busy.get(r["id"], set()),
                )
            )
        return views

    def schedule_pending(self) -> None:
        with self.store.tx() as c:
            views = self._worker_views()
            jobs = self.store.query(
                "SELECT * FROM jobs WHERE state IN (?,?) ORDER BY created_at",
                (JobState.QUEUED.value, JobState.PROVISIONING.value),
            )
            for row in jobs:
                job_id = row["id"]
                spec = JobSpec.model_validate_json(row["spec"])
                if row["cancel_requested"]:
                    self._set_job_state(job_id, JobState.CANCELLED)
                    continue
                decision = schedule(spec, views, job_id=job_id)
                c.execute(
                    "UPDATE jobs SET scheduling=? WHERE id=?", (decision.model_dump_json(), job_id)
                )
                if decision.placed:
                    self._place(c, job_id, spec, row["attempts"] + 1, decision, views)
                    continue
                if spec.placement.provisioning and JobState(row["state"]) == JobState.QUEUED:
                    self.provisioning.mark(job_id, spec, decision)
                elif (
                    spec.placement.on_unschedulable == "fail"
                    and not spec.placement.provisioning
                    and not ever_schedulable(spec, views, job_id=job_id)
                ):
                    self._set_job_state(
                        job_id,
                        JobState.FAILED,
                        error=f"unschedulable: {decision.summary}",
                    )
        # Provider calls (provision, terminate) happen after the transaction is closed.
        self.provisioning.pump()

    def _place(
        self,
        c: Any,
        job_id: str,
        spec: JobSpec,
        attempt: int,
        decision: SchedulingDecision,
        views: list[WorkerView],
    ) -> None:
        """Create one task per node and reserve its devices (updating ``views`` so later
        jobs in the same scheduling pass see the reservation)."""
        lease = ts(self.now() + timedelta(seconds=self.config.lease_ttl_seconds))
        by_id = {v.worker_id: v for v in views}
        for rank, worker_id in enumerate(decision.worker_ids):
            view = by_id[worker_id]
            devices = pick_devices(spec.resources, view.capabilities, view.busy_devices)
            if devices is None:  # schedule() checked this; never place on a busy device
                raise Conflict(f"devices on {worker_id} were taken during placement")
            view.running_tasks += 1
            view.busy_devices.update(devices)
            task_id = _new_id("task")
            c.execute(
                "INSERT INTO tasks (id, job_id, attempt, node_rank, worker_id, state, "
                "lease_expires_at, devices) VALUES (?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    job_id,
                    attempt,
                    rank,
                    worker_id,
                    TaskState.ASSIGNED.value,
                    lease,
                    json.dumps(devices) if devices else None,
                ),
            )
        c.execute("UPDATE jobs SET attempts=?, rendezvous=NULL WHERE id=?", (attempt, job_id))
        self._set_job_state(job_id, JobState.SCHEDULED, message=decision.summary)

    # ======================================================= worker registry

    def register_worker(
        self, name: str, capabilities: WorkerCapabilities, *, instance_token: str | None = None
    ) -> Registration:
        """Caller must already be authorized to join (static join token or embedded trust),
        except provider workers, which must present their single-use instance token."""
        if capabilities.origin == WorkerOrigin.PROVIDER:
            capabilities = self.provisioning.verify_and_bind(capabilities, instance_token)
        worker_id = _new_id("wkr")
        token = secrets.token_urlsafe(32)
        now = ts(self.now())
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO workers (id, name, state, capabilities, registered_at, "
                "last_heartbeat_at, token_hash) VALUES (?,?,?,?,?,?,?)",
                (
                    worker_id,
                    name,
                    WorkerState.IDLE.value,
                    capabilities.model_dump_json(),
                    now,
                    now,
                    _hash_token(token),
                ),
            )
            self.store.add_event(
                "worker.registered",
                f"worker {name} registered ({capabilities.origin.value}, "
                f"{len(capabilities.accelerators)} accelerator(s))",
                worker_id=worker_id,
            )
        self.reconcile()
        self.schedule_pending()
        return Registration(
            worker_id=worker_id,
            session_token=token,
            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
            lease_ttl_seconds=self.config.lease_ttl_seconds,
        )

    def authenticate_worker(self, token: str) -> str:
        row = self.store.one(
            "SELECT id, revoked FROM workers WHERE token_hash=?", (_hash_token(token),)
        )
        if row is None:
            raise Unauthorized("unknown worker token")
        if row["revoked"]:
            raise Forbidden("worker has been revoked")
        return str(row["id"])

    def _worker_row(self, worker_id: str) -> Any:
        row = self.store.one("SELECT * FROM workers WHERE id=?", (worker_id,))
        if row is None:
            raise NotFound(f"worker {worker_id} not found")
        return row

    def list_workers(self) -> list[WorkerStatus]:
        self.reconcile()
        running: dict[str, list[str]] = {}
        for r in self.store.query(
            "SELECT id, worker_id FROM tasks WHERE state IN (?,?)", ACTIVE_TASK_STATES
        ):
            running.setdefault(r["worker_id"], []).append(r["id"])
        return [
            WorkerStatus(
                worker_id=r["id"],
                name=r["name"],
                state=WorkerState(r["state"]),
                capabilities=WorkerCapabilities.model_validate_json(r["capabilities"]),
                registered_at=parse_ts(r["registered_at"]),
                last_heartbeat_at=parse_ts(r["last_heartbeat_at"]),
                running_task_ids=running.get(r["id"], []),
            )
            for r in self.store.query("SELECT * FROM workers ORDER BY registered_at")
        ]

    def heartbeat(
        self, worker_id: str, *, draining: bool = False, metrics: WorkerMetrics | None = None
    ) -> HeartbeatResponse:
        row = self._worker_row(worker_id)
        if row["revoked"]:
            return HeartbeatResponse(lease_ttl_seconds=self.config.lease_ttl_seconds, revoked=True)
        now = self.now()
        lease = ts(now + timedelta(seconds=self.config.lease_ttl_seconds))
        with self.store.tx() as c:
            if metrics is not None:
                self._record_sample(c, worker_id, now, metrics)
            active = c.execute(
                "SELECT COUNT(*) FROM tasks WHERE worker_id=? AND state IN (?,?)",
                (worker_id, *ACTIVE_TASK_STATES),
            ).fetchone()[0]
            state = (
                WorkerState.DRAINING
                if draining
                else (WorkerState.BUSY if active else WorkerState.IDLE)
            )
            if WorkerState(row["state"]) == WorkerState.OFFLINE:
                self.store.add_event(
                    "worker.recovered", "heartbeat after going offline", worker_id=worker_id
                )
            c.execute(
                "UPDATE workers SET last_heartbeat_at=?, state=? WHERE id=?",
                (ts(now), state.value, worker_id),
            )
            c.execute(
                "UPDATE tasks SET lease_expires_at=? WHERE worker_id=? AND state IN (?,?)",
                (lease, worker_id, *ACTIVE_TASK_STATES),
            )
            cancel = [
                r[0]
                for r in c.execute(
                    "SELECT t.id FROM tasks t JOIN jobs j ON j.id=t.job_id WHERE t.worker_id=? "
                    "AND ((t.state=? AND (t.cancel_requested=1 OR j.cancel_requested=1)) "
                    "OR t.state=?)",
                    (worker_id, TaskState.RUNNING.value, TaskState.LOST.value),
                ).fetchall()
            ]
        return HeartbeatResponse(
            lease_ttl_seconds=self.config.lease_ttl_seconds, cancel_task_ids=cancel
        )

    def deregister_worker(self, worker_id: str) -> None:
        self._worker_row(worker_id)
        with self.store.tx() as c:
            c.execute(
                "UPDATE workers SET state=? WHERE id=?", (WorkerState.OFFLINE.value, worker_id)
            )
            self.store.add_event("worker.offline", "worker deregistered", worker_id=worker_id)
        self._lose_tasks_of(worker_id, "worker deregistered with active tasks")

    def revoke_worker(self, worker_id: str) -> None:
        self._worker_row(worker_id)
        with self.store.tx() as c:
            c.execute(
                "UPDATE workers SET revoked=1, state=? WHERE id=?",
                (WorkerState.OFFLINE.value, worker_id),
            )
            self.store.add_event(
                "worker.revoked", "worker credentials revoked", worker_id=worker_id
            )
        self._lose_tasks_of(worker_id, "worker revoked")

    # ======================================================= task protocol

    def _owned_task(self, worker_id: str, task_id: str) -> Any:
        row = self.store.one("SELECT * FROM tasks WHERE id=?", (task_id,))
        if row is None:
            raise NotFound(f"task {task_id} not found")
        if row["worker_id"] != worker_id:
            raise Forbidden("task is not assigned to this worker")
        return row

    # ============================================================ telemetry

    SAMPLE_INTERVAL = 5.0  # seconds; keep at most one stored sample per worker per interval
    SAMPLE_RETENTION = timedelta(hours=24)

    def _record_sample(self, c: Any, worker_id: str, now: datetime, m: WorkerMetrics) -> None:
        last = c.execute(
            "SELECT MAX(ts) FROM worker_samples WHERE worker_id=?", (worker_id,)
        ).fetchone()[0]
        last_at = parse_ts(last) if last else None
        if last_at is not None and (now - last_at).total_seconds() < self.SAMPLE_INTERVAL:
            return
        c.execute(
            "INSERT INTO worker_samples (worker_id, ts, data) VALUES (?,?,?)",
            (worker_id, ts(now), m.model_dump_json(exclude_none=True)),
        )
        c.execute("DELETE FROM worker_samples WHERE ts < ?", (ts(now - self.SAMPLE_RETENTION),))

    def latest_samples(self, max_age_seconds: float = 60.0) -> dict[str, WorkerMetrics]:
        """The newest sample of each worker, if it is recent."""
        since = ts(self.now() - timedelta(seconds=max_age_seconds))
        rows = self.store.query(
            "SELECT s.worker_id, s.data FROM worker_samples s JOIN (SELECT worker_id, MAX(ts) m "
            "FROM worker_samples WHERE ts >= ? GROUP BY worker_id) l "
            "ON l.worker_id=s.worker_id AND l.m=s.ts",
            (since,),
        )
        return {r["worker_id"]: WorkerMetrics.model_validate_json(r["data"]) for r in rows}

    def metrics_history(self, minutes: float = 60.0, points: int = 120) -> dict[str, Any]:
        """Time series for the dashboard: per-worker resource samples, and jobs running and
        waiting per time bucket (derived from job and task timestamps)."""
        minutes = min(max(minutes, 1.0), 24 * 60)
        now = self.now()
        start = now - timedelta(minutes=minutes)
        names = {r["id"]: r["name"] for r in self.store.query("SELECT id, name FROM workers")}
        series: dict[str, list[dict[str, Any]]] = {}
        for r in self.store.query(
            "SELECT worker_id, ts, data FROM worker_samples WHERE ts >= ? ORDER BY ts",
            (ts(start),),
        ):
            sample = json.loads(r["data"])
            sample["t"] = r["ts"]
            series.setdefault(r["worker_id"], []).append(sample)
        workers = [
            {"worker_id": wid, "name": names.get(wid, wid), "samples": s}
            for wid, s in sorted(series.items(), key=lambda kv: names.get(kv[0], kv[0]))
        ]
        # Job activity: one interval for waiting (submit -> first start) and one for running
        # (first start -> end), then count overlaps at each bucket boundary.
        terminal = [s.value for s in JobState if s.terminal]
        rows = self.store.query(
            "SELECT j.id, j.state, j.created_at, j.updated_at, MIN(t.started_at) started "
            "FROM jobs j LEFT JOIN tasks t ON t.job_id=j.id "
            f"WHERE j.updated_at >= ? OR j.state NOT IN ({','.join('?' * len(terminal))}) "
            "GROUP BY j.id",
            (ts(start), *terminal),
        )
        spans: list[tuple[datetime, datetime, datetime | None]] = []
        for r in rows:
            created = parse_ts(r["created_at"])
            # Active jobs have no end yet; extend past `now` so the last bucket counts them.
            end = parse_ts(r["updated_at"]) if r["state"] in terminal else now + timedelta(days=1)
            if created is None or end is None:
                continue
            spans.append((created, end, parse_ts(r["started"]) if r["started"] else None))
        step = (now - start) / max(points - 1, 1)
        jobs = []
        for i in range(points):
            t = start + step * i
            running = sum(1 for _, end, s in spans if s is not None and s <= t < end)
            waiting = sum(
                1 for created, end, s in spans if created <= t < (s if s is not None else end)
            )
            jobs.append({"t": ts(t), "running": running, "waiting": waiting})
        return {"now": ts(now), "minutes": minutes, "workers": workers, "jobs": jobs}

    # ============================================================ change signal

    def signal(self) -> None:
        with self._changed:
            self._generation += 1
            self._changed.notify_all()

    def _wait_for_change(self, generation: int, timeout: float) -> None:
        with self._changed:
            if self._generation == generation and timeout > 0:
                self._changed.wait(timeout)

    def acquire_task(
        self, worker_id: str, *, wait: float = 0.0, stop: threading.Event | None = None
    ) -> TaskAssignment | None:
        """Return an assignment for this worker, if any. Safe to call repeatedly.

        With ``wait`` > 0 this is a long poll: it returns as soon as a task is assigned to
        the worker, or None after ``wait`` seconds (capped at MAX_LONG_POLL).
        """
        deadline = time.monotonic() + min(max(wait, 0.0), MAX_LONG_POLL)
        while True:
            generation = self._generation
            assignment = self._acquire_once(worker_id)
            remaining = deadline - time.monotonic()
            if assignment is not None or remaining <= 0 or (stop is not None and stop.is_set()):
                return assignment
            self._wait_for_change(generation, min(remaining, POLL_SLICE))

    def wait_job(self, job_id: str, timeout: float) -> JobStatus:
        """Return the job once it is terminal, or its current status after ``timeout``."""
        deadline = time.monotonic() + min(max(timeout, 0.0), MAX_LONG_POLL)
        while True:
            generation = self._generation
            status = self.get_job(job_id)
            remaining = deadline - time.monotonic()
            if status.state.terminal or remaining <= 0:
                return status
            self._wait_for_change(generation, min(remaining, POLL_SLICE))

    def _acquire_once(self, worker_id: str) -> TaskAssignment | None:
        if self._worker_row(worker_id)["revoked"]:
            raise Forbidden("worker has been revoked")
        self.reconcile()
        self.schedule_pending()
        rows = self.store.query(
            "SELECT t.*, j.spec, j.rendezvous FROM tasks t JOIN jobs j ON j.id=t.job_id "
            "WHERE t.worker_id=? AND t.state=? ORDER BY j.created_at, t.node_rank",
            (worker_id, TaskState.ASSIGNED.value),
        )
        for r in rows:
            spec = JobSpec.model_validate_json(r["spec"])
            rdzv = RendezvousInfo.model_validate_json(r["rendezvous"]) if r["rendezvous"] else None
            if isinstance(spec.workload, DistributedJob) and r["node_rank"] > 0 and rdzv is None:
                continue  # wait for rank 0 to publish the rendezvous endpoint
            if rdzv is not None:
                rdzv = rdzv.model_copy(update={"node_rank": r["node_rank"]})
            return TaskAssignment(
                task_id=r["id"],
                job_id=r["job_id"],
                attempt=r["attempt"],
                spec=spec,
                node_rank=r["node_rank"],
                lease_expires_at=parse_ts(r["lease_expires_at"]),
                rendezvous=rdzv,
                devices=json.loads(r["devices"]) if r["devices"] else None,
            )
        return None

    def publish_rendezvous(
        self, worker_id: str, task_id: str, master_addr: str, master_port: int
    ) -> RendezvousInfo:
        task = self._owned_task(worker_id, task_id)
        if task["node_rank"] != 0:
            raise Forbidden("only node rank 0 publishes the rendezvous endpoint")
        row = self._job_row(task["job_id"])
        spec = JobSpec.model_validate_json(row["spec"])
        if not isinstance(spec.workload, DistributedJob):
            raise ValidationFailed("rendezvous only applies to distributed jobs")
        info = RendezvousInfo(
            master_addr=master_addr,
            master_port=master_port,
            num_nodes=spec.workload.num_workers,
            procs_per_node=spec.workload.procs_per_worker,
            node_rank=0,
        )
        with self.store.tx() as c:
            current = c.execute(
                "SELECT rendezvous FROM jobs WHERE id=?", (task["job_id"],)
            ).fetchone()[0]
            if current and RendezvousInfo.model_validate_json(current) != info:
                raise Conflict("rendezvous endpoint already published")
            c.execute(
                "UPDATE jobs SET rendezvous=? WHERE id=?", (info.model_dump_json(), task["job_id"])
            )
            self.signal()
            self.store.add_event(
                "job.rendezvous",
                f"rendezvous at {master_addr}:{master_port}",
                job_id=task["job_id"],
                task_id=task_id,
            )
        return info

    def start_task(self, worker_id: str, task_id: str) -> None:
        task = self._owned_task(worker_id, task_id)
        state = TaskState(task["state"])
        if state == TaskState.RUNNING:
            return  # idempotent
        if state != TaskState.ASSIGNED:
            raise Conflict(f"task is {state.value}; do not start it")
        spec = JobSpec.model_validate_json(self._job_row(task["job_id"])["spec"])
        now = self.now()
        deadline = (
            ts(now + timedelta(seconds=spec.timeout_seconds)) if spec.timeout_seconds else None
        )
        with self.store.tx() as c:
            c.execute(
                "UPDATE tasks SET state=?, started_at=?, deadline=?, lease_expires_at=? WHERE id=?",
                (
                    TaskState.RUNNING.value,
                    ts(now),
                    deadline,
                    ts(now + timedelta(seconds=self.config.lease_ttl_seconds)),
                    task_id,
                ),
            )
            self.store.add_event(
                "task.running",
                f"node rank {task['node_rank']} started",
                job_id=task["job_id"],
                task_id=task_id,
                worker_id=worker_id,
            )
            job = self._job_row(task["job_id"])
            if JobState(job["state"]) == JobState.SCHEDULED:
                if job["attempts"] == 1:
                    created = parse_ts(job["created_at"]) or now
                    wait = (now - created).total_seconds()
                    observe(
                        self.store,
                        "gpulayer_job_queue_wait_seconds",
                        max(wait, 0.0),
                        {"target": spec.placement.target.value},
                    )
                self._set_job_state(task["job_id"], JobState.RUNNING)

    def append_logs(self, worker_id: str, task_id: str, lines: list[tuple[str, str]]) -> None:
        task = self._owned_task(worker_id, task_id)
        now = ts(self.now())
        with self.store.tx() as c:
            c.executemany(
                "INSERT INTO logs (job_id, task_id, node_rank, stream, line, ts) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (
                        task["job_id"],
                        task_id,
                        task["node_rank"],
                        stream if stream in ("stdout", "stderr", "system") else "system",
                        line[:MAX_LOG_LINE],
                        now,
                    )
                    for stream, line in lines
                ],
            )

    def finish_task(self, worker_id: str, task_id: str, outcome: TaskOutcome) -> None:
        task = self._owned_task(worker_id, task_id)
        if TaskState(task["state"]).terminal:
            self.store.add_event(
                "task.late_report",
                f"ignored {outcome.state.value} report for {task['state']} task",
                job_id=task["job_id"],
                task_id=task_id,
                worker_id=worker_id,
            )
            return
        for ref in outcome.outputs:
            if not self.artifacts.exists(ref.sha256):
                raise ValidationFailed(f"output artifact {ref.uri} was not uploaded")
        with self.store.tx() as c:
            c.execute(
                "UPDATE tasks SET state=?, exit_code=?, error=?, finished_at=?, outcome=? "
                "WHERE id=?",
                (
                    outcome.state.value,
                    outcome.exit_code,
                    outcome.error,
                    ts(self.now()),
                    outcome.model_dump_json(),
                    task_id,
                ),
            )
            self.store.add_event(
                f"task.{outcome.state.value}",
                f"node rank {task['node_rank']} {outcome.state.value}"
                + (f" (exit {outcome.exit_code})" if outcome.exit_code is not None else ""),
                job_id=task["job_id"],
                task_id=task_id,
                worker_id=worker_id,
                data={"error": outcome.error} if outcome.error else None,
            )
        self._update_job(task["job_id"])
        self.schedule_pending()

    # ====================================================== job resolution

    def _update_job(self, job_id: str) -> None:
        """Derive the job state from the tasks of its current attempt."""
        with self.store.tx() as c:
            row = self._job_row(job_id)
            state = JobState(row["state"])
            if state.terminal or state in (JobState.QUEUED, JobState.PROVISIONING):
                if state in (JobState.QUEUED, JobState.PROVISIONING) and row["cancel_requested"]:
                    self._set_job_state(job_id, JobState.CANCELLED)
                return
            spec = JobSpec.model_validate_json(row["spec"])
            tasks = self._tasks(job_id, row["attempts"])
            states = [t.state for t in tasks]
            distributed = spec.workload_type.value == "distributed"

            bad = [
                t
                for t in tasks
                if t.state in (TaskState.FAILED, TaskState.LOST, TaskState.TIMED_OUT)
            ]
            if distributed and bad and not all(s.terminal for s in states):
                # fail_fast: one failed node invalidates the group; stop the others.
                c.execute(
                    "UPDATE tasks SET cancel_requested=1 WHERE job_id=? AND attempt=? AND state=?",
                    (job_id, row["attempts"], TaskState.RUNNING.value),
                )
                c.execute(
                    "UPDATE tasks SET state=?, finished_at=? WHERE job_id=? AND attempt=? "
                    "AND state=?",
                    (
                        TaskState.CANCELLED.value,
                        ts(self.now()),
                        job_id,
                        row["attempts"],
                        TaskState.ASSIGNED.value,
                    ),
                )
                self._resolve_terminal(job_id, spec, tasks, row)
                return
            if not all(s.terminal for s in states):
                return
            self._resolve_terminal(job_id, spec, tasks, row)

    def _resolve_terminal(
        self, job_id: str, spec: JobSpec, tasks: list[TaskStatus], row: Any
    ) -> None:
        states = {t.state for t in tasks}
        retry = spec.retry
        if states == {TaskState.SUCCEEDED}:
            self._set_job_state(job_id, JobState.SUCCEEDED)
            return
        if row["cancel_requested"]:
            self._set_job_state(job_id, JobState.CANCELLED)
            return
        reason_map = {TaskState.LOST: "worker_lost", TaskState.FAILED: "failed"}
        failed = [t for t in tasks if t.state in reason_map]
        if (
            spec.workload_type.value == "independent"
            and failed
            and reason_map[failed[0].state] in retry.retry_on
            and row["attempts"] < retry.max_attempts
            and not row["cancel_requested"]
        ):
            with self.store.tx():
                self._set_job_state(
                    job_id,
                    JobState.QUEUED,
                    message=f"retrying after {reason_map[failed[0].state]} (attempt "
                    f"{row['attempts']} of {retry.max_attempts}); side effects may repeat",
                )
            self.schedule_pending()
            return
        if TaskState.TIMED_OUT in states and not failed:
            self._set_job_state(job_id, JobState.TIMED_OUT, error="job exceeded timeout_seconds")
            return
        if TaskState.CANCELLED in states and not failed and TaskState.TIMED_OUT not in states:
            self._set_job_state(job_id, JobState.CANCELLED)
            return
        culprits = (TaskState.FAILED, TaskState.LOST, TaskState.TIMED_OUT)
        first = next((t for t in tasks if t.state in culprits), tasks[0] if tasks else None)
        detail = f"node rank {first.node_rank} {first.state.value}" if first else "failed"
        if first and first.error:
            detail += f": {first.error}"
        self._set_job_state(job_id, JobState.FAILED, error=detail)

    def _finalize(self, job_id: str) -> None:
        """Persist the JobResult once the job is terminal and release provisioned capacity."""
        row = self._job_row(job_id)
        tasks = self._tasks(job_id, row["attempts"]) if row["attempts"] else []
        self.store.inc("gpulayer_jobs_finished_total", {"state": row["state"]})
        started = [t.started_at for t in tasks if t.started_at]
        if started:
            run = (self.now() - min(started)).total_seconds()
            observe(self.store, "gpulayer_job_run_seconds", max(run, 0.0), {"state": row["state"]})
        outcomes = {
            r["id"]: TaskOutcome.model_validate_json(r["outcome"])
            for r in self.store.query(
                "SELECT id, outcome FROM tasks WHERE job_id=? AND "
                "attempt=? AND outcome IS NOT NULL",
                (job_id, row["attempts"]),
            )
        }
        rank0 = next((t for t in tasks if t.node_rank == 0), None)
        rank0_out = outcomes.get(rank0.task_id) if rank0 else None
        exit_codes = [t.exit_code for t in tasks if t.exit_code is not None]
        outputs = [o for out in outcomes.values() for o in out.outputs]
        result = JobResult(
            job_id=job_id,
            state=JobState(row["state"]),
            exit_code=next(
                (c for c in exit_codes if c != 0), exit_codes[0] if exit_codes else None
            ),
            return_value=rank0_out.return_value if rank0_out else None,
            error=row["error"],
            outputs=outputs,
            task_results=tasks,
            costs=self._costs(job_id),
        )
        with self.store.tx() as c:
            c.execute("UPDATE jobs SET result=? WHERE id=?", (result.model_dump_json(), job_id))
        self.provisioning.release_for_job(job_id)

    # ========================================================== reconciliation

    def reconcile(self) -> None:
        """Expire leases and heartbeats, enforce deadlines, advance provisioning."""
        now = self.now()
        cutoff = ts(now - timedelta(seconds=self.config.lease_ttl_seconds))
        with self.store.tx() as c:
            for r in c.execute(
                "SELECT id FROM workers WHERE state NOT IN (?) AND revoked=0 AND "
                "last_heartbeat_at < ?",
                (WorkerState.OFFLINE.value, cutoff),
            ).fetchall():
                c.execute(
                    "UPDATE workers SET state=? WHERE id=?", (WorkerState.OFFLINE.value, r[0])
                )
                self.store.add_event("worker.offline", "heartbeat lease expired", worker_id=r[0])
            expired = c.execute(
                "SELECT id, job_id, worker_id FROM tasks WHERE state IN (?,?) AND "
                "lease_expires_at < ?",
                (*ACTIVE_TASK_STATES, ts(now)),
            ).fetchall()
            for t in expired:
                self._mark_task(
                    c,
                    t["id"],
                    TaskState.LOST,
                    "worker lease expired; the process may still be running",
                )
            overdue = c.execute(
                "SELECT id, job_id FROM tasks WHERE state=? AND deadline IS NOT NULL "
                "AND deadline < ?",
                (
                    TaskState.RUNNING.value,
                    ts(now - timedelta(seconds=self.config.timeout_grace_seconds)),
                ),
            ).fetchall()
            for t in overdue:
                self._mark_task(
                    c, t["id"], TaskState.TIMED_OUT, "deadline passed without a worker report"
                )
                c.execute("UPDATE tasks SET cancel_requested=1 WHERE id=?", (t["id"],))
        for job_id in {t["job_id"] for t in [*expired, *overdue]}:
            self._update_job(job_id)
        self.provisioning.reconcile()
        self.provisioning.pump()

    def _mark_task(self, c: Any, task_id: str, state: TaskState, error: str) -> None:
        row = c.execute("SELECT job_id, worker_id FROM tasks WHERE id=?", (task_id,)).fetchone()
        c.execute(
            "UPDATE tasks SET state=?, error=?, finished_at=? WHERE id=?",
            (state.value, error, ts(self.now()), task_id),
        )
        self.store.add_event(
            f"task.{state.value}",
            error,
            job_id=row["job_id"],
            task_id=task_id,
            worker_id=row["worker_id"],
        )
        if state == TaskState.LOST:
            self.store.inc("gpulayer_tasks_lost_total")

    def _lose_tasks_of(self, worker_id: str, reason: str) -> None:
        with self.store.tx() as c:
            rows = c.execute(
                "SELECT id, job_id FROM tasks WHERE worker_id=? AND state IN (?,?)",
                (worker_id, *ACTIVE_TASK_STATES),
            ).fetchall()
            for r in rows:
                self._mark_task(c, r["id"], TaskState.LOST, reason)
        for job_id in {r["job_id"] for r in rows}:
            self._update_job(job_id)

    # ============================================================ artifacts/costs

    def put_artifact(
        self, data: bytes, name: str = "", kind: ArtifactKind = ArtifactKind.INPUT
    ) -> ArtifactReference:
        return self.artifacts.put(data, name, kind)

    # ============================================================ observability

    def metrics_text(self) -> str:
        """Prometheus text exposition of the current state and counters."""
        from gpulayer.observability.metrics import render

        return render(self)

    def whoami(self) -> dict[str, str]:
        return {"principal": "local", "role": "admin"}  # embedded mode: the local OS user

    # ============================================================ retention

    def gc(
        self, older_than_seconds: float, *, delete_jobs: bool = False, dry_run: bool = False
    ) -> dict[str, int]:
        """Remove old data. Returns counts of what was (or, with ``dry_run``, would be) removed.

        * logs and events of jobs that finished before the cutoff;
        * with ``delete_jobs``, those jobs with their tasks, costs and released instances;
        * offline or revoked workers not seen since the cutoff;
        * artifacts that no remaining job references and that are older than the cutoff
          (a fresh upload that has not been submitted yet is never removed).
        """
        cutoff = self.now() - timedelta(seconds=older_than_seconds)
        terminal = tuple(st.value for st in JobState if st.terminal)
        marks = ",".join("?" * len(terminal))
        old_jobs = [
            r["id"]
            for r in self.store.query(
                f"SELECT id FROM jobs WHERE state IN ({marks}) AND updated_at < ?",
                (*terminal, ts(cutoff)),
            )
        ]
        report = {
            "jobs": 0,
            "logs": 0,
            "events": 0,
            "workers": 0,
            "artifacts": 0,
            "artifact_bytes": 0,
        }
        with self.store.tx() as c:
            for chunk in _chunks(old_jobs, 500):
                q = ",".join("?" * len(chunk))
                report["logs"] += c.execute(
                    f"SELECT COUNT(*) FROM logs WHERE job_id IN ({q})", chunk
                ).fetchone()[0]
                report["events"] += c.execute(
                    f"SELECT COUNT(*) FROM events WHERE job_id IN ({q})", chunk
                ).fetchone()[0]
                if not dry_run:
                    c.execute(f"DELETE FROM logs WHERE job_id IN ({q})", chunk)
                    c.execute(f"DELETE FROM events WHERE job_id IN ({q})", chunk)
                    if delete_jobs:
                        c.execute(f"DELETE FROM tasks WHERE job_id IN ({q})", chunk)
                        c.execute(f"DELETE FROM costs WHERE job_id IN ({q})", chunk)
                        c.execute(
                            f"DELETE FROM instances WHERE job_id IN ({q}) AND state IN (?,?)",
                            (*chunk, "terminated", "failed"),
                        )
                        c.execute(f"DELETE FROM jobs WHERE id IN ({q})", chunk)
            if delete_jobs:
                report["jobs"] = len(old_jobs)
            stale_workers = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM workers WHERE (state=? OR revoked=1) AND "
                    "COALESCE(last_heartbeat_at, registered_at) < ? AND id NOT IN "
                    "(SELECT worker_id FROM tasks WHERE state IN (?,?) AND worker_id IS NOT NULL)",
                    (WorkerState.OFFLINE.value, ts(cutoff), *ACTIVE_TASK_STATES),
                ).fetchall()
            ]
            report["workers"] = len(stale_workers)
            if not dry_run:
                for chunk in _chunks(stale_workers, 500):
                    c.execute(
                        f"DELETE FROM workers WHERE id IN ({','.join('?' * len(chunk))})", chunk
                    )
        removed_jobs = set(old_jobs) if delete_jobs and not dry_run else set()
        keep = self._referenced_digests(exclude=set(old_jobs) if delete_jobs else set())
        for art in list(self.artifacts.list()):
            if art.sha256 in keep or art.stored_at >= cutoff:
                continue
            report["artifacts"] += 1
            report["artifact_bytes"] += art.size_bytes
            if not dry_run:
                self.artifacts.delete(art.sha256)
        self.store.add_event(
            "maintenance.gc",
            ("would remove " if dry_run else "removed ")
            + ", ".join(f"{v} {k}" for k, v in report.items() if v),
            data={
                "older_than_seconds": older_than_seconds,
                "dry_run": dry_run,
                "deleted_jobs": len(removed_jobs),
            },
        )
        return report

    def _referenced_digests(self, exclude: set[str]) -> set[str]:
        keep: set[str] = set()
        for r in self.store.query("SELECT id, spec, result FROM jobs"):
            if r["id"] in exclude:
                continue
            spec = JobSpec.model_validate_json(r["spec"])
            keep.update(ref.sha256 for ref in self._referenced_artifacts(spec))
            if r["result"]:
                keep.update(o.sha256 for o in JobResult.model_validate_json(r["result"]).outputs)
        for r in self.store.query("SELECT outcome FROM tasks WHERE outcome IS NOT NULL"):
            keep.update(o.sha256 for o in TaskOutcome.model_validate_json(r["outcome"]).outputs)
        return keep

    def worker_can_read_artifact(self, worker_id: str, sha256: str) -> bool:
        """Workers may only fetch sources and inputs of tasks currently assigned to them."""
        for r in self.store.query(
            "SELECT j.spec FROM tasks t JOIN jobs j ON j.id=t.job_id "
            "WHERE t.worker_id=? AND t.state IN (?,?)",
            (worker_id, *ACTIVE_TASK_STATES),
        ):
            spec = JobSpec.model_validate_json(r["spec"])
            if any(ref.sha256 == sha256 for ref in self._referenced_artifacts(spec)):
                return True
        return False

    def get_artifact(self, sha256: str) -> bytes:
        return self.artifacts.get(sha256)

    def add_cost(self, job_id: str, record: CostRecord) -> None:
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO costs (job_id, record) VALUES (?,?)",
                (job_id, record.model_dump_json()),
            )

    def _costs(self, job_id: str) -> list[CostRecord]:
        return [
            CostRecord.model_validate_json(r["record"])
            for r in self.store.query(
                "SELECT record FROM costs WHERE job_id=? ORDER BY id", (job_id,)
            )
        ]


def _chunks(items: list[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(0, len(items), n):
        yield tuple(items[i : i + n])
