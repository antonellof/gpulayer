"""Provider-backed capacity: offer selection, provisioning, bootstrap trust, release.

Flow for a job with ``placement.provisioning`` that no registered worker can take:

1. filter provider offers with the SAME compatibility check the scheduler uses;
2. enforce ``max_hourly_cost`` (an offer without a reliable price is rejected then);
3. provision ``num_nodes`` instances, each with a single-use join token bound to it;
4. the job waits in ``provisioning``; the booted worker registers with that token and
   is labelled reserved-for the job, so other jobs cannot take it;
5. the scheduler places the job only after the worker *reports* compatible capabilities
   (provisioning success is not proof of GPU readiness);
6. on a terminal job state the instances are terminated (``release=terminate_on_completion``).

Provider API calls never run inside a metadata transaction. Inside transactions the control
plane only records intent (job waiting for capacity, instance requested, instance to release);
``pump()`` then calls the provider with no lock held and records each outcome in a short
transaction. ``pump()`` also retries retryable capacity errors until the provisioning
deadline, retries failed terminations, and sweeps instances whose job has already finished.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from gpulayer.contracts import (
    CostRecord,
    ExecutionTarget,
    JobSpec,
    JobState,
    SchedulingDecision,
    WorkerCapabilities,
)
from gpulayer.control_plane.scheduler import RESERVED_LABEL, compatibility_reasons
from gpulayer.errors import Forbidden, Unauthorized
from gpulayer.providers.base import (
    InstanceOffer,
    InstanceState,
    ProviderAdapter,
    ProviderError,
    WorkerBootstrap,
)
from gpulayer.storage.sqlite import parse_ts, ts

if TYPE_CHECKING:
    from gpulayer.control_plane.service import ControlPlane

_LIVE = (
    InstanceState.REQUESTED.value,
    InstanceState.PROVISIONING.value,
    InstanceState.RUNNING.value,
)


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# An intent older than this with no provider id means the control plane stopped between
# recording the intent and hearing back from the provider: the instance may exist.
STALE_INTENT_SECONDS = 600


class ProvisioningManager:
    def __init__(self, cp: ControlPlane, providers: dict[str, ProviderAdapter]) -> None:
        self.cp = cp
        self.providers = providers
        self._pump_lock = threading.Lock()

    def has(self, name: str) -> bool:
        return name in self.providers

    # --------------------------------------------------------------- selection

    def select_offer(
        self, spec: JobSpec, adapter: ProviderAdapter
    ) -> tuple[InstanceOffer | None, list[str]]:
        p = spec.placement
        rejected: list[str] = []
        ok: list[InstanceOffer] = []
        for offer in adapter.list_offers():
            caps = offer.capabilities.model_copy(
                update={"provider": adapter.name, "region": offer.region}
            )
            reasons = compatibility_reasons(spec, caps, target=ExecutionTarget.CLOUD)
            if p.max_hourly_cost is not None:
                if offer.price is None:
                    reasons.append("no reliable price; max_hourly_cost cannot be enforced")
                elif offer.price.currency != p.currency:
                    reasons.append(f"price in {offer.price.currency}, policy in {p.currency}")
                elif offer.price.amount * spec.num_nodes > p.max_hourly_cost:
                    reasons.append(
                        f"estimated {offer.price.amount * spec.num_nodes:g} "
                        f"{p.currency}/h exceeds {p.max_hourly_cost:g}"
                    )
            if reasons:
                rejected.append(f"{offer.offer_id}: {'; '.join(reasons)}")
            else:
                ok.append(offer)
        # Deterministic: cheapest known price first, unknown price last, then offer id.
        ok.sort(key=lambda o: (o.price is None, o.price.amount if o.price else 0.0, o.offer_id))
        return (ok[0] if ok else None), rejected

    # ------------------------------------------------------------ provisioning

    def mark(self, job_id: str, spec: JobSpec, decision: SchedulingDecision) -> None:
        """Database only (safe inside a transaction): the job now waits for capacity."""
        policy = spec.placement.provisioning
        assert policy is not None
        deadline = self.cp.now() + timedelta(seconds=policy.provisioning_timeout_seconds)
        simulated = " (SIMULATED provider)" if self.providers[policy.provider].simulated else ""
        with self.cp.store.tx() as c:
            c.execute("UPDATE jobs SET provisioning_deadline=? WHERE id=?", (ts(deadline), job_id))
            self.cp._set_job_state(
                job_id,
                JobState.PROVISIONING,
                message=f"{decision.summary}; requesting {spec.num_nodes} instance(s) from "
                f"{policy.provider}{simulated}",
            )

    def _provision_missing(self, job_id: str, spec: JobSpec) -> None:
        """Provider calls happen here, outside any transaction."""
        policy = spec.placement.provisioning
        assert policy is not None
        adapter = self.providers[policy.provider]
        now = self.cp.now()
        rows = self.cp.store.query("SELECT state, offer FROM instances WHERE job_id=?", (job_id,))
        live = [r for r in rows if r["state"] in _LIVE]
        missing = spec.num_nodes - len(live)
        if missing <= 0:
            return
        waits = [
            parse_ts(json.loads(r["offer"]).get("retry_after"))
            for r in rows
            if r["state"] == InstanceState.FAILED.value
        ]
        if any(w and w > now for w in waits):
            return  # backing off after a retryable capacity error
        try:
            offer, rejected = self.select_offer(spec, adapter)
        except ProviderError as e:
            self._capacity_error(job_id, e, adapter)
            return
        if offer is None:
            self._fail_if_waiting(
                job_id,
                f"no compatible offer from provider '{adapter.name}': "
                + (" | ".join(rejected) or "provider returned no offers"),
            )
            return
        if not rows and offer.price is not None:
            self.cp.add_cost(
                job_id,
                CostRecord(
                    kind="estimate",
                    amount=offer.price.amount * spec.num_nodes,
                    currency=offer.price.currency,
                    unit="per_hour",
                    source=offer.price.source,
                    note=f"{spec.num_nodes} x {offer.instance_type}; estimate, not a charge",
                ),
            )
        for _ in range(missing):
            if not self._provision_one(job_id, adapter, offer, policy.release):
                return

    def _provision_one(
        self, job_id: str, adapter: ProviderAdapter, offer: InstanceOffer, release: str
    ) -> bool:
        ref = f"inst-{uuid.uuid4().hex[:12]}"
        token = secrets.token_urlsafe(32)
        now = ts(self.cp.now())
        record = {
            "offer": offer.model_dump(mode="json"),
            "token_hash": _h(token),
            "token_used": False,
            "provider_instance_id": None,
            "error": None,
        }
        with self.cp.store.tx() as c:  # 1. record the intent
            c.execute(
                "INSERT INTO instances (id, provider, job_id, offer, state, created_at, "
                "updated_at, release) VALUES (?,?,?,?,?,?,?,?)",
                (
                    ref,
                    adapter.name,
                    job_id,
                    json.dumps(record),
                    InstanceState.REQUESTED.value,
                    now,
                    now,
                    release,
                ),
            )
        bootstrap = WorkerBootstrap(
            control_plane_url=self.cp.config.public_url,
            tls_fingerprint=self.cp.config.tls_fingerprint,
            instance_ref=ref,
            join_token=token,
            labels={RESERVED_LABEL: job_id},
        )
        try:  # 2. call the provider, no lock held
            inst = adapter.provision(offer.offer_id, bootstrap)
        except ProviderError as e:  # 3. record the outcome
            retry_after = None
            if e.retryable:
                retry_after = ts(
                    self.cp.now() + timedelta(seconds=self.cp.config.provisioning_retry_seconds)
                )
            self._update(ref, InstanceState.FAILED, error=str(e), retry_after=retry_after)
            self._capacity_error(job_id, e, adapter)
            return False
        self._update(ref, inst.state, provider_instance_id=inst.provider_instance_id)
        self.cp.store.add_event(
            "instance.provisioned",
            f"{adapter.name} instance {inst.provider_instance_id} ({inst.state.value})",
            job_id=job_id,
        )
        return True

    def _capacity_error(self, job_id: str, e: ProviderError, adapter: ProviderAdapter) -> None:
        if e.retryable:
            self.cp.store.add_event(
                "instance.retry",
                f"{adapter.name}: {type(e).__name__}: {e}; retrying until the provisioning "
                "deadline",
                job_id=job_id,
            )
        else:
            self._fail_if_waiting(job_id, f"provisioning failed ({type(e).__name__}): {e}")

    def _update(self, ref: str, state: InstanceState, **fields: Any) -> None:
        with self.cp.store.tx() as c:
            row = c.execute("SELECT offer FROM instances WHERE id=?", (ref,)).fetchone()
            record = json.loads(row[0])
            record.update(fields)
            c.execute(
                "UPDATE instances SET state=?, offer=?, updated_at=? WHERE id=?",
                (state.value, json.dumps(record), ts(self.cp.now()), ref),
            )

    # ------------------------------------------------------------- bootstrap

    def verify_and_bind(self, caps: WorkerCapabilities, token: str | None) -> WorkerCapabilities:
        """Validate a provider worker's single-use instance token and pin its identity."""
        if not token or not caps.instance_id:
            raise Unauthorized("provider workers must register with their instance token")
        with self.cp.store.tx() as c:
            row = c.execute("SELECT * FROM instances WHERE id=?", (caps.instance_id,)).fetchone()
            if row is None:
                raise Unauthorized("unknown instance")
            record = json.loads(row["offer"])
            if record["token_used"] or not secrets.compare_digest(record["token_hash"], _h(token)):
                raise Unauthorized("invalid or already used instance token")
            if row["state"] not in _LIVE:
                raise Forbidden(f"instance is {row['state']}")
            record["token_used"] = True
            c.execute("UPDATE instances SET offer=? WHERE id=?", (json.dumps(record), row["id"]))
        labels = dict(caps.labels)
        if row["job_id"]:
            labels[RESERVED_LABEL] = row["job_id"]
        # Identity fields come from our record, not from the worker's claims.
        return caps.model_copy(
            update={
                "provider": row["provider"],
                "region": record["offer"].get("region"),
                "labels": labels,
            }
        )

    # ------------------------------------------------------------- lifecycle

    def reconcile(self) -> None:
        """Database only: fail jobs whose provisioning deadline passed."""
        now = self.cp.now()
        for job in self.cp.store.query(
            "SELECT id, provisioning_deadline FROM jobs WHERE state=?",
            (JobState.PROVISIONING.value,),
        ):
            deadline = parse_ts(job["provisioning_deadline"])
            if deadline and deadline < now:
                self._fail_if_waiting(
                    job["id"], "provisioning timed out before a compatible worker registered"
                )

    def _fail_if_waiting(self, job_id: str, error: str) -> None:
        state = JobState(self.cp._job_row(job_id)["state"])
        if state == JobState.PROVISIONING:
            self.cp._set_job_state(job_id, JobState.FAILED, error=error)

    def release_for_job(self, job_id: str) -> None:
        """Database only (called inside the job's final transaction): mark for release."""
        with self.cp.store.tx() as c:
            c.execute(
                "UPDATE instances SET state=?, updated_at=? WHERE job_id=? AND "
                "release='terminate_on_completion' AND state IN (?,?,?)",
                (InstanceState.TERMINATING.value, ts(self.cp.now()), job_id, *_LIVE),
            )

    # ------------------------------------------------------------------ pump

    def pump(self) -> None:
        """Make all pending provider calls. Must not run inside a transaction."""
        if not self.providers or self.cp.store.in_transaction:
            return
        if not self._pump_lock.acquire(blocking=False):
            return  # another thread is already pumping
        try:
            self._sweep_orphans()
            self._expire_stale_intents()
            for job in self.cp.store.query(
                "SELECT id, spec FROM jobs WHERE state=? ORDER BY created_at",
                (JobState.PROVISIONING.value,),
            ):
                spec = JobSpec.model_validate_json(job["spec"])
                if spec.placement.provisioning and self.has(spec.placement.provisioning.provider):
                    self._provision_missing(job["id"], spec)
            self._poll_states()
            self._terminate_pending()
        finally:
            self._pump_lock.release()

    def _sweep_orphans(self) -> None:
        """Live instances whose job already finished (e.g. after a crash): release them."""
        terminal = tuple(s.value for s in JobState if s.terminal)
        with self.cp.store.tx() as c:
            n = c.execute(
                "UPDATE instances SET state=? WHERE release='terminate_on_completion' AND "
                f"state IN (?,?,?) AND job_id IN (SELECT id FROM jobs WHERE state IN "
                f"({','.join('?' * len(terminal))}))",
                (InstanceState.TERMINATING.value, *_LIVE, *terminal),
            ).rowcount
            if n:
                self.cp.store.add_event("instance.orphans", f"{n} orphaned instance(s) to release")

    def _expire_stale_intents(self) -> None:
        cutoff = ts(self.cp.now() - timedelta(seconds=STALE_INTENT_SECONDS))
        for row in self.cp.store.query(
            "SELECT id, job_id, offer FROM instances WHERE state=? AND updated_at < ?",
            (InstanceState.REQUESTED.value, cutoff),
        ):
            if json.loads(row["offer"]).get("provider_instance_id") is None:
                self._update(row["id"], InstanceState.FAILED, error="provisioning outcome unknown")
                self.cp.store.add_event(
                    "instance.unknown",
                    f"instance {row['id']}: no answer recorded from the provider; check the "
                    "provider console for an orphaned machine",
                    job_id=row["job_id"],
                )

    def _poll_states(self) -> None:
        for row in self.cp.store.query(
            "SELECT * FROM instances WHERE state IN (?,?)",
            (InstanceState.REQUESTED.value, InstanceState.PROVISIONING.value),
        ):
            adapter = self.providers.get(row["provider"])
            pid = json.loads(row["offer"]).get("provider_instance_id")
            if adapter is None or pid is None:
                continue
            err: str | None
            try:
                inst = adapter.get_instance(pid)
            except ProviderError as e:
                inst_state, err = InstanceState.FAILED, str(e)
            else:
                inst_state, err = inst.state, inst.error
            if inst_state.value != row["state"]:
                self._update(row["id"], inst_state, error=err)
            if inst_state == InstanceState.FAILED and row["job_id"]:
                self._fail_if_waiting(row["job_id"], f"instance {pid} failed: {err}")

    def _terminate_pending(self) -> None:
        for row in self.cp.store.query(
            "SELECT * FROM instances WHERE state=?", (InstanceState.TERMINATING.value,)
        ):
            adapter = self.providers.get(row["provider"])
            record = json.loads(row["offer"])
            pid = record.get("provider_instance_id")
            if adapter is not None and pid:
                try:
                    adapter.terminate(pid)
                except ProviderError as e:
                    attempts = int(record.get("terminate_attempts", 0)) + 1
                    self._update(
                        row["id"],
                        InstanceState.TERMINATING,
                        terminate_attempts=attempts,
                        error=str(e),
                    )
                    self.cp.store.add_event(
                        "instance.terminate_failed",
                        f"could not terminate {pid} (attempt {attempts}): {e}; will retry",
                        job_id=row["job_id"],
                    )
                    continue
            self._update(row["id"], InstanceState.TERMINATED)
            with self.cp.store.tx() as c:
                for w in c.execute(
                    "SELECT id, capabilities FROM workers WHERE revoked=0"
                ).fetchall():
                    caps = WorkerCapabilities.model_validate_json(w["capabilities"])
                    if caps.instance_id == row["id"]:
                        c.execute(
                            "UPDATE workers SET revoked=1, state='offline' WHERE id=?", (w["id"],)
                        )
            self.cp.store.add_event(
                "instance.terminated",
                f"released instance {pid or row['id']}",
                job_id=row["job_id"],
            )
