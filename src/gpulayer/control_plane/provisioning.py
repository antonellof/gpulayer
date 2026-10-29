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
"""

from __future__ import annotations

import hashlib
import json
import secrets
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


class ProvisioningManager:
    def __init__(self, cp: ControlPlane, providers: dict[str, ProviderAdapter]) -> None:
        self.cp = cp
        self.providers = providers

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

    def request(self, job_id: str, spec: JobSpec, decision: SchedulingDecision) -> None:
        policy = spec.placement.provisioning
        assert policy is not None
        adapter = self.providers[policy.provider]
        offer, rejected = self.select_offer(spec, adapter)
        if offer is None:
            self.cp._set_job_state(
                job_id,
                JobState.FAILED,
                error=f"no compatible offer from provider '{adapter.name}': "
                + (" | ".join(rejected) or "provider returned no offers"),
            )
            return
        deadline = self.cp.now() + timedelta(seconds=policy.provisioning_timeout_seconds)
        with self.cp.store.tx() as c:
            c.execute("UPDATE jobs SET provisioning_deadline=? WHERE id=?", (ts(deadline), job_id))
            self.cp._set_job_state(
                job_id,
                JobState.PROVISIONING,
                message=f"{decision.summary}; provisioning {spec.num_nodes} x "
                f"{offer.instance_type} from {adapter.name}"
                + (" (SIMULATED provider)" if adapter.simulated else ""),
            )
        if offer.price is not None:
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
        for _ in range(spec.num_nodes):
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
        with self.cp.store.tx() as c:
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
            instance_ref=ref,
            join_token=token,
            labels={RESERVED_LABEL: job_id},
        )
        try:
            inst = adapter.provision(offer.offer_id, bootstrap)
        except ProviderError as e:
            self._update(ref, InstanceState.FAILED, error=str(e))
            self.cp._set_job_state(
                job_id, JobState.FAILED, error=f"provisioning failed ({type(e).__name__}): {e}"
            )
            return False
        self._update(ref, inst.state, provider_instance_id=inst.provider_instance_id)
        self.cp.store.add_event(
            "instance.provisioned",
            f"{adapter.name} instance {inst.provider_instance_id} ({inst.state.value})",
            job_id=job_id,
        )
        return True

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
        now = self.cp.now()
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
        rows = self.cp.store.query(
            "SELECT * FROM instances WHERE job_id=? AND release='terminate_on_completion' "
            "AND state NOT IN (?,?)",
            (job_id, InstanceState.TERMINATED.value, InstanceState.FAILED.value),
        )
        for row in rows:
            adapter = self.providers.get(row["provider"])
            pid = json.loads(row["offer"]).get("provider_instance_id")
            if adapter and pid:
                try:
                    adapter.terminate(pid)
                except ProviderError as e:
                    self.cp.store.add_event(
                        "instance.terminate_failed",
                        f"could not terminate {pid}: {e} -- may be orphaned",
                        job_id=job_id,
                    )
                    continue
            self._update(row["id"], InstanceState.TERMINATED)
            for w in self.cp.store.query("SELECT id, capabilities FROM workers WHERE revoked=0"):
                if (
                    WorkerCapabilities.model_validate_json(w["capabilities"]).instance_id
                    == row["id"]
                ):
                    with self.cp.store.tx() as c:
                        c.execute(
                            "UPDATE workers SET revoked=1, state='offline' WHERE id=?", (w["id"],)
                        )
            self.cp.store.add_event(
                "instance.terminated", f"released instance {pid}", job_id=job_id
            )
