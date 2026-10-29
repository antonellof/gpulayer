"""FakeProvider and the provisioning lifecycle. SIMULATED: no real infrastructure."""

import pytest
from conftest import cuda_device, make_caps

from gpulayer.contracts import (
    CommandJob,
    JobSpec,
    JobState,
    PlacementPolicy,
    ProvisioningPolicy,
    ResourceRequest,
    TaskOutcome,
    TaskState,
    WorkerOrigin,
)
from gpulayer.control_plane.scheduler import RESERVED_LABEL
from gpulayer.errors import Unauthorized, ValidationFailed
from gpulayer.providers.base import (
    CapacityUnavailable,
    InstanceOffer,
    InstanceState,
    PriceQuote,
    ProviderError,
    QuotaExceeded,
)
from gpulayer.providers.fake import FakeProvider


def offer(oid, price=None, gpus=1, mem=24.0):
    caps = make_caps(
        origin=WorkerOrigin.PROVIDER, accelerators=[cuda_device(i, mem=mem) for i in range(gpus)]
    )
    return InstanceOffer(
        offer_id=oid,
        provider="fake",
        instance_type=f"type-{oid}",
        region="test-1",
        capabilities=caps,
        price=PriceQuote(amount=price, currency="USD", source="fake-provider (simulated)")
        if price is not None
        else None,
    )


def cloud_spec(**placement):
    return JobSpec(
        workload=CommandJob(command=["x"]),
        resources=ResourceRequest(accelerator="cuda"),
        placement=PlacementPolicy(
            target="cloud", provisioning=ProvisioningPolicy(provider="fake"), **placement
        ),
    )


def make_cp(cp, provider):
    from gpulayer.control_plane.provisioning import ProvisioningManager

    cp.provisioning = ProvisioningManager(cp, {provider.name: provider})
    return cp


def test_fake_provider_lifecycle():
    p = FakeProvider([offer("a", 1.0)], capacity=1)
    from gpulayer.providers.base import WorkerBootstrap

    b = WorkerBootstrap(control_plane_url=None, instance_ref="r", join_token="t")
    inst = p.provision("a", b)
    assert inst.state == InstanceState.RUNNING
    with pytest.raises(CapacityUnavailable):
        p.provision("a", b)
    assert p.terminate(inst.provider_instance_id).state == InstanceState.TERMINATED
    assert p.terminate(inst.provider_instance_id).state == InstanceState.TERMINATED  # idempotent
    assert "token" not in repr(b)


def test_unconfigured_provider_rejected(cp):
    with pytest.raises(ValidationFailed, match="not configured"):
        cp.submit(cloud_spec())


def test_offer_selection_cheapest_compatible(cp):
    p = FakeProvider(
        [offer("cpu-only", 0.1, gpus=0), offer("big", 3.0), offer("small", 1.0), offer("unpriced")]
    )
    make_cp(cp, p)
    chosen, rejected = cp.provisioning.select_offer(cloud_spec(), p)
    assert chosen.offer_id == "small"
    assert any(r.startswith("cpu-only") for r in rejected)


def test_price_cap_rejects_unknown_price(cp):
    p = FakeProvider([offer("unpriced"), offer("pricey", 5.0)])
    make_cp(cp, p)
    chosen, rejected = cp.provisioning.select_offer(cloud_spec(max_hourly_cost=2.0), p)
    assert chosen is None
    assert any("no reliable price" in r for r in rejected)


def test_end_to_end_simulated_cloud_job(cp):
    booted = []
    p = FakeProvider([offer("small", 1.25)], on_provision=lambda i, b: booted.append(b))
    make_cp(cp, p)
    job = cp.submit(cloud_spec())
    assert job.state == JobState.PROVISIONING
    assert len(booted) == 1 and booted[0].labels[RESERVED_LABEL] == job.job_id

    # Act as the machine's boot script: register with the single-use instance token.
    b = booted[0]
    caps = make_caps(
        origin=WorkerOrigin.PROVIDER, instance_id=b.instance_ref, accelerators=[cuda_device()]
    )
    with pytest.raises(Unauthorized):
        cp.register_worker("x", caps, instance_token="wrong")
    w = cp.register_worker("cloud-1", caps, instance_token=b.join_token).worker_id
    with pytest.raises(Unauthorized):  # single use
        cp.register_worker("x", caps, instance_token=b.join_token)

    # A different job cannot steal the reserved worker.
    other = cp.submit(
        JobSpec(
            workload=CommandJob(command=["y"]),
            resources=ResourceRequest(accelerator="cuda"),
            placement=PlacementPolicy(target="cloud"),
        )
    )
    assert other.state == JobState.QUEUED

    a = cp.acquire_task(w)
    assert a.job_id == job.job_id
    cp.start_task(w, a.task_id)
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0))
    res = cp.job_result(job.job_id)
    assert res.state == JobState.SUCCEEDED
    assert [c.kind for c in res.costs] == ["estimate"] and res.costs[0].amount == 1.25
    assert p.active_instances() == []  # released on completion
    assert cp.heartbeat(w).revoked


def test_worker_reports_less_than_offer_promised(cp):
    booted = []
    p = FakeProvider([offer("gpu", 1.0)], on_provision=lambda i, b: booted.append(b))
    make_cp(cp, p)
    job = cp.submit(cloud_spec())
    caps = make_caps(origin=WorkerOrigin.PROVIDER, instance_id=booted[0].instance_ref)  # no GPU
    cp.register_worker("broken", caps, instance_token=booted[0].join_token)
    assert cp.get_job(job.job_id).state == JobState.PROVISIONING  # never placed on it


def test_non_retryable_provider_error_fails_job(cp):
    p = FakeProvider([offer("a", 1.0)], fail_provision=QuotaExceeded("quota is 0"))
    make_cp(cp, p)
    st = cp.submit(cloud_spec())
    assert st.state == JobState.FAILED and "quota is 0" in st.error


def test_capacity_errors_are_retried_until_capacity_appears(cp, clock):
    p = FakeProvider([offer("a", 1.0)], fail_provision=CapacityUnavailable("sold out"))
    make_cp(cp, p)
    job = cp.submit(cloud_spec())
    assert job.state == JobState.PROVISIONING
    assert any(e.type == "instance.retry" for e in cp.events(job_id=job.job_id))
    calls = len(p.calls)
    cp.reconcile()
    assert len(p.calls) == calls  # backing off, no new provider call
    p.fail_provision = None
    clock.advance(cp.config.provisioning_retry_seconds + 1)
    cp.reconcile()
    assert len(p.active_instances()) == 1
    assert cp.get_job(job.job_id).state == JobState.PROVISIONING


def test_provisioning_deadline(cp, clock):
    p = FakeProvider([offer("a", 1.0)])
    make_cp(cp, p)
    job = cp.submit(cloud_spec())
    clock.advance(1000)
    assert cp.get_job(job.job_id).state == JobState.FAILED
    assert p.active_instances() == []


def test_failed_terminate_is_retried(cp):
    booted = []
    p = FakeProvider([offer("a", 1.0)], on_provision=lambda i, b: booted.append(b))
    make_cp(cp, p)
    job = cp.submit(cloud_spec())
    real_terminate = p.terminate
    failures = {"left": 2}

    def flaky_terminate(pid):
        if failures["left"]:
            failures["left"] -= 1
            raise ProviderError("API timeout")
        return real_terminate(pid)

    p.terminate = flaky_terminate
    cp.cancel_job(job.job_id)
    assert len(p.active_instances()) == 1  # first attempt failed
    cp.reconcile()
    cp.reconcile()
    assert p.active_instances() == []
    types = [e.type for e in cp.events(job_id=job.job_id)]
    assert types.count("instance.terminate_failed") == 2 and "instance.terminated" in types


def test_orphan_sweep_releases_instances_of_finished_jobs(cp):
    p = FakeProvider([offer("a", 1.0)])
    make_cp(cp, p)
    job = cp.submit(cloud_spec())
    # Simulate a crash after the job finished but before release was recorded.
    with cp.store.tx() as c:
        c.execute("UPDATE jobs SET state='failed' WHERE id=?", (job.job_id,))
    assert len(p.active_instances()) == 1
    cp.reconcile()
    assert p.active_instances() == []
    assert any(e.type == "instance.orphans" for e in cp.events())


def test_provider_calls_do_not_hold_the_metadata_lock(cp):
    import threading
    import time

    started = threading.Event()

    def slow_boot(inst, bootstrap):
        started.set()
        time.sleep(1.0)

    p = FakeProvider([offer("a", 1.0)], on_provision=slow_boot)
    make_cp(cp, p)
    t = threading.Thread(target=cp.submit, args=(cloud_spec(),))
    t.start()
    assert started.wait(5)
    t0 = time.monotonic()
    cp.list_workers()  # needs the store; must not wait for the provider
    cp.store.add_event("probe", "write while provisioning")
    assert time.monotonic() - t0 < 0.5
    t.join()


def test_provider_worker_needs_instance_token(cp):
    with pytest.raises(Unauthorized):
        cp.register_worker("x", make_caps(origin=WorkerOrigin.PROVIDER, instance_id="inst-1"))
