"""Control-plane lifecycle driven through the worker protocol by SIMULATED workers
(no processes are launched). Covers leases, heartbeats, cancellation, retries and
distributed coordination semantics."""

import pytest
from conftest import TORCH_CAPS, make_caps

from gpulayer.contracts import (
    CommandJob,
    DistributedJob,
    JobSpec,
    JobState,
    PlacementPolicy,
    ResourceRequest,
    RetryPolicy,
    TaskOutcome,
    TaskState,
    WorkerState,
)
from gpulayer.errors import Conflict, Forbidden, ValidationFailed
from gpulayer.runtime.distributed import direct_argv, direct_env, plan_group, torchrun_argv

OK = TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0)


def cmd_spec(**kw):
    return JobSpec(workload=CommandJob(command=["x"]), **kw)


def dist_spec(n=2, src=None):
    return JobSpec(workload=DistributedJob(entrypoint="t.py", num_workers=n, source=src))


def reg(cp, name="w", **caps):
    return cp.register_worker(name, make_caps(**caps)).worker_id


def test_submit_place_run_succeed(cp):
    w = reg(cp)
    job = cp.submit(cmd_spec())
    assert job.state == JobState.SCHEDULED and job.scheduling.placed
    a = cp.acquire_task(w)
    assert a.job_id == job.job_id and cp.acquire_task(w).task_id == a.task_id  # idempotent
    cp.start_task(w, a.task_id)
    cp.start_task(w, a.task_id)  # idempotent
    assert cp.get_job(job.job_id).state == JobState.RUNNING
    cp.finish_task(
        w, a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0, return_value={"x": 1})
    )
    res = cp.job_result(job.job_id)
    assert res.state == JobState.SUCCEEDED and res.return_value == {"x": 1}
    types = [e.type for e in cp.events(job_id=job.job_id)]
    assert types == [
        "job.submitted",
        "job.scheduled",
        "task.running",
        "job.running",
        "task.succeeded",
        "job.succeeded",
    ]


def test_queued_until_compatible_worker_arrives(cp):
    job = cp.submit(cmd_spec())
    assert job.state == JobState.QUEUED and job.scheduling.summary == "no workers online"
    reg(cp)
    assert cp.get_job(job.job_id).state == JobState.SCHEDULED


def test_unschedulable_fail_policy(cp):
    reg(cp)
    job = cp.submit(
        cmd_spec(
            resources=ResourceRequest(accelerator="cuda"),
            placement=PlacementPolicy(on_unschedulable="fail"),
        )
    )
    assert job.state == JobState.FAILED and "unschedulable" in job.error


def test_missing_artifact_rejected(cp):
    from gpulayer.artifacts.store import ref_for

    with pytest.raises(ValidationFailed, match="not been uploaded"):
        cp.submit(dist_spec(src=ref_for(b"nope")))


def test_task_ownership_enforced(cp):
    w1, w2 = reg(cp, "a"), reg(cp, "b")
    job = cp.submit(cmd_spec())
    owner = job.tasks[0].worker_id
    other = w2 if owner == w1 else w1
    a = cp.acquire_task(owner)
    assert cp.acquire_task(other) is None
    with pytest.raises(Forbidden):
        cp.start_task(other, a.task_id)
    with pytest.raises(Forbidden):
        cp.finish_task(other, a.task_id, OK)


def test_failed_task_fails_job(cp):
    w = reg(cp)
    job = cp.submit(cmd_spec())
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    cp.finish_task(
        w, a.task_id, TaskOutcome(state=TaskState.FAILED, exit_code=3, error="exit code 3")
    )
    st = cp.get_job(job.job_id)
    assert st.state == JobState.FAILED and "exit code 3" in st.error
    assert cp.job_result(job.job_id).exit_code == 3


def test_cancel_before_start_and_late_start(cp):
    w = reg(cp)
    job = cp.submit(cmd_spec())
    a = cp.acquire_task(w)
    assert cp.cancel_job(job.job_id).state == JobState.CANCELLED
    with pytest.raises(Conflict):
        cp.start_task(w, a.task_id)
    assert cp.cancel_job(job.job_id).state == JobState.CANCELLED  # idempotent


def test_cancel_running_is_delivered_by_heartbeat(cp):
    w = reg(cp)
    job = cp.submit(cmd_spec())
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    assert cp.cancel_job(job.job_id).state == JobState.RUNNING
    assert cp.heartbeat(w).cancel_task_ids == [a.task_id]
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.CANCELLED, error="cancelled"))
    assert cp.get_job(job.job_id).state == JobState.CANCELLED


def test_cancel_queued_job(cp):
    job = cp.submit(cmd_spec())
    assert cp.cancel_job(job.job_id).state == JobState.CANCELLED


def test_lease_expiry_marks_worker_offline_and_task_lost(cp, clock):
    w = reg(cp)
    job = cp.submit(cmd_spec())
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    clock.advance(5)
    cp.heartbeat(w)  # renews the lease
    clock.advance(8)
    assert cp.get_job(job.job_id).state == JobState.RUNNING
    clock.advance(30)
    st = cp.get_job(job.job_id)
    assert st.state == JobState.FAILED and "lease expired" in st.error
    assert cp.list_workers()[0].state == WorkerState.OFFLINE
    # The worker comes back: it is told to kill the lost task; its late report is ignored.
    assert a.task_id in cp.heartbeat(w).cancel_task_ids
    cp.finish_task(w, a.task_id, OK)
    assert cp.get_job(job.job_id).state == JobState.FAILED
    assert "task.late_report" in [e.type for e in cp.events(job_id=job.job_id)]


def test_retry_on_worker_loss_is_opt_in(cp, clock):
    w1 = reg(cp, "a")
    job = cp.submit(cmd_spec(retry=RetryPolicy(max_attempts=2)))
    a = cp.acquire_task(w1)
    cp.start_task(w1, a.task_id)
    clock.advance(30)
    w2 = reg(cp, "b")
    st = cp.get_job(job.job_id)
    assert st.state == JobState.SCHEDULED and st.attempts == 2
    b = cp.acquire_task(w2)
    assert b.attempt == 2
    cp.start_task(w2, b.task_id)
    cp.finish_task(w2, b.task_id, OK)
    assert cp.get_job(job.job_id).state == JobState.SUCCEEDED
    assert any("side effects may repeat" in e.message for e in cp.events(job_id=job.job_id))


def test_deadline_backstop(cp, clock):
    w = reg(cp)
    job = cp.submit(cmd_spec(timeout_seconds=5))
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    for _ in range(8):  # keep the lease alive, but never report
        clock.advance(5)
        cp.heartbeat(w)
    assert cp.get_job(job.job_id).state == JobState.TIMED_OUT


def test_revoked_worker_cannot_act(cp):
    w = reg(cp)
    cp.revoke_worker(w)
    assert cp.heartbeat(w).revoked
    with pytest.raises(Forbidden):
        cp.acquire_task(w)


def test_state_is_persisted_across_restart(tmp_path, cp):
    from gpulayer.artifacts.store import LocalArtifactStore
    from gpulayer.control_plane.service import ControlPlane
    from gpulayer.storage.sqlite import SqliteStore

    w = reg(cp)
    job = cp.submit(cmd_spec())
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    cp.finish_task(w, a.task_id, OK)
    cp2 = ControlPlane(
        SqliteStore(tmp_path / "db.sqlite"), LocalArtifactStore(tmp_path / "a"), clock=cp.now
    )
    assert cp2.job_result(job.job_id).state == JobState.SUCCEEDED


# ------------------------------------------------------------------- distributed


def two_node(cp):
    """Returns ((rank0_worker, rank1_worker), job)."""
    src = cp.put_artifact(b"bundle")
    for i in range(2):
        reg(cp, f"n{i}", **TORCH_CAPS)
    job = cp.submit(dist_spec(2, src))
    by_rank = {t.node_rank: t.worker_id for t in job.tasks}
    return (by_rank[0], by_rank[1]), job


def test_rank_plan_and_launch_args():
    job = DistributedJob(entrypoint="t.py", num_workers=2, procs_per_worker=4)
    plan = plan_group(job, ["a", "b"])
    assert plan.world_size == 8 and plan.nodes[1].global_ranks == (4, 5, 6, 7)
    with pytest.raises(ValueError):
        plan_group(job, ["a", "a"])
    with pytest.raises(ValueError):
        plan_group(job, ["a"])
    from gpulayer.contracts import RendezvousInfo

    rdzv = RendezvousInfo(
        master_addr="10.0.0.1", master_port=29500, num_nodes=2, procs_per_node=4, node_rank=1
    )
    argv = torchrun_argv("py", rdzv, "/w/t.py", ["--lr", "1"])
    assert "--node-rank=1" in argv and "--master-addr=10.0.0.1" in argv
    assert "--max-restarts=0" in argv and argv[-3:] == ["/w/t.py", "--lr", "1"]

    assert job.launcher == "direct"  # the default
    argv = direct_argv("py", "/w/rank_launcher.py", rdzv, "/w/t.py", ["--lr", "1"])
    assert argv == ["py", "/w/rank_launcher.py", "4", "4", "py", "/w/t.py", "--lr", "1"]
    env = direct_env("job-1", rdzv)
    assert env["MASTER_ADDR"] == "10.0.0.1" and env["MASTER_PORT"] == "29500"
    assert env["WORLD_SIZE"] == "8" and env["LOCAL_WORLD_SIZE"] == "4"
    assert env["GROUP_RANK"] == "1" and env["TORCHELASTIC_RUN_ID"] == "job-1"
    # Rank 0 must host the TCPStore itself; with this set it would wait for an agent's store.
    assert "TORCHELASTIC_USE_AGENT_STORE" not in env


def test_distributed_rendezvous_gates_non_zero_ranks(cp):
    (w0, w1), job = two_node(cp)
    assert cp.get_job(job.job_id).state == JobState.SCHEDULED
    assert cp.acquire_task(w1) is None  # rank 1 waits for rank 0's endpoint
    a0 = cp.acquire_task(w0)
    assert a0.node_rank == 0 and a0.rendezvous is None
    with pytest.raises(Forbidden):
        cp.publish_rendezvous(w1, a0.task_id, "x", 1)
    info = cp.publish_rendezvous(w0, a0.task_id, "127.0.0.1", 29511)
    assert info.world_size == 2
    a1 = cp.acquire_task(w1)
    assert a1.node_rank == 1 and a1.rendezvous.master_port == 29511
    assert a1.rendezvous.node_rank == 1
    for w, a in ((w0, a0), (w1, a1)):
        cp.start_task(w, a.task_id)
    cp.finish_task(w0, a0.task_id, OK)
    assert cp.get_job(job.job_id).state == JobState.RUNNING  # waits for every node
    cp.finish_task(w1, a1.task_id, OK)
    assert cp.get_job(job.job_id).state == JobState.SUCCEEDED


def test_distributed_fail_fast_cancels_siblings(cp):
    (w0, w1), job = two_node(cp)
    a0 = cp.acquire_task(w0)
    cp.publish_rendezvous(w0, a0.task_id, "127.0.0.1", 29512)
    a1 = cp.acquire_task(w1)
    cp.start_task(w0, a0.task_id)
    cp.start_task(w1, a1.task_id)
    cp.finish_task(
        w1, a1.task_id, TaskOutcome(state=TaskState.FAILED, exit_code=1, error="exit code 1")
    )
    st = cp.get_job(job.job_id)
    assert st.state == JobState.FAILED and "node rank 1" in st.error
    assert cp.heartbeat(w0).cancel_task_ids == [a0.task_id]


def test_distributed_worker_loss_fails_group_without_retry(cp, clock):
    (w0, w1), job = two_node(cp)
    a0 = cp.acquire_task(w0)
    cp.publish_rendezvous(w0, a0.task_id, "127.0.0.1", 29513)
    a1 = cp.acquire_task(w1)
    cp.start_task(w0, a0.task_id)
    cp.start_task(w1, a1.task_id)
    for _ in range(4):
        clock.advance(4)
        cp.heartbeat(w0)  # only node 0 stays alive
    st = cp.get_job(job.job_id)
    assert st.state == JobState.FAILED and st.attempts == 1


def test_full_cluster_submit_only_evaluates_new_jobs_and_keeps_fifo(cp):
    """With every slot busy, a submit gives the new job its first decision without
    re-evaluating the queue; when a slot frees, the oldest queued job is placed first."""
    w = reg(cp, max_concurrent_tasks=2)
    jobs = [cp.submit(cmd_spec()) for _ in range(6)]
    assert [cp.get_job(j.job_id).state for j in jobs[:2]] == [JobState.SCHEDULED] * 2
    waiting = cp.get_job(jobs[2].job_id).scheduling
    assert waiting is not None and not waiting.placed
    assert "no free task slot" in str(waiting.model_dump())
    late = cp.submit(cmd_spec())
    assert cp.get_job(late.job_id).scheduling is not None  # new jobs still get a reason
    # Queued jobs were not re-decided by the later submits.
    assert cp.get_job(jobs[2].job_id).scheduling.decided_at == waiting.decided_at
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    cp.finish_task(w, a.task_id, OK)
    cp.schedule_pending()
    assert cp.get_job(jobs[2].job_id).state == JobState.SCHEDULED
    assert all(cp.get_job(j.job_id).state == JobState.QUEUED for j in jobs[3:])


def test_served_control_plane_waiters_sleep_until_signalled(cp, monkeypatch):
    """Behind a server, an idle long-polling worker does no scheduling work while it waits,
    and wakes as soon as a job is submitted."""
    import threading
    import time as _time

    w = reg(cp)
    cp.served = True
    passes = []
    original = cp.schedule_pending
    monkeypatch.setattr(cp, "schedule_pending", lambda: (passes.append(1), original())[1])
    got = []
    th = threading.Thread(target=lambda: got.append(cp.acquire_task(w, wait=5)))
    th.start()
    _time.sleep(0.6)  # three POLL_SLICEs: an embedded control plane would have re-checked
    assert passes == []
    t0 = _time.monotonic()
    cp.submit(cmd_spec())  # schedules once and signals the waiter
    th.join(5)
    assert got and got[0] is not None and _time.monotonic() - t0 < 1.0
    assert len(passes) == 1


def test_placement_wakes_only_the_chosen_worker(cp, monkeypatch):
    """A served control plane with many idle long-polling workers: assigning a task wakes
    that worker's acquire, not every waiter (no thundering herd)."""
    import threading
    import time as _time

    cp.served = True
    workers = [reg(cp, f"w{i}") for i in range(10)]
    checks = []
    original = cp._acquire_once
    monkeypatch.setattr(cp, "_acquire_once", lambda w: (checks.append(w), original(w))[1])
    got = {}
    threads = [
        threading.Thread(target=lambda w=w: got.setdefault(w, cp.acquire_task(w, wait=3)))
        for w in workers
    ]
    for t in threads:
        t.start()
    _time.sleep(0.3)
    assert sorted(checks) == sorted(workers)  # each checked once, then sleeps
    checks.clear()
    cp.submit(cmd_spec())
    _time.sleep(0.3)
    assert len(checks) == 1  # only the worker the job was placed on woke up
    assert got[checks[0]] is not None
    for t in threads:
        t.join(5)
