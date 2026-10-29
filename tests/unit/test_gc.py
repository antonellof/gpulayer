"""Retention and garbage collection (ControlPlane.gc)."""

import os

from conftest import make_caps

from gpulayer.contracts import ArtifactKind, CommandJob, JobSpec, TaskOutcome, TaskState

DAY = 86400.0
OLD = 1_600_000_000  # 2020: older than any cutoff used below


def age(cp, ref):
    os.utime(cp.artifacts._path(ref.sha256), (OLD, OLD))


def finished_job(cp, source, output_data=b"result"):
    w = cp.register_worker("w", make_caps()).worker_id
    job = cp.submit(JobSpec(workload=CommandJob(command=["x"], source=source)))
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    cp.append_logs(w, a.task_id, [("stdout", "hello")] * 3)
    out = cp.put_artifact(output_data, "out.txt", ArtifactKind.OUTPUT)
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0, outputs=[out]))
    cp.deregister_worker(w)
    return job.job_id, out


def test_gc_removes_old_logs_and_unreferenced_artifacts(cp, clock):
    source = cp.put_artifact(b"bundle", "src", ArtifactKind.SOURCE)
    job_id, out = finished_job(cp, source)
    orphan = cp.put_artifact(b"uploaded and never used")
    fresh = cp.put_artifact(b"uploaded a moment ago, not submitted yet")
    for ref in (source, out, orphan):
        age(cp, ref)
    clock.advance(10 * DAY)

    dry = cp.gc(7 * DAY, dry_run=True)
    assert dry["logs"] == 3 and dry["artifacts"] == 1 and dry["workers"] == 1
    assert len(cp.job_logs(job_id)) == 3 and cp.artifacts.exists(orphan.sha256)

    report = cp.gc(7 * DAY)
    assert report["logs"] == 3 and report["events"] > 0 and report["jobs"] == 0
    assert cp.job_logs(job_id) == []
    assert cp.get_job(job_id).state.value == "succeeded"  # job record kept
    assert cp.job_result(job_id).outputs == [out]
    assert not cp.artifacts.exists(orphan.sha256)
    for kept in (source, out, fresh):  # still referenced, or too recent
        assert cp.artifacts.exists(kept.sha256)
    assert cp.list_workers() == []


def test_gc_delete_jobs_frees_their_artifacts(cp, clock):
    source = cp.put_artifact(b"bundle-2", "src", ArtifactKind.SOURCE)
    job_id, out = finished_job(cp, source, b"result-2")
    age(cp, source)
    age(cp, out)
    clock.advance(30 * DAY)
    report = cp.gc(7 * DAY, delete_jobs=True)
    assert report["jobs"] == 1 and report["artifacts"] == 2
    assert cp.list_jobs() == []
    assert not cp.artifacts.exists(source.sha256) and not cp.artifacts.exists(out.sha256)


def test_gc_never_touches_unfinished_jobs(cp, clock):
    w = cp.register_worker("w", make_caps()).worker_id
    job = cp.submit(JobSpec(workload=CommandJob(command=["x"])))
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    cp.append_logs(w, a.task_id, [("stdout", "still running")])
    clock.advance(30 * DAY)
    cp.heartbeat(w)  # keeps the worker and its task alive
    cp.gc(7 * DAY, delete_jobs=True)
    assert cp.get_job(job.job_id).state.value == "running"
    assert len(cp.job_logs(job.job_id)) == 1
    assert [x.worker_id for x in cp.list_workers()] == [w]
