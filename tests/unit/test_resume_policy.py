"""Keeping a job's newest checkpoint and restarting from it (ADR-0025), on the control plane
with simulated workers and a fake clock."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpulayer.contracts import (
    ArtifactKind,
    CommandJob,
    DistributedJob,
    JobSpec,
    JobState,
    ResumePolicy,
    RetryPolicy,
    TaskOutcome,
    TaskState,
)
from gpulayer.contracts.models import RESUME_INPUT
from gpulayer.errors import Conflict


@pytest.fixture
def register_worker(cp):
    from conftest import TORCH_CAPS, make_caps

    names = iter(range(100))

    def register() -> str:
        return cp.register_worker(f"w{next(names)}", make_caps(**TORCH_CAPS)).worker_id

    return register


def spec(**kw) -> JobSpec:
    kw.setdefault("workload", CommandJob(command=["python", "t.py"]))
    return JobSpec(name="j", **kw)


def resumable(attempts: int = 3) -> dict:
    return {
        "retry": RetryPolicy(max_attempts=attempts, retry_on=["worker_lost", "preempted"]),
        "resume": ResumePolicy(),
    }


def group(cp=None) -> dict:
    source = cp.put_artifact(b"bundle") if cp is not None else None
    job = DistributedJob(entrypoint="t.py", num_workers=2, procs_per_worker=1, source=source)
    return {"workload": job}


def start_group(cp, workers):
    """Start a group of two: returns (worker, assignment) of node ranks 0 and 1."""
    rank0 = next((w, a) for w in workers if (a := cp.acquire_task(w)) is not None)
    cp.start_task(rank0[0], rank0[1].task_id)
    cp.publish_rendezvous(rank0[0], rank0[1].task_id, "10.0.0.1", 1234)
    other = next(w for w in workers if w != rank0[0])
    rank1 = (other, cp.acquire_task(other))
    cp.start_task(other, rank1[1].task_id)
    return rank0, rank1


def run(cp, worker, job):
    a = cp.acquire_task(worker)
    assert a is not None and a.job_id == job
    cp.start_task(worker, a.task_id)
    return a


def preempted(cp, worker, a):
    cp.finish_task(worker, a.task_id, TaskOutcome(state=TaskState.FAILED, exit_code=143))


def checkpoint(cp, worker, a, data=b"state"):
    ref = cp.put_artifact(data, "last.ckpt", ArtifactKind.OUTPUT)
    cp.checkpoint_task(worker, a.task_id, ref)
    return ref


def test_a_job_without_resume_stays_valid_for_older_workers():
    dumped = spec().model_dump(mode="json")
    assert "resume" not in dumped
    assert "resume" not in spec().model_dump_json()
    assert spec(**resumable()).model_dump(mode="json")["resume"]["path"]
    assert JobSpec.model_validate(dumped).resume is None


def test_distributed_retry_needs_resume():
    with pytest.raises(ValidationError, match="needs resume"):
        spec(retry=RetryPolicy(max_attempts=2), **group())
    assert spec(**resumable(), **group()).resume is not None


@pytest.mark.parametrize("path", ["/etc/passwd", "../x", "a/../../b", ""])
def test_resume_path_stays_inside_outputs(path):
    with pytest.raises(ValidationError):
        ResumePolicy(path=path)


def test_next_attempt_gets_the_newest_checkpoint(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec(**resumable())).job_id
    first = run(cp, w, job)
    assert first.spec.inputs == []
    old = checkpoint(cp, w, first, b"older")
    new = checkpoint(cp, w, first, b"newer")
    assert not cp.artifacts.exists(old.sha256), "only the newest is kept"
    preempted(cp, w, first)
    assert cp.get_job(job).state in (JobState.QUEUED, JobState.SCHEDULED)
    second = run(cp, w, job)
    assert second.attempt == 2
    assert [(i.path, i.artifact.sha256) for i in second.spec.inputs] == [(RESUME_INPUT, new.sha256)]


def test_kept_checkpoint_replaces_an_earlier_resume_input(cp, register_worker):
    from gpulayer.contracts.models import InputArtifact

    w = register_worker()
    start = cp.put_artifact(b"from the user", "x", ArtifactKind.INPUT)
    job = cp.submit(
        spec(inputs=[InputArtifact(artifact=start, path=RESUME_INPUT)], **resumable())
    ).job_id
    first = run(cp, w, job)
    kept = checkpoint(cp, w, first)
    preempted(cp, w, first)
    second = run(cp, w, job)
    assert [i.artifact.sha256 for i in second.spec.inputs] == [kept.sha256]


def test_a_crash_in_the_script_is_not_retried(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec(**resumable())).job_id
    a = run(cp, w, job)
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.FAILED, exit_code=1))
    assert cp.get_job(job).state == JobState.FAILED


def test_attempts_run_out_and_the_checkpoint_comes_back(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec(**resumable(2))).job_id
    for _ in range(2):
        a = run(cp, w, job)
        ref = checkpoint(cp, w, a, f"attempt {a.attempt}".encode())
        preempted(cp, w, a)
    result = cp.job_result(job)
    assert result.state == JobState.FAILED
    assert [(o.name, o.sha256) for o in result.outputs] == [("resume/last.ckpt", ref.sha256)]


def test_success_drops_the_kept_checkpoint(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec(**resumable())).job_id
    a = run(cp, w, job)
    ref = checkpoint(cp, w, a)
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0))
    assert not cp.artifacts.exists(ref.sha256)


def test_success_keeps_it_when_it_is_also_an_output(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec(**resumable())).job_id
    a = run(cp, w, job)
    ref = checkpoint(cp, w, a)
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0, outputs=[ref]))
    assert cp.artifacts.exists(ref.sha256)


def test_a_stale_attempt_cannot_replace_the_checkpoint(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec(**resumable())).job_id
    first = run(cp, w, job)
    kept = checkpoint(cp, w, first, b"kept")
    preempted(cp, w, first)
    second = run(cp, w, job)
    checkpoint(cp, w, first, b"late report from attempt one")
    assert second.spec.inputs[0].artifact.sha256 == kept.sha256
    row = cp.store.one("SELECT resume_artifact FROM jobs WHERE id=?", (job,))
    assert kept.sha256 in row["resume_artifact"]


def test_checkpoints_need_a_resume_policy(cp, register_worker):
    w = register_worker()
    job = cp.submit(spec()).job_id
    a = run(cp, w, job)
    with pytest.raises(Conflict, match="resume"):
        checkpoint(cp, w, a)


def test_group_restarts_only_after_every_rank_stopped(cp, register_worker):
    w0, w1 = register_worker(), register_worker()
    job = cp.submit(spec(**resumable(), **group(cp))).job_id
    (w_a, a0), (w_b, a1) = start_group(cp, [w0, w1])
    kept = checkpoint(cp, w_a, a0)
    preempted(cp, w_a, a0)
    assert cp.get_job(job).state == JobState.RUNNING, "rank 1 is still running"
    assert cp.acquire_task(w_a) is None and cp.acquire_task(w_b) is None
    cp.finish_task(w_b, a1.task_id, TaskOutcome(state=TaskState.CANCELLED, exit_code=143))
    assert cp.get_job(job).state != JobState.FAILED
    got = next(a for w in (w0, w1) if (a := cp.acquire_task(w)) is not None)
    assert got.attempt == 2 and got.spec.inputs[0].artifact.sha256 == kept.sha256


def test_group_without_retries_left_fails_at_once(cp, register_worker):
    w0, w1 = register_worker(), register_worker()
    job = cp.submit(spec(**resumable(1), **group(cp))).job_id
    (w_a, a0), _ = start_group(cp, [w0, w1])
    preempted(cp, w_a, a0)
    assert cp.get_job(job).state == JobState.FAILED
