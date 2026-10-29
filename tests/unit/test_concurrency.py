"""Several tasks per worker: device reservation, default slots and the store migration."""

import sqlite3

from conftest import cuda_device, make_caps, mps_device

from gpulayer.contracts import (
    Accelerator,
    CommandJob,
    JobSpec,
    ResourceRequest,
    TaskOutcome,
    TaskState,
)
from gpulayer.control_plane.scheduler import default_max_tasks, pick_devices


def gpu_job(n=1, **kw):
    return JobSpec(
        workload=CommandJob(command=["x"]),
        resources=ResourceRequest(accelerator=Accelerator.CUDA, gpu_count=n, **kw),
    )


def four_gpu_worker(cp, slots=4):
    caps = make_caps(accelerators=[cuda_device(i) for i in range(4)], max_concurrent_tasks=slots)
    return cp.register_worker("big", caps).worker_id


def test_pick_devices_skips_busy_and_unfit_devices():
    caps = make_caps(accelerators=[cuda_device(0, mem=8), cuda_device(1), cuda_device(2)])
    r = ResourceRequest(accelerator=Accelerator.CUDA, gpu_count=1, gpu_memory_gb=16)
    assert pick_devices(r, caps, set()) == [1]
    assert pick_devices(r, caps, {1}) == [2]
    assert pick_devices(r, caps, {1, 2}) is None
    cpu = ResourceRequest()
    assert pick_devices(cpu, caps, {0, 1, 2}) == []


def test_default_slots_follow_discrete_gpus():
    assert default_max_tasks(make_caps()) == 1
    assert default_max_tasks(make_caps(accelerators=[mps_device()])) == 1
    assert default_max_tasks(make_caps(accelerators=[cuda_device(i) for i in range(8)])) == 8


def test_one_gpu_jobs_share_a_worker_without_sharing_gpus(cp):
    w = four_gpu_worker(cp, slots=8)  # more slots than GPUs: the GPUs are the limit
    jobs = [cp.submit(gpu_job()) for _ in range(5)]
    got = []
    while (a := cp.acquire_task(w)) is not None:
        cp.start_task(w, a.task_id)
        got.append(a)
    assert sorted(d for a in got for d in a.devices) == [0, 1, 2, 3]
    fifth = cp.get_job(jobs[4].job_id)
    assert fifth.state.value == "queued"
    assert any("devices are in use" in r for c in fifth.scheduling.candidates for r in c.reasons)
    # Freeing one GPU lets the fifth job run on exactly that GPU.
    first = got[0]
    cp.finish_task(w, first.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0))
    nxt = cp.acquire_task(w)
    assert nxt.job_id == jobs[4].job_id and nxt.devices == first.devices


def test_multi_gpu_job_waits_for_enough_free_gpus(cp):
    w = four_gpu_worker(cp)
    small = [cp.submit(gpu_job()) for _ in range(2)]
    big = cp.submit(gpu_job(3))
    held = []
    for _ in small:
        held.append(cp.acquire_task(w))
        cp.start_task(w, held[-1].task_id)
    assert cp.acquire_task(w) is None  # only 2 GPUs free, the 3-GPU job must wait
    assert cp.get_job(big.job_id).state.value == "queued"
    cp.finish_task(w, held[0].task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0))
    a = cp.acquire_task(w)
    assert a.job_id == big.job_id and len(a.devices) == 3
    assert not set(a.devices) & set(held[1].devices)


def test_store_adds_devices_column_to_old_databases(tmp_path):
    from gpulayer.storage.sqlite import SqliteStore

    path = tmp_path / "old.db"
    SqliteStore(path).close()
    con = sqlite3.connect(path)
    con.execute("ALTER TABLE tasks DROP COLUMN devices")
    con.commit()
    con.close()
    SqliteStore(path).close()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(tasks)")}
    assert "devices" in cols
