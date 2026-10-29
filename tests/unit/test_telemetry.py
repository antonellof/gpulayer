"""Worker resource samples: storage, history for the dashboard, Prometheus gauges."""

from conftest import make_caps

from gpulayer.contracts import (
    CommandJob,
    GpuMetrics,
    JobSpec,
    TaskOutcome,
    TaskState,
    WorkerMetrics,
)


def m(load=1.0, **kw):
    return WorkerMetrics(load1=load, cpu_count=8, memory_used_gb=4, memory_total_gb=16, **kw)


def test_samples_are_rate_limited_and_returned_in_history(cp, clock):
    w = cp.register_worker("gpu-box", make_caps()).worker_id
    cp.heartbeat(w, metrics=m(1.0))
    clock.advance(1)
    cp.heartbeat(w, metrics=m(9.0))  # within SAMPLE_INTERVAL: dropped
    clock.advance(10)
    cp.heartbeat(w, metrics=m(2.0, gpus=[GpuMetrics(index=0, utilization_pct=87)]))
    h = cp.metrics_history(minutes=5)
    (worker,) = h["workers"]
    assert worker["name"] == "gpu-box"
    assert [s["load1"] for s in worker["samples"]] == [1.0, 2.0]
    assert worker["samples"][-1]["gpus"][0]["utilization_pct"] == 87
    assert cp.latest_samples()[w].load1 == 2.0


def test_history_counts_running_and_waiting_jobs(cp, clock):
    job = cp.submit(JobSpec(workload=CommandJob(command=["x"])))  # no worker yet: it waits
    clock.advance(60)
    w = cp.register_worker("w", make_caps()).worker_id
    a = cp.acquire_task(w)
    assert a.job_id == job.job_id
    cp.start_task(w, a.task_id)
    for _ in range(6):  # keep the lease alive while it runs for a minute
        clock.advance(10)
        cp.heartbeat(w)
    cp.finish_task(w, a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0))
    clock.advance(60)
    h = cp.metrics_history(minutes=3, points=4)  # buckets at -3, -2, -1, 0 minutes
    assert [p["waiting"] for p in h["jobs"]] == [1, 0, 0, 0]
    assert [p["running"] for p in h["jobs"]] == [0, 1, 0, 0]


def test_prometheus_exposes_worker_gauges(cp):
    w = cp.register_worker("box", make_caps()).worker_id
    cp.heartbeat(w, metrics=m(3.5, gpus=[GpuMetrics(index=1, utilization_pct=50)]))
    text = cp.metrics_text()
    assert f'gpulayer_worker_load1{{worker="box",worker_id="{w}"}} 3.5' in text
    assert f'gpulayer_gpu_utilization_ratio{{gpu="1",worker="box",worker_id="{w}"}} 0.5' in text
    assert "gpulayer_worker_memory_total_bytes" in text


def test_history_counts_active_jobs_in_the_latest_bucket(cp, clock):
    w = cp.register_worker("w", make_caps()).worker_id
    cp.submit(JobSpec(workload=CommandJob(command=["x"])))
    a = cp.acquire_task(w)
    cp.start_task(w, a.task_id)
    clock.advance(5)
    cp.heartbeat(w)
    assert cp.metrics_history(minutes=1, points=3)["jobs"][-1]["running"] == 1
