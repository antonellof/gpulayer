"""Deterministic placement tests against fake worker views (no processes)."""

from conftest import TORCH_CAPS, cuda_device, make_caps, mps_device

from gpulayer.contracts import (
    AcceleratorDevice,
    CommandJob,
    DistributedJob,
    ExecutionTarget,
    JobSpec,
    PlacementPolicy,
    ResourceRequest,
    WorkerOrigin,
    WorkerState,
)
from gpulayer.control_plane.scheduler import (
    RESERVED_LABEL,
    WorkerView,
    compatibility_reasons,
    ever_schedulable,
    schedule,
)


def view(wid, state=WorkerState.IDLE, running=0, **caps):
    return WorkerView(
        worker_id=wid, state=state, capabilities=make_caps(**caps), running_tasks=running
    )


def spec(resources=None, placement=None, workload=None):
    return JobSpec(
        workload=workload or CommandJob(command=["x"]),
        resources=resources or ResourceRequest(),
        placement=placement or PlacementPolicy(),
    )


def test_cpu_job_first_fit_is_deterministic():
    ws = [view("w-b"), view("w-a")]
    d = schedule(spec(), ws)
    assert d.placed and d.worker_ids == ["w-a"]
    assert schedule(spec(), list(reversed(ws))).worker_ids == ["w-a"]


def test_cuda_job_rejects_cpu_only_and_explains():
    d = schedule(spec(ResourceRequest(accelerator="cuda")), [view("w1")])
    assert not d.placed
    assert "usable cuda device" in d.candidates[0].reasons[0]


def test_cuda_memory_and_compute_capability():
    r = ResourceRequest(accelerator="cuda", gpu_memory_gb=40, min_cuda_compute_capability="8.0")
    small = view("w1", accelerators=[cuda_device(mem=24, cc="8.6")])
    old = view("w2", accelerators=[cuda_device(mem=80, cc="7.0")])
    good = view("w3", accelerators=[cuda_device(mem=80, cc="9.0")])
    d = schedule(spec(r), [small, old, good])
    assert d.worker_ids == ["w3"]
    reasons = {c.worker_id: " ".join(c.reasons) for c in d.candidates}
    assert ">= 40" in reasons["w1"] and "compute capability" in reasons["w2"]


def test_gpu_count():
    w = view("w1", accelerators=[cuda_device(0), cuda_device(1)])
    assert schedule(spec(ResourceRequest(accelerator="cuda", gpu_count=2)), [w]).placed
    assert not schedule(spec(ResourceRequest(accelerator="cuda", gpu_count=4)), [w]).placed


def test_mps_is_not_cuda():
    mac = view("mac", os="darwin", cpu_arch="arm64", accelerators=[mps_device()])
    assert not schedule(spec(ResourceRequest(accelerator="cuda")), [mac]).placed
    assert schedule(spec(ResourceRequest(accelerator="mps")), [mac]).placed


def test_unified_memory_is_explicit():
    mac = view("mac", accelerators=[mps_device(mem=16)])
    d = schedule(spec(ResourceRequest(accelerator="mps", gpu_memory_gb=24)), [mac])
    assert not d.placed and "unified memory" in d.candidates[0].reasons[0]
    d = schedule(spec(ResourceRequest(accelerator="mps", gpu_memory_kind="dedicated")), [mac])
    assert not d.placed


def test_rocm_detected_but_never_scheduled():
    dev = AcceleratorDevice(
        index=0, accelerator="rocm", vendor="amd", name="MI300", execution_supported=False
    )
    d = schedule(spec(ResourceRequest(accelerator="rocm")), [view("w", accelerators=[dev])])
    assert not d.placed
    assert any("detection-only" in r for r in d.candidates[0].reasons)


def test_framework_and_python_constraints():
    w = view("w", frameworks={"torch": "2.1.0"}, python_version="3.10.4")
    assert not schedule(spec(ResourceRequest(frameworks={"torch": ">=2.2"})), [w]).placed
    assert schedule(spec(ResourceRequest(frameworks={"torch": ">=2.0"})), [w]).placed
    assert not schedule(spec(ResourceRequest(python=">=3.11")), [w]).placed
    assert not schedule(spec(ResourceRequest(frameworks={"jax": ">=0"})), [w]).placed


def test_unknown_memory_never_assumed_sufficient():
    w = view("w", memory_gb=None)
    assert not schedule(spec(ResourceRequest(memory_gb=1)), [w]).placed


def test_target_matching():
    local, remote = view("l"), view("r", origin=WorkerOrigin.EXISTING)
    d = schedule(spec(placement=PlacementPolicy(target="remote")), [local, remote])
    assert d.worker_ids == ["r"] and d.target == ExecutionTarget.REMOTE
    assert not schedule(spec(placement=PlacementPolicy(target="cloud")), [local, remote]).placed


def test_explicit_fallback_only():
    remote = view("r", origin=WorkerOrigin.EXISTING)
    assert not schedule(spec(), [remote]).placed  # local target never silently goes remote
    d = schedule(
        spec(placement=PlacementPolicy(target="local", fallback_targets=["remote"])), [remote]
    )
    assert d.placed and d.target == ExecutionTarget.REMOTE


def test_busy_vs_incompatible_vs_offline():
    s = spec()
    busy = view("busy", running=1)
    offline = view("off", state=WorkerState.OFFLINE)
    d = schedule(s, [busy, offline])
    assert not d.placed
    assert ever_schedulable(s, [busy])  # compatible but busy: worth queueing
    assert not ever_schedulable(s, [offline])


def test_pinning_and_labels():
    ws = [view("a", labels={"pool": "x"}), view("b", labels={"pool": "y"})]
    assert schedule(spec(placement=PlacementPolicy(worker_ids=["b"])), ws).worker_ids == ["b"]
    assert schedule(
        spec(placement=PlacementPolicy(worker_labels={"pool": "y"})), ws
    ).worker_ids == ["b"]


def test_reserved_workers_only_serve_their_job():
    w = view("p", labels={RESERVED_LABEL: "job-1"})
    assert not schedule(spec(), [w], job_id="job-2").placed
    assert schedule(spec(), [w], job_id="job-1").placed


def test_most_free_memory_strategy():
    ws = [view("a", memory_gb=16), view("b", memory_gb=64)]
    assert schedule(
        spec(placement=PlacementPolicy(strategy="most_free_memory")), ws
    ).worker_ids == ["b"]


# --------------------------------------------------------------------- distributed


def dist(n=2, p=1):
    return spec(workload=DistributedJob(entrypoint="t.py", num_workers=n, procs_per_worker=p))


def test_distributed_gang_schedules_whole_group():
    ws = [view(f"w{i}", **TORCH_CAPS) for i in range(3)]
    d = schedule(dist(2), ws)
    assert d.placed and d.worker_ids == ["w0", "w1"]


def test_distributed_no_partial_group():
    ws = [view("w0", **TORCH_CAPS), view("w1", running=1, **TORCH_CAPS)]
    d = schedule(dist(2), ws)
    assert not d.placed and d.worker_ids == []
    assert "needs 2 eligible workers, 1 available" in d.summary


def test_distributed_requires_launcher_backend_and_address():
    no_torch = view("a")
    no_addr = view("b", **{**TORCH_CAPS, "advertise_address": None})
    reasons = compatibility_reasons(dist(2), no_torch.capabilities)
    assert any("torchrun" in r for r in reasons) and any("gloo" in r for r in reasons)
    assert any(
        "advertise_address" in r for r in compatibility_reasons(dist(2), no_addr.capabilities)
    )


def test_distributed_group_has_consistent_torch():
    ws = [
        view("a", **{**TORCH_CAPS, "frameworks": {"torch": "2.3.0"}}),
        view("b", **{**TORCH_CAPS, "frameworks": {"torch": "2.4.0"}}),
        view("c", **{**TORCH_CAPS, "frameworks": {"torch": "2.4.0"}}),
    ]
    assert schedule(dist(2), ws).worker_ids == ["b", "c"]
    assert not schedule(dist(3), ws).placed
