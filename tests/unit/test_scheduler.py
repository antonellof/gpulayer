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
    NETWORK_GROUP_LABEL,
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


def test_rocm_needs_a_usable_device_and_a_rocm_torch():
    detected = AcceleratorDevice(
        index=0, accelerator="rocm", vendor="amd", name="MI300", execution_supported=False
    )
    d = schedule(spec(ResourceRequest(accelerator="rocm")), [view("w", accelerators=[detected])])
    assert not d.placed and "usable rocm device" in d.candidates[0].reasons[0]
    usable = detected.model_copy(update={"execution_supported": True})
    r = ResourceRequest(accelerator="rocm", frameworks={"torch.rocm": ">=0"})
    no_rocm_torch = view("a", accelerators=[usable], frameworks={"torch": "2.15.0"})
    ok = view("b", accelerators=[usable], frameworks={"torch": "2.15.0", "torch.rocm": "2.15.0"})
    d = schedule(spec(r), [no_rocm_torch, ok])
    assert d.worker_ids == ["b"]
    assert "cannot use ROCm" in " ".join(d.candidates[0].reasons)


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
    assert any("direct" in r for r in reasons) and any("gloo" in r for r in reasons)
    assert any(
        "advertise_address" in r for r in compatibility_reasons(dist(2), no_addr.capabilities)
    )


def test_direct_launcher_needs_a_worker_that_supports_it():
    # Workers from v0.1.0 report only torchrun; a direct job never lands there.
    old = view("old", **{**TORCH_CAPS, "launchers": ["torchrun"]})
    assert any(
        "launcher 'direct' unavailable" in r
        for r in compatibility_reasons(dist(1), old.capabilities)
    )
    torchrun = dist(1)
    torchrun.workload.launcher = "torchrun"
    assert compatibility_reasons(torchrun, old.capabilities) == []


def test_distributed_group_has_consistent_torch():
    ws = [
        view("a", **{**TORCH_CAPS, "frameworks": {"torch": "2.3.0"}}),
        view("b", **{**TORCH_CAPS, "frameworks": {"torch": "2.4.0"}}),
        view("c", **{**TORCH_CAPS, "frameworks": {"torch": "2.4.0"}}),
    ]
    assert schedule(dist(2), ws).worker_ids == ["b", "c"]
    assert not schedule(dist(3), ws).placed


def test_offline_workers_are_not_listed_as_candidates():
    gone = view("gone", state=WorkerState.OFFLINE)
    d = schedule(spec(ResourceRequest(frameworks={"jax": ">=0"})), [gone, view("live")])
    assert [c.worker_id for c in d.candidates] == ["live"]
    assert schedule(spec(), [gone]).summary == "no workers online"


def test_failure_explanations_give_actionable_hints():
    from datetime import datetime, timezone

    from gpulayer.contracts import JobState, JobStatus, WorkloadType
    from gpulayer.sdk.remote import explain_failure

    def status(resources):
        s = spec(resources)
        return JobStatus(
            job_id="j",
            name="j",
            state=JobState.FAILED,
            workload_type=WorkloadType.INDEPENDENT,
            spec=s,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            error="unschedulable: no eligible worker",
            scheduling=schedule(s, [view("w")]),
        )

    numpy_msg = explain_failure(status(ResourceRequest(frameworks={"numpy": ">=1.20"})))
    assert "framework 'numpy' not installed" in numpy_msg
    assert "python -m pip install numpy" in numpy_msg
    cuda_msg = explain_failure(
        status(ResourceRequest(accelerator="cuda", frameworks={"torch.cuda": ">=0"}))
    )
    assert "torch on this worker cannot use CUDA" in cuda_msg
    assert "pip install" not in cuda_msg  # no bogus "pip install torch.cuda"


def test_spread_prefers_the_least_loaded_worker():
    ws = [
        view("w-a", running=2, max_concurrent_tasks=3),
        view("w-b", running=1, max_concurrent_tasks=3),
        view("w-c", running=1, max_concurrent_tasks=2),
    ]
    assert schedule(spec(), ws).worker_ids == ["w-b"]  # 1/3 loaded beats 1/2 and 2/3
    packed = schedule(spec(placement=PlacementPolicy(strategy="first_fit")), ws)
    assert packed.worker_ids == ["w-a"]


def test_spread_keeps_room_for_a_distributed_group(cp):
    caps = make_caps(max_concurrent_tasks=2, **TORCH_CAPS)
    a = cp.register_worker("a", caps).worker_id
    b = cp.register_worker("b", caps).worker_id
    for _ in range(2):
        cp.submit(JobSpec(workload=CommandJob(command=["x"])))
    busy = {t.worker_id for j in cp.list_jobs() for t in j.tasks}
    assert busy == {a, b}  # one each, not both on the same worker
    ddp = cp.submit(
        JobSpec(workload=DistributedJob(entrypoint="t.py", num_workers=2, procs_per_worker=1))
    )
    assert cp.get_job(ddp.job_id).state.value == "scheduled"


# ------------------------------------------------------------ topology-aware groups


def placed_view(wid, zone=None, region=None, group=None, torch="2.4.0", running=0):
    labels = {NETWORK_GROUP_LABEL: group} if group else {}
    return view(
        wid,
        running=running,
        **{**TORCH_CAPS, "frameworks": {"torch": torch}},
        zone=zone,
        region=region,
        labels=labels,
    )


def dist_with(n, locality):
    s = dist(n)
    s.placement.group_locality = locality
    return s


def test_group_prefers_one_zone_over_worker_order():
    ws = [
        placed_view("a1", zone="z-a", region="r"),
        placed_view("a2", zone="z-a", region="r"),
        placed_view("b1", zone="z-b", region="r"),
        placed_view("b2", zone="z-b", region="r"),
        placed_view("b3", zone="z-b", region="r"),
    ]
    d = schedule(dist(3), ws)
    assert d.worker_ids == ["b1", "b2", "b3"]
    assert "same zone z-b" in d.summary


def test_group_prefers_a_network_group_inside_a_zone():
    ws = [
        placed_view("w1", zone="z", group="rack-1"),
        placed_view("w2", zone="z", group="rack-2"),
        placed_view("w3", zone="z", group="rack-2"),
    ]
    d = schedule(dist(2), ws)
    assert d.worker_ids == ["w2", "w3"] and "same network group rack-2" in d.summary


def test_group_falls_back_to_region_then_anywhere():
    ws = [
        placed_view("a", zone="z-a", region="r1"),
        placed_view("b", zone="z-b", region="r1"),
        placed_view("c", zone="z-c", region="r2"),
    ]
    d = schedule(dist(2), ws)
    assert d.worker_ids == ["a", "b"] and "same region r1" in d.summary
    d = schedule(dist(3), ws)
    assert d.placed and d.worker_ids == ["a", "b", "c"] and "same" not in d.summary


def test_workers_without_a_location_place_as_before():
    ws = [placed_view(f"w{i}") for i in range(3)]
    d = schedule(dist(2), ws)
    assert d.worker_ids == ["w0", "w1"] and "same" not in d.summary


def test_locality_never_mixes_torch_versions():
    ws = [
        placed_view("a", zone="z", torch="2.3.0"),
        placed_view("b", zone="z", torch="2.4.0"),
        placed_view("c", zone="y", torch="2.4.0"),
    ]
    d = schedule(dist(2), ws)
    assert d.worker_ids == ["b", "c"] and "same" not in d.summary


def test_required_zone_rejects_a_group_split_across_zones():
    ws = [placed_view("a", zone="z-a"), placed_view("b", zone="z-b")]
    s = dist_with(2, "zone")
    d = schedule(s, ws)
    assert not d.placed
    assert "needs 2 eligible workers in one zone" in d.summary and "at most 1" in d.summary
    assert not ever_schedulable(s, ws)
    # Busy but in one zone: wait instead of failing.
    ws = [placed_view("a", zone="z"), placed_view("b", zone="z", running=1)]
    assert not schedule(s, ws).placed and ever_schedulable(s, ws)


def test_required_zone_ignores_workers_without_a_zone():
    ws = [placed_view("a", zone="z"), placed_view("b")]
    assert not schedule(dist_with(2, "zone"), ws).placed


def test_any_locality_keeps_rank_order():
    ws = [
        placed_view("a1", zone="z-a"),
        placed_view("b1", zone="z-b"),
        placed_view("b2", zone="z-b"),
    ]
    assert schedule(dist_with(2, "any"), ws).worker_ids == ["a1", "b1"]


def test_regions_filter_also_accepts_a_zone():
    caps = make_caps(region="europe-west1", zone="europe-west1-b")
    for allowed in (["europe-west1"], ["europe-west1-b"]):
        s = spec(placement=PlacementPolicy(regions=allowed))
        assert compatibility_reasons(s, caps) == []
    s = spec(placement=PlacementPolicy(regions=["us-east1"]))
    assert compatibility_reasons(s, caps)


def test_torch_compile_job_is_rejected_without_a_compiler_with_the_reason():
    s = spec(ResourceRequest(frameworks={"torch.compile": ">=0"}))
    no_cc = view("no-cc", frameworks={"torch": "2.15.0"})
    cc = view("cc", frameworks={"torch": "2.15.0", "torch.compile": "2.15.0"})
    d = schedule(s, [no_cc, cc])
    assert d.worker_ids == ["cc"]
    reasons = {c.worker_id: " ".join(c.reasons) for c in d.candidates}
    assert "needs a C compiler" in reasons["no-cc"]
