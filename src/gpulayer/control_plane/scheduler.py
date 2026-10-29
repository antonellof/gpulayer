"""Capability-aware, deterministic placement.

The scheduler is a pure function of (job spec, worker views). It never talks to
providers, never mutates state, and never weakens a requirement: an incompatible
worker is rejected with a human-readable reason. Distributed jobs are gang-scheduled:
a whole compatible worker group is selected or nothing is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from gpulayer.contracts import (
    Accelerator,
    CandidateEvaluation,
    DistributedJob,
    ExecutionTarget,
    JobSpec,
    MemoryKind,
    ResourceRequest,
    SchedulingDecision,
    WorkerCapabilities,
    WorkerState,
)
from gpulayer.gpu.support import unsupported_reason


@dataclass
class WorkerView:
    """What the scheduler knows about a worker: capability + current availability."""

    worker_id: str
    state: WorkerState
    capabilities: WorkerCapabilities
    running_tasks: int = 0
    reserved: set[str] = field(default_factory=set)  # job ids holding a reservation

    @property
    def free_slots(self) -> int:
        return max(0, self.capabilities.max_concurrent_tasks - self.running_tasks)

    @property
    def available(self) -> bool:
        return self.state in (WorkerState.IDLE, WorkerState.BUSY) and self.free_slots > 0


def _spec_ok(spec: str, version: str | None) -> bool:
    if version is None:
        return False
    try:
        return Version(version) in SpecifierSet(spec)
    except (InvalidSpecifier, InvalidVersion):
        return False


def _cc_tuple(cc: str) -> tuple[int, int]:
    major, minor = cc.split(".")
    return int(major), int(minor)


RESERVED_LABEL = "gpulayer.dev/reserved-for"


def compatibility_reasons(
    spec: JobSpec,
    caps: WorkerCapabilities,
    *,
    target: ExecutionTarget | None = None,
    job_id: str | None = None,
) -> list[str]:
    """Reasons ``caps`` can NEVER run ``spec`` (ignoring current load). Empty == compatible."""
    r: ResourceRequest = spec.resources
    p = spec.placement
    reasons: list[str] = []

    allowed = [target] if target else p.allowed_targets
    if caps.target not in allowed:
        reasons.append(
            f"worker serves target '{caps.target.value}', job allows {[t.value for t in allowed]}"
        )
    reserved = caps.labels.get(RESERVED_LABEL)
    if reserved and reserved != job_id:
        reasons.append(f"worker is reserved for job {reserved}")
    if p.providers and caps.provider not in p.providers:
        reasons.append(f"provider {caps.provider!r} not in allowed providers {p.providers}")
    if p.regions and caps.region not in p.regions:
        reasons.append(f"region {caps.region!r} not in allowed regions {p.regions}")
    for k, v in p.worker_labels.items():
        if caps.labels.get(k) != v:
            reasons.append(f"label {k}={v} required (worker has {caps.labels.get(k)!r})")

    if r.os and caps.os not in r.os:
        reasons.append(f"os '{caps.os}' not in {r.os}")
    if r.cpu_arch and caps.cpu_arch not in r.cpu_arch:
        reasons.append(f"cpu_arch '{caps.cpu_arch}' not in {r.cpu_arch}")
    if caps.cpu_cores < r.cpu_cores:
        reasons.append(f"needs {r.cpu_cores} CPU cores, worker has {caps.cpu_cores}")
    if r.memory_gb is not None:
        if caps.memory_gb is None:
            reasons.append("memory requirement set but worker memory is unknown")
        elif caps.memory_gb < r.memory_gb:
            reasons.append(f"needs {r.memory_gb} GB RAM, worker has {caps.memory_gb}")
    if r.disk_gb is not None:
        if caps.disk_free_gb is None:
            reasons.append("disk requirement set but worker free disk is unknown")
        elif caps.disk_free_gb < r.disk_gb:
            reasons.append(f"needs {r.disk_gb} GB disk, worker has {caps.disk_free_gb}")
    if r.python and not _spec_ok(r.python, caps.python_version):
        reasons.append(f"python {caps.python_version} does not satisfy {r.python}")
    for fw, wanted in r.frameworks.items():
        have = caps.frameworks.get(fw)
        if have is None:
            reasons.append(f"framework '{fw}' not installed")
        elif not _spec_ok(wanted, have):
            reasons.append(f"{fw} {have} does not satisfy {wanted}")

    why = unsupported_reason(r.accelerator, spec.workload_type)
    if why:
        reasons.append(why)
    if r.accelerator != Accelerator.CPU:
        reasons.extend(_accelerator_reasons(r, caps))

    w = spec.workload
    if isinstance(w, DistributedJob):
        if w.launcher not in caps.launchers:
            reasons.append(f"launcher '{w.launcher}' unavailable on worker")
        if w.comm_backend not in caps.distributed_backends:
            reasons.append(f"communication backend '{w.comm_backend}' unavailable")
        if "torch" not in caps.frameworks:
            reasons.append("torch is required for torchrun-based distributed jobs")
        if w.num_workers > 1 and not caps.advertise_address:
            reasons.append("multi-node job needs a worker with an advertise_address")
        if r.accelerator == Accelerator.CPU and w.procs_per_worker > caps.cpu_cores:
            reasons.append(f"{w.procs_per_worker} procs exceed {caps.cpu_cores} CPU cores")
    return reasons


def _accelerator_reasons(r: ResourceRequest, caps: WorkerCapabilities) -> list[str]:
    devices = [d for d in caps.devices(r.accelerator) if d.execution_supported]
    if len(devices) < r.gpu_count:
        return [
            f"needs {r.gpu_count} usable {r.accelerator.value} device(s), worker has {len(devices)}"
        ]
    reasons: list[str] = []
    if r.gpu_memory_kind and any(d.memory_kind != r.gpu_memory_kind for d in devices):
        reasons.append(f"requires {r.gpu_memory_kind.value} GPU memory")
    if r.gpu_memory_gb is not None:
        ok = [d for d in devices if d.memory_gb is not None and d.memory_gb >= r.gpu_memory_gb]
        if len(ok) < r.gpu_count:
            kind = devices[0].memory_kind if devices else MemoryKind.DEDICATED
            suffix = " (unified memory shared with the CPU)" if kind == MemoryKind.UNIFIED else ""
            reasons.append(
                f"needs {r.gpu_count} device(s) with >= {r.gpu_memory_gb} GB{suffix}, "
                f"{len(ok)} qualify"
            )
    if r.min_cuda_compute_capability:
        need = _cc_tuple(r.min_cuda_compute_capability)
        ok = [
            d for d in devices if d.compute_capability and _cc_tuple(d.compute_capability) >= need
        ]
        if len(ok) < r.gpu_count:
            reasons.append(f"needs compute capability >= {r.min_cuda_compute_capability}")
    return reasons


def _rank_key(spec: JobSpec, w: WorkerView) -> tuple:
    # Deterministic: strategy key first, worker_id as the final tiebreak.
    if spec.placement.strategy == "most_free_memory":
        return (-(w.capabilities.memory_gb or 0.0), w.worker_id)
    return (w.worker_id,)


def schedule(
    spec: JobSpec, workers: list[WorkerView], job_id: str | None = None
) -> SchedulingDecision:
    """Select a worker (independent) or a worker group (distributed) for ``spec``."""
    p = spec.placement
    pinned = set(p.worker_ids)
    needed = spec.num_nodes
    candidates: list[CandidateEvaluation] = []

    for target in p.allowed_targets:
        eligible: list[WorkerView] = []
        any_compatible = False
        for w in sorted(workers, key=lambda w: w.worker_id):
            reasons = []
            if pinned and w.worker_id not in pinned:
                reasons.append("not in placement.worker_ids")
            reasons += compatibility_reasons(spec, w.capabilities, target=target, job_id=job_id)
            compatible = not reasons
            if compatible:
                any_compatible = True
                if w.state in (WorkerState.OFFLINE, WorkerState.UNHEALTHY, WorkerState.DRAINING):
                    reasons.append(f"worker is {w.state.value}")
                elif not w.available:
                    reasons.append("compatible but no free task slot")
            candidates.append(
                CandidateEvaluation(worker_id=w.worker_id, eligible=not reasons, reasons=reasons)
            )
            if not reasons:
                eligible.append(w)

        eligible.sort(key=lambda w: _rank_key(spec, w))
        group = _pick_group(spec, eligible, needed)
        if group:
            what = "worker group" if spec.workload_type.value == "distributed" else "worker"
            return SchedulingDecision(
                placed=True,
                target=target,
                worker_ids=[g.worker_id for g in group],
                summary=f"placed on {what} {[g.worker_id for g in group]} (target "
                f"{target.value}, policy {p.strategy})",
                candidates=candidates,
            )
        if any_compatible and len(eligible) < needed:
            continue  # try explicit fallback targets before giving up

    n_ok = sum(1 for c in candidates if c.eligible)
    if not workers:
        summary = "no workers registered"
    elif needed > 1:
        summary = f"distributed job needs {needed} eligible workers, {n_ok} available"
    else:
        summary = "no eligible worker"
    return SchedulingDecision(placed=False, summary=summary, candidates=candidates)


def _pick_group(spec: JobSpec, eligible: list[WorkerView], needed: int) -> list[WorkerView]:
    if len(eligible) < needed:
        return []
    if needed == 1:
        return eligible[:1]
    # Distributed: all nodes must agree on the torch version so collectives interoperate.
    by_torch: dict[str | None, list[WorkerView]] = {}
    for w in eligible:
        by_torch.setdefault(w.capabilities.frameworks.get("torch"), []).append(w)
    for version in sorted(by_torch, key=lambda v: (v is None, v or "")):
        if len(by_torch[version]) >= needed:
            return by_torch[version][:needed]
    return []


def ever_schedulable(spec: JobSpec, workers: list[WorkerView], job_id: str | None = None) -> bool:
    """True if enough registered, non-offline workers are *compatible*, regardless of load."""
    ok = [
        w
        for w in workers
        if w.state != WorkerState.OFFLINE
        and (not spec.placement.worker_ids or w.worker_id in spec.placement.worker_ids)
        and not compatibility_reasons(spec, w.capabilities, job_id=job_id)
    ]
    return len(ok) >= spec.num_nodes
