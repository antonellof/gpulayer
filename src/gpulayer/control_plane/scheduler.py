"""Capability-aware, deterministic placement.

The scheduler is a pure function of (job spec, worker views). It never talks to
providers, never mutates state, and never weakens a requirement: an incompatible
worker is rejected with a human-readable reason. Distributed jobs are gang-scheduled:
a whole compatible worker group is selected or nothing is.
"""

from __future__ import annotations

from collections.abc import Callable
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
    busy_devices: set[int] = field(default_factory=set)  # device indices held by active tasks

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
# User-defined network locality finer than a zone: a rack, a LAN, a switch, one office.
NETWORK_GROUP_LABEL = "gpulayer.dev/network-group"

# Locality levels for distributed groups, tightest first.
_LEVELS = ("network_group", "zone", "region")
_LEVEL_NAME = {"network_group": "network group", "zone": "zone", "region": "region"}


def locality(caps: WorkerCapabilities, level: str) -> str | None:
    """The worker's value at a locality level, or None when it is not known."""
    if level == "network_group":
        return caps.labels.get(NETWORK_GROUP_LABEL) or None
    if level == "zone":
        return caps.zone
    return caps.region


# Pseudo-frameworks reported by discovery: "this worker's torch can drive that GPU".
_TORCH_GPU = {"torch.cuda": "CUDA", "torch.mps": "MPS"}


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
    if p.regions and caps.region not in p.regions and caps.zone not in p.regions:
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
            if fw in _TORCH_GPU:
                reasons.append(f"torch on this worker cannot use {_TORCH_GPU[fw]}")
            else:
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


def pick_devices(r: ResourceRequest, caps: WorkerCapabilities, busy: set[int]) -> list[int] | None:
    """Indices of the devices a task gets on this worker: free devices that meet every
    per-device requirement. Several GPUs are chosen by interconnect when the worker reports it
    (the best-connected set), else lowest index first. ``[]`` for CPU jobs; None if too few
    are free."""
    if r.accelerator == Accelerator.CPU or r.gpu_count == 0:
        return []
    ok = _fitting_free(r, caps, busy)
    if len(ok) < r.gpu_count:
        return None
    if r.gpu_count > 1:
        links = {d.index: d.links for d in caps.devices(r.accelerator)}
        if any(links.get(i) for i in ok):
            return _best_connected(ok, r.gpu_count, links)
    return ok[: r.gpu_count]


def has_free_devices(r: ResourceRequest, caps: WorkerCapabilities, busy: set[int]) -> bool:
    """Whether ``pick_devices`` would succeed, without choosing among the devices."""
    if r.accelerator == Accelerator.CPU or r.gpu_count == 0:
        return True
    return len(_fitting_free(r, caps, busy)) >= r.gpu_count


def _fitting_free(r: ResourceRequest, caps: WorkerCapabilities, busy: set[int]) -> list[int]:
    ok = []
    for d in caps.devices(r.accelerator):
        if not d.execution_supported or d.index in busy:
            continue
        if r.gpu_memory_kind and d.memory_kind != r.gpu_memory_kind:
            continue
        if r.gpu_memory_gb is not None and (d.memory_gb is None or d.memory_gb < r.gpu_memory_gb):
            continue
        if r.min_cuda_compute_capability and not (
            d.compute_capability
            and _cc_tuple(d.compute_capability) >= _cc_tuple(r.min_cuda_compute_capability)
        ):
            continue
        ok.append(d.index)
    ok.sort()
    return ok


def link_rank(link: str | None) -> int:
    """Higher is faster: NVLink (more links better), then the PCIe paths nvidia-smi names,
    from one switch (PIX) to across CPU sockets (SYS). Unknown ranks lowest."""
    if not link:
        return -1
    if link.startswith("NV"):
        n = link[2:]
        return 100 + (int(n) if n.isdigit() else 1)
    return {"PIX": 50, "PXB": 40, "PHB": 30, "NODE": 20, "SYS": 10}.get(link, -1)


def _best_connected(free: list[int], n: int, links: dict[int, dict[int, str]]) -> list[int]:
    """The ``n`` free GPUs whose slowest link between any two is fastest (then the fastest
    links in total, then the lowest indices). Greedy from every starting GPU: a machine has at
    most a few dozen GPUs, so this is cheap and finds e.g. an NVLink island when one is free."""
    if n >= len(free):
        return list(free)
    m = len(free)
    rank = [
        [
            max(
                link_rank(links.get(a, {}).get(b)),
                link_rank(links.get(b, {}).get(a)),
            )
            for b in free
        ]
        for a in free
    ]
    best: tuple[tuple[int, int], list[int]] | None = None
    for seed in range(m):
        group = [seed]
        weakest = rank[seed][:]  # each candidate's slowest link to the group so far
        total = rank[seed][:]  # and the sum of its links to the group
        taken = [False] * m
        taken[seed] = True
        while len(group) < n:
            nxt = max(
                (c for c in range(m) if not taken[c]),
                key=lambda c: (weakest[c], total[c], -c),
            )
            taken[nxt] = True
            group.append(nxt)
            row = rank[nxt]
            for c in range(m):
                if row[c] < weakest[c]:
                    weakest[c] = row[c]
                total[c] += row[c]
        pairs = [rank[a][b] for k, a in enumerate(group) for b in group[k + 1 :]]
        score = (min(pairs), sum(pairs))
        chosen = sorted(free[g] for g in group)
        if best is None or score > best[0] or (score == best[0] and chosen < best[1]):
            best = (score, chosen)
    assert best is not None
    return best[1]


def default_max_tasks(caps: WorkerCapabilities) -> int:
    """One task slot per usable discrete GPU, so a multi-GPU machine runs one single-GPU job
    per GPU at the same time. Machines without such GPUs run one task at a time."""
    n = sum(
        1
        for d in caps.accelerators
        if d.execution_supported and d.accelerator in (Accelerator.CUDA, Accelerator.ROCM)
    )
    return max(1, n)


def _rank_key(spec: JobSpec, w: WorkerView) -> tuple:
    # Deterministic: strategy key first, worker_id as the final tiebreak.
    if spec.placement.strategy == "most_free_memory":
        return (-(w.capabilities.memory_gb or 0.0), w.worker_id)
    if spec.placement.strategy == "spread":
        # Fraction of slots in use, so a 1-slot worker running one task counts as full.
        load = w.running_tasks / max(1, w.capabilities.max_concurrent_tasks)
        return (load, w.running_tasks, w.worker_id)
    return (w.worker_id,)


def schedule(
    spec: JobSpec, workers: list[WorkerView], job_id: str | None = None
) -> SchedulingDecision:
    """Select a worker (independent) or a worker group (distributed) for ``spec``."""
    p = spec.placement
    pinned = set(p.worker_ids)
    needed = spec.num_nodes
    candidates: list[CandidateEvaluation] = []
    locality_miss: str | None = None

    for target in p.allowed_targets:
        eligible: list[WorkerView] = []
        any_compatible = False
        for w in sorted(workers, key=lambda w: w.worker_id):
            if w.state == WorkerState.OFFLINE:
                continue  # gone; listing it would only clutter the explanation
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
                elif not has_free_devices(spec.resources, w.capabilities, w.busy_devices):
                    reasons.append(
                        f"compatible but its {spec.resources.accelerator.value} devices are in "
                        "use by other tasks"
                    )
            candidates.append(
                CandidateEvaluation(worker_id=w.worker_id, eligible=not reasons, reasons=reasons)
            )
            if not reasons:
                eligible.append(w)

        eligible.sort(key=lambda w: _rank_key(spec, w))
        group, where = _pick_group(spec, eligible, needed)
        if group:
            what = "worker group" if spec.workload_type.value == "distributed" else "worker"
            near = f"{where}, " if where else ""
            return SchedulingDecision(
                placed=True,
                target=target,
                worker_ids=[g.worker_id for g in group],
                summary=f"placed on {what} {[g.worker_id for g in group]} ({near}target "
                f"{target.value}, policy {p.strategy})",
                candidates=candidates,
            )
        if where:
            locality_miss = where
        if any_compatible and len(eligible) < needed:
            continue  # try explicit fallback targets before giving up

    n_ok = sum(1 for c in candidates if c.eligible)
    if not any(w.state != WorkerState.OFFLINE for w in workers):
        summary = "no workers online"
    elif locality_miss:
        summary = locality_miss
    elif needed > 1:
        summary = f"distributed job needs {needed} eligible workers, {n_ok} available"
    else:
        summary = "no eligible worker"
    return SchedulingDecision(placed=False, summary=summary, candidates=candidates)


def _first_full(
    workers: list[WorkerView], needed: int, key: Callable[[WorkerView], tuple | None]
) -> list[WorkerView]:
    """Bucket ``workers`` (already in rank order) by ``key`` and return the first bucket to
    reach ``needed`` members. Workers whose key is None are skipped."""
    buckets: dict[tuple, list[WorkerView]] = {}
    for w in workers:
        k = key(w)
        if k is None:
            continue
        b = buckets.setdefault(k, [])
        b.append(w)
        if len(b) == needed:
            return b
    return []


def _locality_levels(spec: JobSpec) -> tuple[tuple[str, ...], bool]:
    """The levels to try, tightest first, and whether one of them is required."""
    mode = spec.placement.group_locality
    if mode == "prefer":
        return _LEVELS, False
    if mode == "any":
        return (), False
    return (mode,), True


def _torch(w: WorkerView) -> str | None:
    return w.capabilities.frameworks.get("torch")


def _pick_group(
    spec: JobSpec, eligible: list[WorkerView], needed: int
) -> tuple[list[WorkerView], str | None]:
    """A group of ``needed`` workers, and a note on its locality (or, with no group, why a
    required locality could not be met)."""
    if len(eligible) < needed:
        return [], None
    if needed == 1:
        return eligible[:1], None
    # Distributed: all nodes must agree on the torch version so collectives interoperate.
    # Then keep the group as close together as possible, so collectives cross the fewest
    # and shortest network links: one network group, else one zone, else one region.
    levels, required = _locality_levels(spec)
    for level in levels:

        def key(w: WorkerView, level: str = level) -> tuple | None:
            loc = locality(w.capabilities, level)
            return None if loc is None else (_torch(w), loc)

        group = _first_full(eligible, needed, key)
        if group:
            return group, f"same {_LEVEL_NAME[level]} {locality(group[0].capabilities, level)}"
    if required:
        level = levels[0]
        counts: dict[str, int] = {}
        for w in eligible:
            loc = locality(w.capabilities, level)
            if loc is not None:
                counts[loc] = counts.get(loc, 0) + 1
        most = max(counts.values(), default=0)
        return [], (
            f"distributed job needs {needed} eligible workers in one {_LEVEL_NAME[level]} "
            f"(group_locality={level}); at most {most} share one"
        )
    return _first_full(eligible, needed, lambda w: (_torch(w),)), None


def ever_schedulable(spec: JobSpec, workers: list[WorkerView], job_id: str | None = None) -> bool:
    """True if enough registered, non-offline workers are *compatible*, regardless of load."""
    ok = [
        w
        for w in workers
        if w.state != WorkerState.OFFLINE
        and (not spec.placement.worker_ids or w.worker_id in spec.placement.worker_ids)
        and not compatibility_reasons(spec, w.capabilities, job_id=job_id)
    ]
    levels, required = _locality_levels(spec)
    if required and spec.num_nodes > 1:
        level = levels[0]
        return bool(
            _first_full(
                ok,
                spec.num_nodes,
                lambda w: (
                    None
                    if locality(w.capabilities, level) is None
                    else (locality(w.capabilities, level),)
                ),
            )
        )
    return len(ok) >= spec.num_nodes
