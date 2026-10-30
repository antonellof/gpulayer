"""Versioned, validated contracts shared by the SDK, CLI, control plane and workers.

Every object that crosses a process or network boundary is defined here. The wire
format is the pydantic JSON representation of these models; ``API_VERSION`` is
carried on job specifications and ``PROTOCOL_VERSION`` on every HTTP request.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

API_VERSION = "gpulayer.dev/v1alpha1"
PROTOCOL_VERSION = "1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, use_enum_values=False)


# --------------------------------------------------------------------------- enums


class ExecutionTarget(str, Enum):
    """Where a job may run. Independent of workload type (see WorkloadType)."""

    LOCAL = "local"  # a worker on this machine / embedded control plane
    REMOTE = "remote"  # an existing worker that joined an authorized control plane
    CLOUD = "cloud"  # a worker obtained through a provider adapter


class WorkloadType(str, Enum):
    INDEPENDENT = "independent"
    DISTRIBUTED = "distributed"


class Accelerator(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    ROCM = "rocm"


class GpuVendor(str, Enum):
    NVIDIA = "nvidia"
    APPLE = "apple"
    AMD = "amd"


class MemoryKind(str, Enum):
    DEDICATED = "dedicated"  # discrete VRAM
    UNIFIED = "unified"  # shared with system RAM (Apple Silicon); NOT equivalent to VRAM


class WorkerOrigin(str, Enum):
    """How a worker came to exist; maps onto the execution target it serves."""

    LOCAL = "local"
    EXISTING = "existing"  # user-operated machine (target=remote)
    PROVIDER = "provider"  # provisioned through a provider adapter (target=cloud)


ORIGIN_TO_TARGET: dict[WorkerOrigin, ExecutionTarget] = {
    WorkerOrigin.LOCAL: ExecutionTarget.LOCAL,
    WorkerOrigin.EXISTING: ExecutionTarget.REMOTE,
    WorkerOrigin.PROVIDER: ExecutionTarget.CLOUD,
}


class JobState(str, Enum):
    QUEUED = "queued"  # validated and persisted, waiting for placement
    PROVISIONING = "provisioning"  # waiting for provider-backed capacity
    SCHEDULED = "scheduled"  # placed on worker(s); assignments not yet started
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_JOB_STATES


TERMINAL_JOB_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT}
)

# Allowed job state transitions. Anything not listed is rejected by the control plane.
JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset(
        {JobState.PROVISIONING, JobState.SCHEDULED, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.PROVISIONING: frozenset(
        {JobState.QUEUED, JobState.SCHEDULED, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.SCHEDULED: frozenset(
        {
            JobState.QUEUED,  # retry after worker loss before start
            JobState.RUNNING,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.SUCCEEDED,  # fast tasks may finish before "started" is observed
            JobState.TIMED_OUT,
        }
    ),
    JobState.RUNNING: frozenset(
        {
            JobState.QUEUED,  # retry after worker loss (independent jobs only)
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMED_OUT,
        }
    ),
}


class TaskState(str, Enum):
    ASSIGNED = "assigned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    LOST = "lost"  # lease expired; the process may still be running (no exactly-once)

    @property
    def terminal(self) -> bool:
        return self not in (TaskState.ASSIGNED, TaskState.RUNNING)


class WorkerState(str, Enum):
    REGISTERING = "registering"
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class SupportLevel(str, Enum):
    """Implementation status vocabulary from CLAUDE.md section 2.5."""

    HARDWARE_VALIDATED = "hardware-validated"
    TESTED = "tested"
    IMPLEMENTED = "implemented"  # execution path exists; not exercised on this hardware
    SIMULATED = "simulated"
    DETECTION_ONLY = "detection-only"
    PLANNED = "planned"
    UNSUPPORTED = "unsupported"


# ----------------------------------------------------------------- resource model

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


# Runtime profiles: named shorthands for common accelerator + framework requirements.
# A profile only fills in requirements; workers are still matched on what they report.
RUNTIME_PROFILES: dict[str, dict[str, Any]] = {
    "pytorch-cpu": {"accelerator": "cpu", "frameworks": {"torch": ">=2.1"}},
    "pytorch-cuda": {
        "accelerator": "cuda",
        "frameworks": {"torch": ">=2.1", "torch.cuda": ">=0"},
    },
    "pytorch-mps": {"accelerator": "mps", "frameworks": {"torch": ">=2.1", "torch.mps": ">=0"}},
    "jax-cuda": {"accelerator": "cuda", "frameworks": {"jax": ">=0.4", "jaxlib": ">=0.4"}},
    "cupy-cuda": {"accelerator": "cuda", "frameworks": {"cupy": ">=12"}},
    "mlx": {"accelerator": "mps", "frameworks": {"mlx": ">=0.10"}},
}


class ResourceRequest(_Model):
    """Hardware/runtime requirements for ONE execution unit (one worker / node)."""

    profile: str | None = Field(
        None,
        description="Runtime profile, e.g. 'pytorch-cuda'. Fills in accelerator and frameworks; "
        "explicit values take precedence.",
    )
    accelerator: Accelerator = Accelerator.CPU
    gpu_count: int = Field(0, ge=0, le=64)
    gpu_memory_gb: float | None = Field(None, gt=0, description="Per-device memory floor.")
    gpu_memory_kind: MemoryKind | None = Field(
        None, description="Require dedicated or unified memory. None accepts either."
    )
    min_cuda_compute_capability: str | None = Field(None, pattern=r"^\d+\.\d+$")
    cpu_cores: float = Field(1, gt=0)
    memory_gb: float | None = Field(None, gt=0)
    disk_gb: float | None = Field(None, gt=0)
    os: list[str] = Field(default_factory=list, description="e.g. ['linux', 'darwin']")
    cpu_arch: list[str] = Field(default_factory=list, description="e.g. ['x86_64', 'arm64']")
    frameworks: dict[str, str] = Field(
        default_factory=dict,
        description="Framework -> PEP 440 specifier, e.g. {'torch': '>=2.1'}.",
    )
    python: str | None = Field(None, description="PEP 440 specifier for the worker Python.")

    @model_validator(mode="before")
    @classmethod
    def _apply_profile(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not data.get("profile"):
            return data
        name = data["profile"]
        if name not in RUNTIME_PROFILES:
            raise ValueError(f"unknown profile {name!r}; known: {sorted(RUNTIME_PROFILES)}")
        prof = RUNTIME_PROFILES[name]
        out = dict(data)
        given = out.get("accelerator")
        if given is not None and str(getattr(given, "value", given)) != prof["accelerator"]:
            raise ValueError(
                f"profile {name!r} needs accelerator {prof['accelerator']!r}, got {given!r}"
            )
        out["accelerator"] = prof["accelerator"]
        out["frameworks"] = {**prof["frameworks"], **(out.get("frameworks") or {})}
        return out

    @model_validator(mode="after")
    def _check_accelerator(self) -> ResourceRequest:
        if self.accelerator == Accelerator.CPU:
            if self.gpu_count:
                raise ValueError("gpu_count must be 0 when accelerator is 'cpu'")
            if self.gpu_memory_gb is not None:
                raise ValueError("gpu_memory_gb requires a GPU accelerator")
        elif self.gpu_count == 0:
            self.gpu_count = 1
        if self.min_cuda_compute_capability and self.accelerator != Accelerator.CUDA:
            raise ValueError("min_cuda_compute_capability only applies to accelerator 'cuda'")
        return self


class ProvisioningPolicy(_Model):
    """Allows the control plane to obtain capacity through a provider adapter."""

    provider: str
    release: Literal["terminate_on_completion", "retain"] = "terminate_on_completion"
    provisioning_timeout_seconds: float = Field(900, gt=0)


class PlacementPolicy(_Model):
    target: ExecutionTarget = ExecutionTarget.LOCAL
    fallback_targets: list[ExecutionTarget] = Field(
        default_factory=list,
        description="Explicitly allowed additional targets, tried in order after `target`.",
    )
    worker_ids: list[str] = Field(default_factory=list)
    worker_labels: dict[str, str] = Field(default_factory=dict)
    providers: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    # spread: least-loaded worker first (keeps machines evenly busy and leaves room for
    # distributed groups); first_fit: lowest worker id with a free slot (packs work).
    strategy: Literal["spread", "first_fit", "most_free_memory"] = "spread"
    on_unschedulable: Literal["queue", "fail"] = "queue"
    max_hourly_cost: float | None = Field(
        None, gt=0, description="Reject offers whose estimated hourly cost exceeds this."
    )
    currency: str = "USD"
    provisioning: ProvisioningPolicy | None = None

    @model_validator(mode="after")
    def _check(self) -> PlacementPolicy:
        if self.target in self.fallback_targets:
            raise ValueError("fallback_targets must not repeat the primary target")
        if self.provisioning and ExecutionTarget.CLOUD not in self.allowed_targets:
            raise ValueError("provisioning requires target or fallback 'cloud'")
        return self

    @property
    def allowed_targets(self) -> list[ExecutionTarget]:
        return [self.target, *self.fallback_targets]


RetryReason = Literal["worker_lost", "failed"]
_WORKER_LOST: RetryReason = "worker_lost"


class RetryPolicy(_Model):
    """Retries are opt-in. A retry may duplicate side effects (no exactly-once)."""

    max_attempts: int = Field(1, ge=1, le=10)
    retry_on: list[RetryReason] = Field(default_factory=lambda: [_WORKER_LOST])


class SecretRef(_Model):
    """A reference resolved on the worker; the value never enters the job record.

    ``source='env'`` reads ``key`` from the worker process environment.
    """

    name: str = Field(description="Environment variable exposed to the workload.")
    source: Literal["env"] = "env"
    key: str

    @field_validator("name", "key")
    @classmethod
    def _env_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
            raise ValueError(f"invalid environment variable name: {v!r}")
        return v


class ArtifactKind(str, Enum):
    SOURCE = "source"
    INPUT = "input"
    OUTPUT = "output"
    LOG = "log"
    CHECKPOINT = "checkpoint"


class ArtifactReference(_Model):
    """Content-addressed artifact. ``uri`` is resolvable by any authorized worker."""

    uri: str = Field(pattern=r"^artifact://sha256/[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    name: str = ""
    kind: ArtifactKind = ArtifactKind.INPUT

    @model_validator(mode="after")
    def _consistent(self) -> ArtifactReference:
        if not self.uri.endswith(self.sha256):
            raise ValueError("artifact uri and sha256 disagree")
        return self


def _safe_relpath(v: str) -> str:
    if not v or v.startswith(("/", "\\")) or ".." in re.split(r"[\\/]", v) or ":" in v:
        raise ValueError(f"path must be relative and stay inside the workspace: {v!r}")
    return v


class InputArtifact(_Model):
    artifact: ArtifactReference
    path: str = Field(description="Relative path under $GPULAYER_INPUT_DIR.")

    _p = field_validator("path")(classmethod(lambda cls, v: _safe_relpath(v)))


# ------------------------------------------------------------------- workloads


class CommandJob(_Model):
    """Run an executable. With ``source`` set, it runs inside the extracted bundle."""

    kind: Literal["command"] = "command"
    command: list[str] = Field(min_length=1)
    source: ArtifactReference | None = None


class FunctionJob(_Model):
    """Call ``module:function`` from a packaged source bundle.

    Arguments and return value must be JSON-serializable. Closures, lambdas and
    pickled objects are deliberately unsupported (see ADR-0002).
    """

    kind: Literal["function"] = "function"
    entrypoint: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    source: ArtifactReference | None = Field(
        None, description="Bundle added to sys.path. Without it, the module must be installed."
    )


class RendezvousSpec(_Model):
    backend: Literal["static"] = Field(
        "static", description="torchrun static rendezvous via MASTER_ADDR/MASTER_PORT of rank 0."
    )
    port: int | None = Field(None, ge=1024, le=65535, description="None: rank-0 worker picks.")
    join_timeout_seconds: float = Field(300, gt=0)


class DistributedJob(_Model):
    """ONE logical workload spanning ``num_workers`` nodes.

    Give exactly one of ``entrypoint`` (a script; ``args`` are its CLI arguments) or
    ``function`` (``module:function`` called on every rank with JSON ``function_args``;
    rank 0's JSON return value becomes the job's return value).
    """

    kind: Literal["distributed"] = "distributed"
    launcher: Literal["direct", "torchrun"] = Field(
        "direct",
        description="direct: GPUlayer starts the ranks with the torchrun environment (faster "
        "start-up and exit); torchrun: start them through torch.distributed.run.",
    )
    entrypoint: str | None = Field(
        None, description="Script path relative to the source bundle root."
    )
    args: list[str] = Field(default_factory=list)
    function: str | None = Field(None, pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    function_args: list[Any] = Field(default_factory=list)
    function_kwargs: dict[str, Any] = Field(default_factory=dict)
    source: ArtifactReference | None = None
    num_workers: int = Field(1, ge=1, le=1024)
    procs_per_worker: int = Field(1, ge=1, le=64)
    comm_backend: Literal["gloo", "nccl"] = "gloo"
    rendezvous: RendezvousSpec = Field(default_factory=RendezvousSpec)
    failure_policy: Literal["fail_fast"] = Field(
        "fail_fast", description="Any rank/node failure fails the logical job."
    )

    _p = field_validator("entrypoint")(
        classmethod(lambda cls, v: v if v is None else _safe_relpath(v))
    )

    @model_validator(mode="after")
    def _one_entry(self) -> DistributedJob:
        if (self.entrypoint is None) == (self.function is None):
            raise ValueError("give exactly one of 'entrypoint' (script) or 'function'")
        if self.function is None and (self.function_args or self.function_kwargs):
            raise ValueError("function_args/function_kwargs require 'function'")
        return self

    @property
    def world_size(self) -> int:
        return self.num_workers * self.procs_per_worker


Workload = Annotated[CommandJob | FunctionJob | DistributedJob, Field(discriminator="kind")]


class JobSpec(_Model):
    api_version: Literal["gpulayer.dev/v1alpha1"] = "gpulayer.dev/v1alpha1"
    name: str = "job"
    workload: Workload
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    placement: PlacementPolicy = Field(default_factory=PlacementPolicy)
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[SecretRef] = Field(default_factory=list)
    inputs: list[InputArtifact] = Field(default_factory=list)
    timeout_seconds: float | None = Field(None, gt=0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
        return v

    @field_validator("env")
    @classmethod
    def _env(cls, v: dict[str, str]) -> dict[str, str]:
        for k in v:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise ValueError(f"invalid environment variable name: {k!r}")
            if k.startswith("GPULAYER_") or k in _RESERVED_ENV:
                raise ValueError(f"environment variable {k!r} is reserved")
        return v

    @model_validator(mode="after")
    def _check(self) -> JobSpec:
        w = self.workload
        if isinstance(w, DistributedJob):
            if self.retry.max_attempts > 1:
                raise ValueError(
                    "automatic retry of distributed jobs is not supported: a partially "
                    "executed worker group is not safe to replay"
                )
            if self.resources.accelerator == Accelerator.MPS:
                raise ValueError("distributed execution on MPS is unsupported (ADR-0006)")
            if w.comm_backend == "nccl" and self.resources.accelerator != Accelerator.CUDA:
                raise ValueError("comm_backend 'nccl' requires accelerator 'cuda'")
            if (
                self.resources.accelerator != Accelerator.CPU
                and w.procs_per_worker > self.resources.gpu_count
            ):
                raise ValueError("procs_per_worker cannot exceed gpu_count per worker")
        names = [s.name for s in self.secrets]
        if len(names) != len(set(names)) or set(names) & set(self.env):
            raise ValueError("secret names must be unique and must not collide with env")
        return self

    @property
    def workload_type(self) -> WorkloadType:
        if isinstance(self.workload, DistributedJob):
            return WorkloadType.DISTRIBUTED
        return WorkloadType.INDEPENDENT

    @property
    def num_nodes(self) -> int:
        return self.workload.num_workers if isinstance(self.workload, DistributedJob) else 1


_RESERVED_ENV = {"MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "PYTHONPATH"}


# ------------------------------------------------------------- worker contracts


class AcceleratorDevice(_Model):
    index: int
    accelerator: Accelerator
    vendor: GpuVendor
    name: str
    memory_gb: float | None = Field(None, description="None when not reliably known.")
    memory_kind: MemoryKind = MemoryKind.DEDICATED
    compute_capability: str | None = None
    driver_version: str | None = None
    execution_supported: bool = Field(
        True, description="False when detected but GPUlayer has no validated execution path."
    )


class WorkerCapabilities(_Model):
    """What a worker CAN support. Availability lives in WorkerStatus."""

    hostname: str
    os: str
    cpu_arch: str
    cpu_cores: int
    memory_gb: float | None = None
    disk_free_gb: float | None = None
    python_version: str
    accelerators: list[AcceleratorDevice] = Field(default_factory=list)
    frameworks: dict[str, str] = Field(default_factory=dict)
    distributed_backends: list[str] = Field(default_factory=list)
    launchers: list[str] = Field(default_factory=list)
    advertise_address: str | None = Field(
        None, description="Address other workers use for rendezvous. None: not reachable."
    )
    origin: WorkerOrigin = WorkerOrigin.LOCAL
    provider: str | None = None
    instance_id: str | None = None
    region: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    max_concurrent_tasks: int = Field(1, ge=1)
    notes: list[str] = Field(
        default_factory=list, description="Detection uncertainty, reported instead of guessed."
    )

    @property
    def target(self) -> ExecutionTarget:
        return ORIGIN_TO_TARGET[self.origin]

    def devices(self, accelerator: Accelerator) -> list[AcceleratorDevice]:
        return [d for d in self.accelerators if d.accelerator == accelerator]


class WorkerStatus(_Model):
    worker_id: str
    name: str
    state: WorkerState
    capabilities: WorkerCapabilities
    registered_at: datetime
    last_heartbeat_at: datetime | None = None
    running_task_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------- job observation


class CandidateEvaluation(_Model):
    worker_id: str
    eligible: bool
    reasons: list[str] = Field(default_factory=list)


class SchedulingDecision(_Model):
    """Explainable result of one scheduling pass for one job."""

    placed: bool
    target: ExecutionTarget | None = None
    worker_ids: list[str] = Field(default_factory=list)
    summary: str
    candidates: list[CandidateEvaluation] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=utcnow)


class TaskStatus(_Model):
    task_id: str
    job_id: str
    attempt: int
    node_rank: int = 0
    worker_id: str | None
    state: TaskState
    exit_code: int | None = None
    error: str | None = None
    lease_expires_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class CostRecord(_Model):
    """Never present an estimate as a charge: ``kind`` makes the difference explicit."""

    kind: Literal["estimate", "provider_usage", "provider_charge", "internal"]
    amount: float
    currency: str
    unit: Literal["per_hour", "total"]
    source: str
    recorded_at: datetime = Field(default_factory=utcnow)
    note: str = ""


class JobResult(_Model):
    job_id: str
    state: JobState
    exit_code: int | None = None
    return_value: Any = None
    error: str | None = None
    outputs: list[ArtifactReference] = Field(default_factory=list)
    task_results: list[TaskStatus] = Field(default_factory=list)
    costs: list[CostRecord] = Field(default_factory=list)


class JobStatus(_Model):
    job_id: str
    name: str
    state: JobState
    workload_type: WorkloadType
    spec: JobSpec
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    cancel_requested: bool = False
    submitted_by: str | None = Field(None, description="Principal that submitted the job.")
    error: str | None = None
    scheduling: SchedulingDecision | None = None
    tasks: list[TaskStatus] = Field(default_factory=list)


class ExecutionEvent(_Model):
    seq: int
    job_id: str | None = None
    task_id: str | None = None
    worker_id: str | None = None
    timestamp: datetime
    type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class LogLine(_Model):
    seq: int
    task_id: str
    node_rank: int = 0
    stream: Literal["stdout", "stderr", "system"]
    line: str
    timestamp: datetime
    # Global rank of the process that wrote the line (distributed jobs); None otherwise.
    rank: int | None = None

    @property
    def origin(self) -> str:
        """Short label for multi-process output: ``rank3`` when known, else ``node1``."""
        return f"rank{self.rank}" if self.rank is not None else f"node{self.node_rank}"


# ------------------------------------------------------------ worker protocol


class RendezvousInfo(_Model):
    master_addr: str
    master_port: int
    num_nodes: int
    procs_per_node: int
    node_rank: int

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.procs_per_node


class TaskAssignment(_Model):
    """Handed to exactly one worker. Rank > 0 assignments carry rendezvous info."""

    task_id: str
    job_id: str
    attempt: int
    spec: JobSpec
    node_rank: int = 0
    lease_expires_at: datetime
    rendezvous: RendezvousInfo | None = None
    # Indices of the accelerator devices reserved for this task on the worker (for example
    # CUDA_VISIBLE_DEVICES). None from control planes that do not assign devices.
    devices: list[int] | None = None


class TaskOutcome(_Model):
    state: TaskState
    exit_code: int | None = None
    error: str | None = None
    return_value: Any = None
    outputs: list[ArtifactReference] = Field(default_factory=list)

    @field_validator("state")
    @classmethod
    def _terminal(cls, v: TaskState) -> TaskState:
        if not v.terminal or v == TaskState.LOST:
            raise ValueError("a worker may only report succeeded/failed/cancelled/timed_out")
        return v


class GpuMetrics(_Model):
    index: int
    utilization_pct: float | None = Field(None, ge=0, le=100)
    memory_used_gb: float | None = Field(None, ge=0)
    memory_total_gb: float | None = Field(None, ge=0)


class WorkerMetrics(_Model):
    """Resource sample a worker sends with some heartbeats. Every field is optional:
    workers report what their platform exposes without extra privileges."""

    load1: float | None = Field(None, ge=0)  # 1-minute load average
    cpu_count: int | None = Field(None, ge=1)
    memory_used_gb: float | None = Field(None, ge=0)
    memory_total_gb: float | None = Field(None, ge=0)
    processes: int | None = Field(None, ge=0)  # OS processes belonging to running tasks
    running_tasks: int | None = Field(None, ge=0)
    gpus: list[GpuMetrics] = Field(default_factory=list, max_length=64)


class HeartbeatResponse(_Model):
    lease_ttl_seconds: float
    cancel_task_ids: list[str] = Field(default_factory=list)
    revoked: bool = False


class Registration(_Model):
    worker_id: str
    session_token: str = Field(repr=False)
    heartbeat_interval_seconds: float
    lease_ttl_seconds: float
