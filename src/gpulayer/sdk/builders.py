"""Convenience constructors for JobSpec. They only build validated contracts."""

from __future__ import annotations

from typing import Any

from gpulayer.contracts import (
    Accelerator,
    ArtifactReference,
    CommandJob,
    DistributedJob,
    ExecutionTarget,
    FunctionJob,
    JobSpec,
    PlacementPolicy,
    ResourceRequest,
    SecretRef,
)


def _spec(
    workload: Any,
    *,
    name: str,
    target: str | ExecutionTarget,
    resources: ResourceRequest | None,
    placement: PlacementPolicy | None,
    accelerator: str | Accelerator | None,
    gpus: int | None,
    **kw: Any,
) -> JobSpec:
    if resources is None:
        resources = ResourceRequest(
            accelerator=Accelerator(accelerator or "cpu"),
            gpu_count=gpus or 0,
        )
    if placement is None:
        placement = PlacementPolicy(target=ExecutionTarget(target))
    secrets = kw.pop("secrets", None) or []
    secrets = [s if isinstance(s, SecretRef) else SecretRef(**s) for s in secrets]
    return JobSpec(
        name=name,
        workload=workload,
        resources=resources,
        placement=placement,
        secrets=secrets,
        **{k: v for k, v in kw.items() if v is not None},
    )


def command(
    argv: list[str],
    *,
    name: str = "command",
    target: str = "local",
    source: ArtifactReference | None = None,
    resources: ResourceRequest | None = None,
    placement: PlacementPolicy | None = None,
    accelerator: str | None = None,
    gpus: int | None = None,
    **kw: Any,
) -> JobSpec:
    return _spec(
        CommandJob(command=argv, source=source),
        name=name,
        target=target,
        resources=resources,
        placement=placement,
        accelerator=accelerator,
        gpus=gpus,
        **kw,
    )


def function(
    entrypoint: str,
    *args: Any,
    source: ArtifactReference | None = None,
    kwargs: dict[str, Any] | None = None,
    name: str = "function",
    target: str = "local",
    resources: ResourceRequest | None = None,
    placement: PlacementPolicy | None = None,
    accelerator: str | None = None,
    gpus: int | None = None,
    **kw: Any,
) -> JobSpec:
    return _spec(
        FunctionJob(entrypoint=entrypoint, args=list(args), kwargs=kwargs or {}, source=source),
        name=name,
        target=target,
        resources=resources,
        placement=placement,
        accelerator=accelerator,
        gpus=gpus,
        **kw,
    )


def distributed(
    entrypoint: str,
    *,
    source: ArtifactReference,
    num_workers: int = 1,
    procs_per_worker: int = 1,
    args: list[str] | None = None,
    comm_backend: str = "gloo",
    name: str = "distributed",
    target: str = "local",
    resources: ResourceRequest | None = None,
    placement: PlacementPolicy | None = None,
    accelerator: str | None = None,
    gpus: int | None = None,
    **kw: Any,
) -> JobSpec:
    """ONE coordinated workload across ``num_workers`` nodes (not a batch of jobs)."""
    return _spec(
        DistributedJob(
            entrypoint=entrypoint,
            source=source,
            num_workers=num_workers,
            procs_per_worker=procs_per_worker,
            args=args or [],
            comm_backend=comm_backend,
        ),
        name=name,
        target=target,
        resources=resources,
        placement=placement,
        accelerator=accelerator,
        gpus=gpus,
        **kw,
    )
