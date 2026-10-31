"""Convenience constructors for JobSpec. They only build validated contracts."""

from __future__ import annotations

from typing import Any

from gpulayer.contracts import (
    Accelerator,
    ArtifactReference,
    CommandJob,
    DistributedJob,
    EnvironmentSpec,
    ExecutionTarget,
    FunctionJob,
    JobSpec,
    PlacementPolicy,
    ResourceRequest,
    SecretRef,
    ServiceJob,
)


def pop_environment(kw: dict[str, Any]) -> EnvironmentSpec | None:
    """Turn ``requirements=[...]`` / ``requirements_file=...`` options into an
    EnvironmentSpec (ADR-0022), removing them from ``kw``."""
    reqs = kw.pop("requirements", None)
    req_file = kw.pop("requirements_file", None)
    if isinstance(reqs, str):
        reqs = [reqs]
    if not reqs and not req_file:
        return kw.pop("environment", None)
    return EnvironmentSpec(requirements=list(reqs or []), requirements_file=req_file)


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
    environment = pop_environment(kw)
    if environment is not None:
        kw["environment"] = environment
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


def service(
    argv: list[str],
    *,
    replicas: int = 1,
    port: bool = True,
    restart: str = "always",
    max_restarts: int | None = None,
    name: str = "service",
    target: str = "local",
    source: ArtifactReference | None = None,
    resources: ResourceRequest | None = None,
    placement: PlacementPolicy | None = None,
    accelerator: str | None = None,
    gpus: int | None = None,
    **kw: Any,
) -> JobSpec:
    """A long-running service (ADR-0019): ``replicas`` copies of ``argv`` that stay up. With
    ``port`` each gets ``$PORT``; ``Job.endpoints()`` lists where they listen."""
    return _spec(
        ServiceJob(
            command=argv,
            source=source,
            replicas=replicas,
            port=port,
            restart=restart,
            max_restarts=max_restarts,
        ),
        name=name,
        target=target,
        resources=resources,
        placement=placement,
        accelerator=accelerator,
        gpus=gpus,
        **kw,
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
    launcher: str = "direct",
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
            launcher=launcher,
        ),
        name=name,
        target=target,
        resources=resources,
        placement=placement,
        accelerator=accelerator,
        gpus=gpus,
        **kw,
    )
