"""``@gl.remote``: run a top-level Python function on any compatible worker.

    @gl.remote(accelerator="mps", frameworks={"torch": ">=2.0"})
    def train(lr: float) -> dict: ...

    train(0.01)                         # plain local call, no GPUlayer involved
    train.run(0.01)                     # placed on a compatible worker; logs streamed
    train.map([(0.1,), (0.01,)])        # independent jobs in parallel (a sweep)
    train.options(target="remote").run(0.01)

    @gl.remote(nodes=2, procs_per_node=2)   # one torchrun job across a worker group
    def train_ddp() -> dict: ...

The function is shipped as SOURCE (the directory of its file) and called by name on the
worker; nothing is pickled (ADR-0002). Arguments and return value must be JSON.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import re
import sys
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from gpulayer.contracts import (
    RUNTIME_PROFILES,
    ArtifactReference,
    DistributedJob,
    FunctionJob,
    JobResult,
    JobSpec,
    JobState,
    JobStatus,
    PlacementPolicy,
    ResourceRequest,
)
from gpulayer.errors import GpuLayerError, ValidationFailed
from gpulayer.sdk.client import Client, Job

_default_client: Client | None = None
_default_lock = threading.Lock()


def connect(
    url: str, token: str, *, fingerprint: str | None = None, ca_file: str | None = None
) -> Client:
    """Use a GPUlayer server for everything in this process (Trainer, launch, @gl.remote).

    Copy the arguments from what ``gpulayer up`` prints.
    """
    global _default_client
    client = Client(url=url, token=token, fingerprint=fingerprint, ca_file=ca_file)
    client.workers()  # fail now, not on the first job, if the URL or credentials are wrong
    with _default_lock:
        if _default_client is not None:
            _default_client.close()
        _default_client = client
    return client


def default_client() -> Client:
    """Process-wide client: set by ``connect()``, else $GPULAYER_URL, else embedded."""
    global _default_client
    with _default_lock:
        if _default_client is None:
            _default_client = Client()
            atexit.register(_default_client.close)
        return _default_client


class RemoteError(GpuLayerError):
    """A remote run did not succeed. ``result`` and ``status`` carry the details."""

    def __init__(self, message: str, job: Job) -> None:
        super().__init__(message)
        self.job = job


def _function_ref(fn: Callable[..., Any]) -> tuple[str, Path]:
    """Return ('module:qualname', source_dir) for a top-level function defined in a file."""
    if not inspect.isfunction(fn):
        raise ValidationFailed("@gl.remote only supports plain functions")
    if "<locals>" in fn.__qualname__ or "." in fn.__qualname__ or fn.__name__ == "<lambda>":
        raise ValidationFailed(
            f"{fn.__qualname__} must be defined at module top level: GPUlayer ships source "
            "code and calls functions by name, it does not pickle closures (ADR-0002)"
        )
    try:
        path = Path(inspect.getfile(fn)).resolve()
    except TypeError:
        raise ValidationFailed(f"{fn.__name__} has no source file") from None
    module = fn.__module__
    if module == "__main__":
        module = path.stem
        if not module.isidentifier():
            raise ValidationFailed(f"script name {path.name!r} is not an importable module name")
    # Package-relative modules (pkg.sub.mod) need the package root on sys.path.
    root = path.parent
    for _ in range(module.count(".")):
        root = root.parent
    return f"{module}:{fn.__name__}", root


class RemoteFunction:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        client: Client | None = None,
        source: str | Path | None = None,
        nodes: int = 1,
        procs_per_node: int = 1,
        comm_backend: str | None = None,
        **spec_options: Any,
    ) -> None:
        self.fn = fn
        functools.update_wrapper(self, fn)
        self.ref, default_root = _function_ref(fn)
        self.source_dir = Path(source) if source else default_root
        self._client = client
        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.comm_backend = comm_backend
        self.spec_options = spec_options
        self._bundle: ArtifactReference | None = None
        if nodes < 1 or procs_per_node < 1:
            raise ValidationFailed("nodes and procs_per_node must be >= 1")

    # Calling the object directly is a plain local call: debug with your normal tools.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    @property
    def distributed(self) -> bool:
        return self.nodes > 1 or self.procs_per_node > 1

    @property
    def client(self) -> Client:
        return self._client or default_client()

    def options(self, **overrides: Any) -> RemoteFunction:
        """A copy with different placement/resources, e.g. ``.options(target="remote")``."""
        merged = {
            "client": self._client,
            "source": self.source_dir,
            "nodes": self.nodes,
            "procs_per_node": self.procs_per_node,
            "comm_backend": self.comm_backend,
            **self.spec_options,
            **overrides,
        }
        return RemoteFunction(self.fn, **merged)

    # ------------------------------------------------------------------ spec

    def spec(self, *args: Any, **kwargs: Any) -> JobSpec:
        opts = dict(self.spec_options)
        resources = opts.pop("resources", None)
        placement = opts.pop("placement", None)
        profile = opts.pop("profile", None)
        accelerator = opts.pop("accelerator", None)
        if accelerator is None:
            accelerator = (
                RUNTIME_PROFILES[profile]["accelerator"] if profile in RUNTIME_PROFILES else "cpu"
            )
        if resources is None:
            gpus = opts.pop("gpus", None)
            if gpus is None and accelerator != "cpu" and self.distributed:
                gpus = self.procs_per_node
            resources = ResourceRequest(
                profile=profile,
                accelerator=accelerator,
                gpu_count=gpus or 0,
                gpu_memory_gb=opts.pop("gpu_memory_gb", None),
                memory_gb=opts.pop("memory_gb", None),
                cpu_cores=opts.pop("cpu_cores", 1),
                frameworks=opts.pop("frameworks", {}),
                python=opts.pop("python", None),
            )
        if placement is None:
            placement = PlacementPolicy(
                target=opts.pop("target", None) or default_target(self.client),
                fallback_targets=opts.pop("fallback_targets", []),
                worker_labels=opts.pop("worker_labels", {}),
                on_unschedulable=opts.pop("on_unschedulable", "fail"),
                group_locality=opts.pop("group_locality", "prefer"),
                provisioning=opts.pop("provisioning", None),
            )
        name = opts.pop("name", self.fn.__name__.replace("_", "-")[:63])
        source = self._source()
        if self.distributed:
            backend = self.comm_backend or (
                "nccl" if resources.accelerator.value in ("cuda", "rocm") else "gloo"
            )
            workload: Any = DistributedJob(
                function=self.ref,
                function_args=list(args),
                function_kwargs=kwargs,
                source=source,
                num_workers=self.nodes,
                procs_per_worker=self.procs_per_node,
                comm_backend=backend,
            )
        else:
            workload = FunctionJob(
                entrypoint=self.ref, args=list(args), kwargs=kwargs, source=source
            )
        from gpulayer.sdk.builders import pop_environment

        environment = pop_environment(opts)
        return JobSpec(
            name=name,
            workload=workload,
            resources=resources,
            placement=placement,
            environment=environment,
            **opts,
        )

    def _source(self) -> ArtifactReference:
        if self._bundle is None:
            self._bundle = self.client.package(self.source_dir)
        return self._bundle

    # --------------------------------------------------------------- running

    def submit(self, *args: Any, **kwargs: Any) -> Job:
        return self.client.submit(self.spec(*args, **kwargs))

    def run(self, *args: Any, stream_logs: bool = True, **kwargs: Any) -> Any:
        """Run remotely, stream logs, and return the function's return value."""
        job = self.submit(*args, **kwargs)
        if stream_logs:
            for line in job.follow_logs():
                if line.stream != "system":
                    prefix = f"[{line.origin}] " if self.distributed else ""
                    print(
                        prefix + line.line,
                        file=sys.stderr if line.stream == "stderr" else sys.stdout,
                        flush=True,
                    )
        else:
            job.wait()
        return _value_or_raise(job)

    def map(
        self, arg_tuples: Iterable[tuple[Any, ...] | Any], *, parallelism: int | None = None
    ) -> list[Any]:
        """Run independent jobs, one per argument tuple, and return results in order.

        In embedded mode this starts ``parallelism`` local workers (default: up to 4).
        """
        calls = [a if isinstance(a, tuple) else (a,) for a in arg_tuples]
        if not calls:
            return []
        c = self.client
        if c.mode == "embedded":
            c.ensure_local_workers(parallelism or min(len(calls), 4))
        jobs = [self.submit(*a) for a in calls]
        return [_value_or_raise(j, wait=True) for j in jobs]


def default_target(client: Client) -> str:
    """Jobs go to the server's workers when connected to one, else to this machine."""
    return "remote" if client.mode == "remote" else "local"


def _value_or_raise(job: Job, *, wait: bool = False) -> Any:
    if wait:
        job.wait()
    result: JobResult = job.result()
    if result.state == JobState.SUCCEEDED:
        return result.return_value
    status = job.status()
    raise RemoteError(f"job {job.id} {result.state.value}: {explain_failure(status)}", job)


def explain_failure(status: JobStatus) -> str:
    """The job's error plus, when it was never placed, why each worker was rejected."""
    detail = status.error or status.state.value
    if status.scheduling and not status.scheduling.placed:
        reasons = "; ".join(
            f"{c.worker_id}: {', '.join(c.reasons)}"
            for c in status.scheduling.candidates
            if c.reasons
        )
        detail += f" ({reasons})" if reasons else ""
        missing = sorted(
            {
                m.group(1)
                for c in status.scheduling.candidates
                for r in c.reasons
                if (m := re.match(r"framework '([^']+)' not installed", r))
            }
        )
        if missing:
            detail += (
                f". Install {', '.join(missing)} in the Python that runs jobs on a worker "
                f"(for local jobs, the Python running this script: "
                f"`python -m pip install {' '.join(missing)}`), then start the worker again."
            )
    return detail


def remote(fn: Callable[..., Any] | None = None, **options: Any) -> Any:
    """Decorator. Usable bare (``@gl.remote``) or with options (``@gl.remote(...)``)."""
    if fn is not None:
        return RemoteFunction(fn, **options)
    return lambda f: RemoteFunction(f, **options)
