"""GpuLayer: one job model, one capability system, one scheduling layer, many environments."""

from gpulayer.contracts import (
    API_VERSION,
    Accelerator,
    ExecutionTarget,
    JobSpec,
    JobState,
    PlacementPolicy,
    ResourceRequest,
)
from gpulayer.sdk.builders import command, distributed, function
from gpulayer.sdk.client import Client, Job
from gpulayer.sdk.remote import RemoteError, RemoteFunction, default_client, remote

__version__ = "0.1.0.dev0"

__all__ = [
    "API_VERSION",
    "Accelerator",
    "Client",
    "ExecutionTarget",
    "Job",
    "JobSpec",
    "JobState",
    "PlacementPolicy",
    "RemoteError",
    "RemoteFunction",
    "ResourceRequest",
    "command",
    "default_client",
    "distributed",
    "function",
    "remote",
]
