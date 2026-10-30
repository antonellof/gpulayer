"""GPUlayer: one job model, one capability system, one scheduling layer, many environments."""

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
from gpulayer.sdk.remote import RemoteError, RemoteFunction, connect, default_client, remote

__version__ = "0.3.0"

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
    "connect",
    "default_client",
    "distributed",
    "function",
    "remote",
]


def __getattr__(name: str) -> object:
    # `gl.pytorch` is imported lazily so CPU-only installs without torch keep working.
    if name == "pytorch":
        import importlib

        return importlib.import_module("gpulayer.pytorch")
    raise AttributeError(f"module 'gpulayer' has no attribute {name!r}")
