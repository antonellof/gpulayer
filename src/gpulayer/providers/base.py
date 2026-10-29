"""Provider adapter interface.

Adapters translate GPUlayer's capability model into an infrastructure provider's API.
They expose offers as ``WorkerCapabilities`` templates, so the core scheduler never
sees provider-specific instance names. Each adapter declares what it actually supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from gpulayer.contracts import WorkerCapabilities, utcnow


class PriceQuote(BaseModel):
    """A price *estimate* as published by a source. Never an actual charge."""

    model_config = ConfigDict(extra="forbid")
    amount: float = Field(ge=0)
    currency: str
    unit: str = "per_hour"
    source: str = Field(description="Where the number came from, e.g. provider pricing API.")
    as_of: datetime = Field(default_factory=utcnow)


class InstanceOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    offer_id: str
    provider: str
    instance_type: str = Field(description="Provider-specific name; metadata only.")
    region: str | None = None
    capabilities: WorkerCapabilities = Field(
        description="Expected worker capabilities. Verified again when the worker registers."
    )
    price: PriceQuote | None = Field(None, description="None when no reliable price exists.")


class InstanceState(str, Enum):
    REQUESTED = "requested"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_instance_id: str
    provider: str
    offer_id: str
    state: InstanceState
    error: str | None = None


class WorkerBootstrap(BaseModel):
    """What a provisioned machine needs to join the control plane.

    ``join_token`` is single-use and bound to ``instance_ref``; adapters must deliver it
    through the provider's secret/user-data mechanism and must never log it.
    """

    model_config = ConfigDict(extra="forbid")
    control_plane_url: str | None
    instance_ref: str
    join_token: str = Field(repr=False)
    labels: dict[str, str] = Field(default_factory=dict)


class ProviderFeatures(BaseModel):
    list_offers: bool = False
    pricing: bool = False
    provision: bool = False
    terminate: bool = False
    worker_bootstrap: bool = False


class ProviderError(Exception):
    retryable = False


class CapacityUnavailable(ProviderError):
    retryable = True


class QuotaExceeded(ProviderError):
    pass


class ProviderAdapter(ABC):
    name: str
    simulated: bool = False

    @abstractmethod
    def features(self) -> ProviderFeatures: ...

    @abstractmethod
    def list_offers(self) -> list[InstanceOffer]: ...

    @abstractmethod
    def provision(self, offer_id: str, bootstrap: WorkerBootstrap) -> Instance: ...

    @abstractmethod
    def get_instance(self, provider_instance_id: str) -> Instance: ...

    @abstractmethod
    def terminate(self, provider_instance_id: str) -> Instance:
        """Idempotent: terminating a terminated instance succeeds."""
