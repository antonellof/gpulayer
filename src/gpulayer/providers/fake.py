"""Deterministic in-memory provider for tests and demos. SIMULATED: provisions nothing.

``on_provision`` lets a test play the role of the machine's boot script, e.g. by
starting an in-process worker agent that registers with the bootstrap token.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from gpulayer.providers.base import (
    CapacityUnavailable,
    Instance,
    InstanceOffer,
    InstanceState,
    ProviderAdapter,
    ProviderError,
    ProviderFeatures,
    WorkerBootstrap,
)


class FakeProvider(ProviderAdapter):
    simulated = True

    def __init__(
        self,
        offers: list[InstanceOffer],
        *,
        name: str = "fake",
        capacity: int | None = None,
        on_provision: Callable[[Instance, WorkerBootstrap], None] | None = None,
        fail_provision: ProviderError | None = None,
        start_state: InstanceState = InstanceState.RUNNING,
    ) -> None:
        self.name = name
        self._offers = {o.offer_id: o for o in offers}
        self._instances: dict[str, Instance] = {}
        self._capacity = capacity
        self._on_provision = on_provision
        self.fail_provision = fail_provision
        self._start_state = start_state
        self._lock = threading.Lock()
        self._seq = 0
        self.calls: list[tuple[str, str]] = []

    def features(self) -> ProviderFeatures:
        return ProviderFeatures(
            list_offers=True, pricing=True, provision=True, terminate=True, worker_bootstrap=True
        )

    def list_offers(self) -> list[InstanceOffer]:
        return [o.model_copy(deep=True) for o in self._offers.values()]

    def provision(self, offer_id: str, bootstrap: WorkerBootstrap) -> Instance:
        with self._lock:
            self.calls.append(("provision", offer_id))
            if self.fail_provision:
                raise self.fail_provision
            if offer_id not in self._offers:
                raise ProviderError(f"unknown offer {offer_id}")
            active = [i for i in self._instances.values() if i.state != InstanceState.TERMINATED]
            if self._capacity is not None and len(active) >= self._capacity:
                raise CapacityUnavailable("fake provider capacity exhausted")
            self._seq += 1
            inst = Instance(
                provider_instance_id=f"fake-i-{self._seq:04d}",
                provider=self.name,
                offer_id=offer_id,
                state=self._start_state,
            )
            self._instances[inst.provider_instance_id] = inst
        if self._on_provision:
            self._on_provision(inst.model_copy(), bootstrap)
        return inst.model_copy()

    def set_state(
        self, provider_instance_id: str, state: InstanceState, error: str | None = None
    ) -> None:
        with self._lock:
            inst = self._instances[provider_instance_id]
            inst.state, inst.error = state, error

    def get_instance(self, provider_instance_id: str) -> Instance:
        with self._lock:
            if provider_instance_id not in self._instances:
                raise ProviderError(f"unknown instance {provider_instance_id}")
            return self._instances[provider_instance_id].model_copy()

    def terminate(self, provider_instance_id: str) -> Instance:
        with self._lock:
            self.calls.append(("terminate", provider_instance_id))
            inst = self._instances.get(provider_instance_id)
            if inst is None:
                raise ProviderError(f"unknown instance {provider_instance_id}")
            inst.state = InstanceState.TERMINATED
            return inst.model_copy()

    def active_instances(self) -> list[Instance]:
        with self._lock:
            return [
                i.model_copy()
                for i in self._instances.values()
                if i.state != InstanceState.TERMINATED
            ]
