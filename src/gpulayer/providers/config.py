"""Load provider adapters from a JSON file (``gpulayer up --providers FILE``).

    {"gcp": {"type": "gcp", "project": "my-project", "zone": "europe-west1-b",
             "offers": [{"id": "t4", "machine_type": "n1-standard-8",
                         "gpu_type": "nvidia-tesla-t4", "gpu_count": 1, "spot": true,
                         "packages": ["torch"], "frameworks": {"torch": "2.15"}}]}}

Each key is the provider name jobs ask for (``--provider gcp``).
"""

from __future__ import annotations

import json
from pathlib import Path

from gpulayer.errors import ValidationFailed
from gpulayer.providers.base import ProviderAdapter


def load_providers(path: Path) -> list[ProviderAdapter]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ValidationFailed(f"cannot read provider config {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValidationFailed(f"{path}: expected an object of named providers")
    adapters: list[ProviderAdapter] = []
    for name, conf in data.items():
        kind = conf.get("type") if isinstance(conf, dict) else None
        if kind == "gcp":
            from gpulayer.providers.gcp import GCPConfig, GCPProvider

            adapters.append(GCPProvider(GCPConfig.model_validate(conf), name=name))
        else:
            raise ValidationFailed(f"{path}: provider {name!r} has unknown type {kind!r}")
    return adapters
