"""Control-plane credentials.

Three credential types, never interchangeable:

* client token   -- submit/inspect/cancel jobs, list/revoke workers, artifacts;
* join token     -- may only call ``POST /v1/workers/register``;
* worker session -- minted per worker at registration; may only act on that worker's
                    own heartbeat/assignments (checked by the control plane).

Provider-provisioned workers instead present a single-use instance token (see
control_plane/provisioning.py). Only SHA-256 hashes of tokens are written to disk.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class ServerCredentials:
    client_token_hashes: list[str] = field(default_factory=list)
    join_token_hashes: list[str] = field(default_factory=list)

    def is_client(self, token: str) -> bool:
        return _matches(token, self.client_token_hashes)

    def is_join(self, token: str) -> bool:
        return _matches(token, self.join_token_hashes)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "client_token_hashes": self.client_token_hashes,
                    "join_token_hashes": self.join_token_hashes,
                },
                indent=2,
            )
        )
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ServerCredentials:
        data = json.loads(path.read_text())
        return cls(
            list(data.get("client_token_hashes", [])), list(data.get("join_token_hashes", []))
        )


def _matches(token: str, hashes: list[str]) -> bool:
    h = token_hash(token)
    found = False
    for candidate in hashes:  # constant-time per comparison; no early exit
        found |= hmac.compare_digest(h, candidate)
    return found


def generate_credentials() -> tuple[ServerCredentials, str, str]:
    """Returns (credentials-with-hashes, client_token, join_token). Tokens are shown once."""
    client, join = "glc_" + secrets.token_urlsafe(32), "glj_" + secrets.token_urlsafe(32)
    return ServerCredentials([token_hash(client)], [token_hash(join)]), client, join
