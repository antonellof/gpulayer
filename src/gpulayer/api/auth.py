"""Control-plane credentials.

Named tokens, stored only as SHA-256 hashes, each with one role:

* ``admin``     -- everything a client can do, plus revoking workers, managing tokens and
                   maintenance (garbage collection);
* ``submitter`` -- submit jobs, read all jobs and workers, cancel only its own jobs;
* ``join``      -- may only call ``POST /v1/workers/register``.

Tokens can expire and can be revoked. Registered workers then get a per-worker session
token (checked by the control plane), and provider-provisioned workers use a single-use
instance token (control_plane/provisioning.py). The server re-reads the credentials file
when it changes, so ``gpulayer token create/revoke`` take effect without a restart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROLES = ("admin", "submitter", "join")
CLIENT_ROLES = ("admin", "submitter")
_PREFIX = {"admin": "glc_", "submitter": "glc_", "join": "glj_"}


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TokenEntry:
    name: str
    role: str
    hash: str
    created_at: str
    expires_at: str | None = None
    revoked: bool = False

    def active(self, now: datetime | None = None) -> bool:
        if self.revoked:
            return False
        expired = bool(
            self.expires_at and datetime.fromisoformat(self.expires_at) <= (now or _now())
        )
        return not expired


@dataclass
class ServerCredentials:
    tokens: list[TokenEntry] = field(default_factory=list)
    path: Path | None = None
    _mtime: float | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------ lookup

    def authenticate(self, token: str, roles: tuple[str, ...]) -> TokenEntry | None:
        """The active entry for ``token`` if its role is in ``roles`` (constant-time match)."""
        self._maybe_reload()
        h = token_hash(token)
        match: TokenEntry | None = None
        for entry in self.tokens:  # compare against every entry; no early exit
            if hmac.compare_digest(h, entry.hash):
                match = entry
        if match is None or match.role not in roles or not match.active():
            return None
        return match

    def is_client(self, token: str) -> bool:
        return self.authenticate(token, CLIENT_ROLES) is not None

    def is_join(self, token: str) -> bool:
        return self.authenticate(token, ("join",)) is not None

    # ------------------------------------------------------------ management

    def create(self, name: str, role: str, ttl: timedelta | None = None) -> str:
        """Add a token and return its plaintext value (shown once)."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if any(t.name == name and t.active() for t in self.tokens):
            raise ValueError(f"an active token named {name!r} already exists")
        token = _PREFIX[role] + secrets.token_urlsafe(32)
        now = _now()
        self.tokens.append(
            TokenEntry(
                name=name,
                role=role,
                hash=token_hash(token),
                created_at=now.isoformat(),
                expires_at=(now + ttl).isoformat() if ttl else None,
            )
        )
        return token

    def revoke(self, name: str) -> int:
        n = 0
        for t in self.tokens:
            if t.name == name and not t.revoked:
                t.revoked = True
                n += 1
        return n

    # ------------------------------------------------------------ persistence

    def save(self, path: Path | None = None) -> None:
        path = path or self.path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tokens": [asdict(t) for t in self.tokens]}, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        self.path = path
        self._mtime = path.stat().st_mtime

    @classmethod
    def load(cls, path: Path) -> ServerCredentials:
        creds = cls(tokens=_read(path), path=path)
        creds._mtime = path.stat().st_mtime
        return creds

    def _maybe_reload(self) -> None:
        if self.path is None:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            with self._lock:
                self.tokens = _read(self.path)
                self._mtime = mtime


def _read(path: Path) -> list[TokenEntry]:
    data = json.loads(path.read_text())
    if "tokens" in data:
        return [TokenEntry(**t) for t in data["tokens"]]
    # Format written before named tokens: unnamed admin and join hashes.
    created = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    entries = [
        TokenEntry(name=f"client-{i}", role="admin", hash=h, created_at=created)
        for i, h in enumerate(data.get("client_token_hashes", []))
    ]
    entries += [
        TokenEntry(name=f"join-{i}", role="join", hash=h, created_at=created)
        for i, h in enumerate(data.get("join_token_hashes", []))
    ]
    return entries


def generate_credentials() -> tuple[ServerCredentials, str, str]:
    """A fresh set with one admin and one join token. Returns (creds, client, join)."""
    creds = ServerCredentials()
    client = creds.create("admin", "admin")
    join = creds.create("join", "join")
    return creds, client, join


def parse_ttl(text: str) -> timedelta:
    """'90m', '12h', '7d' -> timedelta."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if not text or text[-1] not in units or not text[:-1].isdigit():
        raise ValueError("TTL looks like 90m, 12h or 7d")
    return timedelta(seconds=int(text[:-1]) * units[text[-1]])
