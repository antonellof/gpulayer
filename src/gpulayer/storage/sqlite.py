"""SQLite metadata store (jobs, tasks, workers, events, logs, instances, costs).

Large payloads never live here: artifacts are referenced by content hash and stored
by an ``ArtifactStore``. The schema is accessed only through this module so a shared
database (e.g. PostgreSQL, ADR-0008) can replace it behind the same methods.

Concurrency: WAL mode, one connection guarded by a lock, and ``BEGIN IMMEDIATE``
transactions, so several local processes (CLI, local worker) can share one file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from gpulayer.contracts import utcnow

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL, spec TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT, scheduling TEXT, result TEXT,
  rendezvous TEXT, provisioning_deadline TEXT, submitted_by TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created_at);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL, attempt INTEGER NOT NULL,
  node_rank INTEGER NOT NULL DEFAULT 0, worker_id TEXT, state TEXT NOT NULL,
  exit_code INTEGER, error TEXT, lease_expires_at TEXT, started_at TEXT, finished_at TEXT,
  outcome TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0, deadline TEXT
);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks(job_id);
CREATE INDEX IF NOT EXISTS tasks_worker ON tasks(worker_id, state);
CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL, capabilities TEXT NOT NULL,
  registered_at TEXT NOT NULL, last_heartbeat_at TEXT, token_hash TEXT NOT NULL,
  revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, task_id TEXT, worker_id TEXT,
  ts TEXT NOT NULL, type TEXT NOT NULL, message TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, seq);
CREATE TABLE IF NOT EXISTS logs (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, task_id TEXT NOT NULL,
  node_rank INTEGER NOT NULL, stream TEXT NOT NULL, line TEXT NOT NULL, ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS logs_job ON logs(job_id, seq);
CREATE TABLE IF NOT EXISTS instances (
  id TEXT PRIMARY KEY, provider TEXT NOT NULL, job_id TEXT, offer TEXT NOT NULL,
  state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, release TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS costs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, record TEXT NOT NULL
);
"""


def ts(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def parse_ts(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


class SqliteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=30
        )
        self._conn.row_factory = sqlite3.Row
        self._depth = 0
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"metadata schema {row[0]} != supported {SCHEMA_VERSION}; migrate first"
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Re-entrant write transaction (BEGIN IMMEDIATE takes the cross-process lock)."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self._conn
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self._conn.execute("COMMIT")

    def query(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, args).fetchall())

    def one(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ helpers

    def add_event(
        self,
        type_: str,
        message: str,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO events (job_id, task_id, worker_id, ts, type, message, data) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    job_id,
                    task_id,
                    worker_id,
                    ts(utcnow()),
                    type_,
                    message,
                    json.dumps(data or {}, default=str),
                ),
            )
