# ADR-0008: Storage and artifact management

- **Status:** Accepted (milestone 1) for SQLite metadata and the local CAS. Shared database and object storage are Proposed.
- **Date:** 2026-10-29

## Context

Job metadata (small, transactional) and artifacts (large, immutable) have different
needs. Remote workers do not share a filesystem with the submitter or the control plane.

## Decision

- **Metadata:** `SqliteStore` (`src/gpulayer/storage/sqlite.py`) holds jobs, tasks,
  workers, events, logs, instances and costs. It uses WAL mode, `busy_timeout` 30 s, a
  process-local re-entrant lock, and `BEGIN IMMEDIATE` transactions, so several local
  processes can safely share one file. There is a schema version in a `meta` table, and
  a mismatch refuses to open.
- **Artifacts:** `LocalArtifactStore` (`artifacts/store.py`) is content-addressed by
  SHA-256 (`artifact://sha256/<hex>`). Writes are atomic (temp file + rename), and every
  read verifies the digest. The `ArtifactStore` protocol lets an object store replace it.
- Remote workers fetch artifacts **over the API** (`GET /v1/artifacts/<sha>`), verify the
  digest, and cache them per worker. Outputs written to `$GPULAYER_OUTPUT_DIR` are
  uploaded after the task (limits: 1000 files, 512 MiB per task) and referenced in
  `JobResult.outputs`.
- Logs are stored as rows (lines truncated at 8 KiB), not as artifacts.

## Alternatives considered

- **Shared filesystem (NFS):** not available for cloud or remote workers. Possibly a
  future optimization, declared as a capability.
- **PostgreSQL from day one:** an operational burden for the local mode. It is the
  expected choice for multi-replica deployments (Proposed).
- **S3-compatible object storage:** the expected remote implementation (Proposed).
  Presigned URLs would offload transfer from the control plane.

## Consequences and trade-offs

- All artifact bytes pass through the control plane in served mode, which limits it to
  modest data sizes.
- No garbage collection: artifacts, logs and events grow without bound.
- Restart recovery works because every state is persisted (see
  `test_state_is_persisted_across_restart`). In-flight provider calls are not journaled.

## Security and operational implications

- Artifact access is by digest. Any authenticated client or worker can read any
  artifact whose digest it knows (ADR-0003 gap).
- Back up `$GPULAYER_HOME` (`metadata.db*`, `artifacts/`, `server-credentials.json`).

## Follow-up

- `PostgresStore` and an `S3ArtifactStore` with presigned transfers, both behind the
  existing interfaces.
- Retention and GC policies, plus log archival to artifacts.
- Data-locality hints in placement.
