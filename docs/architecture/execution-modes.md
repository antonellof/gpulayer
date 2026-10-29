# Execution modes

Two orthogonal choices: **target** (where) and **workload type** (what).

## Targets

| Target | Served by workers with origin | How workers join |
|---|---|---|
| `local` | `local` | in-process (SDK/CLI) or `gpulayer worker start` without `--url` |
| `remote` | `existing` | `gpulayer --url URL worker start` with a join token |
| `cloud` | `provider` | booted by a provider adapter with a single-use instance token |

`placement.fallback_targets` lists more targets to try, in order. Without it, a `local` job
only ever runs on a local worker.

## Workload kinds

| Kind | Type | Runs | Packaging |
|---|---|---|---|
| `CommandJob` | independent | `command` argv in the source bundle (or an empty workspace) | optional source bundle |
| `FunctionJob` | independent | `module:function(*args, **kwargs)` via a stdlib runner | source bundle on `PYTHONPATH`, or an installed module |
| `DistributedJob` | distributed | `torchrun` on every node | source bundle (required for the entrypoint) |

Function arguments and return values must be JSON. Closures, lambdas and pickled objects are
unsupported by design (ADR-0002), because they break across machines, Python versions and trust
boundaries.

`Client.submit_batch([...])` submits independent jobs that are scheduled separately.
`gl.distributed(...)` submits one job whose nodes start together and fail together.

## Local execution

CPU-only machines work with no GPU libraries. The CLI's in-process worker lives as long as the
command; `--detach` leaves the job queued for a long-running `gpulayer worker start`.

## Remote execution

See [remote-workers.md](remote-workers.md). Same agent, HTTP channel instead of in-process.

## Cloud execution

A cloud job either lands on an already-registered provider worker or, with
`placement.provisioning`, triggers the provisioning lifecycle in
[providers.md](providers.md). The only provider today is the simulated `FakeProvider`.

## Distributed execution

See [distributed.md](distributed.md).
