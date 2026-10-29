# GpuLayer — Initial Implementation Task

You are the lead engineer implementing **GpuLayer**, an open-source, local-first GPU execution platform. Work directly in the current repository.

Before writing code, read `CLAUDE.md` and `architecture_plan.md` in full. Treat them as the project requirements and architectural guidance.

## Objective

Build the first working end-to-end vertical slice of GpuLayer: a developer can submit a Python function or command as a job, have it execute locally, inspect its status and logs, and retrieve its result.

The first version must work on an ordinary CPU-only machine without an account, cloud provider, hosted control plane, or GPU.

Design the interfaces so GPU backends, Apple Metal/MPS, private worker meshes, distributed jobs, and external GPU providers can be added incrementally without rewriting the core.

## Phase 1: Inspect and plan

1. Inspect the repository, Git status, existing source files, tests, dependency configuration, and documentation.
2. Read `CLAUDE.md` and `architecture_plan.md`.
3. Identify what already exists and what is missing.
4. Do not overwrite existing work or introduce duplicate implementations.
5. Create a short implementation plan based on the actual repository state.
6. If the repository is empty, establish a minimal Python project using the stack recommended in the architecture plan.
7. Record major architectural decisions in `docs/decisions/`. Keep unresolved decisions marked `Proposed`.

After inspection, proceed with implementation. Do not stop at producing a plan unless a genuine blocker requires clarification.

## Phase 2: Implement the local execution vertical slice

Implement the smallest coherent version of these capabilities:

### 1. Project and developer tooling
- Set up the Python package and development environment.
- Add a CLI.
- Add formatting, linting, type checking, and tests.
- Keep dependencies minimal.
- Ensure installation and tests are documented.

### 2. Job contracts
Implement typed, validated models for:
- `JobSpec`
- `FunctionJob` or an equivalent supported function-job representation
- `CommandJob`
- `JobResult`
- `JobStatus`
- `ResourceRequest`

Include schema versioning and validation errors that are clear to users.

Do not overengineer the contract. Implement only the fields required for the initial working slice, while keeping the design extensible.

### 3. Local execution engine
Implement a local runner that can:
- Accept a valid job specification.
- Execute a supported Python function or command.
- Capture standard output and standard error.
- Track job status.
- Capture exit codes and exceptions.
- Return structured results.
- Handle cancellation where practical.
- Clean up temporary files and subprocesses safely.

Use a subprocess or another suitable isolation boundary for command execution. Do not claim that this provides a secure sandbox for hostile code.

For Python functions, define and document the initial supported serialization and packaging constraints. Do not silently depend on arbitrary closure serialization or make unsupported claims about moving any Python function between machines.

### 4. Job persistence
- Use SQLite for local job metadata.
- Persist job IDs, lifecycle states, timestamps, exit status, and result/artifact references as appropriate.
- Ensure the application can recover its recorded job state after restarting.
- Keep database access behind a small interface so PostgreSQL can be added later.
- Use safe, idempotent state transitions where practical.

### 5. Artifacts and logs
- Store local job logs and results using a clearly defined local storage layout.
- Return a stable job ID and provide a way to retrieve logs and results.
- Avoid storing large outputs directly in database rows.
- Validate paths and prevent jobs from accidentally writing outside their designated working directory where applicable.

### 6. CLI
Provide a small, usable CLI. The exact command syntax may follow existing project conventions, but aim for equivalents of:

```bash
gpulayer start
gpulayer status
gpulayer submit --help
gpulayer job status <job-id>
gpulayer job logs <job-id>
gpulayer job result <job-id>
gpulayer job cancel <job-id>
```

Do not add commands that are only placeholders. If a command is not implemented, do not advertise it as working.

Prefer a simple foreground local mode initially if a persistent daemon would significantly complicate the first vertical slice. Document that choice.

### 7. Python SDK
Expose a small Python API for submitting and inspecting jobs.

Keep the public API simple and typed. Follow the architecture plan's proposed API where practical, but prioritize a working, tested implementation over matching an illustrative API exactly.

Add examples that can actually run against the implemented code.

## Phase 3: Prepare GPU backend discovery

Implement the discovery interface without requiring GPU hardware for ordinary tests.

The first version should:
- Detect the CPU baseline.
- Introduce backend capability models.
- Keep discovery separate from scheduling and execution.
- Report unknown capabilities honestly.
- Avoid requiring CUDA, ROCm, or macOS-specific packages on CPU-only systems.

### Apple Metal / PyTorch MPS

Add an isolated discovery implementation for Apple Silicon Macs.

When PyTorch is installed in the selected environment, use its MPS APIs to check whether MPS is built and available.

Requirements:
- Skip cleanly on non-macOS systems.
- Handle PyTorch being absent.
- Provide clear diagnostic output.
- Add unit tests using mocks so CI does not require a Mac.
- Add an optional hardware-gated MPS smoke test.
- Do not automatically move CUDA-specific workloads to MPS.
- Do not claim MPS execution is implemented until an actual supported workload can execute and pass a test.

Represent Apple unified memory separately from discrete GPU VRAM.

For this first milestone, it is acceptable to implement MPS discovery and capability reporting without completing GPU-aware scheduling. Clearly document what is implemented and what remains.

Do not implement distributed multi-Mac MPS execution.

## Phase 4: Define future extension points

Create clear interfaces, but do not fully implement these features yet:

- Worker agent and private mesh.
- Capability-aware scheduler.
- Provider adapter protocol.
- `FakeProvider`.
- Distributed job launcher.
- PostgreSQL and S3-compatible storage.
- Hosted control plane.

Only add an interface when it has a clear purpose in the current architecture. Avoid speculative abstractions and empty modules.

The system must distinguish independent jobs from distributed jobs in its contracts, but distributed execution itself can remain a later milestone.

## Phase 5: Website and ADR structure

Add the initial repository structure for:

```text
site/
docs/
└── decisions/
    ├── README.md
    ├── ADR-0001-local-first-control-plane.md
    ├── ADR-0002-job-and-function-packaging.md
    ├── ADR-0003-worker-protocol-versioning.md
    ├── ADR-0004-distributed-launch-and-rendezvous.md
    ├── ADR-0005-gpu-capability-and-runtime-profiles.md
    ├── ADR-0006-metal-mps-support-boundaries.md
    ├── ADR-0007-storage-and-persistence.md
    ├── ADR-0008-security-and-trust-model.md
    ├── ADR-0009-open-source-license-and-governance.md
    ├── ADR-0010-provider-accounts-and-commercial-terms.md
    ├── ADR-0011-cost-and-retry-semantics.md
    └── ADR-0012-website-and-documentation-stack.md
```

For the first iteration:
- Add a `site/README.md` explaining the website goals and planned pages.
- Add an ADR index.
- Create proposed ADR starter documents with context, decision, alternatives, consequences, and follow-up sections.
- Populate an ADR with a real decision only when there is enough information to justify it.
- Do not install a full frontend toolchain until the website framework has been selected through the appropriate ADR.
- Keep website dependencies separate from the Python runtime package.

## Phase 6: Testing and quality

Write tests as you implement, not at the end.

At minimum, test:
- Valid and invalid job specifications.
- Successful local execution.
- Exceptions and nonzero exit codes.
- Captured logs.
- Result retrieval.
- Persistence and restart recovery.
- Cancellation and cleanup where implemented.
- Duplicate submissions or idempotency behavior where applicable.
- CPU-only execution without GPU dependencies.
- MPS detection when PyTorch is absent, when MPS is unavailable, and when MPS is available using mocks.

Add an optional hardware-gated MPS test, clearly marked as requiring suitable Apple Silicon hardware and software.

Run the actual test suite, linter, formatter checks, and type checker that are configured. Fix failures introduced by your changes.

Do not report tests as passing if they were not run.

## Engineering constraints

- Follow `CLAUDE.md` and `architecture_plan.md`.
- Use type hints and clear interfaces.
- Keep the implementation small and maintainable.
- Avoid unnecessary abstractions and premature microservices.
- Do not add mandatory telemetry.
- Do not require a hosted account.
- Do not invent provider integrations or pricing.
- Do not claim universal CUDA, ROCm, or Metal portability.
- Do not expose secrets in logs.
- Do not claim containers alone provide strong isolation for untrusted workloads.
- Keep documentation consistent with the implemented behavior.
- Preserve existing user changes and do not run destructive Git commands.

## Definition of done

The first milestone is complete when:

1. A developer can install the project in a clean environment.
2. A simple supported job can be submitted and executed locally.
3. The job receives a stable ID and its status can be inspected.
4. Logs and results can be retrieved.
5. Job metadata persists across application restarts.
6. The core tests pass on a CPU-only machine.
7. MPS capability discovery is isolated and tested without requiring Mac hardware in CI.
8. The website and ADR structure exists.
9. `CLAUDE.md` and `architecture_plan.md` match the implementation.
10. Known limitations are documented.

## Final response

When finished, provide:

1. A summary of what you implemented.
2. The important files created or modified.
3. Commands to install, run, and test the project.
4. The exact checks you ran and their results.
5. Features that remain unimplemented.
6. Any architectural decisions that still require approval.
7. The next recommended implementation milestone.

Start by inspecting the repository, then implement the first complete vertical slice. Prioritize working, tested local execution over scaffolding the entire platform.