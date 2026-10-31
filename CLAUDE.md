# CLAUDE.md

## 1. Project mission

**Product focus (ADR-0023): Train PyTorch on every GPU you can reach: one Trainer, from your
laptop to many machines.**

The PyTorch `Trainer` is the product people adopt; the platform below is what makes it run on
every GPU and machine. Prioritise work that makes training faster, simpler or able to reach
more hardware, and keep every public claim backed by a measurement in `benchmarks/results/`.
"Every GPU" means the backends validated on hardware (today NVIDIA CUDA and Apple Metal, plus
CPU); others (AMD ROCm) are named as implemented until validated. Speed against PyTorch
Lightning appears only in performance comparisons, as "the GPUlayer Trainer", with the cases
measured, including any where GpuLayer is not faster; never in taglines.

GpuLayer is an open-source GPU execution and orchestration platform that unifies **local, cloud, and distributed GPU compute** behind a consistent developer interface.

The goal is to let developers package, submit, schedule, execute, observe, and manage GPU workloads across different environments without writing separate orchestration logic for each infrastructure provider.

GpuLayer must support a path toward:

- Local CPU and GPU execution.
- Remote execution on existing machines and GPU servers.
- Cloud GPU provisioning through provider adapters.
- Distributed workloads across multiple GPUs and machines.
- Self-hosted and hosted control-plane deployments.
- Heterogeneous infrastructure, including NVIDIA CUDA, Apple Metal through PyTorch MPS for supported workloads, and potentially AMD ROCm.
- Reproducible execution environments, artifact management, logs, resource reporting, and cost accounting.

The project must not become a local-only execution tool with cloud and distributed features represented by empty interfaces.

## 2. Core architectural principles

### 2.1 Unified platform

Use a common job model, SDK, CLI, capability model, and lifecycle across supported execution targets.

A user should not need a different programming interface for every provider or machine type.

### 2.2 Local, cloud, and distributed are first-class

Support three execution targets:

- `local`: execute on a local machine or local worker.
- `remote`: execute on an existing worker or managed remote machine.
- `cloud`: use an infrastructure provider to obtain suitable compute and execute the workload.

Support at least two workload types:

- `independent`: a workload that runs as a single execution unit on one worker or a suitable single-node allocation.
- `distributed`: one logical workload coordinated across multiple processes, GPUs, or machines.

These concepts are different. A cloud job is not necessarily distributed, and a distributed job is not necessarily cloud-hosted.

A placement policy may allow fallback or selection across multiple eligible environments, but only when explicitly configured.

### 2.3 One core, multiple execution backends

Separate platform contracts and scheduling from provider-specific APIs, GPU runtimes, and execution mechanisms.

Do not create separate, unrelated implementations for local, cloud, and distributed workloads.

### 2.4 Capability-driven scheduling

Scheduling must be based on declared workload requirements and reported worker capabilities, including:

- GPU vendor, architecture, and available memory.
- GPU count and topology.
- CPU, RAM, disk, and network capacity.
- Operating system and runtime compatibility.
- Framework and dependency requirements.
- Distributed communication requirements.
- Data locality.
- Availability, user constraints, and estimated cost where reliable data exists.

Never infer that a workload is compatible merely because a worker has a GPU.

### 2.5 Explicit implementation status

Distinguish clearly between:

- **Implemented:** the code exists and works within its documented scope.
- **Tested:** automated tests cover the stated behavior.
- **Hardware-validated:** the behavior has been verified on the relevant hardware.
- **Simulated:** behavior is represented by fakes, mocks, or in-memory implementations.
- **Planned:** the behavior has not been implemented.

Never describe a simulated provider or distributed executor as a real integration.

### 2.6 Speed, parallelism and distribution come first

GpuLayer exists to get GPU work done sooner. When choosing between designs, prefer the one
that is faster end to end, keeps more GPUs busy, or distributes work better, unless it costs
correctness or security. See ADR-0016.

- Do not put fixed sleeps or polling on hot paths (dispatch, worker loop, process supervision,
  log and artifact transfer, job waits). Use events, long polls or blocking waits.
- Never block a job on the control plane: ship logs and results asynchronously.
- Do not force host-device synchronisation in training loops (`.item()`, `.cpu()`, printing
  tensors) except where a value is actually needed.
- Use the resources a job was granted: all its GPUs, a fair share of CPU threads, overlapping
  data loading with compute.
- Prefer running independent work in parallel (more tasks per worker, more workers) and
  placing distributed groups where communication is cheapest.
- Measure before and after. Changes to hot paths come with a benchmark result in `benchmarks/`
  or a test that guards the behaviour. Report the hardware, versions and method with numbers.
- Never claim GpuLayer is faster than another system without a reproducible comparison, and
  never present CPU or single-machine results as GPU or multi-node results.

## 3. Read the architecture first

Before implementing a feature:

1. Read `architecture_plan.md`.
2. Inspect the relevant ADRs in `docs/decisions/`.
3. Inspect the existing code, tests, and public interfaces.
4. Check Git status and preserve user changes.
5. Identify dependencies and compatibility constraints.
6. Make a small implementation plan for nontrivial changes.

If the repository differs from the architecture plan, identify the discrepancy before making broad changes.

Do not stop after planning when implementation can safely proceed. Ask for clarification only when an important unresolved product, security, or compatibility decision blocks progress.

## 4. Architectural boundaries

Keep these responsibilities logically separate:

- **SDK and CLI:** user-facing submission and management interfaces.
- **API and contracts:** versioned schemas and protocol definitions.
- **Control plane:** job lifecycle, worker registry, scheduling, coordination, and authorization.
- **Worker agent:** registration, heartbeat, workload preparation, execution, and reporting.
- **Execution runtime:** process management, environment preparation, logs, results, and cleanup.
- **GPU discovery:** hardware and runtime capability detection.
- **Scheduler:** placement decisions using job requirements and worker capabilities.
- **Distributed execution:** worker-group allocation, ranks, rendezvous, launch, monitoring, and coordinated termination.
- **Provider adapters:** external infrastructure provisioning, state, capacity, and termination.
- **Artifact storage:** metadata, inputs, outputs, checkpoints, and logs.
- **Observability and accounting:** events, resource usage, execution outcomes, and cost records.

These are logical boundaries, not a requirement to deploy separate microservices. Prefer a modular implementation until independent deployment is justified.

## 5. Execution and deployment models

Support architectural paths for:

- Local development and single-machine execution without a hosted dependency.
- A self-hosted control plane managing multiple workers.
- A hosted control plane managing authorized workers.
- Existing remote machines that join an authorized deployment.
- Cloud-provisioned workers through provider adapters.
- Distributed jobs spanning a compatible worker group.

Do not require every worker to share a filesystem. Use explicit artifact transfer or shared-storage capabilities.

Do not assume every worker can connect directly to every other worker. Distributed networking and rendezvous requirements must be explicit.

A hosted control plane must not automatically imply that workloads or user data are sent to a GpuLayer-operated service. Document the actual control-plane and data-plane paths.

## 6. Job contracts and public API

Use typed, validated, versioned contracts.

The core model should cover:

- `JobSpec`
- `FunctionJob`
- `CommandJob`
- `DistributedJob`
- `ResourceRequest`
- `PlacementPolicy`
- `WorkerCapabilities`
- `WorkerStatus`
- `JobStatus`
- `JobResult`
- `ArtifactReference`
- `ExecutionEvent`

Adapt names to established project conventions instead of creating duplicate models.

Represent workload type, execution target, runtime requirements, hardware requirements, artifact references, timeout, retry policy, and relevant security settings explicitly.

Distributed jobs must express the intended worker count, processes or GPUs per worker, distributed runtime, and rendezvous requirements.

Do not promise arbitrary Python-function portability. Define supported code packaging, serialization, dependency installation, and environment constraints.

Do not make breaking changes to public SDK or protocol contracts without documenting migration requirements.

## 7. Worker lifecycle and remote communication

Workers must have a defined lifecycle:

1. Configure identity and trust.
2. Register with an authorized control plane.
3. Report capabilities and health.
4. Receive eligible assignments.
5. Acquire or renew execution leases as appropriate.
6. Prepare the runtime and artifacts.
7. Execute and report events, logs, and results.
8. Handle cancellation and shutdown.
9. Clean up resources.

Use explicit protocol versions, authentication, authorization, timeouts, and well-defined error handling.

Do not expose unauthenticated worker control endpoints or place secrets in logs.

A worker lease does not guarantee exactly-once execution. Document duplicate execution and recovery behavior.

## 8. Scheduling and distributed execution

The scheduler must reject incompatible placements instead of silently weakening requirements.

Keep scheduling policy independent of provider-specific APIs.

Distributed scheduling must allocate a compatible group of workers and account for networking, runtime consistency, GPU topology, and artifact accessibility.

Distributed execution must define:

- Rank and world-size assignment.
- Rendezvous configuration.
- Process launch and initialization.
- Inter-worker communication requirements.
- Readiness and failure detection.
- Coordinated cancellation and termination.
- Per-rank logs and status.
- Checkpoint and artifact behavior.
- Retry and partial-failure semantics.

Prefer established frameworks and launch mechanisms, such as PyTorch distributed and `torchrun`, where they satisfy the requirements.

Do not implement custom collective communication unless there is a demonstrated need.

Do not claim multi-node execution works without an end-to-end integration test in a documented supported environment.

## 9. GPU backend rules

### NVIDIA CUDA

Keep detection, runtime compatibility, process launch, and distributed communication separately testable.

Hardware-specific tests must be clearly marked.

### Apple Metal / PyTorch MPS

For supported Apple Silicon workloads, use PyTorch MPS APIs to detect availability and validate execution.

Handle missing PyTorch, unsupported platforms, and unavailable MPS gracefully.

Do not assume CUDA compatibility. Treat memory reporting, operators, dtypes, extensions, and distributed capabilities as backend-specific.

Do not claim distributed multi-Mac MPS support until it has been independently implemented and validated.

### AMD ROCm

Keep the architecture extensible for ROCm, but do not report it as supported until actual execution and compatibility tests exist.

### General rules

Do not require all GPU libraries on CPU-only installations. Use optional dependencies and isolated discovery where appropriate.

Never infer support from a backend's name alone. Maintain an explicit capability and compatibility model.

## 10. Provider integrations

Define provider adapters for external infrastructure operations, including applicable capabilities such as:

- Resource and instance-type discovery.
- Capacity checks.
- Cost estimates.
- Provisioning and termination.
- Instance-state inspection.
- Worker bootstrap and connectivity.
- Provider-specific errors and limitations.

Implement a `FakeProvider` for deterministic tests.

Use real provider integrations only after validating current APIs, credentials, lifecycle behavior, and failure handling.

Never invent pricing, available GPU types, capacity, or integration support.

Keep provider credentials out of source code, job logs, and persisted job payloads. Prefer secret references and scoped credentials.

## 11. Storage, artifacts, and accounting

Separate job metadata from large payloads.

Use interfaces that support local development and future remote databases and object storage.

Persist enough state to explain execution outcomes and recover from restarts. Define how artifact references remain valid across workers.

Distinguish cost estimates from actual charges. Use explicit currency, units, source, and timestamps. Do not present estimated costs as billed amounts.

Do not assume that an interrupted job can be resumed unless the runtime and workload support checkpointing.

## 12. Security and trust

Treat remote execution as a security boundary.

- Authenticate workers and control-plane clients.
- Authorize job submission, cancellation, and worker management.
- Protect remote communication.
- Handle secrets explicitly.
- Validate job specifications and artifact paths.
- Limit resource usage and enforce timeouts where applicable.
- Document the execution isolation model.
- Do not claim that subprocesses or containers alone make arbitrary hostile workloads safe.
- Do not expose public multi-tenant execution without an explicit isolation and threat-model review.
- Keep telemetry disabled by default unless a documented product decision establishes otherwise.

Security-sensitive changes require tests and documentation.

## 13. Testing requirements

Write tests with each implementation change.

Maintain:

- Unit tests for contracts, state transitions, and scheduling.
- Integration tests for local execution and persistence.
- Simulated tests using fake workers and providers.
- Remote-worker integration tests.
- Distributed execution tests.
- Optional hardware-gated CUDA, MPS, and ROCm tests.
- Tests for authentication, authorization, cancellation, and failure handling.

CPU-only CI must remain usable without GPU hardware or optional GPU libraries.

Clearly identify tests that are mocked, simulated, multi-process, multi-node, or hardware-validated.

A passing unit test is not evidence of cloud provisioning or multi-node GPU compatibility.

## 14. Repository and implementation discipline

- Use the existing language, framework, and conventions unless a justified architectural change is needed.
- Keep changes incremental and reviewable.
- Use type hints and clear error messages.
- Avoid unnecessary dependencies, abstractions, and microservices.
- Do not add empty modules just to match a diagram.
- Update documentation with behavior changes.
- Add or update ADRs when major architectural decisions change.
- Do not modify unrelated files.
- Do not run destructive Git commands.
- Never discard user changes.
- Run applicable formatting, linting, type checking, and tests.
- Report commands actually executed and their real results.
- Do not claim tests pass if they were not run.

## 15. Documentation

Keep `architecture_plan.md`, `site/`, and `docs/decisions/` aligned with the implementation.

Every change that alters behavior, capability, status, or measured results must update all
places that describe it in the same change, on GitHub and on the website:

- `README.md` and `CHANGELOG.md`.
- `docs/` (guides, reference, architecture, `docs/project/status.md`), which the site renders.
- The site's own pages in `site/src/` (home page features and status list, quickstart,
  compare page, structured data).
- `docs/decisions/` and `architecture_plan.md` when a decision or the architecture changes.
- Charts and benchmark summaries in `docs/assets/perf/` and `benchmarks/results/` when numbers
  change: show the latest results and comparisons with other tools, not before/after values
  of GPUlayer's own fixes.

Before finishing, search the docs and site for statements the change makes stale (feature
names, "planned", "not yet", version numbers, measured values) and fix them.

Document supported execution modes, setup instructions, provider requirements, networking, compatibility limits, and security assumptions.

Distinguish stable public APIs from experimental interfaces.

Do not advertise planned functionality as available.

## 16. Definition of done

A feature is complete only when:

1. Its behavior is implemented within a defined scope, and its performance impact on hot paths
   has been measured.
2. Its public contracts and failure behavior are documented.
3. Appropriate tests are added and run.
4. Relevant compatibility constraints are explicit.
5. Security implications have been considered.
6. Documentation reflects the actual implementation.
7. Remaining gaps are reported honestly.

For broad features such as cloud provisioning or distributed GPU execution, report partial completion at the capability level instead of declaring the entire feature complete.

## 17. Agent completion report

For every substantial task, report:

- What changed and why.
- Files created or modified.
- Public API or protocol changes.
- Commands and tests run, with actual results.
- Which behaviors are implemented, simulated, or hardware-validated.
- Known limitations and security considerations.
- Architectural decisions that need review.
- The next recommended step.

Prioritize a coherent, tested, end-to-end implementation over large amounts of scaffolding.

The long-term objective is a unified local, cloud, and distributed GPU execution platform, built incrementally without misrepresenting its current capabilities.