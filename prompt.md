# GpuLayer — Core Platform Implementation Prompt

You are the lead engineer building **GpuLayer**, an open-source GPU execution and orchestration platform that unifies local GPUs, cloud GPU providers, and distributed multi-node GPU compute behind a consistent developer interface.

Work directly in the current repository.

Before coding, read `CLAUDE.md` and `architecture_plan.md` in full. Inspect the existing codebase and preserve useful existing work.

## 1. Product vision

GpuLayer enables developers to run GPU workloads wherever suitable compute is available:

- **Local:** GPUs and CPUs on a developer's workstation or server, including NVIDIA CUDA and Apple Silicon Metal through PyTorch MPS.
- **Cloud:** GPUs provided by external infrastructure providers, rented GPU machines, and self-hosted cloud infrastructure.
- **Distributed:** A single workload coordinated across multiple GPUs and machines, whether in a local cluster, private network, self-hosted environment, or supported cloud GPU infrastructure.
- **Hybrid:** A unified control plane that can discover, select, coordinate, and manage eligible compute resources across supported environments.

These are first-class platform capabilities, not separate products or an architecture where cloud support is postponed indefinitely.

The platform must distinguish between an independent job, which can execute on one worker, and a distributed job, which requires coordinated execution across multiple workers or GPUs.

The architecture must support all of these execution targets through consistent job contracts, capability-aware scheduling, worker protocols, provider adapters, observability, and lifecycle management.

The initial implementation may support a limited subset of execution backends, but its architecture must make the path to all three execution modes explicit and testable.

## 2. Core design principles

1. **Unified interface:** Developers submit and manage jobs through one SDK and CLI, regardless of execution target.
2. **Multiple execution targets:** Local, cloud, and distributed execution are explicit scheduling choices.
3. **Local development without cloud dependencies:** Local execution must work without an account or hosted control plane.
4. **Cloud is a first-class target:** Provider adapters, remote provisioning boundaries, authentication interfaces, and remote worker communication must be designed into the platform.
5. **Distributed execution is a first-class workload type:** The system must model worker groups, ranks, rendezvous, communication requirements, health, and coordinated failure.
6. **Open and extensible:** Keep provider-specific functionality behind adapters and avoid locking the core to a single cloud vendor.
7. **Capability-aware placement:** Schedule based on hardware, memory, runtime, software dependencies, topology, and workload requirements rather than GPU count alone.
8. **Explicit behavior:** Never imply that a workload can run on a backend until its requirements and compatibility have been validated.
9. **Observable execution:** Provide job state, worker state, logs, events, failure reasons, and resource usage where available.
10. **Security by design:** Treat remote execution, credentials, worker registration, and cross-network communication as core design concerns.

## 3. First inspect the repository

Before implementation:

- Read `CLAUDE.md` and `architecture_plan.md`.
- Inspect the repository, source tree, dependencies, tests, and Git status.
- Identify existing interfaces and avoid duplicate implementations.
- Identify architectural gaps in local execution, remote execution, scheduling, distributed execution, and provider integration.
- Create a concise, repository-specific implementation plan.
- Record important decisions as ADRs, keeping unresolved decisions marked as proposed.

Then proceed with implementation. Do not stop after producing a plan unless a genuine blocker requires clarification.

## 4. Target architecture

Design GpuLayer around these logical components. They may initially run in one process or deployment; do not create unnecessary microservices.

### A. Developer interfaces

- Python SDK.
- CLI.
- Versioned job and worker API contracts.
- Clear submission, status, logs, result, cancellation, and resource-inspection operations.

### B. Control plane

Responsible for:

- Job lifecycle and metadata.
- Resource discovery and worker registry.
- Capability-aware scheduling.
- Placement decisions.
- Worker leases and heartbeats.
- Distributed job coordination.
- Provider lifecycle operations.
- Authentication and authorization.
- Usage accounting and observability.

Support a locally hosted control plane and a remotely hosted or self-hosted control plane. Local execution must not require an external service.

### C. Worker and execution plane

Workers run on local machines, cloud GPU instances, or cluster nodes.

A worker should be able to:

- Register with an authorized control plane.
- Report hardware and runtime capabilities.
- Receive compatible work.
- Prepare an execution environment.
- Execute supported workloads.
- Stream logs, state, and resource observations.
- Report completion and failure.
- Handle leases, cancellation, and worker loss safely.

Do not require every worker to have the same GPU vendor.

### D. Scheduler

Implement scheduling as a separate logical component.

It should consider:

- Execution target: local, cloud, or distributed.
- GPU vendor and architecture.
- GPU count and memory requirements.
- CPU, RAM, disk, and network requirements.
- Runtime and framework compatibility.
- Container or environment requirements.
- Interconnect and topology requirements for distributed jobs.
- Data and artifact location.
- Provider availability and user-defined placement constraints.
- Estimated cost, when reliable pricing information is available.

For distributed jobs, the scheduler must select a compatible worker group rather than schedule each rank independently without coordination.

Initially, implement a deterministic, testable scheduler with a small set of placement policies. Keep advanced optimization for later.

### E. Provider adapters

Define a provider interface that separates GpuLayer's core from infrastructure-specific APIs.

The interface should accommodate, where supported:

- Discovering available resources and instance types.
- Estimating resource costs.
- Provisioning and terminating GPU instances.
- Checking instance state.
- Bootstrapping or connecting workers.
- Reporting provider-specific errors and capacity limitations.

Implement a `FakeProvider` first for deterministic tests.

Do not claim real cloud provisioning is supported until an actual provider adapter has been implemented and tested. Do not invent credentials, pricing, APIs, or provider capabilities.

### F. Storage and artifacts

Separate job metadata from large artifacts.

Support a local development implementation and define interfaces that can later support remote object storage and a shared database.

Artifacts must be accessible to workers according to the chosen execution mode. Do not assume every worker shares the same local filesystem.

### G. Observability

Provide a consistent view of:

- Job and task status.
- Worker registration and health.
- Execution logs.
- Distributed rank status.
- Failures and retry decisions.
- Resource usage.
- Estimated and actual costs, when available.

## 5. Unified job model

Implement versioned, typed contracts for at least:

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

Use clear, validated models. Keep the initial implementation minimal, but avoid a schema that assumes every job runs on a single local machine.

A job specification should be able to express:

- Workload type.
- Code, command, or workload package.
- Runtime and dependency requirements.
- Required hardware and memory.
- Execution target or placement constraints.
- Environment configuration and secret references.
- Input and output artifacts.
- Timeouts and retry policy.
- For distributed jobs, worker count, GPU count per worker, distributed runtime, and rendezvous requirements.

Do not silently serialize arbitrary Python closures and promise that they will run on any remote worker. Define supported packaging and serialization constraints explicitly.

## 6. Execution modes

Expose the three primary execution modes through one coherent SDK and CLI.

### Local execution

Run a supported workload on a selected local worker. CPU-only machines must work without GPU dependencies.

### Cloud execution

Allow a job to target a compatible cloud worker or provider-backed resource. Design the complete provisioning-to-execution lifecycle, but only advertise provider support where the relevant adapter and integration are implemented.

Support both pre-existing remote workers and provider-provisioned instances as distinct concepts where appropriate.

### Distributed execution

Represent one logical job that coordinates multiple workers or GPUs.

The architecture must account for:

- Worker-group allocation.
- Rank and world-size assignment.
- Rendezvous and process-group initialization.
- Compatible runtime and dependency environments.
- Inter-worker communication.
- Coordinated startup and termination.
- Worker failures and timeouts.
- Log aggregation and per-rank diagnostics.
- Artifact and checkpoint handling.

Prioritize a supported PyTorch distributed path, such as `torchrun` with an appropriate rendezvous mechanism, rather than inventing a custom collective-communication system.

Do not claim distributed execution works until an end-to-end multi-process or multi-node test demonstrates it.

## 7. GPU backend strategy

Create an extensible capability and runtime model.

### NVIDIA CUDA

Design the initial primary path for NVIDIA GPU execution. Separate GPU detection, runtime compatibility, workload execution, and distributed communication.

### Apple Metal / PyTorch MPS

Support capability discovery for Apple Silicon and prepare an independent-job execution path using PyTorch MPS.

Check MPS availability through PyTorch's supported APIs. Handle missing PyTorch and unavailable MPS gracefully.

MPS is not CUDA. Do not assume CUDA kernels, extensions, dtypes, distributed collectives, or memory semantics are portable to MPS.

Treat multi-Mac distributed MPS as unsupported until independently validated.

### AMD ROCm

Keep the capability model extensible for ROCm, but do not claim execution support before it is implemented and tested.

## 8. Initial implementation milestone

Build a complete, testable foundation rather than attempting to finish every provider and distributed feature at once.

Implement in this order:

1. **Contracts and runtime interfaces:** Define jobs, workers, capabilities, execution targets, and results.
2. **Local execution:** Complete a real CPU-only job lifecycle through the SDK and CLI.
3. **Worker registry and scheduler:** Implement capability reporting, a fake worker registry, and deterministic placement tests.
4. **Remote-worker path:** Define and implement the smallest authenticated worker communication path needed to execute a job on a second machine. If a real remote execution path is not completed in this milestone, document the exact gap.
5. **Distributed job foundation:** Implement distributed job contracts, worker-group planning, rank assignment, and a fake distributed executor with tests.
6. **NVIDIA execution foundation:** Implement GPU capability detection and a hardware-gated local smoke test.
7. **MPS discovery:** Implement isolated MPS detection with mock-based CPU CI tests.
8. **Provider foundation:** Implement `FakeProvider` and lifecycle interfaces; do not fake real provisioning.
9. **Documentation and ADRs:** Explain the architecture, supported capabilities, and next integration milestones.

Keep the project runnable after each step. Do not leave the core unusable while adding future-facing scaffolding.

If scope requires splitting work, finish a coherent vertical slice first and report which steps are complete versus only designed.

## 9. CLI and SDK expectations

Aim for commands equivalent to:

```bash
gpulayer worker start
gpulayer worker status
gpulayer resources list
gpulayer submit --target local ...
gpulayer submit --target cloud ...
gpulayer submit --target distributed ...
gpulayer job status <job-id>
gpulayer job logs <job-id>
gpulayer job result <job-id>
gpulayer job cancel <job-id>
```

Adjust syntax to match the implementation and avoid placeholder commands.

The SDK should provide a consistent way to submit, inspect, and cancel jobs. Users should not need separate SDKs for local versus cloud execution.

For distributed jobs, distinguish a request to execute one distributed workload from a batch of independent jobs.

## 10. Security and reliability

Address these concerns in the architecture and implementation:

- Authenticated worker registration and control-plane communication.
- Authorization for job submission and worker management.
- Secure handling of credentials and secret references.
- TLS for remote connections where applicable.
- Worker leases and heartbeats.
- Idempotent lifecycle operations where practical.
- Timeouts and cancellation.
- Clear retry semantics.
- Isolation boundaries and documented limitations.
- No mandatory telemetry.
- No secrets in logs.
- No assumption that public, untrusted jobs can safely execute on arbitrary workers.

Do not claim exactly-once execution. Document the behavior around worker loss, retries, partial distributed execution, and potentially duplicated side effects.

## 11. Testing strategy

Write tests throughout implementation.

Include:

- Job contract validation.
- Local CPU execution.
- Failure and cancellation handling.
- Persistence and result retrieval.
- Worker registration and heartbeat behavior.
- Capability-aware scheduling.
- Incompatible resource rejection.
- Provider adapter behavior using `FakeProvider`.
- Distributed rank assignment and worker-group validation.
- Remote communication and authentication where implemented.
- MPS discovery using mocks.
- Hardware-gated CUDA and MPS smoke tests.
- A real distributed integration test when suitable hardware or a supported multi-process test environment is available.

CPU-only CI must not require CUDA, ROCm, or Apple-specific dependencies.

Clearly distinguish unit tests, simulated integration tests, and tests that require actual GPU hardware or multiple machines.

## 12. Website and architecture decisions

Maintain the planned `site/` directory and `docs/decisions/` structure.

Ensure documentation covers:

- Product overview and architecture.
- Local execution.
- Cloud execution and provider adapters.
- Distributed execution.
- Worker deployment and private networking.
- GPU compatibility and runtime profiles.
- Apple Metal/MPS boundaries.
- Security and trust model.
- Cost accounting and scheduling.
- Self-hosting and hosted control-plane options.
- Contributing and release process.

Use ADRs for major decisions, especially control-plane topology, worker communication, job packaging, distributed launch, provider abstraction, storage, security, licensing, and website technology.

Do not prematurely select a cloud vendor, license, or commercial model without evaluating alternatives.

## 13. Engineering rules

- Read existing code before modifying it.
- Follow `CLAUDE.md` and `architecture_plan.md`.
- Prefer small, maintainable components over premature microservices.
- Use type hints and explicit contracts.
- Keep provider-specific logic out of the core scheduler.
- Do not hardcode credentials, prices, or provider assumptions.
- Do not advertise unimplemented functionality.
- Preserve existing user changes.
- Do not run destructive Git commands.
- Update documentation alongside behavior.
- Run the available tests and quality checks, and report the actual results.

## 14. Definition of done for the first milestone

The milestone is complete when:

1. The project installs and runs on a CPU-only machine.
2. A job can be submitted, executed, inspected, and retrieved through the SDK or CLI.
3. Execution targets and worker capabilities are represented in the contracts.
4. The scheduler has deterministic tests for compatible and incompatible placements.
5. Distributed jobs have explicit contracts and tested worker-group/rank planning.
6. MPS discovery works without making Mac hardware mandatory for CI.
7. Provider interfaces and `FakeProvider` are tested.
8. Remote and distributed capabilities are accurately classified as implemented, simulated, or planned.
9. Documentation and ADRs reflect the actual architecture.
10. Tests and quality checks have been run, with failures disclosed honestly.

## 15. Final report

When finished, report:

- What works end-to-end.
- What exists only as an interface or simulation.
- Which local, cloud, and distributed execution paths are actually functional.
- Files created or changed.
- Commands to install, run, and test.
- Exact test and quality-check results.
- Security and compatibility limitations.
- Decisions requiring approval.
- The next recommended milestone toward real multi-node GPU execution and cloud provider integration.

Begin by inspecting the repository, then implement the first coherent vertical slice. Build toward a unified local, cloud, and distributed GPU platform—not a local-only runner with future features represented by empty interfaces.