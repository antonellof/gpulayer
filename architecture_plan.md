# GpuLayer — Architecture Plan

**Status:** Proposed architecture  
**Project:** GpuLayer  
**Category:** Open-source GPU execution and orchestration platform

## 1. Vision

**Product focus (2026-10-31, ADR-0023):** Train PyTorch on every GPU you can reach: one
Trainer, from your laptop to many machines. The architecture below is
the engine under that Trainer: placement, workers, distributed launch and providers serve
PyTorch training first, and remain usable for any job.

GpuLayer provides a unified way to execute GPU workloads across local machines, existing remote workers, cloud GPU providers, and distributed GPU clusters.

It aims to make infrastructure selection and execution management consistent while preserving access to the capabilities and limitations of each underlying environment.

A developer should be able to describe a workload and its requirements, choose an execution target or placement policy, submit it through the SDK or CLI, and observe its execution without writing provider-specific orchestration code.

The platform should support independent jobs and distributed workloads, with a clear path from local development to remote execution and multi-node GPU computing.

## 2. Goals

- Run GPU work as fast as the hardware allows: minimal dispatch latency, high GPU utilisation,
  parallel execution of independent work, and well-placed distributed groups (ADR-0016).
- Provide a consistent SDK and CLI for job submission and management.
- Support local CPU and GPU execution.
- Support execution on existing remote workers.
- Integrate external GPU infrastructure through provider adapters.
- Support distributed jobs across multiple GPUs and machines.
- Schedule work based on hardware and runtime compatibility.
- Support reproducible workload packaging and artifact transfer.
- Expose job, worker, and resource status.
- Provide extensible storage, security, and accounting interfaces.
- Support self-hosted and hosted control-plane deployments.
- Keep provider-specific behavior outside the core scheduling logic.
- Maintain a practical CPU-only development and CI path.

## 3. Non-goals

The initial architecture does not promise:

- Universal portability of every CUDA workload across GPU vendors.
- Automatic conversion of CUDA workloads to Apple MPS or ROCm.
- Distributed multi-Mac MPS execution without validated support.
- A proprietary distributed tensor or collective-communication framework.
- Exactly-once job execution under arbitrary failures.
- Automatic migration of arbitrary running processes between machines.
- A secure public multi-tenant sandbox merely by using subprocesses or containers.
- A single scheduler that optimally solves every cost, latency, and topology problem.
- Support for every cloud provider in the first release.
- Transparent access to all remote filesystems.
- Automatic checkpointing for workloads that do not support it.

## 4. Fundamental domain model

Keep these concepts separate throughout the API and implementation.

### 4.1 Workload type

**Independent job:** A single logical execution unit that runs on an eligible worker or allocation. Examples include model inference, data processing, model training on one machine, and batch tasks.

**Distributed job:** One logical workload coordinated across multiple processes, GPUs, or workers. Examples include multi-GPU training and multi-node distributed computation.

A distributed job may create multiple worker tasks and processes, but those tasks belong to one coordinated workload.

### 4.2 Execution target

**Local:** Execution on the local machine or a local worker.

**Remote:** Execution on an existing worker that is reachable through an authorized deployment.

**Cloud:** Execution on infrastructure managed through a provider adapter. A cloud worker may be provisioned specifically for a job or drawn from an existing worker pool.

These targets are independent of workload type. For example, an independent job can execute on a cloud GPU, and a distributed job can execute on an existing private cluster.

### 4.3 Placement policy

Placement describes where a job is eligible to run.

Possible policies include:

- Explicit target or worker selection.
- Required hardware and runtime constraints.
- Allowed providers or regions.
- Required GPU count and memory.
- Distributed worker-group requirements.
- Data locality and network constraints.
- Cost or latency preferences.
- Explicitly configured fallback rules.

Fallback must not violate required constraints or silently change a workload's execution semantics.

### 4.4 Capability versus availability

A worker's **capability** describes what it can support. Its **availability** describes what resources it can currently offer.

The scheduler needs both. A worker can be compatible but busy, or available but incompatible.

## 5. High-level architecture

<box gap={2}>
  <box border radius="lg" padding={3} gap={1}>
    <title size="lg">Developer interface</title>
    Python SDK · CLI · Job specification · API client
  </box>
  <box align="center">
    <icon name="arrow-down" size="lg" />
  </box>
  <box border radius="lg" padding={3} gap={1}>
    <title size="lg">Control plane</title>
    Job lifecycle · API · Worker registry · Scheduler · Authorization · Coordination · Accounting
  </box>
  <box align="center">
    <icon name="arrow-down" size="lg" />
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="lg" padding={3} gap={1} height="100%">
        <title>Execution plane</title>
        Local worker · Remote worker · Cloud worker · Runtime manager
      </box>
    </grid-item>
    <grid-item>
      <box border radius="lg" padding={3} gap={1} height="100%">
        <title>Infrastructure adapters</title>
        Provider APIs · Instance lifecycle · Worker bootstrap · Capacity
      </box>
    </grid-item>
  </grid>
  <box align="center">
    <icon name="arrow-down" size="lg" />
  </box>
  <grid columns={3} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} gap={1} height="100%">
        **Local**
        <text size="sm">CPU, CUDA, supported MPS workloads</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} gap={1} height="100%">
        **Cloud**
        <text size="sm">Provider-managed GPU resources</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} gap={1} height="100%">
        **Distributed**
        <text size="sm">Coordinated multi-process and multi-node execution</text>
      </box>
    </grid-item>
  </grid>
</box>

The diagram represents logical responsibilities. It does not require each box to be a separate service or deployment.

## 6. Control plane

The control plane manages desired state and execution coordination.

### Responsibilities

- Authenticate clients and workers.
- Validate job specifications.
- Persist job metadata and lifecycle events.
- Maintain worker registrations, heartbeats, capabilities, and leases.
- Schedule independent jobs and distributed worker groups.
- Coordinate provider operations.
- Track distributed job state and worker ranks.
- Support cancellation, timeouts, and retry policies.
- Maintain artifact references and execution metadata.
- Expose job and resource information through the API.

### Deployment modes

The architecture should support:

1. **Embedded or local mode:** Minimal setup for development and single-machine use.
2. **Self-hosted control plane:** An operator runs GpuLayer and manages multiple workers.
3. **Hosted control plane:** A managed service coordinates authorized workers and configured providers.

These modes should share the same core contracts and scheduling concepts. The implementation may vary in persistence, authentication, availability, and deployment topology.

Local execution must remain possible without a hosted service.

The platform must clearly document which metadata, logs, artifacts, and credentials are transmitted to the control plane in each deployment mode.

## 7. Execution plane and workers

Workers execute workloads on local or remote compute resources.

### Worker responsibilities

- Establish authenticated identity.
- Register with the control plane.
- Report capabilities and current resource availability.
- Receive and validate assignments.
- Prepare runtime environments.
- Retrieve required artifacts.
- Execute supported workloads.
- Stream logs and lifecycle events.
- Report resource observations and execution outcomes.
- Handle cancellation and lease expiry.
- Clean up processes and temporary resources.

### Worker state model

The implemented state model (`WorkerState`) is:

- `idle`, `busy`: online, with or without running tasks.
- `draining`: finishing its tasks, taking no new ones.
- `offline`: heartbeats stopped or the worker deregistered.
- `registering`, `unhealthy`: defined in the contract but not produced yet. Registration is
  atomic (one request), and there are no health checks beyond heartbeats; the scheduler
  already skips `unhealthy` workers so health checks can be added without a contract change.

Job state and worker state must remain separate. A worker can recover while a job has failed, and a job can continue through other workers after an individual worker fails if the workload's semantics allow it.

### Leases and recovery

Assignments should have explicit ownership and lease semantics.

If a worker disappears, the control plane must determine whether to mark the execution as failed, wait for recovery, or attempt a retry.

A lease cannot guarantee that the previous process has stopped. Retrying may duplicate execution or external side effects. The platform must document these risks and avoid promising exactly-once execution.

## 8. Job contracts and lifecycle

All execution modes should share versioned job contracts.

### Proposed contracts

- `JobSpec`: top-level submission.
- `FunctionJob`: supported packaged Python workload.
- `CommandJob`: supported command or executable workload.
- `DistributedJob`: one coordinated multi-process or multi-worker workload.
- `ResourceRequest`: compute and hardware requirements.
- `PlacementPolicy`: target and placement constraints.
- `WorkerCapabilities`: hardware, runtime, and environment capabilities.
- `WorkerStatus`: health, availability, and current assignments.
- `JobResult`: exit status, outputs, errors, and references.
- `ArtifactReference`: input, output, log, or checkpoint location.
- `ExecutionEvent`: timestamped lifecycle and diagnostic information.

The final names and fields should be refined against the existing codebase before implementation.

### Job lifecycle

The implemented job lifecycle (`JobState`) is:

`queued → [provisioning] → scheduled → running → succeeded | failed | cancelled | timed_out`

- Validation is synchronous at submission: an invalid spec is rejected and never stored, so
  `submitted` and `validated` are not observable states.
- `provisioning` is used when a provider adapter is creating capacity for the job.
- Preparation (fetching source and inputs) and finalization (uploading outputs) are phases of
  each task, reported as execution events, not job states.
- Each task has its own state (`TaskState`: `assigned`, `running`, terminal states, and `lost`
  when its lease expired). Retries create new task attempts under the same job.

State transitions are validated, persisted, and observable through events.

### Job packaging

Remote execution requires a defined representation of code and dependencies.

Possible packaging strategies include:

- A versioned source bundle with declared dependencies.
- A container image.
- A preinstalled runtime profile.
- An explicit command and environment specification.

The first supported strategy should be selected through an ADR and implemented end-to-end. Avoid promising arbitrary Python closures, local paths, or environments are portable across machines.

## 9. Scheduler and placement engine

The scheduler maps a job's requirements to suitable available resources.

### Scheduling inputs

- Workload type.
- Execution target and allowed environments.
- GPU vendor, architecture, and memory.
- GPU count and topology.
- CPU, RAM, and disk requirements.
- Runtime, framework, and dependency compatibility.
- Network and interconnect requirements.
- Artifact availability and data locality.
- Worker health and current load.
- Region and provider restrictions.
- Cost estimates and policy preferences when reliable.

### Scheduling stages

1. Validate requirements.
2. Identify eligible workers or provider-backed allocations.
3. Filter out incompatible resources.
4. Select a suitable worker or worker group.
5. Reserve or lease resources.
6. Prepare the execution environment.
7. Launch the workload.
8. Observe progress and handle failures.
9. Release resources and finalize results.

For cloud execution, provisioning may occur between resource selection and worker readiness. The design must distinguish selecting an instance type from successfully provisioning a working GPU worker.

### Initial policies

Start with deterministic policies:

- Explicit worker or target.
- First eligible worker.
- Simple resource-aware selection.
- User-specified constraints.

Add cost-aware or topology-aware optimization after the platform has reliable capability, availability, and pricing data.

Scheduling decisions should be explainable: users should be able to understand why a worker was selected or rejected.

## 10. Distributed execution architecture

Distributed execution is a core workload type, not merely a batch of independent jobs.

### Distributed job specification

A distributed job should describe:

- Worker count.
- GPU or process count per worker.
- Required backend and framework.
- Runtime and dependency requirements.
- Communication and network requirements.
- Rendezvous mechanism.
- Entry point and arguments.
- Environment configuration and secret references.
- Timeout, cancellation, and failure policy.
- Inputs, outputs, and checkpoint behavior where supported.

### Execution flow

1. Validate the distributed workload.
2. Identify a compatible worker group.
3. Reserve the group as a coordinated allocation.
4. Assign ranks and world size.
5. Prepare consistent execution environments.
6. Distribute code and required artifacts.
7. Establish rendezvous.
8. Start the distributed processes.
9. Track worker readiness and per-rank state.
10. Stream logs and execution events.
11. Coordinate completion or failure.
12. Release allocations and collect outputs.

### Distributed runtime

Prioritize established mechanisms such as PyTorch distributed with `torchrun` and an appropriate rendezvous service.

Keep the orchestration layer responsible for allocation and lifecycle management, while established frameworks handle collective communication.

Do not assume a distributed job can start successfully merely because enough GPUs are available. Network reachability, firewall rules, interface selection, runtime compatibility, and communication support must be validated.

### Failure semantics

Distributed failures can invalidate the entire worker group.

The first implementation should use conservative semantics: a fatal worker or rank failure fails the logical distributed job unless the selected runtime explicitly supports recovery.

Checkpoint-based restart can be added when supported by the workload and runtime.

Automatic retry of a partially executed distributed job must not be assumed safe.

### Initial scope

Begin with a real, testable multi-process distributed path. Progress to multi-node execution using an explicitly supported environment.

A simulated worker group is useful for testing placement logic but is not proof of distributed execution support.

## 11. Cloud and provider architecture

Cloud infrastructure must be a first-class target, but integrations must be explicit and verifiable.

### Provider adapter responsibilities

An adapter may support:

- Listing supported instance types.
- Querying capacity or availability.
- Estimating cost.
- Creating and terminating instances.
- Inspecting instance state.
- Configuring access and worker bootstrap.
- Waiting for worker registration.
- Reporting provider errors and resource identifiers.

Not all providers expose every operation. Each adapter should declare its actual capabilities.

### Provisioning lifecycle

A cloud job may follow this sequence:

1. Validate the job.
2. Resolve provider constraints.
3. Select a compatible resource configuration.
4. Obtain an estimate if reliable pricing is available.
5. Request provisioning.
6. Wait for the instance to become reachable.
7. Authenticate and register the worker.
8. Verify its reported capabilities.
9. Schedule and execute the job.
10. Collect logs, results, and accounting information.
11. Release or retain the resource according to explicit policy.

Provisioning success does not prove GPU runtime readiness. The worker must report verified capabilities before receiving incompatible work.

### Existing and managed workers

Support two concepts:

- **Existing workers:** machines already operated by the user or organization.
- **Managed workers:** machines provisioned or managed through a provider adapter.

This distinction allows GpuLayer to orchestrate a private cluster without taking responsibility for provisioning every machine.

### Provider independence

The core scheduler should use GpuLayer's capability model rather than hardcoding provider-specific instance names.

Provider-specific fields may be retained as metadata, but they should not leak into core scheduling logic unless required by a declared extension.

Implement `FakeProvider` for deterministic testing. Add a real provider only after documenting authentication, supported regions or instance types, lifecycle behavior, cleanup, and failure cases.

## 12. GPU and runtime compatibility

Model hardware capability separately from the framework and runtime available to a worker.

### Capability dimensions

- GPU vendor and architecture.
- Device count and visible devices.
- Memory capacity and available-memory observations.
- Operating system and CPU architecture.
- Driver and runtime versions.
- Framework and supported versions.
- Supported operators, dtypes, and relevant acceleration capabilities.
- Multi-GPU and distributed communication support.
- Interconnect and topology information where available.

Capability discovery should report uncertainty rather than fabricate precision.

### NVIDIA CUDA

Use a clear distinction between detecting hardware, validating the driver/runtime, preparing the environment, and successfully executing the workload.

Local GPU tests and distributed CUDA tests should be independently gated.

### Apple Metal / PyTorch MPS

Use PyTorch's MPS APIs to detect whether MPS is built and available in the selected environment.

MPS is a distinct backend, not a CUDA compatibility layer.

The platform must account for differences in operators, dtypes, extensions, memory reporting, and distributed support.

Apple Silicon uses unified memory, which must not be treated as equivalent to discrete GPU VRAM.

The initial MPS target is supported independent workloads on eligible Apple Silicon machines. Distributed MPS across multiple Macs is out of scope until a dedicated implementation and validation effort proves the required behavior.

### AMD ROCm

Maintain an extensible capability model, but do not mark ROCm execution as supported until a concrete runtime path has been implemented and tested.

### Runtime profiles

Runtime profiles can describe known combinations of framework, driver, operating system, and hardware requirements.

A profile is a compatibility aid, not a guarantee. Actual worker checks and workload validation remain necessary.

## 13. Storage and artifact management

Metadata, artifacts, logs, and checkpoints have different storage requirements.

### Metadata storage

Use a database abstraction for job metadata, worker records, execution events, and resource allocations.

A local SQLite implementation is suitable for initial development. A shared deployment can use a database selected through an ADR.

### Artifact storage

Define an interface for:

- Source and package bundles.
- Input datasets and files.
- Job outputs.
- Logs where appropriate.
- Distributed checkpoints.
- Metadata and integrity information.

Support local storage first as a working implementation and define remote object storage integration separately.

Do not assume that local paths are accessible from remote workers. Artifact staging, upload, download, and integrity checks must be explicit.

### Data locality

Where possible, the scheduler should account for where large inputs reside.

The first implementation can use explicit artifact locations and simple locality constraints. Advanced data placement and caching can follow later.

## 14. Security and trust model

Security requirements depend on deployment topology and whether workers execute workloads from mutually trusted users.

### Required design considerations

- Authentication and authorization.
- Worker identity and registration.
- Encrypted remote communication.
- Scoped credentials and secret references.
- API and protocol versioning.
- Resource limits and timeouts.
- Artifact access control and path validation.
- Isolation of processes and workloads.
- Secure provisioning and worker bootstrap.
- Log redaction and audit events.
- Worker revocation and lease expiration.

### Trust boundaries

Treat the control plane, workers, provider APIs, and workload processes as distinct trust boundaries.

A self-hosted single-user setup may use a simpler trust configuration than a multi-tenant service, but the difference must be explicit.

Containers and subprocesses are not, by themselves, a guarantee of strong isolation against hostile code.

Before supporting public, untrusted multi-tenant workloads, perform a dedicated threat-model and isolation review.

### Secrets

Never embed provider credentials in job specifications or log output.

Use scoped credentials or secret references with documented access behavior. Do not forward all control-plane credentials to every worker.

## 15. Observability and cost accounting

### Observability

Expose:

- Job status and lifecycle events.
- Worker status and heartbeats.
- Scheduling decisions.
- Preparation and launch errors.
- Per-rank distributed logs.
- Exit codes and failure reasons.
- Resource usage where available.
- Artifact and checkpoint references.
- Cancellation and retry events.

The first version may use structured logs and persisted events. A dedicated metrics backend can be added later.

### Cost accounting

Keep these separate:

- Cost estimates.
- Provider-reported usage.
- Provider-reported charges.
- GpuLayer-calculated internal accounting.

Every amount should have a currency, unit, timestamp, and source. Estimates must never be presented as actual charges.

Do not invent missing pricing information. Report unknown cost when there is no reliable source.

## 16. Proposed repository layout

Adapt this layout to the existing repository. Do not create every module in advance merely to match the tree.

The current layout is flatter than a module per box in the diagram, following the "no empty
modules" rule:

```text
gpulayer/
├── CLAUDE.md
├── architecture_plan.md
├── pyproject.toml
├── README.md
├── src/
│   └── gpulayer/
│       ├── __init__.py
│       ├── cli/            # gpulayer command
│       ├── sdk/            # Client, @gl.remote, job builders
│       ├── api/            # HTTP server and client, TLS, auth
│       ├── contracts/      # versioned models (JobSpec, WorkerCapabilities, CostRecord, ...)
│       ├── control_plane/  # service.py (jobs, workers, leases), scheduler.py, provisioning.py
│       ├── worker/         # agent, channel, telemetry
│       ├── runtime/        # processes, function runners, distributed.py (ranks, rendezvous)
│       ├── gpu/            # discovery and the support matrix
│       ├── providers/      # adapter interface, FakeProvider, Google Cloud
│       ├── storage/        # metadata store (SQLite), events and logs
│       ├── artifacts/      # content-addressed artifact store
│       ├── observability/  # Prometheus metrics
│       ├── pytorch/        # Trainer, Module, Runtime, launch()
│       └── agents/         # MCP server
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── distributed/
│   └── hardware/
├── docs/
│   ├── architecture/
│   ├── development/
│   └── decisions/
└── site/
    └── README.md
```

Distributed execution lives in `runtime/distributed.py` and the scheduler rather than a
`gpulayer.distributed` package, which would also shadow the public `gpulayer.distributed()`
builder. Events and logs are kept in the metadata store, and cost records are contracts
(`CostRecord`); separate `observability`/`accounting` modules grow only when there is code for
them.

The initial implementation may be a modular monolith. Split services only when deployment, scale, security, or operational requirements justify it.

## 17. Testing and validation strategy

### CPU-only CI

Must validate:

- Contract validation.
- Job lifecycle and state transitions.
- Local CPU execution.
- Scheduler behavior with fake workers.
- Artifact and metadata persistence.
- Failure and cancellation semantics.
- Provider behavior with `FakeProvider`.
- Distributed allocation and rank planning.
- MPS discovery using mocks.
- API authentication and authorization where implemented.

### Integration tests

Validate the actual execution paths:

- Local job submission through the public SDK or CLI.
- Worker registration and remote execution.
- Artifact staging between machines.
- Distributed multi-process execution.
- Provider lifecycle against an explicitly configured test environment.

### Hardware-gated tests

Run only when suitable infrastructure is available:

- NVIDIA GPU execution.
- Multi-GPU distributed execution.
- Multi-node distributed execution.
- Apple MPS execution.
- ROCm execution when implemented.

Test results must state the environment and scope. Mock-based tests cannot establish hardware compatibility.

## 18. Implementation roadmap

The milestones are intentionally overlapping architectural concerns, not three separate products.

### Milestone 1 — Core contracts and local vertical slice

- Define versioned job and capability contracts.
- Implement local CPU execution.
- Provide a working SDK and CLI path.
- Persist job metadata and results.
- Add tests and basic observability.

**Exit criteria:** a user can submit, inspect, and retrieve a real local job through the public interface.

### Milestone 2 — Worker protocol and scheduler

- Implement worker registration and heartbeats.
- Define authenticated worker communication.
- Implement deterministic capability-aware scheduling.
- Add fake workers and placement tests.
- Support explicit local and existing-remote-worker concepts.

**Exit criteria:** the control plane can place work on eligible workers using real reported capabilities, with the remote execution path validated before being advertised.

### Milestone 3 — GPU execution

- Implement NVIDIA GPU detection and a supported execution path.
- Add optional hardware smoke tests.
- Implement Apple MPS discovery and supported independent execution as a separate backend.
- Document runtime compatibility and hardware requirements.

**Exit criteria:** each advertised backend has a concrete tested workload path and clearly documented limitations.

### Milestone 4 — Distributed execution

- Implement distributed job contracts.
- Allocate worker groups.
- Assign ranks and configure rendezvous.
- Launch a supported PyTorch distributed workload.
- Add coordinated status, logs, cancellation, and failure handling.

**Exit criteria:** a documented end-to-end distributed integration test passes in a supported environment. Multi-process single-node testing is a useful step, but does not establish multi-node support.

### Milestone 5 — Cloud provider integration

- Implement provider adapter contracts and `FakeProvider`.
- Select a real provider through an ADR.
- Implement provisioning, bootstrap, registration, execution, and cleanup.
- Add resource estimates when supported by reliable data.
- Test failure and orphaned-resource cleanup.

**Exit criteria:** a real cloud GPU can be provisioned or acquired, verified, used for an actual workload, and released according to policy.

### Milestone 6 — Unified placement

- Combine local, remote, and cloud worker pools.
- Implement explicit placement constraints and policies.
- Support distributed worker-group selection across approved environments.
- Add cost and data-locality preferences where reliable information exists.

**Exit criteria:** a single job interface can target multiple supported environments while respecting compatibility and security constraints.

### Milestone 7 — Production hardening and hosted deployment

- Harden authentication and authorization.
- Document trust boundaries and deployment models.
- Add operational metrics, audit events, and failure recovery.
- Evaluate shared databases and remote object storage.
- Add service deployment and operational documentation.

**Exit criteria:** a documented deployment model can operate with explicit reliability, security, and recovery guarantees.

### Milestone 8 — Website and public beta

- Select the website stack through an ADR.
- Publish installation, architecture, compatibility, and deployment documentation.
- Document actual provider and distributed support.
- Establish contribution, release, and security-reporting procedures.

**Exit criteria:** the public website and documentation accurately represent the product's validated capabilities.

## 19. Architecture Decision Records

Store ADRs in `docs/decisions/`.

Each ADR should include:

- Title and status.
- Context and problem.
- Decision.
- Alternatives considered.
- Consequences and trade-offs.
- Security and operational implications.
- Follow-up work.

Suggested initial ADRs:

1. `ADR-0001-control-plane-and-deployment-model.md`
2. `ADR-0002-job-contracts-and-packaging.md`
3. `ADR-0003-worker-protocol-and-authentication.md`
4. `ADR-0004-scheduling-and-placement-policy.md`
5. `ADR-0005-gpu-capabilities-and-runtime-profiles.md`
6. `ADR-0006-apple-metal-mps-boundaries.md`
7. `ADR-0007-distributed-launch-and-rendezvous.md`
8. `ADR-0008-storage-and-artifact-management.md`
9. `ADR-0009-security-and-workload-isolation.md`
10. `ADR-0010-provider-adapters-and-provisioning.md`
11. `ADR-0011-cost-accounting-and-retry-semantics.md`
12. `ADR-0012-licensing-and-project-governance.md`
13. `ADR-0013-website-and-documentation-stack.md`
14. `ADR-0014-pytorch-integration.md`
15. `ADR-0015-workload-agnostic-core.md`
16. `ADR-0016-performance-parallelism-and-distribution.md`
17. `ADR-0017-distribution-and-installation.md`

Create an ADR index. Use `Proposed` for decisions that still need validation; do not present an undecided option as established policy.

## 20. Open questions

Answered so far (see the ADRs):

- License: Apache-2.0 (ADR-0012).
- First real provider: Google Cloud Compute Engine (ADR-0010).
- Default packaging: source bundle, with JSON arguments and results for functions (ADR-0002).
- Remote protocol: HTTP and JSON, workers polling with long polls (ADR-0003, ADR-0016).
- Worker and client authentication: bearer tokens: client, join, per-worker session, and
  single-use instance tokens for provisioned workers (ADR-0003).
- Distributed runtime: PyTorch distributed with static rendezvous, started by GPUlayer's direct
  launcher or `torchrun` (ADR-0007); groups placed by network locality (ADR-0021).
- Storage for a single control plane: SQLite and a local content-addressed store (ADR-0008).

Still open; resolve before the relevant milestone, rather than prematurely committing the entire project:

- Which database and object storage implementations are needed for a multi-node control plane?
- What isolation guarantees are required for single-user, team, and hosted multi-tenant deployments?
- Which cost and placement policies are necessary as more providers are added?
- Record the website stack: the site is built with Next.js as a static export, but ADR-0013
  still lists the options as Proposed.

Decisions should be made using implementation cost, security, operability, compatibility, and user needs—not by maximizing abstraction.

## 21. Success criteria

GpuLayer is successful when a developer can use a consistent interface to run supported jobs on local resources, remote workers, and cloud GPU infrastructure, and can submit a supported distributed workload across a compatible worker group.

The platform must make placement decisions explainable, keep provider-specific behavior modular, report execution status accurately, and expose limitations honestly.

The core architectural promise is:

**One job model, one capability system, one scheduling layer, and multiple execution environments.**

Local, cloud, and distributed execution are integrated parts of the same platform. Their implementation may mature at different rates, but none should be relegated to an unrelated future architecture.