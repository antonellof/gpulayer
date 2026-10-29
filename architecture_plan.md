# GpuLayer Architecture Plan

**Status:** Proposed architecture  
**Project:** GpuLayer — working name, subject to trademark, package, and domain checks  
**Product model:** Open source, local-first, optional hosted control plane  
**Execution model:** Independent jobs and explicitly distributed jobs

## 1. Vision

GpuLayer is an open-source GPU execution platform that lets developers submit workloads through a consistent API without hard-coding their applications to one GPU provider.

The platform should allow a developer to:

- Run a GPU job on their current computer.
- Use Apple Silicon GPUs through Metal/MPS for compatible workloads.
- Combine trusted local computers into a private execution mesh.
- Submit independent jobs to available compatible workers.
- Run distributed workloads across multiple GPUs or machines where the framework, network, and hardware support it.
- Add external GPU providers through adapters.
- Self-host the control plane.
- Optionally use a hosted control plane without making it a requirement for local execution.

The long-term goal is a provider-neutral GPU execution layer. The initial goal is a reliable local execution engine that can grow into a private mesh and provider-routing system.

## 2. Product principles

1. **Local-first:** Local execution must not require a hosted service, account, or mandatory telemetry.
2. **Open source:** The core platform should use an OSI-approved license.
3. **Provider-neutral:** Provider integrations belong behind stable adapter interfaces.
4. **Explicit execution semantics:** Independent jobs and distributed jobs are different job types.
5. **Capability-aware scheduling:** The scheduler matches declared job requirements against validated worker capabilities.
6. **Honest portability:** A common API does not imply universal compatibility between CUDA, ROCm, Metal/MPS, and CPU.
7. **Optional hosting:** Self-hosted and hosted control planes should use the same core contracts where practical.
8. **Security by design:** Trusted private meshes come before arbitrary public multi-tenant execution.
9. **Incremental delivery:** Build and validate each milestone before expanding the supported matrix.

## 3. Supported execution models

### 3.1 Independent jobs

An independent job executes on one selected worker or provider allocation.

Examples include:

- PyTorch inference.
- Embedding generation.
- Image and video processing.
- Batch data processing.
- Python functions.
- Custom scripts and supported containers.
- Other workloads that fit a declared runtime profile.

The scheduler may place independent jobs on local hardware, a private mesh worker, or a compatible external provider.

### 3.2 Distributed jobs

A distributed job represents one logical workload that requires multiple coordinated processes.

Examples include:

- Multi-GPU training.
- Multi-node PyTorch distributed training.
- Distributed inference where the framework and topology are supported.
- Other explicitly supported distributed workloads.

A distributed job must declare its worker count, GPUs per worker, runtime, network requirements, and failure policy.

The system must reserve a feasible group of workers and coordinate launch, rendezvous, monitoring, and cleanup.

**Distributed execution is not simply independent scheduling repeated several times.**

## 4. Deployment modes

### Mode A: Local single machine

Components:

- CLI.
- Python SDK.
- Local coordinator/daemon.
- SQLite metadata database.
- Local artifact storage.
- Local worker/runtime process.

Properties:

- Works without an account.
- No hosted control plane dependency.
- CPU jobs run on ordinary development machines.
- NVIDIA CUDA is available when a compatible environment is installed.
- AMD ROCm is supported only on validated configurations.
- Apple Silicon GPUs are supported through PyTorch MPS for eligible workloads.

### Mode B: Private mesh

Components:

- One coordinator.
- Multiple authenticated worker agents.
- Resource discovery and scheduling.
- Job logs and artifact exchange.
- Optional shared artifact storage.

Initially, assume the machines are trusted and controlled by the same person or organization.

Workers should preferably establish outbound authenticated connections to the coordinator. Distributed jobs may additionally require peer connectivity.

### Mode C: Self-hosted control plane

Optional components:

- PostgreSQL.
- S3-compatible artifact storage.
- Coordinator service.
- Worker agents.
- Monitoring and backup procedures.

The self-hosted deployment must not depend on GpuLayer-operated infrastructure.

### Mode D: Optional hosted control plane

The hosted offering may provide:

- Remote coordination.
- Worker and provider management.
- Usage reporting.
- Job history.
- Multi-user access control.

It must remain optional. Local mode must continue to work independently.

Any paid-provider or hosted spending must be explicitly configured by the user.

## 5. Proposed SDK

### Independent job example

```python
import gpulayer as gpu

@gpu.function(gpu="auto", gpu_memory_gb=12)
def embed_batch(texts: list[str]) -> dict:
    import torch
    model = load_model()
    return {"embeddings": model.encode(texts).tolist()}

job = embed_batch.submit(
    ["hello", "world"],
    target="local",
)

result = job.result()
print(result.output)
```

### Distributed job example

```python
from gpulayer import DistributedJob, ResourceRequest

job = DistributedJob(
    entrypoint="train.py",
    command=["--config", "configs/train.yaml"],
    workers=2,
    gpus_per_worker=4,
    runtime="pytorch-distributed",
    resources=ResourceRequest(
        accelerator="nvidia",
        gpu_memory_gb=40,
        network="multi-node",
    ),
)

handle = gpu.submit(job, target="mesh")
handle.wait()

print(handle.status())
print(handle.logs())
```

These are design sketches. They must not be documented as implemented APIs until the corresponding implementation exists.

The API should make the execution target explicit when useful, while allowing a scheduler to select among compatible resources when the user requests automatic placement.

## 6. High-level architecture

```text
Python SDK / CLI
       |
       v
Versioned JobSpec
       |
       v
Coordinator
       |
       +--> Job state machine
       |
       +--> Scheduler and resource leases
       |
       +--> Capability registry
       |
       +--> Local worker
       |
       +--> Private mesh workers
       |
       +--> Provider adapters
                  |
                  +--> External GPU providers

Execution runtime
       |
       +--> Logs and events
       +--> Artifacts and results
       +--> Usage accounting
       +--> Reconciliation
```

The initial system should be a modular monolith. Split services only when there is a clear operational or scaling reason.

## 7. Core domain types

Expected types include:

- `JobSpec`
- `FunctionJob`
- `CommandJob`
- `DistributedJob`
- `ResourceRequest`
- `WorkerCapabilities`
- `ProviderOffer`
- `JobAttempt`
- `JobResult`
- `ArtifactRef`
- `UsageRecord`

A job specification should include:

- Schema version.
- Job type.
- Entrypoint and arguments.
- Runtime and dependency requirements.
- Resource constraints.
- Backend compatibility requirements.
- Input artifact references.
- Output and log references.
- Execution target or placement policy.
- Retry and cancellation policy.

Avoid serializing arbitrary Python closures as the only job format. Use an explicit packaging and execution contract with documented limitations.

## 8. Scheduling design

### 8.1 Independent jobs

The scheduling flow is:

1. Validate the job specification.
2. Determine hard resource and runtime constraints.
3. Discover compatible workers and provider offers.
4. Filter incompatible backends and runtime profiles.
5. Rank the remaining candidates.
6. Reserve resources.
7. Dispatch the job.
8. Track status and collect logs/results.
9. Release reservations and reconcile failures.

Candidate ranking can eventually consider:

- Availability.
- GPU model and count.
- Runtime compatibility.
- Memory requirements.
- Data locality.
- Network constraints.
- User-defined placement preferences.
- Estimated cost, when reliable cost information is available.

The scheduler must explain why a target was selected or rejected.

### 8.2 Distributed jobs

Distributed scheduling must:

1. Validate runtime and topology.
2. Find a feasible worker group.
3. Reserve the group.
4. Validate network connectivity.
5. Establish rendezvous and rank assignment.
6. Launch all required processes.
7. Monitor every rank.
8. Collect logs and status.
9. Clean up the whole group when a partial launch fails.

Do not assume that independent workers can automatically form a valid distributed training group.

A provider adapter must explicitly advertise placement-group and networking support before it can be selected for a distributed job.

## 9. GPU backend and runtime compatibility

GpuLayer should use explicit runtime profiles.

| Profile | Intended scope | Initial status |
|---|---|---|
| `cpu` | CPU workloads | Baseline |
| `pytorch-nvidia` | Compatible PyTorch workloads on NVIDIA CUDA | Validate incrementally |
| `pytorch-distributed-nvidia` | Supported distributed PyTorch configurations | Start with tested configurations |
| `pytorch-metal-mps` | Compatible PyTorch workloads on Apple Silicon | Support local independent jobs |
| `pytorch-amd` | Compatible PyTorch workloads on AMD ROCm | Add after validation |

The profile name is not proof of compatibility. Each profile must define supported versions and pass automated smoke tests.

### 9.1 Apple Metal / PyTorch MPS support

Apple Silicon Macs must be supported as local GPU workers through PyTorch's MPS backend when available.

#### Discovery

Probe the selected Python environment using:

```python
import torch

mps_built = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_built()
)

mps_available = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)
```

Guard the probe so missing PyTorch, missing APIs, and non-macOS systems produce a clear capability result instead of crashing the daemon.

A successful availability probe should be followed by a small runtime smoke test where practical.

#### Capability metadata

Capture, where reliably available:

- Operating system and version.
- CPU architecture.
- Python version.
- PyTorch version.
- MPS build and availability status.
- Device details.
- Backend-specific memory information.
- Probe errors and compatibility diagnostics.

Apple Silicon uses unified memory. Do not represent its memory capacity as if it were directly equivalent to discrete NVIDIA VRAM.

Use backend-aware fields such as `memory_kind`, `memory_confidence`, and optional recommended working-set information.

#### Compatibility rules

- Metal/MPS is a distinct backend, not a CUDA implementation.
- CUDA-only packages and custom CUDA extensions cannot be assumed to work.
- Not every PyTorch operator or dtype is necessarily supported.
- Performance and numerical behavior must not be assumed identical to CUDA.
- Unsupported workloads should fail with useful diagnostics.
- Never silently change a CUDA-specific job to MPS.

#### Scheduling rules

Eligible MPS jobs should run locally on Apple Silicon Macs. These Macs may also participate in a private mesh for independent jobs.

For the initial release, **do not schedule distributed jobs across multiple MPS workers**.

Before enabling that capability, verify:

- Supported process-group and collective behavior.
- Framework and backend compatibility.
- Inter-node networking and rendezvous.
- Failure and cleanup behavior.
- Dedicated multi-node integration tests.

Do not advertise multi-Mac distributed training as supported before those requirements are satisfied.

### 9.2 NVIDIA CUDA

Detect supported NVIDIA hardware and runtime versions through platform-appropriate tooling.

Validate:

- Driver and CUDA compatibility.
- PyTorch runtime compatibility.
- GPU count and device visibility.
- Memory availability.
- Distributed launch support for the selected configuration.

Do not assume that every NVIDIA device or CUDA extension works with every runtime.

### 9.3 AMD ROCm

Treat AMD ROCm as a distinct runtime. Add support after testing actual supported hardware and software combinations.

Do not infer CUDA compatibility from the presence of an AMD GPU.

### 9.4 Capability discovery rules

- Keep backend probes modular.
- Isolate macOS-specific logic.
- Missing tooling means capability is unknown.
- Report runtime and probe errors explicitly.
- Avoid hard-coding hardware support from machine names alone.
- Make scheduler compatibility depend on validated profiles and job requirements.

## 10. Provider adapter protocol

A provider adapter should expose operations equivalent to:

- `health_check`
- `list_offers`
- `get_capabilities`
- `submit`
- `get_status`
- `fetch_logs`
- `cancel`
- `get_usage`

Start with `FakeProvider` so scheduling, lifecycle, retry, and error handling can be tested without external accounts.

Add real providers incrementally.

Keep provider-specific API details and pricing out of the core scheduler.

Distributed workloads require explicit support for worker groups, placement, networking, and coordinated launch. An adapter that only supports independent jobs must not be used for distributed jobs.

## 11. Worker security and networking

The worker protocol should support:

- Authentication.
- Enrollment and revocation.
- Heartbeats.
- Capability updates.
- Job dispatch.
- Logs and state updates.
- Cancellation.
- Resource leases.
- Version compatibility.

Prefer outbound worker connections to the coordinator.

Distributed jobs may need additional worker-to-worker connectivity. Document firewall and VPN requirements.

The first private mesh should target trusted machines. Public multi-tenant arbitrary-code execution requires a separate security review.

Do not expose:

- Host secrets.
- Provider credentials.
- Cloud metadata endpoints.
- Docker sockets.
- Unrestricted host filesystem access.

Containers alone must not be treated as sufficient isolation for hostile workloads.

## 12. Job lifecycle, persistence, and recovery

Use a durable state machine.

Example lifecycle:

`queued → validating → scheduled → starting → running → succeeded`

Failure and cancellation should be explicit terminal states.

Local mode should use SQLite and local artifact storage by default.

Self-hosted multi-user deployments may use PostgreSQL and S3-compatible object storage.

Requirements:

- Database is the source of truth.
- Queues are delivery mechanisms, not the sole record of job state.
- Submissions and state transitions are idempotent where practical.
- Resource leases expire or are reconciled.
- Coordinator restarts must not lose all knowledge of active jobs.
- Worker loss must produce a clear state transition.
- Retry semantics must acknowledge at-least-once execution and possible duplicate side effects.

## 13. Artifacts, logs, and results

Define an artifact abstraction supporting local storage and, later, S3-compatible storage.

The system should eventually support:

- Input bundles.
- Output artifacts.
- Structured results.
- Standard output and standard error.
- Worker and scheduler events.
- Artifact retention policies.
- Explicit cleanup.
- Artifact transfer between workers when required.

Do not upload source code, inputs, or outputs to a hosted service unless the user has explicitly configured that behavior.

## 14. Cost and usage accounting

Distinguish:

- Local resource usage.
- Provider-reported usage.
- Estimated cost.
- Final provider cost.
- Any explicitly configured platform charge.

Use decimal arithmetic for monetary calculations.

Never invent provider prices. Estimates must be labeled as estimates.

Local execution should not require billing. Hosted or paid-provider use must be opt-in and configured.

For distributed jobs, aggregate usage across workers while preserving per-worker records.

## 15. Open-source governance

Choose an OSI-approved license before public release. Apache-2.0 is a candidate, but the final decision belongs in an ADR.

Prepare:

- `LICENSE`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- Issue and pull-request templates.
- Dependency and license review.
- Release and versioning policy.
- A process for reporting security vulnerabilities.

Telemetry should be disabled by default or explicitly opt-in.

The open-source core must remain usable without a GpuLayer-hosted account.

## 16. Website: `site/`

Create a `site/` folder for the project's public website.

The site should be independent of the Python package and build locally without hosted-service credentials.

Proposed structure:

```text
site/
├── README.md
├── package.json             # After selecting a framework
├── src/
│   ├── pages/
│   ├── components/
│   └── styles/
└── public/
```

The exact structure depends on the chosen static-site framework.

### Initial pages

1. Home and product overview.
2. Quickstart.
3. Installation and CLI reference.
4. Supported hardware and backend compatibility matrix.
5. Apple Silicon Metal/MPS setup and limitations.
6. Private mesh installation and security.
7. Distributed jobs and supported topologies.
8. Provider adapters.
9. Self-hosted deployment.
10. Optional hosted control plane.
11. Architecture and ADR index.
12. Contributing and code of conduct.
13. License and security reporting.

A static-site framework such as Astro is a candidate, not a finalized choice.

Record the framework decision in an ADR. Consider accessibility, search, documentation reuse, deployment simplicity, maintenance burden, and contributor experience.

Avoid maintaining duplicate versions of the same instructions. Define whether website content is generated from Markdown in `docs/` or maintained separately.

Add site build and broken-link checks to CI once the framework is selected.

## 17. Architecture Decision Records

Create:

```text
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

Each ADR must contain:

- Status: `Proposed`, `Accepted`, or `Superseded`.
- Date.
- Context.
- Decision.
- Alternatives considered.
- Consequences.
- Follow-up work.

### Initial ADR topics

| ADR | Decision area |
|---|---|
| 0001 | Local-first, self-hosted, and hosted control planes |
| 0002 | Job packaging, arguments/results, and runtime reproducibility |
| 0003 | Worker protocol versioning and compatibility |
| 0004 | Distributed launch, rendezvous, networking, and failure behavior |
| 0005 | GPU capabilities, runtime profiles, and scheduler matching |
| 0006 | Metal/MPS support boundaries and unified-memory semantics |
| 0007 | SQLite/PostgreSQL and artifact storage |
| 0008 | Worker trust, workload isolation, and security |
| 0009 | Open-source license and governance |
| 0010 | Provider credentials and commercial integration model |
| 0011 | Cost accounting, idempotency, and retry semantics |
| 0012 | Website framework, docs source of truth, hosting, and CI |

All ADRs start as proposed. Do not mark them accepted until alternatives and consequences have been evaluated.

## 18. Testing strategy

### Always-on CI

Tests must run without GPU hardware.

Include:

- Job specification validation.
- State machine transitions.
- Local CPU execution.
- Success, failure, and cancellation.
- Duplicate submissions.
- Artifact handling.
- Scheduler constraint matching.
- Worker disconnection and recovery.
- Fake provider integration.
- Invalid backend/profile rejection.
- Configuration validation.

### Hardware-gated tests

Run where compatible hardware is available:

- NVIDIA GPU discovery and smoke tests.
- Local multi-GPU distributed tests.
- Two-node distributed tests.
- Apple Silicon MPS smoke tests.
- AMD ROCm tests once supported.

A skipped hardware test is not evidence of successful hardware support.

### Failure injection

Test:

- Worker termination during execution.
- Coordinator restart.
- Expired resource leases.
- Partial distributed launch.
- Provider submission timeouts.
- Duplicate requests.
- Artifact transfer failures.
- Unsupported MPS operators and runtime failures.

## 19. Development stack

Suggested initial choices:

- Python 3.11 or newer.
- `uv` for Python dependency and environment management.
- FastAPI and Pydantic for the control API and contracts.
- Typer for the CLI.
- SQLite for local metadata.
- PostgreSQL as an optional self-hosted database.
- Local filesystem and later S3-compatible artifact storage.
- Pytest for tests.
- Ruff for linting and formatting.
- Mypy or Pyright for type checking.
- OpenTelemetry-compatible instrumentation, with telemetry opt-in.
- `torchrun` for validated PyTorch distributed execution.
- A static-site framework selected through ADR for `site/`.

Do not add every suggested dependency immediately. Choose only what the current milestone needs.

## 20. Implementation roadmap

### Milestone 0 — foundations

- Inspect the existing repository.
- Confirm package structure and supported Python versions.
- Establish CI and basic quality checks.
- Create the ADR process.
- Decide the license before public release.
- Define the initial job contract and local security model.

**Exit criteria:** the repository has reproducible development setup, tests, and documented implementation boundaries.

### Milestone 1 — local execution

- Implement the job contract and lifecycle.
- Run CPU command/function jobs locally.
- Add logs, results, artifacts, status, and cancellation.
- Add local coordinator and SQLite persistence.
- Add GPU discovery interfaces.
- Add macOS MPS detection and a hardware-gated smoke test.

**Exit criteria:** a local job can be submitted, executed, inspected, and retrieved without an account or hosted service. MPS discovery degrades cleanly when unavailable.

### Milestone 2 — local GPU execution

- Add validated NVIDIA GPU execution.
- Add the `pytorch-metal-mps` profile for eligible Apple Silicon jobs.
- Add backend-aware resource matching.
- Document MPS limitations and memory semantics.
- Add compatibility tests.

**Exit criteria:** supported local GPU workloads run on declared backends and incompatible jobs are rejected clearly.

### Milestone 3 — local distributed execution

- Add the `DistributedJob` contract.
- Launch validated multi-GPU PyTorch jobs.
- Implement rank assignment, rendezvous, monitoring, and cleanup.
- Add failure injection.

**Exit criteria:** a documented distributed smoke test works on a supported multi-GPU configuration. The initial target is validated NVIDIA/CUDA hardware; distributed MPS is not implied.

### Milestone 4 — private mesh independent jobs

- Implement worker enrollment and authentication.
- Add heartbeat and capability reporting.
- Add scheduling and resource leases.
- Support independent jobs across trusted machines.
- Add logs, artifacts, and recovery.

**Exit criteria:** compatible independent jobs can execute across a private mesh.

### Milestone 5 — two-node distributed execution

- Add multi-node rendezvous and network validation.
- Reserve worker groups.
- Launch and monitor all ranks.
- Handle partial launch and group cleanup.
- Document LAN/VPN requirements.

**Exit criteria:** distributed execution works on a documented and tested topology.

### Milestone 6 — first external provider

- Implement the provider adapter contract.
- Add `FakeProvider`.
- Add one real provider after verifying its current API and terms.
- Support independent jobs first.
- Add usage and cancellation handling.

**Exit criteria:** an external provider can be used without provider-specific logic leaking into the core scheduler.

### Milestone 7 — provider routing

- Add additional providers.
- Normalize capabilities.
- Improve placement explanations.
- Add configurable cost and placement preferences.
- Add explicit support checks for distributed provider workloads.

**Exit criteria:** the scheduler can select among compatible offers without misrepresenting capabilities or costs.

### Milestone 8 — self-hosting

- Add PostgreSQL support where needed.
- Add S3-compatible artifact storage.
- Document deployment, upgrades, backup, and restore.
- Verify hosted features are not required for local operation.

**Exit criteria:** users can run the control plane under their own infrastructure.

### Milestone 9 — website and public documentation

- Add `site/` and choose its framework through ADR.
- Publish the landing page and quickstart.
- Document hardware/runtime compatibility.
- Publish local, mesh, distributed, and self-hosting guides.
- Publish contribution, license, and security information.
- Add site build checks to CI.

**Exit criteria:** contributors can build the site locally, and users can understand supported capabilities and limitations.

### Milestone 10 — public beta hardening

- Review workload isolation and security.
- Validate provider integrations and terms.
- Improve metering and reconciliation.
- Complete installation and upgrade documentation.
- Verify licensing and dependencies.
- Publish known limitations and a compatibility matrix.
- Run end-to-end tests on supported hardware.

**Exit criteria:** the supported feature set is documented, reproducible, and tested.

## 21. Non-goals for the initial release

Do not initially promise:

- Universal CUDA-to-ROCm or CUDA-to-Metal portability.
- Arbitrary distributed training across any internet-connected computers.
- Multi-node distributed MPS across Macs.
- Transparent remote execution of individual PyTorch tensor operations.
- Exactly-once job execution.
- Secure public execution of arbitrary untrusted code without a reviewed isolation architecture.
- Universal compatibility across every provider and GPU.
- Accurate cost estimates where provider pricing or usage data is unavailable.

## 22. Initial priorities

The recommended first implementation sequence is:

1. Local CPU execution.
2. Versioned job contracts and reliable lifecycle.
3. Local artifact and log handling.
4. GPU discovery and runtime compatibility.
5. NVIDIA and Apple Silicon MPS independent-job support.
6. Validated local multi-GPU distributed execution.
7. Private mesh independent jobs.
8. Two-node distributed execution.
9. Provider adapter and first provider.
10. Self-hosting, website, and public beta hardening.

The design should anticipate all required execution modes, but implementation must remain narrow enough that every advertised feature can be tested and supported.
