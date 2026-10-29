# CLAUDE.md

## Mission

Build GpuLayer, an open-source, local-first GPU execution platform with an optional hosted control plane.

GpuLayer enables developers to execute GPU workloads through a consistent SDK and API across local computers, private clusters, and supported external GPU providers.

The platform must support two execution models:

1. **Independent jobs:** Run functions, scripts, inference workloads, and other jobs on a compatible GPU worker.
2. **Distributed jobs:** Coordinate multiple GPUs or multiple machines to execute one logical workload, such as distributed PyTorch training.

The project must also support Apple Silicon GPU execution through Apple's Metal framework using PyTorch's MPS backend for compatible workloads.

Implement features incrementally. Do not attempt to build the entire roadmap in one change.

## 1. Core principles

- Local execution must work without an account or hosted control plane.
- Users must be able to self-host the control plane.
- Hosted services are optional and must use the same core protocol.
- The core project must remain open source under an OSI-approved license.
- Independent and distributed jobs must be explicit concepts in the API.
- GPU compatibility must be based on validated capabilities, not assumptions.
- NVIDIA CUDA, AMD ROCm, Apple Metal/MPS, and CPU execution are distinct backends.
- Do not promise that arbitrary CUDA code runs on AMD or Apple GPUs.
- Do not silently move a job to a different backend unless its runtime requirements explicitly permit it.
- Do not add mandatory telemetry, account registration, or cloud dependencies to local execution.
- Do not claim that a feature works unless it has been tested or its limitation is documented.

## 2. Working rules for Claude Code

Before making changes:

1. Inspect the repository structure, existing code, tests, dependencies, and configuration.
2. Read `architecture_plan.md` and relevant ADRs under `docs/decisions/`.
3. Identify the smallest useful implementation step.
4. Avoid rewriting working subsystems without a concrete reason.
5. Add tests alongside implementation.
6. Run the relevant checks and report which checks passed, failed, or were skipped.
7. Update documentation when behavior or public APIs change.
8. Record major architectural decisions as ADRs.
9. Never invent provider APIs, capabilities, prices, or service terms.
10. Do not implement every milestone at once.

If an architecture decision is unresolved, propose alternatives and record the decision before introducing a hard-to-reverse dependency.

## 3. Product model

### Independent jobs

Independent jobs execute without coordinating distributed processes.

Examples:

- Python functions.
- PyTorch inference.
- Embedding generation.
- Image processing.
- Batch data processing.
- Custom scripts and supported container workloads.
- Jobs submitted to one worker in a private mesh or external provider.

The SDK should eventually support an interface similar to:

```python
import gpulayer as gpu

@gpu.function(gpu="auto")
def process_batch(items: list[str]) -> dict:
    # User workload
    return {"count": len(items)}

job = process_batch.submit(
    ["a", "b", "c"],
    target="local",
)

result = job.result()
print(result.output)
```

This is a proposed API, not an existing implementation. Verify the actual code before documenting it as available.

### Distributed jobs

A distributed job is one logical workload that launches coordinated processes across a specified group of GPUs or machines.

Examples:

- Multi-GPU PyTorch training.
- Multi-node distributed training.
- Framework-supported distributed inference.
- Other explicitly supported distributed workloads.

Distributed jobs must specify their resource and topology requirements. They must not be implemented as a hidden extension of independent-job scheduling.

The architecture should support both models from the beginning, but implementations and compatibility guarantees must be staged.

## 4. Suggested module boundaries

Keep the initial implementation a modular monolith with clear interfaces.

Suggested structure:

```text
src/gpulayer/
├── api/             # Public Python SDK
├── cli/             # Command-line interface
├── contracts/       # Versioned job and worker contracts
├── control/         # Coordinator and job state machine
├── scheduler/       # Resource matching and placement
├── worker/          # Worker agent, authentication, heartbeats
├── runtime/         # Execution environments and launchers
├── discovery/       # Hardware and runtime capability probes
├── providers/       # External GPU provider adapters
├── artifacts/       # Input and output artifact storage
├── observability/   # Logs, events, usage and metrics
└── config/          # Configuration and settings
```

Adjust this structure to fit the repository. Do not create empty modules just to match the proposed tree.

Workers and job runners should eventually run as separate processes, so coordinator failures and job execution are not unnecessarily coupled.

## 5. Job contracts and lifecycle

Use typed, versioned contracts.

Core domain types should include:

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

A job specification should capture the entrypoint, arguments, runtime, environment requirements, resource constraints, artifact references, and relevant execution policy.

Use an explicit lifecycle, for example:

`queued → validating → scheduled → starting → running → succeeded`

Terminal failure states should include `failed` and `cancelled`.

Requirements:

- Make state transitions explicit and testable.
- Support idempotency keys for submissions.
- Persist enough state to recover after coordinator restarts.
- Reconcile jobs whose workers disconnect or crash.
- Do not promise exactly-once execution.
- Design retries around the possibility of duplicate side effects.
- Preserve logs and failure reasons when possible.

## 6. Independent-job scheduling

The scheduler must:

1. Validate the job specification.
2. Identify hard resource and runtime constraints.
3. Discover compatible workers and provider offers.
4. Reject incompatible backends.
5. Rank feasible targets according to configured policy.
6. Reserve resources using leases or equivalent mechanisms.
7. Dispatch the job.
8. Track execution and collect results.
9. Release reservations and reconcile failures.

Support explainable placement decisions.

Potential scheduling inputs include:

- Backend and runtime profile.
- Available GPU count.
- GPU memory or backend-specific memory capacity.
- CPU and RAM requirements.
- Worker availability.
- Network and topology requirements.
- Data locality.
- Estimated execution cost, when available.
- User-defined placement preferences.

A missing capability probe means the capability is unknown. It must not be interpreted as proof that hardware is absent.

## 7. Distributed execution

Distributed jobs must explicitly declare:

- Worker count.
- GPUs per worker.
- Runtime and framework.
- Entry point and arguments.
- Network requirements.
- Rendezvous configuration.
- Resource constraints.
- Failure and cleanup policy.

The distributed scheduler must:

1. Validate the requested topology and runtime.
2. Find a feasible group of workers.
3. Reserve the group before launch.
4. Verify required network connectivity.
5. Establish rendezvous and rank assignments.
6. Launch coordinated processes.
7. Monitor all ranks.
8. Collect logs and results.
9. Clean up partially launched groups on failure.

Do not restart a single rank independently unless the chosen distributed runtime explicitly supports that recovery behavior.

Implementation order:

1. Local multi-GPU PyTorch on validated hardware.
2. Two-node execution on a private LAN or VPN.
3. General private cluster support.
4. External providers with explicit placement-group and networking support.
5. Additional frameworks after compatibility testing.

Do not promise distributed training across arbitrary machines on the public internet.

## 8. GPU backends and portability

Use explicit runtime profiles instead of claiming universal GPU portability.

Initial profile names may include:

- `cpu`
- `pytorch-nvidia`
- `pytorch-distributed-nvidia`
- `pytorch-metal-mps`
- `pytorch-amd`, only after ROCm support is validated

Profiles must describe tested software/backend combinations, not merely the presence of a GPU.

The scheduler must match job requirements against worker capabilities and the selected runtime profile.

### 8.1 NVIDIA CUDA

- Detect NVIDIA devices using supported platform tools.
- Capture CUDA and driver compatibility information when available.
- Distinguish GPU discovery from a successfully tested runtime.
- Support distributed CUDA workloads only on validated configurations.
- Do not assume every NVIDIA GPU supports every model or workload.

### 8.2 AMD ROCm

- Add AMD support only after testing supported hardware and ROCm/PyTorch combinations.
- Keep ROCm discovery and runtime setup separate from CUDA-specific code.
- Do not claim CUDA extensions work on ROCm.

### 8.3 Apple Metal and PyTorch MPS

Support compatible independent GPU jobs on Apple Silicon Macs through PyTorch's MPS backend.

Metal/MPS requirements:

- Detect MPS inside the selected Python/PyTorch environment.
- Use `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` when those APIs are available.
- Keep macOS-specific probing isolated so it skips cleanly on Linux and Windows.
- Record macOS version, CPU architecture, Python version, PyTorch version, MPS availability, and device details where reliably available.
- Use an actual smoke test in addition to hardware detection.
- Report unsupported operations and runtime failures clearly.
- Treat MPS as a distinct backend, not as CUDA compatibility.
- Do not assume CUDA-only libraries, custom CUDA extensions, every operator, every dtype, or CUDA performance characteristics are supported.
- Do not silently route a CUDA-specific job to MPS.

Apple Silicon uses unified memory rather than conventional discrete GPU VRAM. Represent memory with backend-aware metadata such as:

- `memory_kind`
- `memory_capacity_bytes`, if reliably known
- `recommended_max_memory_bytes`, if available
- `memory_confidence`

Do not treat Apple unified memory as directly equivalent to NVIDIA discrete VRAM when scheduling workloads.

Eligible MPS jobs should work in local mode. An Apple Silicon Mac may also participate in a private mesh for independent jobs.

**Do not enable distributed training across multiple MPS workers in the initial release.** Require verified framework support, process-group and collective compatibility, networking validation, and dedicated multi-node tests before advertising that capability.

### 8.4 Capability discovery

Implement hardware detection behind a stable interface.

For each worker, report:

- Operating system and architecture.
- Available backend profiles.
- Runtime versions.
- GPU count and device identifiers where available.
- Memory information and its confidence.
- Supported execution modes.
- Probe status and diagnostic errors.

Keep probing resilient. A failure to import PyTorch must not crash the entire daemon.

## 9. Worker protocol and private mesh

Workers must authenticate to the coordinator.

The worker protocol should support:

- Enrollment and revocation.
- Heartbeats.
- Capability reporting.
- Job dispatch.
- Logs and status updates.
- Resource leases.
- Cancellation.
- Protocol version compatibility.
- Worker health and reconciliation.

Prefer outbound worker connections to the coordinator to reduce inbound firewall requirements.

Distributed jobs may require additional peer-to-peer connectivity. Document these network requirements explicitly.

Initially assume a trusted private mesh of machines controlled by the same user or organization. Do not imply that the system safely runs hostile multi-tenant workloads until isolation and security have been reviewed.

## 10. Provider adapters

Define a provider-neutral adapter interface.

Expected operations may include:

- `health_check`
- `list_offers`
- `get_capabilities`
- `submit`
- `get_status`
- `fetch_logs`
- `cancel`
- `get_usage`

Start with a `FakeProvider` for deterministic tests.

Add real provider integrations incrementally. Do not hardcode provider-specific behavior into the core scheduler.

Distributed jobs require providers that explicitly support the necessary worker grouping, placement, and network semantics. Do not force distributed workloads into an adapter that only supports independent single-worker jobs.

Keep user-owned credentials separate from platform credentials.

## 11. Storage and persistence

Local mode:

- SQLite for job metadata.
- Local filesystem for artifacts.
- No external database or hosted service requirement.

Self-hosted multi-user mode:

- PostgreSQL for shared job metadata.
- Filesystem or S3-compatible artifact storage.

Requirements:

- Keep the database as the source of truth.
- Treat a queue as a delivery mechanism, not the sole record of job state.
- Use idempotent transitions and reconciliation.
- Define artifact lifecycle and cleanup policies.
- Document backup and restore for self-hosted deployments.

## 12. Security and privacy

- Authenticate worker connections.
- Support worker revocation.
- Scope credentials to the smallest possible set of jobs.
- Do not expose host secrets, provider credentials, Docker sockets, or cloud metadata to workloads.
- Do not treat containers alone as sufficient isolation for hostile code.
- Do not expose arbitrary public job execution until the isolation model has been reviewed.
- Make telemetry disabled by default or explicitly opt-in.
- Never upload source code, job inputs, or artifacts without a clear user-configured action.
- Document trust assumptions for local and mesh deployments.
- Provide a security reporting process before public beta.

## 13. Cost and usage accounting

Separate:

- Local resource usage.
- Provider-reported usage.
- Estimated cost.
- Final provider cost.
- Any explicitly configured platform/customer charge.

Use decimal arithmetic for monetary values.

Do not invent provider prices. Do not present estimates as final charges.

Local execution should not require billing. Any hosted or paid-provider usage must be explicitly configured by the user.

## 14. Website and documentation

Keep the public website in `site/`, separate from the Python runtime package.

The site should build and preview locally without cloud credentials.

Initial pages:

- Product overview.
- Quickstart.
- Installation and CLI reference.
- Hardware/backend compatibility matrix.
- Apple Silicon MPS guide and limitations.
- Private mesh setup.
- Distributed execution requirements and supported topologies.
- Provider adapter documentation.
- Self-hosting guide.
- Architecture and ADR index.
- Contributing guide.
- License and code of conduct.
- Security reporting.

Choose the static-site framework through an ADR. Astro is a candidate, not a locked decision.

Avoid maintaining duplicate versions of the same documentation. Establish a clear source of truth for Markdown and website content.

Add site build and link checks to CI after the framework is selected.

## 15. Architecture Decision Records

Store ADRs in `docs/decisions/` using names such as:

`ADR-0001-local-first-control-plane.md`

Each ADR must include:

- Status: `Proposed`, `Accepted`, or `Superseded`.
- Date.
- Context.
- Decision.
- Alternatives considered.
- Consequences.
- Follow-up work.

Initial ADRs:

1. Local-first control plane.
2. Job and function packaging.
3. Worker protocol versioning.
4. Distributed launch and rendezvous.
5. GPU capability and runtime profiles.
6. Metal/MPS support boundaries.
7. Storage and persistence.
8. Security and trust model.
9. Open-source license and governance.
10. Provider accounts and commercial terms.
11. Cost and retry semantics.
12. Website and documentation stack.

Do not mark an ADR accepted without recording the rationale and relevant trade-offs.

## 16. Testing requirements

Tests must include:

- Job contract validation.
- Job lifecycle transitions.
- Successful and failed local execution.
- Cancellation.
- Duplicate submissions.
- Worker loss and reconciliation.
- Resource reservation and release.
- Scheduler rejection of incompatible backends.
- Fake provider behavior.
- Artifact and log handling.
- Distributed launch and partial-launch cleanup where supported.
- MPS discovery behavior, including the unavailable case.
- A hardware-gated MPS smoke test on eligible Apple Silicon environments.

CPU-only CI must not require a GPU.

MPS tests should skip cleanly when MPS is unavailable. Do not treat skipped hardware tests as proof that the backend works.

Hardware and multi-node tests should be separately documented.

## 17. Coding standards

- Use type hints for public APIs and core domain models.
- Prefer small, testable modules.
- Keep provider-specific code behind adapters.
- Use explicit exceptions and useful diagnostic messages.
- Avoid hidden global state.
- Avoid premature abstractions that have no real use case.
- Keep configuration validated and documented.
- Do not log secrets or sensitive job inputs.
- Keep dependencies minimal.
- Add compatibility tests before expanding supported runtime profiles.

## 18. Implementation roadmap

1. Inspect the existing repository and document current state.
2. Add versioned job contracts and lifecycle.
3. Implement local CPU command/function execution end-to-end.
4. Add logs, results, artifacts, status, and cancellation.
5. Add GPU discovery, local resource-aware scheduling, and MPS detection.
6. Add tested local multi-GPU distributed execution.
7. Add authenticated workers and private mesh independent jobs.
8. Add two-node distributed execution on a supported network.
9. Add `FakeProvider` and then one real provider.
10. Add optional PostgreSQL/S3 self-hosting.
11. Add the `site/` website and initial ADRs.
12. Harden security, packaging, metering, documentation, and releases.

## 19. Definition of done

A feature is done only when:

- The implementation is complete for its declared scope.
- Relevant tests have been added.
- Relevant tests have actually been run.
- Errors and unsupported configurations are documented.
- Public API and architecture documentation match the implementation.
- Security and compatibility implications have been considered.
- No unsupported backend or topology is advertised.
- The final report states exactly what changed and what remains incomplete.
