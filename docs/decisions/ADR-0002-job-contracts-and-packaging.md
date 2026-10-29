# ADR-0002: Job contracts and packaging

- **Status:** Accepted (milestone 1). Container-image packaging and runtime profiles are Proposed.
- **Date:** 2026-10-29

## Context

Every job must be describable without knowing where it will run. Remote workers do not
share the submitter's filesystem, Python environment or memory. Pickled closures are not
portable across Python versions or library versions, and they are unsafe to deserialize.

## Decision

- **Contracts** (`src/gpulayer/contracts/models.py`) are pydantic v2 models with
  `extra="forbid"`, carried with `api_version = "gpulayer.dev/v1alpha1"`.
  `JobSpec.workload` is a discriminated union:
  - `CommandJob`: an argv, optionally run inside an extracted source bundle. A leading
    `python` is replaced with the worker's configured interpreter (documented behavior).
  - `FunctionJob`: `module:function` plus JSON `args`/`kwargs`. The return value must be
    JSON-serializable. It runs through a stdlib-only runner
    (`runtime/function_runner.py`) that is copied into the workspace, so the workload
    interpreter does not need gpulayer installed.
  - `DistributedJob`: one coordinated torchrun workload (ADR-0007).
- **Workload type** (independent or distributed) is derived from the workload kind.
  **Execution target** (local, remote or cloud) lives in `PlacementPolicy`. The two are
  independent.
- **Packaging.** `build_source_bundle` (`artifacts/store.py`) produces a deterministic
  zip: sorted entries, fixed timestamps, VCS/cache dirs and symlinks excluded. It is
  uploaded as a content-addressed artifact (`artifact://sha256/<hex>`) and verified by
  digest before extraction on the worker.
- Validation happens at construction and again at the control-plane boundary
  (`ControlPlane.submit` re-validates and checks that referenced artifacts exist).
  Contradictions such as CPU with gpu_count>0, distributed MPS, NCCL without CUDA,
  retrying distributed jobs, reserved env names and unsafe relative paths are rejected.
- GpuLayer does not install dependencies. The worker's interpreter (`--python`)
  must already provide them. Requirements are declared through
  `ResourceRequest.frameworks` and `python` (PEP 440 specifiers) and checked by the
  scheduler.

## Alternatives considered

- **cloudpickle closures:** convenient, but non-portable and unsafe. Rejected.
- **Container images as the only format:** strong reproducibility, but requires a
  container runtime everywhere, including Apple Silicon/MPS, where GPU passthrough is
  unavailable. Kept as a future packaging option.
- **Per-job virtualenv creation:** slow and network-dependent. Possibly later as an
  opt-in, cached runtime profile.

## Consequences and trade-offs

- A job is reproducible only as far as the worker's environment is. Two workers with
  different library versions may behave differently even though both satisfy the
  specifier.
- JSON-only function I/O excludes tensors and objects. Large data goes through artifacts.
- `v1alpha1` may change. Any breaking change needs a documented migration (CLAUDE.md §6).

## Security and operational implications

- Bundles are extracted with zip-slip, absolute-path and symlink checks
  (`extract_bundle`). Input paths are validated as relative.
- Packaging code to a remote worker means that worker runs it. See ADR-0009.

## Follow-up

- Container-image workloads (`ImageJob`) with digest pinning, plus an ADR for the
  container runtime.
- Runtime profiles naming known-good interpreter and library combinations (ADR-0005).
- Bundle size limits per deployment, and garbage collection (ADR-0008).
