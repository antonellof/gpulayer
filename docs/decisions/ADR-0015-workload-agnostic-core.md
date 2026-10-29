# ADR-0015: A workload-agnostic GPU layer, with framework integrations on top

- Status: Accepted (milestone 2)
- Date: 2026-10-29

## Context

GPUlayer's core (job contracts, scheduler, workers, artifacts, server) never depended on a
particular framework, but two things pulled the project towards being seen as a PyTorch training
tool. First, the PyTorch integration (ADR-0014) became the most visible API. Second, GPU discovery
treated an NVIDIA GPU as usable only when a CUDA build of torch was installed, and found Apple
GPUs only through torch's MPS backend. A CUDA machine running JAX, CuPy, a CUDA binary or
`ffmpeg`, or a Mac running MLX, looked like a machine without a GPU.

Training frameworks already cover how a model trains inside the machines they are started on.
What is missing across GPU work in general (training, inference batches, rendering,
simulation, data processing) is a layer that decides which machine runs a workload, gets code
and data there, starts it, and returns results.

## Decision

1. GPUlayer is a workload-agnostic GPU layer. Its primary workload kinds are commands in any
   language and Python functions using any library. Distributed jobs are a coordination
   feature, launched with torchrun today.
2. Hardware capability and framework capability are separate:
   - an accelerator is usable when its driver or platform is (`nvidia-smi` for NVIDIA,
     Apple Silicon for the Metal GPU, reported as `mps`);
   - installed GPU frameworks are reported as `frameworks` with versions, read from package
     metadata without importing them (`torch`, `jax`, `jaxlib`, `tensorflow`, `mlx`, `cupy`,
     `onnxruntime`, `tensorrt`, `triton`, `vllm`, `transformers`, `numpy`);
   - framework-specific GPU support is a pseudo-framework, for example `torch.cuda` and
     `torch.mps`, meaning "this torch can drive that GPU".
3. Jobs state framework requirements explicitly (`frameworks={"jax": ">=0.4.30"}`). Framework
   integrations add their own requirements: the PyTorch layer requires `torch` and
   `torch.cuda` or `torch.mps` for the accelerator it asks for.
4. Framework integrations are optional layers on the core. PyTorch is the first
   (`gpulayer.pytorch`, ADR-0014). The core never imports a GPU framework.
5. Documentation, examples and the website present GPUlayer by workload first (commands,
   functions, distributed jobs) and by framework second.

## Alternatives considered

- Stay a PyTorch-first tool. Smaller scope, but it overlaps with existing training frameworks
  and leaves out most GPU work.
- Import each framework to test GPU access during discovery. More precise (for example JAX's
  view of devices), but slow, memory-hungry, and fragile when a framework is broken.
  Package metadata plus driver detection is cheap and does not fail; a per-framework probe can
  be added later behind the same `frameworks` mechanism, as was done for torch.
- Ship container images as the unit of portability. Useful later (ADR-0002 lists it as
  proposed), but it would make a container runtime mandatory on every worker.

## Consequences

- A CUDA worker without torch now accepts CUDA jobs. PyTorch jobs still require `torch.cuda`, so
  they do not land there.
- The scheduler can explain framework mismatches (`framework 'jax' not installed`,
  `mlx 0.9.0 does not satisfy >=0.10`).
- Long-running services (for example an inference server that must stay up) are not covered by
  the job model yet; they need ports, health checks and restarts, and are planned separately.
- Distributed launchers other than torchrun (JAX multi-host, MPI) are planned.

## Validation

MLX on the Apple GPU, NumPy batches, shell jobs and PyTorch on CPU and MPS have been run on an
Apple Silicon Mac. JAX, CuPy, TensorFlow and ONNX Runtime on NVIDIA use the same mechanism but
have not been run in this repository yet.

## Follow-up

- Run the framework examples on NVIDIA hardware.
- Optional per-framework device probes (JAX, CuPy) when package presence is not enough.
- A launcher abstraction for non-torch distributed runtimes.
- A service workload type for long-running GPU servers.
