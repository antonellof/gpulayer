# ADR-0023: Product focus on PyTorch training

- **Status:** Accepted
- **Date:** 2026-10-31

## Context

GPUlayer was presented as a general platform: "run any workload on every GPU (and CPU) you can
reach". That puts it against Ray, SkyPilot, Slurm and Modal, where it has little evidence of
being better. Its measured advantages are all in PyTorch training: against PyTorch Lightning,
with the same model, data and optimizer, its Trainer was faster in 10 of 11 GPU benchmarks
(Tesla T4, 1 to 4 RTX 3070s, Apple M2 Pro GPU) and in every CPU benchmark; it trains across
machines from one `Trainer(num_nodes=N)` call; and placement by zone (1.38x) and by GPU
interconnect serve distributed training. Claims about other libraries were weaker: JAX, CuPy,
TensorFlow, ONNX Runtime and Transformers are detected on workers but were not run on a GPU.

## Decision

1. The product statement is: **Train PyTorch on every GPU you can reach: one Trainer, from
   your laptop to many machines.** The site, README, docs, package
   metadata and repository description lead with it.
2. The general job engine stays. The Trainer's multi-GPU and multi-machine runs, placement,
   GPU checks, logs and checkpoints go through it, and commands, functions and services are
   built and tested. It is presented second ("Not only PyTorch").
3. "Every GPU" names what is validated on hardware (NVIDIA CUDA, Apple Metal, CPU). AMD ROCm
   has an execution path (implemented 2026-10-31) and is named as awaiting validation.
4. Speed against PyTorch Lightning is stated only in performance comparisons (benchmark
   sections, the performance page, the compare page), never in the tagline, always as "the
   GPUlayer Trainer" with the measured cases, including the one where it was slower (1-GPU
   CNN on one RTX 3070 host). A new speed claim needs a log in `benchmarks/results/`.
5. Logos and lists of libraries show only what has been run as GPUlayer jobs.
6. Work is prioritised by what makes PyTorch training faster, simpler or able to reach more
   hardware: per-step overhead, start-up, multi-GPU and multi-machine scaling, NCCL across
   machines, AMD validation.

## Alternatives considered

- **Keep the general positioning:** broader, but the claims were harder to back and the
  competition is larger and more mature.
- **Drop the general engine:** the Trainer's distributed and remote modes depend on it.

## Consequences

- Examples, docs and site sections that only served the general positioning and were never
  validated on NVIDIA (Blender with CUDA, NVENC, JAX/CuPy on CUDA) are removed or moved behind
  the PyTorch material.
- Benchmarks target the Trainer against Lightning on every backend we claim.
