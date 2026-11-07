# Status and roadmap

GPUlayer's focus: train PyTorch on every GPU you can reach, with one Trainer, from your laptop
to many machines ([ADR-0023](../decisions/ADR-0023-product-focus-pytorch-training.md)).

## What works today

"Tested" means automated tests cover it. "Validated" means it ran on the real hardware or
machines named. Everything else says what is missing.

| Capability | Status |
|---|---|
| Commands, Python functions and the PyTorch Trainer on this machine: logs, results, artifacts, cancel, timeout | tested |
| Scheduling by requirements, with a reason for every rejected worker; distributed groups placed all at once | tested |
| A server with workers on other machines (`gpulayer up`, `join`), TLS with a pinned certificate, named tokens | validated on 30 GCP machines in two regions |
| Distributed PyTorch with gloo | validated on up to 16 machines |
| Distributed groups kept in one network group, zone or region | validated on GCP: 1.38x faster than a group split across two regions |
| NVIDIA CUDA | validated on Tesla T4, RTX 3070 and A100 |
| Several GPUs per machine with NCCL, choosing the best-connected GPUs | validated on 4× RTX 3070 (PCIe) and 2× A100 SXM4 (NVLink) |
| NCCL across machines | validated on 2, 4 and 8 AWS machines with one Tesla T4 each: DDP with `sync_batchnorm`, manual optimization, FSDP ([results](../../benchmarks/results/2026-11-01-aws-8xt4/README.md)) |
| GPUlayer Trainer speed against Lightning | measured (0.8, 2026-11-01): faster in 13 of 17 GPU cases and every CPU case; Lightning faster on a CNN on one RTX 3070 and on a CNN across machines over NCCL ([details](../get-started/performance-and-comparison.md)) |
| Apple Silicon (PyTorch MPS, MLX) | validated on an M2 Pro |
| AMD ROCm | implemented (devices, `HIP_VISIBLE_DEVICES`, Trainer `accelerator="rocm"`, RCCL); not run on AMD hardware |
| Other GPU libraries (JAX, CuPy, TensorFlow, ONNX Runtime, Transformers) | detected on workers; not validated on a GPU (a validation run was stopped before it finished) |
| Trainer FSDP (`strategy="fsdp"`) | experimental; validated over NCCL on 2 and 8 T4 machines (32-true; each rank held 1/N of the parameters) |
| Trainer Lightning parity (ADR-0024): hooks, manual optimization, DataModule, scheduler forms, "-true" precisions, DDP options, callbacks (checkpointing, early stopping, SWA, EMA, fine-tuning, pruning, timers, monitors), CSV logger, profilers, tuner | tested on CPU, one process and two DDP processes; step time within 1.4% of 0.7 on an M2 Pro CPU |
| Lightning's public names: every name of a live `Trainer` and `Module` (compared at runtime) and of its callbacks, loggers, profilers, plugins and utilities. `trainer.strategy` has the methods code usually calls, not Lightning's internals; `trainer.fit_loop` is a small view of the state | tested on CPU; SIGTERM checkpoints, `init_module`, eval/train mode hooks tested in one process |
| Same behaviour as Lightning: the same code runs on both in `tests/conformance` (results, weights, hooks, logging, schedules, checkpoints both ways) | tested in CI on CPU, one process, against Lightning 2.6. One known difference: with a validation loader and shuffling, the same seed draws different random numbers |
| Automatic resume after a lost worker or SIGTERM, also for a group of machines (ADR-0025) | tested with real processes on one machine and simulated workers; not run on a reclaimed cloud VM |
| Exact resume: the same weights as a run that never stopped, in one process; the same data order across processes | tested on CPU; not measured on a GPU |
| Checkpoints and logs in object stores through fsspec | tested with fsspec's in-memory filesystem; not run against S3, GCS or Azure |
| FSDP options (sharding strategies, activation checkpointing, CPU offload, sharded checkpoints merged across machines), `ModelParallelStrategy` (tensor parallelism), `to_tensorrt` | validated on 2 Tesla T4 machines over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-parallel/README.md)) |
| DDP communication hooks (fp16 compression) and post-local SGD | validated on 2 Tesla T4 machines over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-ddp-hooks/README.md)) |
| Trainer `DeepSpeedStrategy` (ZeRO 1 to 3, CPU offload) | validated on Tesla T4s, one machine and two over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-deepspeed/README.md)) |
| `BitsandbytesPrecision` (nf4, int8) | validated on a Tesla T4 |
| `TransformerEnginePrecision`, `XLAProfiler`, Comet, W&B, MLflow, TensorBoard and Lit loggers | implemented; tested against fake packages only |
| Trainer `sync_batchnorm`, `torch.compile` with Inductor, 16-mixed with manual optimization, "-true" precisions, `PyTorchProfiler` on CUDA | validated on Tesla T4 (one GPU, and up to 8 machines for `sync_batchnorm` and manual optimization) |
| TensorBoard, Weights & Biases and MLflow loggers | implemented; not tested (optional packages) |
| Trainer: several dataloaders (`CombinedLoader`, `dataloader_idx`), distributed `validate`/`test`/`predict`, `LightningCLI`, plugins (precision, `CheckpointIO`, `AsyncCheckpointIO`) | tested on CPU, distributed cases with two processes |
| Trainer `accelerator="tpu"` (one device, every device of a TPU VM through `xmp.spawn`, or a TPU pod) | implemented and tested with a simulated torch_xla; not run on a TPU |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Per-job requirements, installed once per worker and reused (ADR-0022) | tested; measured on an M2 Pro (`timm`: 6.6 s first, 0.00 s after) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- Validate AMD ROCm on AMD hardware, and TPU training.
- Why the 1-GPU CNN's CPU work ran about 10% slower than Lightning's on one RTX 3070 host (same GPU kernels).
- NCCL across multi-GPU machines (only 32 vCPUs of AWS G instances are approved: one GPU each).
- Why Lightning is faster on the CNN over NCCL across machines.
- A second cloud provider.
- A CUDA container image.

## Later

- A shared database and object storage, so the control plane can run on several machines.
- A hosted control plane, and an isolation review before running jobs for people who do not
  trust each other.
- Container packaging for jobs.
