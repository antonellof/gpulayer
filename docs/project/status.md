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
| Every public name of Lightning's Trainer, Module, hooks, callbacks, loggers, profilers, plugins, strategies, accelerators and utilities (`scripts/compare_lightning_api.py` finds none missing) | tested on CPU; SIGTERM checkpoints, `init_module`, eval/train mode hooks tested in one process |
| FSDP options (sharding strategies, activation checkpointing, CPU offload, sharded checkpoints), `ModelParallelStrategy` (tensor parallelism), DDP communication hooks and post-local SGD | tested with two CPU processes (gloo); not run on GPUs |
| `BitsandbytesPrecision`, `TransformerEnginePrecision`, `to_tensorrt`, `XLAProfiler`, Comet, W&B, MLflow, TensorBoard and Lit loggers | implemented; tested against fake packages only |
| Trainer `sync_batchnorm`, `torch.compile` with Inductor, 16-mixed with manual optimization, "-true" precisions, `PyTorchProfiler` on CUDA | validated on Tesla T4 (one GPU, and up to 8 machines for `sync_batchnorm` and manual optimization) |
| TensorBoard, Weights & Biases and MLflow loggers | implemented; not tested (optional packages) |
| Trainer: several dataloaders (`CombinedLoader`, `dataloader_idx`), distributed `validate`/`test`/`predict`, `LightningCLI`, plugins (precision, `CheckpointIO`, `AsyncCheckpointIO`) | tested on CPU, distributed cases with two processes |
| Trainer `DeepSpeedStrategy` and `accelerator="tpu"` (one device, every device of a TPU VM through `xmp.spawn`, or a TPU pod) | implemented and tested with simulated libraries; not run (need DeepSpeed with CUDA GPUs, and torch_xla on a TPU VM) |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Per-job requirements, installed once per worker and reused (ADR-0022) | tested; measured on an M2 Pro (`timm`: 6.6 s first, 0.00 s after) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- Validate AMD ROCm on AMD hardware, and DeepSpeed and TPU training.
- Re-run manual optimization under DDP on GPUs: its gradient averaging changed after the 8×T4 run.
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
