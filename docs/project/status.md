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
| Distributed PyTorch with gloo | validated on up to 16 machines; GPUlayer Trainer 17.8% faster than Lightning on 8 AWS machines (0.8, 2026-11-01) |
| Distributed groups kept in one network group, zone or region | validated on GCP: placed by the scheduler, 42.5 s with `prefer` (one zone) against 58.8 s with `any` (two regions), median of five |
| NVIDIA CUDA | validated on Tesla T4, RTX 3070 and A100; the GPUlayer Trainer 1.08 to 1.38x faster per step than Lightning on one T4 (0.8, 2026-11-01) |
| Several GPUs per machine with NCCL, choosing the best-connected GPUs | validated on 4× RTX 3070 (PCIe) and 2× A100 SXM4 (NVLink) |
| GPUlayer Trainer against Lightning on an Apple M2 Pro (GPU and CPU) | GPUlayer Trainer faster in all 5 cases (1.02 to 1.80x) |
| GPUlayer Trainer on several GPUs against Lightning | measured on 1, 2 and 4 RTX 3070s: GPUlayer Trainer faster in 5 of 6 cases (1.26 to 1.60x), 1.12x slower on the 1-GPU CNN on one host |
| GPUlayer Trainer across machines with NCCL against Lightning | measured on 2, 4 and 8 machines with one T4 each: GPUlayer Trainer 1.08 to 1.29x faster on an MLP, Lightning 1.01 to 1.09x faster on a CNN |
| NCCL across machines | validated on 2, 4 and 8 AWS machines with one Tesla T4 each: DDP with `sync_batchnorm`, manual optimization, FSDP; identical weights on every rank ([results](../../benchmarks/results/2026-11-01-aws-8xt4/README.md)) |
| Apple Silicon (PyTorch MPS, MLX) | validated on an M2 Pro |
| AMD ROCm | implemented (devices, `HIP_VISIBLE_DEVICES`, Trainer `accelerator="rocm"`, RCCL); not run on AMD hardware |
| Other GPU libraries (JAX, CuPy, TensorFlow, ONNX Runtime, Transformers) | detected on workers; not validated on a GPU (a validation run was stopped before it finished) |
| Trainer FSDP (`strategy="fsdp"`) | experimental; validated over NCCL on 2 and 8 T4 machines (32-true; each rank held 1/N of the parameters) |
| Trainer Lightning parity (ADR-0024): hooks, manual optimization, DataModule, scheduler forms, "-true" precisions, DDP options, callbacks (checkpointing, early stopping, SWA, EMA, fine-tuning, pruning, timers, monitors), CSV logger, profilers, tuner | tested on CPU, one process and two DDP processes; step time within 1.4% of 0.7 on an M2 Pro CPU |
| Trainer `sync_batchnorm`, `torch.compile` with Inductor, 16-mixed with manual optimization, "-true" precisions, `PyTorchProfiler` on CUDA | validated on Tesla T4 (one GPU, and up to 8 machines for `sync_batchnorm` and manual optimization) |
| TensorBoard, Weights & Biases and MLflow loggers | implemented; not tested (optional packages) |
| Trainer: several dataloaders (`CombinedLoader`, `dataloader_idx`), distributed `validate`/`test`/`predict`, `LightningCLI`, plugins (precision, `CheckpointIO`, `AsyncCheckpointIO`) | tested on CPU, distributed cases with two processes |
| Trainer `DeepSpeedStrategy` and `accelerator="tpu"` | implemented; not run (need DeepSpeed with CUDA GPUs, and torch_xla on a TPU VM) |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Per-job requirements, installed once per worker and reused (ADR-0022) | tested; measured on an M2 Pro (`timm`: 6.6 s first, 0.00 s after) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- Validate AMD ROCm on AMD hardware.

- Why the 1-GPU CNN's CPU work ran about 10% slower than Lightning's on one RTX 3070 host (same GPU kernels).
- NCCL across multi-GPU machines (only 32 vCPUs of AWS G instances are approved: one GPU each).
- Speed against Lightning with NCCL across machines.
- A second cloud provider.
- A CUDA container image.

## Later

- A shared database and object storage, so the control plane can run on several machines.
- A hosted control plane, and an isolation review before running jobs for people who do not
  trust each other.
- Container packaging for jobs.
