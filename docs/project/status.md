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
| Distributed PyTorch with gloo | validated on up to 16 machines; GPUlayer Trainer 11.9% faster than Lightning on 8 machines |
| Distributed groups kept in one network group, zone or region | validated on GCP: placed by the scheduler, 42.5 s with `prefer` (one zone) against 58.8 s with `any` (two regions), median of five |
| NVIDIA CUDA | validated on one Tesla T4, including the Trainer (1.14 to 1.39x faster per step than Lightning) |
| Several GPUs per machine with NCCL, choosing the best-connected GPUs | validated on 4× RTX 3070 (PCIe) and 2× A100 SXM4 (NVLink) |
| GPUlayer Trainer against Lightning on an Apple M2 Pro (GPU and CPU) | GPUlayer Trainer faster in all 5 cases (1.05 to 1.75x) |
| GPUlayer Trainer on several GPUs against Lightning | measured on 1, 2 and 4 RTX 3070s: GPUlayer Trainer faster in 5 of 6 cases (1.11 to 1.76x), 1.14x slower on the 1-GPU CNN on one host |
| NCCL across machines | implemented; not run yet |
| Apple Silicon (PyTorch MPS, MLX) | validated on an M2 Pro |
| AMD ROCm | implemented (devices, `HIP_VISIBLE_DEVICES`, Trainer `accelerator="rocm"`, RCCL); not run on AMD hardware |
| Other GPU libraries (JAX, CuPy, TensorFlow, ONNX Runtime, Transformers) | detected on workers; not validated on a GPU (a validation run was stopped before it finished) |
| Trainer FSDP (`strategy="fsdp"`) | experimental; tested on CPU only |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Per-job requirements, installed once per worker and reused (ADR-0022) | tested; measured on an M2 Pro (`timm`: 6.6 s first, 0.00 s after) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- Validate AMD ROCm on AMD hardware.

- NCCL across machines.
- Why the 1-GPU CNN's CPU work ran about 10% slower than Lightning's on one RTX 3070 host (same GPU kernels).
- FSDP and `torch.compile` on CUDA.
- A second cloud provider.
- A CUDA container image.

## Later

- A shared database and object storage, so the control plane can run on several machines.
- A hosted control plane, and an isolation review before running jobs for people who do not
  trust each other.
- Container packaging for jobs.
