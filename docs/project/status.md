# Status and roadmap

## What works today

"Tested" means automated tests cover it. "Validated" means it ran on the real hardware or
machines named. Everything else says what is missing.

| Capability | Status |
|---|---|
| Commands, Python functions and the PyTorch Trainer on this machine: logs, results, artifacts, cancel, timeout | tested |
| Scheduling by requirements, with a reason for every rejected worker; distributed groups placed all at once | tested |
| A server with workers on other machines (`gpulayer up`, `join`), TLS with a pinned certificate, named tokens | validated on 30 GCP machines in two regions |
| Distributed PyTorch with gloo | validated on up to 16 machines; DDP training on 8 machines 11.9% faster than Lightning |
| Distributed groups kept in one network group, zone or region | validated on GCP: a job split across two regions ran 16 to 24% slower than in one region; the scheduler kept it in one zone |
| NVIDIA CUDA | validated on one Tesla T4, including the Trainer (1.14 to 1.39x faster per step than Lightning) |
| Several GPUs per machine with NCCL, choosing the best-connected GPUs | validated on 4× RTX 3070 (PCIe) and 2× A100 SXM4 (NVLink) |
| Trainer on several GPUs against Lightning | measured on 1, 2 and 4 RTX 3070s: faster in 4 of 6 cases, slower on the 2-GPU CNN with defaults (under investigation) |
| NCCL across machines | implemented; not run yet |
| Apple Silicon (PyTorch MPS, MLX) | validated on an M2 Pro |
| AMD ROCm | detected, not run |
| JAX and CuPy on NVIDIA GPUs | detection tested; not run on a GPU |
| Trainer FSDP (`strategy="fsdp"`) | experimental; tested on CPU only |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Per-job requirements, installed once per worker and reused (ADR-0022) | tested; measured on an M2 Pro (`timm`: 6.6 s first, 0.00 s after) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- NCCL across machines.
- Why the 2-GPU CNN picks slower cuDNN kernels with GPUlayer's defaults than with Lightning's.
- FSDP and `torch.compile` on CUDA.
- A second cloud provider.
- A CUDA container image.

## Later

- A shared database and object storage, so the control plane can run on several machines.
- A hosted control plane, and an isolation review before running jobs for people who do not
  trust each other.
- Container packaging for jobs.
