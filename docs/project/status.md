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
| Several GPUs per machine, NCCL, choosing the best-connected GPUs | implemented; GPU choice tested with recorded `nvidia-smi topo -m` output; not run on a multi-GPU machine |
| Apple Silicon (PyTorch MPS, MLX) | validated on an M2 Pro |
| AMD ROCm | detected, not run |
| JAX and CuPy on NVIDIA GPUs | detection tested; not run on a GPU |
| Trainer FSDP (`strategy="fsdp"`) | experimental; tested on CPU only |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- Several GPUs per machine and NCCL across machines, including the best-connected GPU choice.
  Waiting on more GPU quota from Google Cloud.
- FSDP and `torch.compile` on CUDA.
- A second cloud provider.
- A CUDA container image.

## Later

- A shared database and object storage, so the control plane can run on several machines.
- A hosted control plane, and an isolation review before running jobs for people who do not
  trust each other.
- Container packaging for jobs.
