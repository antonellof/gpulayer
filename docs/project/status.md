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
| NVIDIA CUDA | validated on Tesla T4, RTX 3070, A100, L4, H200 and RTX PRO 6000 Blackwell: the test suite, CUDA checks and the Trainer's GPU checks on each ([results](../../benchmarks/results/2026-11-09-hf-gpus/README.md)) |
| Several GPUs per machine with NCCL, choosing the best-connected GPUs | validated on 4× RTX 3070 (PCIe), 4× L4 (PCIe), 8× RTX PRO 6000 (PCIe, two CPU sockets), 2× A100 SXM4 (NVLink) and 2× H200 (NVLink, `NV18`); DDP, FSDP, tensor parallelism and DeepSpeed ZeRO 1 to 3 on all 8 ([results](../../benchmarks/results/2026-11-09-hf-models-services/README.md)) |
| NCCL across machines | validated on 2, 4 and 8 AWS machines with one Tesla T4 each: DDP with `sync_batchnorm`, manual optimization, FSDP ([results](../../benchmarks/results/2026-11-01-aws-8xt4/README.md)) |
| GPUlayer Trainer speed against Lightning | measured: faster in 13 of 17 GPU cases and every CPU case (0.8, 2026-11-01), in all 14 small-model cases on L4s and H200s, and in 10 of 12 cases with a GPT-2-size model and ResNet-50 (0.11, 2026-11-09). Lightning faster on a CNN on one RTX 3070, on a CNN across machines over NCCL, on ResNet-50 on an H200 (1%) and per step on the GPT model on 8 GPUs (1.15x) ([details](../get-started/performance-and-comparison.md)) |
| Apple Silicon (PyTorch MPS, MLX) | validated on an M2 Pro |
| AMD ROCm | implemented (GPUs found through a ROCm build of torch, `HIP_VISIBLE_DEVICES`, Trainer `accelerator="rocm"`); not run on AMD hardware. `backend="nccl"` on ROCm is assumed, not documented by AMD. [`scripts/validate_rocm.sh`](../guides/amd-rocm.md#validating-a-machine) is ready for the first machine |
| AMD Ryzen AI NPUs: serving models through FastFlowLM (`gl.flm_service`, ADR-0026). No training | implemented; tested with a stand-in `flm` on a real worker; not run on an NPU |
| Other GPU libraries (JAX, CuPy, TensorFlow, ONNX Runtime, Transformers) | validated as GPUlayer jobs on an NVIDIA L4, each installed with per-job requirements (`tensorflow[and-cuda]`, `onnxruntime-gpu[cuda,cudnn]`); jobs find the CUDA libraries their packages install ([results](benchmarks/results/2026-11-09-hf-strategies-speed/README.md)) |
| Trainer FSDP (`strategy="fsdp"`) | experimental; validated over NCCL on 2 and 8 T4 machines (32-true; each rank held 1/N of the parameters) |
| Trainer Lightning parity (ADR-0024): hooks, manual optimization, DataModule, scheduler forms, "-true" precisions, DDP options, callbacks (checkpointing, early stopping, SWA, EMA, fine-tuning, pruning, timers, monitors), CSV logger, profilers, tuner | tested on CPU, one process and two DDP processes; step time within 1.4% of 0.7 on an M2 Pro CPU. On GPUs: `torchmetrics` objects and values logged in `on_validation_epoch_end` combined over NCCL on 4 L4s, and the full test suite with the conformance tests passes on an L4, H200 and RTX PRO 6000 |
| Lightning's public names: every name of a live `Trainer` and `Module` (compared at runtime) and of its callbacks, loggers, profilers, plugins and utilities. `trainer.strategy` has the methods code usually calls, not Lightning's internals; `trainer.fit_loop` is a small view of the state | tested on CPU; SIGTERM checkpoints, `init_module`, eval/train mode hooks tested in one process |
| Same behaviour as Lightning: the same code runs on both in `tests/conformance` (results, weights, hooks, logging, schedules, checkpoints both ways) | tested in CI on CPU, one process, against Lightning 2.6. One known difference: with a validation loader and shuffling, the same seed draws different random numbers |
| Automatic resume after a lost worker or SIGTERM, also for a group of machines (ADR-0025) | tested with real processes on one machine and simulated workers; not run on a reclaimed cloud VM |
| Exact resume: the same weights as a run that never stopped, in one process; the same data order across processes | validated on an NVIDIA L4 (identical weights, with dropout on the GPU) and over NCCL on 4 L4s (the data order) |
| Checkpoints and logs in object stores through fsspec | validated against a private Hugging Face bucket (`hf://`): checkpoints, CSV logs, resume, `load_from_checkpoint`, two DDP processes; not run against S3, GCS or Azure |
| FSDP options (sharding strategies, activation checkpointing, CPU offload, sharded checkpoints merged across machines), `ModelParallelStrategy` (tensor parallelism), `to_tensorrt` | validated on 2 Tesla T4 machines over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-parallel/README.md)), and on the 4 L4s of one machine (FSDP and tensor parallelism also on 2 H200s; [results](benchmarks/results/2026-11-09-hf-strategies-speed/README.md)) |
| DDP communication hooks (fp16 compression) and post-local SGD | validated on 2 Tesla T4 machines over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-ddp-hooks/README.md)), and on the 4 L4s of one machine |
| Trainer `DeepSpeedStrategy` (ZeRO 1 to 3, CPU offload) | validated on Tesla T4s, one machine and two over NCCL ([results](../../benchmarks/results/2026-11-01-aws-2xt4-deepspeed/README.md)), and on 4 L4s in one machine (DeepSpeed 0.19.9) |
| `BitsandbytesPrecision` (nf4, int8) | validated on a Tesla T4 and an L4 |
| `TransformerEnginePrecision` (FP8) | validated on an NVIDIA L4, H200 and RTX PRO 6000 Blackwell: FP8 on in the forward pass, the loss within 2% of bf16-true, checkpoints load (Transformer Engine 2.20 asks for `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1` first) |
| `XLAProfiler`, Comet, W&B, MLflow, TensorBoard and Lit loggers | implemented; tested against fake packages only |
| Trainer `sync_batchnorm`, `torch.compile` with Inductor, 16-mixed with manual optimization, "-true" precisions, `PyTorchProfiler` on CUDA | validated on Tesla T4 (one GPU, and up to 8 machines for `sync_batchnorm` and manual optimization) |
| TensorBoard, Weights & Biases and MLflow loggers | implemented; not tested (optional packages) |
| Trainer: several dataloaders (`CombinedLoader`, `dataloader_idx`), distributed `validate`/`test`/`predict`, `LightningCLI`, plugins (precision, `CheckpointIO`, `AsyncCheckpointIO`) | tested on CPU, distributed cases with two processes |
| Trainer `accelerator="tpu"` (one device, every device of a TPU VM through `xmp.spawn`, or a TPU pod) | implemented and tested with a simulated torch_xla; not run on a TPU |
| Notebooks: `devices=N` forks one process per device (Linux; refused on macOS, where forks of a kernel crash) | validated in a real Jupyter kernel on 4 NVIDIA L4s; tested in CI on CPU |
| SLURM: `srun` tasks train in place | simulated with SLURM's variables on one machine; not run on a cluster |
| Long-running services: replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); a PyTorch model and vLLM served from an NVIDIA L4 (vLLM installed per job), restarted and cancelled; no load balancer |
| Google Cloud provisioning | validated: a CPU VM and a Spot T4 created for a job, joined, ran it and were deleted |
| Kubernetes Helm chart | tested in CI on kind (jobs, a distributed job, upgrade, server restart); GPU worker pool not run |
| Container images on GitHub Container Registry | published with each release (slim and `-torch`; no CUDA image) |
| Per-job requirements, installed once per worker and reused (ADR-0022) | tested; measured on an M2 Pro (`timm`: 6.6 s first, 0.00 s after) |
| Web dashboard, Prometheus metrics, Grafana dashboard | tested |
| MCP server for AI agents | tested |

## Next

- AMD: validate ROCm on AMD hardware, and serving on a Ryzen AI NPU.
- TPU training on a TPU.
- Run automatic resume on a reclaimed Spot VM, object stores on S3, GCS and Azure, and `srun` on
  a SLURM cluster: each is tested only in simulation or on one machine.
- Why the 1-GPU CNN's CPU work ran about 10% slower than Lightning's on one RTX 3070 host (same GPU kernels).
- NCCL across multi-GPU machines (only 32 vCPUs of AWS G instances are approved: one GPU each).
- Why Lightning is faster on the CNN over NCCL across machines.
- Why GPUlayer's DDP step is 1.15x slower than Lightning's on the GPT-2-size model on 8 GPUs in one machine (ResNet-50 is 1.16x faster there).
- A second cloud provider.
- A CUDA container image.

## Later

- A shared database and object storage, so the control plane can run on several machines.
- A hosted control plane, and an isolation review before running jobs for people who do not
  trust each other.
- Container packaging for jobs.
