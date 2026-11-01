# NVIDIA CUDA

## Requirements

An NVIDIA driver (`nvidia-smi` works) and, for PyTorch jobs, a CUDA build of torch in the
worker's Python (`python -c "import torch; print(torch.cuda.is_available())"`).

```bash
gpulayer resources detect
```

This lists each GPU with memory and compute capability. Without the `torch.cuda` capability it
prints a note and PyTorch jobs are not placed there.

If convolutions fail with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` (or similar cuDNN or cuBLAS
errors) while plain tensor operations work, `LD_LIBRARY_PATH` points at a system CUDA that
clashes with the PyTorch wheel (AWS Deep Learning AMIs set it in `/etc/profile.d/dlami.sh`).
Start the worker without it: `env -u LD_LIBRARY_PATH gpulayer join ...` (or `gpulayer up`).

## Train

```python
trainer = glt.Trainer(max_epochs=10, accelerator="cuda", devices=4, precision="bf16-mixed")
trainer.fit(model, train_loader, val_loader)
```

`devices` is GPUs (one process each) per node. Multi-GPU and multi-node runs use DDP with
`nccl`. `16-mixed` uses a gradient scaler; `bf16-mixed` needs bfloat16 support.

## Placement constraints

```python
@gl.remote(accelerator="cuda", gpus=2, gpu_memory_gb=40)
def finetune(...): ...
```

The same fields work in `launch(...)` and a `JobSpec`.
`ResourceRequest(min_cuda_compute_capability="8.0")` restricts jobs to Ampere or newer.

## Several GPUs, one or more workers

A worker offers all its GPUs and runs one single-GPU job per GPU at a time, setting
`CUDA_VISIBLE_DEVICES` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` (indices match `nvidia-smi`). A
multi-GPU job gets the free GPUs with the fastest links between them
([Scheduling](../architecture/scheduling.md#gpus-inside-a-machine)).

One worker per machine is usually right. To split a machine, give each worker its own GPUs,
keeping GPUs that share NVLink or a PCIe switch together:

```bash
gpulayer join URL --token T --name box-a --gpus 0,1
gpulayer join URL --token T --name box-b --gpus 2,3
gpulayer join URL --token T --name box-cpu --gpus none --concurrency 8   # CPU jobs only
```

`python scripts/validate_multigpu.py` checks the parsed interconnect against
`nvidia-smi topo -m`, multi-GPU job assignment, and an NCCL all-reduce across all GPUs.
`benchmarks/cloud/vast.sh` rents multi-GPU machines.

## Validation status

| Hardware | Validated |
|---|---|
| Tesla T4 (GCP `n1-standard-8`, driver 580, torch 2.15, CUDA 13.2) | detection, jobs on their reserved GPU, one GPU job at a time, Trainer in `32-true`, `bf16-mixed`, `16-mixed` and with `torch.compile`, in process and as a job |
| 4× RTX 3070, AMD EPYC 7551 (Vast.ai, 2026-10-31, torch 2.15, CUDA 13); topology 0-1 and 2-3 `PHB`, the rest `SYS` | 8/8 CUDA checks; NCCL all-reduce across 4 GPUs as a GPUlayer job; a 2-GPU job with GPU 0 busy got GPUs 2-3 (`PHB`), not 1-2 (`SYS`) |
| 2× A100 SXM4 40 GB (Vast.ai, 2026-10-31); `NV4` (NVLink) | 8/8 CUDA checks; NCCL all-reduce over NVLink |
| 8 AWS `g4dn.xlarge`, one T4 each, one subnet, driver 595 (2026-11-01) | DDP with 16-mixed and `sync_batchnorm` on 2, 4 and 8 machines (identical weights on every rank); manual optimization on 8; FSDP on 2 and 8; on one T4: Inductor `torch.compile`, 16-mixed manual optimization, "-true" precisions, `PyTorchProfiler` |

The Vast.ai machines needed `apt-get install gcc` for `torch.compile`. Logs:
`benchmarks/results/2026-10-30-gcp/t4/`, `benchmarks/results/2026-10-31-vast-4x3070/`,
`benchmarks/results/2026-10-31-vast-2xa100sxm4/` and `benchmarks/results/2026-11-01-aws-8xt4/`. Numbers are in
[Performance and comparison](../get-started/performance-and-comparison.md).

Repeat the validation on any NVIDIA machine (test suite, CUDA hardware test,
`scripts/validate_cuda.py`, `scripts/validate_multigpu.py` on multi-GPU machines, benchmarks):

```bash
bash scripts/validate_gpu.sh 2>&1 | tee validate.log
```
