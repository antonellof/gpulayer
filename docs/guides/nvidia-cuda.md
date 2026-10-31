# NVIDIA CUDA

## Requirements

- An NVIDIA driver (`nvidia-smi` works).
- A CUDA build of PyTorch in the interpreter that runs jobs. Check with
  `python -c "import torch; print(torch.cuda.is_available())"`.

```bash
gpulayer resources detect
```

Each GPU is listed with its memory and compute capability. Any CUDA workload can use them once
the driver works: CUDA binaries, CuPy, JAX, TensorRT, PyTorch. PyTorch jobs additionally need
the `torch.cuda` capability (a CUDA build of torch in the worker's Python); if it is missing,
`resources detect` prints a note and PyTorch jobs are not placed there.

## Train

```python
trainer = glt.Trainer(max_epochs=10, accelerator="cuda", devices=4, precision="bf16-mixed")
trainer.fit(model, train_loader, val_loader)
```

- `devices` is the number of GPUs (one process per GPU) on each node. The job gets
  `CUDA_VISIBLE_DEVICES` set to that many GPUs.
- Multi-GPU and multi-node runs use DDP with the `nccl` backend.
- `precision="16-mixed"` uses a gradient scaler; `bf16-mixed` needs a GPU with bfloat16 support.

## Placement constraints

Ask for specific hardware through `@gl.remote(...)`, `launch(...)` or a `JobSpec`:

```python
@gl.remote(accelerator="cuda", gpus=2, gpu_memory_gb=40)
def finetune(...): ...
```

`ResourceRequest(min_cuda_compute_capability="8.0")` restricts jobs to Ampere or newer.

## Several GPUs, one or more workers

A worker offers all of its machine's GPUs and runs one single-GPU job per GPU at a time,
reserving each job's GPUs and setting `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
so the indices match `nvidia-smi`). One worker per machine is usually right.

A job that asks for several GPUs (`--gpus 4`, or `devices=4` in the Trainer) gets the free
GPUs with the fastest links between them, read from `nvidia-smi topo -m`: an NVLink island
when one is free, otherwise GPUs behind one PCIe switch, and across CPU sockets only when
nothing better is free. When splitting a machine with `--gpus`, keep GPUs that share NVLink
or a PCIe switch in the same worker. See
[Scheduling](../architecture/scheduling.md#gpus-inside-a-machine).

On a machine with two or more GPUs, `python scripts/validate_multigpu.py` checks the parsed
interconnect against `nvidia-smi topo -m`, which GPUs a multi-GPU job is given, and an NCCL
all-reduce across all of them (`scripts/validate_gpu.sh` runs it automatically).
`benchmarks/cloud/vast.sh` rents such machines.

To split a machine between workers, for example separate queues or Python environments, give
each worker its own GPUs with `--gpus`, so no GPU is offered twice:

```bash
gpulayer join URL --token T --name box-a --gpus 0,1
gpulayer join URL --token T --name box-b --gpus 2,3
gpulayer join URL --token T --name box-cpu --gpus none --concurrency 8   # CPU jobs only
```

## Validation status

Validated on a Tesla T4 (GCP `n1-standard-8`, driver 580, torch 2.15 with CUDA 13.2):
detection, CUDA jobs with their reserved GPU, one GPU job at a time on a one-GPU worker, and
the PyTorch Trainer in `32-true`, `bf16-mixed`, `16-mixed` and with `torch.compile`, both in
process and as a job.

Validated on two multi-GPU machines rented on Vast.ai (2026-10-31, torch 2.15 with CUDA 13):

| Machine | Interconnect read from `nvidia-smi topo -m` | Checks |
|---|---|---|
| 4× RTX 3070, AMD EPYC 7551 | 0–1 and 2–3 `PHB`, the rest `SYS` | 8/8 CUDA checks; NCCL all-reduce across 4 GPUs as a GPUlayer job; a 2-GPU job with GPU 0 busy got GPUs 2–3 (`PHB`), not 1–2 (`SYS`) |
| 2× A100 SXM4 40 GB | `NV4` (NVLink) | 8/8 CUDA checks; NCCL all-reduce across both GPUs over NVLink |

Both machines needed a C compiler for `torch.compile` (`apt-get install gcc`); rented
containers often lack one. NCCL across machines has not been run yet. Logs:
`benchmarks/results/2026-10-31-vast-4x3070/` and `2026-10-31-vast-2xa100sxm4/`.

Repeat the validation on any NVIDIA machine:

```bash
bash scripts/validate_gpu.sh 2>&1 | tee validate.log
```

It runs the test suite, the CUDA hardware test, `scripts/validate_cuda.py` and the
benchmarks. Results from 2026-10-30 are in `benchmarks/results/2026-10-30-gcp/t4/`, and
[Performance and comparison](../get-started/performance-and-comparison.md) has the numbers.
