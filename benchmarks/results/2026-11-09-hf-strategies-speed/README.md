# Parallel strategies on one machine, speed against Lightning, NVLink, libraries, object store

2026-11-09, Hugging Face Jobs (one container per run, NVIDIA driver 580.178.04), GPUlayer at
`cbcac28` (0.11.1) plus the scripts' `--devices` option and the library-path change in this
commit range. Logs are kept outside the repository (`gpulayer-validation-results/2026-11-09-hf-*`).

## 1. Parallel strategies on the 4 GPUs of one machine

4× NVIDIA L4 (PCIe, `NODE` links), NVIDIA's `nvcr.io/nvidia/pytorch:26.10-py3` (torch
2.15.0a0, DeepSpeed 0.19.9). Until now these ran only across machines with one T4 each.
`python scripts/validate_parallel.py --case ... --devices 4` (and `validate_deepspeed.py`,
`validate_nccl.py`): an MLP of 8 blocks of width 1024, batch 512 per GPU, 200 steps. ms per step
after a 10-step warm-up; memory is rank 0's peak. Each rank trains a full batch, so steps on
4 GPUs do 4 times the work of the 1-GPU baseline: these are checks, not a scaling measurement.

| Case | Checks | ms/step | Peak MB (rank 0) | Parameters per rank |
|---|---|---|---|---|
| 1 GPU (baseline) | 3/3 | 55.3 | 1283 | all |
| DDP | 4/4 (identical weights) | 181.5 | 1539 | all |
| DDP, fp16 compression hook | 5/5 | 132.8 | 1539 | all |
| DDP, post-local SGD | 6/6 | 225.8 | 1539 | all |
| FSDP `FULL_SHARD` | 4/4 | 253.9 | 470 | 1/4 |
| FSDP `SHARD_GRAD_OP` | 3/3 | 202.3 | 655 | 1/4 |
| FSDP, activation checkpointing | 5/5 | 257.3 | 392 | 1/4 |
| FSDP, CPU offload | 5/5 | 323.3 | 270 | 1/4 |
| FSDP sharded checkpoint (100 steps), then resumed to 200 | 5/5, 6/6 | 226.1, 220.6 | 470 | 1/4 |
| Tensor parallel (`ModelParallelStrategy`, 4-way) | 4/4 (`fc1` split 1024×1024) | 55.6 | 337 | 1/4 |
| DeepSpeed ZeRO 1 / 2 / 3, 16-mixed | 4/4 each | 46.9 / 46.7 / 33.0 | 883 / 883 / 524 | |
| DeepSpeed ZeRO 2 / 3 with CPU offload | 4/4 each | 55.9 / 52.3 | 840 / 453 | |
| `validate_nccl.py`: DDP with `sync_batchnorm`, manual optimization, FSDP | 5/5, 4/4, 4/4 | 5.8, 3.9, 16.7 | | |
| `BitsandbytesPrecision` nf4, int8 (1 GPU) | 4/4 each | | | |
| `to_tensorrt` (1 GPU) | 2/2: max difference 0.0006 | eager 3.47, TensorRT 3.64 | | |

The NCCL script's cases ran in a second job (`python:3.12`, torch 2.15.0+cu132), since the first
job's log filter dropped their `[rank0]`-prefixed lines.

## 3. Two H200s with NVLink

`h200x2` (2× NVIDIA H200, `NV18` between them), `python:3.12`, torch 2.15.0+cu132.
`scripts/validate_multigpu.py`: the topology was read as `NV18` both ways, the pair was chosen,
`CUDA_VISIBLE_DEVICES` set, and NCCL ran 1 node × 2 processes (5.3 s): all passed. With
`--devices 2`: FSDP (64.7 ms, half the parameters per rank), tensor parallelism (51.0 ms,
`fc1` 2048×1024 per rank) and DDP (33.7 ms, identical weights) passed.

## 2. Speed against PyTorch Lightning

`benchmarks/trainer_comparison/run.py`, as in the earlier comparisons: same model, data,
optimizer and logged loss; each trainer with its defaults (Lightning without logger, progress
bar, summary or checkpoints; GPUlayer without checkpoints). Each case runs 100 and 600 steps
three times in fresh processes, one at a time with nothing else on the GPUs (checked); ms per
step = (median wall(600) - median wall(100)) / 500. GPUlayer 0.11.1, Lightning 2.6.6, torch
2.15.0+cu132, `python:3.12`.

| Case | 4× L4 (48 vCPUs) GPUlayer / Lightning ms | Result |
|---|---|---|
| MLP fp32, 1 GPU | 3.58 / 3.72 | GPUlayer 1.04x faster |
| MLP fp16, 1 GPU | 3.25 / 4.80 | GPUlayer 1.48x faster |
| CNN fp16, 1 GPU | 6.40 / 6.91 | GPUlayer 1.08x faster |
| MLP fp32, 2 GPUs | 7.30 / 11.12 | GPUlayer 1.52x faster |
| MLP fp32, 4 GPUs | 10.24 / 15.67 | GPUlayer 1.53x faster |
| CNN fp16, 2 GPUs | 9.14 / 14.20 | GPUlayer 1.55x faster |
| CNN fp16, 4 GPUs | 13.40 / 22.45 | GPUlayer 1.68x faster |

| Case | H200 (`h200`, 23 vCPUs) | 2× H200 (`h200x2`, 46 vCPUs) |
|---|---|---|
| MLP fp32, 1 GPU | 1.74 / 4.36: GPUlayer 2.50x faster | 2.30 / 3.39: GPUlayer 1.47x faster |
| MLP fp16, 1 GPU | 2.95 / 3.36: GPUlayer 1.14x faster | |
| CNN fp16, 1 GPU | 5.51 / 6.06: GPUlayer 1.10x faster | 5.20 / 6.43: GPUlayer 1.24x faster |
| MLP fp32, 2 GPUs | | 3.01 / 6.67: GPUlayer 2.21x faster |
| CNN fp16, 2 GPUs | | 9.25 / 13.58: GPUlayer 1.47x faster |

GPUlayer was faster in all 14 cases. On an H200 these models take a few milliseconds per step,
so the trainers' CPU work per step decides the result, and it varies by host: the same 1-GPU
MLP case gave 2.50x on one H200 host and 1.47x on another (Lightning 4.36 against 3.39 ms).
Small models, not a measure of large-model training.

## 4. Other libraries as GPUlayer jobs

One L4. JAX 0.12.0.1, CuPy 14.2.0, TensorFlow 2.21.0 (`tensorflow[and-cuda]`), ONNX Runtime
1.31.1 (`onnxruntime-gpu[cuda,cudnn]`) and Transformers 5.22.2, each installed per job, all ran on
the GPU (`scripts/validate_libraries.py`), and the worker's PyTorch ran a convolution on the GPU
in the TensorFlow job's process. Before the change, TensorFlow and ONNX Runtime ran on the CPU.
Two diagnostic runs showed why, without GPUlayer: in a plain virtualenv TensorFlow 2.21 cannot
load `libcusolver.so.11` from its own pip packages, and ONNX Runtime next to torch's CUDA 13
libraries lacks `libcurand.so.10`. Both work once `LD_LIBRARY_PATH` lists the `nvidia/*/lib`
folders, which the worker now does for each job (its own packages first, then the worker's).

## 5. Object store

`python scripts/validate_object_store.py hf://buckets/antonellof/gpulayer-validation` from an
M2 Pro (CPU, `huggingface_hub` 2.4.0, fsspec 2026.9.0) to a private Hugging Face bucket: fit
with `default_root_dir`, `ModelCheckpoint(dirpath=...)` and `CSVLogger` in the bucket, resume
from a URL and from `"last"`, `load_from_checkpoint`, `save_checkpoint`, and two DDP processes
checkpointing into it: all passed. Three short epochs with three checkpoints took 35 s, because
saving waits for the upload. Not S3, GCS or Azure.
