# DeepSpeed, bitsandbytes and manual optimization on Tesla T4s (AWS)

2026-11-01. 2x `g4dn.xlarge` Spot (4 vCPU, 1x Tesla T4 16 GB), us-east-1c, one subnet; AWS
Deep Learning Base OSS NVIDIA driver AMI (Ubuntu 22.04, driver 595.112.07, CUDA 13.2, nvcc
13.2); Python 3.12, torch 2.15.0+cu132, DeepSpeed 0.19.9, bitsandbytes 0.50.2, ninja 1.13.2,
GPUlayer at commit `2cc8675` plus the three fixes below. Set up with
`benchmarks/cloud/aws.sh`. One server (`gpulayer up`) on machine 0, which also runs jobs;
machine 1 joined over the private network. Jobs ran one at a time.

## DeepSpeed (scripts/validate_deepspeed.py)

An MLP of 17.8M parameters (5 layers of 2048), synthetic data, batch 64 per GPU, 16-mixed
(DeepSpeed fp16), 200 steps; `--offload` uses DeepSpeed's CPU Adam. Every case runs as a
GPUlayer job (`target="remote"`, one GPU per machine). ms/step is timed from step 20 with
`torch.cuda.synchronize()` on rank 0; memory is rank 0's peak `torch.cuda.max_memory_allocated`.

| ZeRO | Machines | Peak GPU MB | ms/step | Loss (step 1 -> 200) | Checks |
|---|---|---|---|---|---|
| stage 1 | 1 | 1036 | 18.36 | 0.693 -> 0.150 | 4/4 |
| stage 2 | 1 | 1036 | 18.42 | 0.693 -> 0.150 | 4/4 |
| stage 3 | 1 | 790 | 26.27 | 0.693 -> 0.150 | 4/4 |
| stage 2, optimizer on CPU | 1 | 838 | 55.25 | 0.693 -> 0.150 | 4/4 |
| stage 3, optimizer and parameters on CPU | 1 | 458 | 120.58 | 0.693 -> 0.150 | 4/4 |
| stage 1 | 2 (NCCL) | 934 | 66.32 | 0.693 -> 0.070 | 4/4 |
| stage 2 | 2 | 934 | 65.67 | 0.693 -> 0.070 | 4/4 |
| stage 3 | 2 | 590 | 59.42 | 0.693 -> 0.069 | 4/4 |
| stage 2, optimizer on CPU | 2 | 840 | 87.36 | 0.693 -> 0.069 | 4/4 |
| stage 3, optimizer and parameters on CPU | 2 | 453 | 112.42 | 0.693 -> 0.070 | 4/4 |

Checks: world size; the loss falls; the checkpoint holds all 17,840,130 parameters (gathered
from both ranks with stage 3) and loads into a fresh model; the trained weights came back to
the launching process. Steps across machines are communication-bound (TCP, no EFA); this is a
validation, not a speed result. Logs: `ds<stage>[offload]-<machines>.log`.

## Bugs found and fixed in this run

1. DeepSpeed in one process failed with `No module named 'mpi4py'`: without a process group
   DeepSpeed tries MPI discovery. The Trainer now starts a one-process group for it.
2. Then `DeepSpeed requires the LOCAL_RANK environment variable`. The Trainer sets it for the
   DeepSpeed run and removes it afterwards.
3. The offload cases failed with `Ninja is required to load C++ extensions`: ninja was in the
   worker's venv, but jobs did not have that venv's `bin` on `PATH`. Workers now put the job
   Python's `bin` directory first on `PATH`, as an activated venv does.

Each has a regression test (`tests/pytorch/test_lightning_parity_3.py`,
`tests/unit/test_concurrency.py`). All rows above ran with the fixes.

## Manual optimization under DDP (scripts/validate_nccl.py --case manual --nodes 2)

Re-run because `manual_backward`'s gradient averaging changed after the 8-machine run of
2026-11-01 (it no longer relies on DDP's reducer). 2 machines, NCCL, 16-mixed with
`clip_gradients`: loss 2.29 -> 0.15, identical weights on both ranks (506.9764), 7.79 ms/step.
Log: `manual-2.log`.

## bitsandbytes (scripts/validate_deepspeed.py --case bnb, one T4, in-process)

The same MLP with `BitsandbytesPrecision(mode, dtype=torch.float16, ignore_modules={"head"})`:
the body's `nn.Linear` layers are quantized, the head stays trainable.

| Mode | Layers | Peak GPU MB (32-true -> quantized) | ms/step (32-true -> quantized) | Body output cosine vs fp32 | Checks |
|---|---|---|---|---|---|
| nf4 | `Linear4bit` | 145 -> 27 | 2.92 -> 2.77 | 0.9969 | 4/4 |
| int8 | `Linear8bitLt` | 145 -> 36 | 2.91 -> 3.88 | 1.0000 | 4/4 |

Checks: layers quantized; the quantized body's output matches fp32 (cosine > 0.95); the
trainable head's weights changed; less GPU memory than 32-true. Logs: `bnb-*.log`.

Not run here: Transformer Engine (FP8 needs Hopper or Ada GPUs), TensorRT, TPUs.
