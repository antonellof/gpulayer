# Realistic models, 8 GPUs, GPU services, FP8 on Hopper and Blackwell, remote checkpoints

2026-11-09, Hugging Face Jobs (one container per run, NVIDIA driver 580.178.04), GPUlayer 0.11.2
plus this commit range (benchmark cases and checks). Logs are kept outside the repository
(`gpulayer-validation-results/2026-11-09-*`).

## 1. Speed against Lightning on realistic models

`benchmarks/trainer_comparison/run.py` with two new cases: a GPT-2-size decoder (torch's own
layers: 12 pre-norm blocks of width 768, 12 heads, causal attention, 32k vocabulary, tied
embeddings, 110M parameters; 512 tokens, batch 8 per GPU) and torchvision's ResNet-50 (224x224,
batch 64 per GPU), both bf16-mixed with AdamW; `-compile` wraps the model in `torch.compile` in
both trainers. Same method as before (fresh processes, three short and three long runs
interleaved, nothing else on the GPUs, checked), with 20 and 120 steps; a compiled case first
runs once unmeasured, since the first run fills torch.compile's on-disk cache (without it, one
H200 case came out at a negative step time, and those first compiled runs are discarded).
Lightning 2.6.6; torch 2.15.0+cu132 and torchvision 0.29.2 (`python:3.12`), or NVIDIA's 26.10
container on the 8-GPU machine.

| Case | L4 GPUlayer / Lightning ms | Result | H200 GPUlayer / Lightning ms | Result |
|---|---|---|---|---|
| GPT, bf16 | 133.4 / 136.8 | GPUlayer 1.03x faster | 16.8 / 19.2 | GPUlayer 1.15x faster |
| ResNet-50, bf16 | 166.8 / 170.2 | GPUlayer 1.02x faster | 50.7 / 50.1 | Lightning 1.01x faster |
| GPT, compiled | 126.0 / 130.6 | GPUlayer 1.04x faster | 6.8 / 10.2 | GPUlayer 1.51x faster |
| ResNet-50, compiled | 128.5 / 132.2 | GPUlayer 1.03x faster | 47.9 / 51.0 | GPUlayer 1.07x faster |

On the L4 the GPU does most of the work (130 to 170 ms a step) and the trainers are within 4%.
On the H200 the compiled GPT takes 7 to 10 ms a step, and per-step CPU work decides.

8× RTX PRO 6000 Blackwell (PCIe, two CPU sockets: GPUs 0-3 on one, 4-7 on the other):

| Case | GPUlayer / Lightning ms | Result |
|---|---|---|
| GPT, 1 GPU | 23.8 / 25.3 | GPUlayer 1.06x faster |
| ResNet-50, 1 GPU | 34.0 / 37.3 | GPUlayer 1.10x faster |
| GPT, 8 GPUs (20 and 520 steps) | 51.8 / 45.1 | **Lightning 1.15x faster** |
| ResNet-50, 8 GPUs (20 and 520 steps) | 45.2 / 52.2 | GPUlayer 1.16x faster |

The 8-GPU cases first ran with 20 and 120 steps (GPT: Lightning 1.11x faster; ResNet-50:
GPUlayer 1.31x faster) and were run again with 500 steps between short and long runs, every
wall time printed. The second run is the one in the table: its long runs agree within 1.7%.
On GPT with 8 GPUs, GPUlayer started 3.3 s sooner (13.2 s against 16.5 s for 20 steps) but each
step took 6.7 ms longer, so at 520 steps both took 39.0 s. GPUlayer builds DDP with Lightning's
defaults and sets no NCCL variables; why its DDP step is slower on this model on this machine
is not known yet.

## 2. Eight GPUs in one machine

8× RTX PRO 6000 Blackwell Server Edition (96 GB each, PCIe: `PIX` pairs, `NODE` within a socket,
`SYS` across sockets), NVIDIA's `nvcr.io/nvidia/pytorch:26.10-py3`, DeepSpeed 0.19.9. No 8× A100
or 8× L40S machine became free in over two hours, so this ran on the next 8-GPU flavor. All 50
checks passed:

| Check | Result |
|---|---|
| Topology, `scripts/validate_multigpu.py` | every link read (`PIX`, `NODE`, `SYS`); for 2 to 8 GPUs the chosen set had the best slowest link; `CUDA_VISIBLE_DEVICES` as chosen; NCCL 1 node × 8 processes in 11.7 s |
| DDP, 8 GPUs | identical weights on every rank; 113.1 ms/step |
| FSDP `FULL_SHARD` / `SHARD_GRAD_OP` | 1/8 of the parameters per GPU; 147.3 / 146.7 ms; 394 / 570 MB on rank 0 |
| Tensor parallelism, 8-way | `fc1` split into 512×1024 per GPU; 47.5 ms; 214 MB |
| `validate_nccl.py`: DDP with `sync_batchnorm`, manual optimization | 5/5 and 4/4; 4.95 and 3.43 ms |
| DeepSpeed ZeRO 1 / 2 / 3, 16-mixed | 11.5 / 12.0 / 15.9 ms; 867 / 867 / 493 MB |

The `validate_parallel.py` model is 8 blocks of width 1024 with batch 512 per GPU, so each step
on 8 GPUs trains 8 times the data of one: these are checks, not scaling measurements.

## 3. A long-running service on a GPU

`scripts/validate_service_gpu.py` on one L4, with a local worker: `examples/gpu_service` (a
PyTorch model behind HTTP) as `gl.service(..., accelerator="cuda", gpus=1)` came up in 1.8 s on
the L4 and answered 200 sequential requests (0.92 ms median, 1.11 ms p95, 0.28 ms on the GPU).
Killed, it was started again on a new port within seconds, with `service exited with code -15;
restart 1 in 1s` in the job's log; cancelled, its port closed.

VLLM

## 4. FP8 on Hopper and Blackwell

`scripts/validate_te_fp8.py` in NVIDIA's 26.10 container (Transformer Engine 2.20.2), all five
checks on each GPU, as on the L4 before:

| GPU | FP8 in the forward pass | Loss after 3 epochs, FP8 / bf16-true | Checkpoint |
|---|---|---|---|
| H200 (sm_90) | every epoch | 4.160 / 4.094 | loads with `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`, same outputs |
| RTX PRO 6000 Blackwell (sm_120) | every epoch | 4.157 / 4.095 | same |

## 5. Saving checkpoints to a remote store

`benchmarks/remote_checkpoint.py hf://buckets/antonellof/gpulayer-validation` from an M2 Pro
(CPU, `huggingface_hub` 2.4.0) to a private Hugging Face bucket: a 50 MB model, three
checkpoints per run, modes alternating, median of three runs.

| Steps | No checkpoints | Saving in the loop | `AsyncCheckpointIO` |
|---|---|---|---|
| 60 (checkpoint every 20) | 2.8 s | 11.3 s | 9.1 s |
| 600 (every 200) | 6.6 s | 16.0 s | 10.6 s |

With `AsyncCheckpointIO` the uploads run while training goes on; `fit` still waits for the
last one, which is most of what remains. Every checkpoint of the async runs was in the bucket
and loaded.
