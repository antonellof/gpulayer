# GPUlayer 0.11 on L4, H200 and RTX PRO 6000 (Hugging Face Jobs)

2026-11-09. Hugging Face Jobs: one container per run, `python:3.12` (Debian 13) with the
repository at `a010b0a` (0.11.0) plus the checks and fixes on branch `validate/hf-gpus`, torch
2.15.0+cu132 from PyPI, NVIDIA driver 580.178.04, Linux 6.12. FP8 ran in NVIDIA's
`nvcr.io/nvidia/pytorch:26.10-py3` container (torch 2.15.0a0, Transformer Engine 2.20.2).
These are correctness checks, not speed comparisons: no GPUlayer-against-Lightning timings
were taken here.

| Machine | Flavor | GPUs |
|---|---|---|
| 4× NVIDIA L4 (Ada, sm_89, 22.5 GB each, PCIe, `NODE` links) | `l4x4` | 4 |
| NVIDIA H200 (Hopper, sm_90, 140 GB) | `h200` | 1 |
| NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB) | `rtx-pro-6000` | 1 |
| NVIDIA L4 | `l4x1` | 1 (FP8) |

Method: `bash scripts/validate_gpu.sh` (with `LIBRARIES=1` on the L4s): the test suite with the
Lightning conformance tests, the `hardware_cuda` tests, `validate_cuda.py`,
`validate_trainer_gpu.py`, `validate_parity_gpu.py`, `validate_multigpu.py` and
`validate_libraries.py`. FP8: `python scripts/validate_te_fp8.py`.

## Results

| Check | L4 | H200 | RTX PRO 6000 |
|---|---|---|---|
| Test suite, with the conformance tests | 638 passed | 638 passed | 637 passed, 1 failed (bug 2); with the fixes 640 passed, 1 failed (a new unit test that assumed CUDA was not initialized, since corrected) |
| CUDA checks: detection, fp32/bf16/fp16, `torch.compile`, GPU per job | 8/8 | 8/8 | 8/8 |
| Trainer GPU checks: compile, 16-mixed manual optimization, 16/bf16-true, profiler, EMA, device stats | all passed | all passed | all passed |
| Exact resume in one process (identical weights, dropout on the GPU) | passed | passed | passed |
| Resume over NCCL, 4 GPUs (each sample trained once) | passed | | |
| `torchmetrics` object synced over NCCL, 4 GPUs; `on_validation_epoch_end` value averaged over ranks | passed (0.7910 = 0.7910) | | |
| Notebook: `devices=4` forked from a Jupyter kernel | failed (bug 1); passed with the fix: trained weights back at step 8, one rank on each of `cuda:0` to `cuda:3`, CUDA unused in the kernel, and the new messages for both mistakes | | |
| Interconnect, GPU choice, NCCL 1 node × 4 processes | passed | | |
| Automatic resume after SIGTERM, 20 runs | | | 20/20 with the fix |

FP8 on one L4 (`TransformerEnginePrecision`): `nn.Linear` and `nn.LayerNorm` became Transformer
Engine layers in bfloat16, and a 15-wide layer was kept; FP8 was on in every epoch's forward
pass; the loss went from 32.25 to 4.16 in 3 epochs, against 4.09 in bf16-true. A checkpoint
loads into a new model with identical outputs, but Transformer Engine 2.20 first refuses its
FP8 scaling state, a pickle, unless `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1` is set.

Other libraries as GPUlayer jobs on an L4 (cold environment, then warm):

| Library | Result |
|---|---|
| JAX 0.12.0.1 | ran on the L4 (42.3 s, then 0.0 s to prepare) |
| CuPy 14.2.0 | ran on the L4 (18.9 s, then 0.0 s) |
| Transformers 5.22.2 | ran on the L4 (10.3 s, then 0.0 s) |
| TensorFlow | no GPU: "Cannot dlopen some GPU libraries" |
| ONNX Runtime 1.31.1 | CPU only: the CUDA provider needs cuDNN 9 and CUDA 13 libraries it did not find |

TensorFlow and ONNX Runtime were installed per job next to torch's CUDA 13 pip libraries and
found neither their own nor torch's; not investigated further.

## Bugs found and fixed

1. **Notebook forks could not use CUDA.** Every forked rank failed with "Cannot re-initialize
   CUDA in forked subprocess". `torch.cuda.is_available()` asks the CUDA driver, after which a
   forked child cannot use CUDA, although `torch.cuda.is_initialized()` stays False, so the
   Trainer's own guard did not see it. The Trainer now asks NVML
   (`PYTORCH_NVML_BASED_CUDA_CHECK`), as Lightning does, and a rank forked after a cell asked
   the driver explains how to avoid it.
2. **Automatic resume could repeat steps.** After a SIGTERM at step 12, the next attempt
   resumed at step 10. The worker decides whether a checkpoint is new by its modification time
   and size; Linux stamps files from a coarse clock, and the epoch's checkpoint and the SIGTERM
   one, written about 3 ms apart with the same size, looked the same. The worker now compares
   contents. Passed on the L4 and H200 by timing luck; failed on the RTX PRO 6000 host.

Logs: kept outside the repository (`gpulayer-validation-results/2026-11-09-hf-*`).
