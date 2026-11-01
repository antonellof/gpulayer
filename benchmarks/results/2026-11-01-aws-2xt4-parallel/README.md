# FSDP options, tensor parallelism and TensorRT on Tesla T4s (AWS)

2026-11-01. 2x `g4dn.xlarge` Spot (1x Tesla T4 16 GB each), us-east-1c, one subnet; driver
595.112.07; Python 3.12, torch 2.15.0+cu132, GPUlayer at commit `b97ae6a`. One server
(`gpulayer up`) on machine 0, machine 1 joined; jobs ran one at a time; NCCL across the two
machines (TCP, no EFA). Harness: `scripts/validate_parallel.py`.

Model: 8 residual MLP blocks of width 1024 (fc1 1024 -> 4096, fc2 4096 -> 1024, LayerNorm), a
2-class head, 67.2M parameters; synthetic data, batch 512 per GPU, AdamW, 32-true, 200 steps.
Peak memory is rank 0's `torch.cuda.max_memory_allocated`; ms/step is timed from the 10th
step of each run.

| Case | GPUs | Peak MB | ms/step | Loss (first -> last logged) | Checks |
|---|---|---|---|---|---|
| one GPU (baseline) | 1 | 1283 | 288 | 0.785 -> 0.008 | 3/3 |
| tensor parallel (`ModelParallelStrategy`, fc1/fc2 split) | 2 | 657 | 312 | 0.778 -> 0.009 | 4/4 |
| FSDP `FULL_SHARD`, each Block a unit | 2 | 671 | 841 | 0.785 -> 0.016 | 4/4 |
| same + activation checkpointing per Block | 2 | 643 | 853 | 0.785 -> 0.016 | 5/5 |
| FSDP `SHARD_GRAD_OP` | 2 | 836 | 733 | 0.785 -> 0.016 | 3/3 |
| FSDP `FULL_SHARD` + CPU offload | 2 | 270 | 987 | 0.785 -> 0.016 | 5/5 |
| FSDP, sharded checkpoint, 100 steps | 2 | 671 | 629 | 0.785 -> 0.014 | 5/5 |
| resumed from that folder to 200 steps | 2 | 671 | 632 | 0.010 -> 0.006 | 6/6 |

What the checks show:

- Tensor parallelism: each GPU holds half of every fc1/fc2 (`fc1` local shape [2048, 1024] of
  [4096, 1024]), half the parameters and about half the peak memory of one GPU, and the loss
  follows the one-GPU run. Full weights are gathered into the checkpoint and load into a fresh
  model.
- FSDP: each rank holds half of the parameters; activation checkpointing wraps every Block
  (this model's activations are small, so it saves 28 MB here); `SHARD_GRAD_OP` keeps
  gathered parameters after the forward (more memory, 13% faster steps); CPU offload keeps
  the parameters in CPU memory (270 MB on the GPU). All FSDP variants give the same losses.
- Sharded checkpoints across machines: each node uploaded its own shard files; the launching
  process merged them into one `last.ckpt/` folder (no shared filesystem), it loaded into a
  fresh model, and a second job resumed from it and trained exactly the remaining 100 steps.

Steps across machines are communication-bound on this network; this is a validation, not a
speed result.

## Bug found and fixed in this run

The first attempt returned no outputs from the joined machine: the 67M-parameter model's
checkpoint (about 800 MB with AdamW state) passed the worker's 512 MiB output cap, and the
upload loop stopped there, dropping the small result files after it, without a message. Files
over a limit are now skipped and named in the task's system log, the others still upload, and
the default cap is 4 GiB per task (`GPULAYER_MAX_OUTPUT_BYTES`), 2 GiB per file. All rows above
ran with the fix.

## TensorRT (`Module.to_tensorrt`, one T4)

In a separate venv, because torch-tensorrt 2.14.0 requires torch 2.14.1 (cu130), TensorRT
11.1.0. The same model (batch 512, fp32) compiled through Torch-TensorRT: the result holds a
`TorchTensorRTModule` engine, the exported program was saved, and outputs match eager PyTorch
(max difference 9.5e-7 on fresh inputs; `tensorrt-engine-check.txt`). In fp32 on a T4 it ran at
the same speed as eager (19.3 against 19.6 ms per batch); this run checks the export, not a
speed-up.

Logs: one `.log` per row, `tensorrt-1.log`, `nvidia-smi.txt`, `pip-freeze*.txt`.
