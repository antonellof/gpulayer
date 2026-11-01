# NCCL across machines and GPU-only Trainer features: 8 AWS machines with one Tesla T4 each

2026-11-01. 8x `g4dn.xlarge` Spot (4 vCPU, 1x Tesla T4 16 GB, "up to 25 Gbps"), us-east-1c, one
VPC and subnet, AWS Deep Learning Base OSS NVIDIA driver AMI (Ubuntu 22.04, driver 595.112.07),
Python 3.12, torch 2.15.0+cu132, Lightning 2.6.6, GPUlayer 0.8.0 (commits `331124f` to
`8c32442`). Set up with `benchmarks/cloud/aws.sh` (first-boot install, `aws-boot.sh`).

One GPUlayer server (`gpulayer up`) on machine 0, which also runs jobs; machines 1-7 joined with
`gpulayer join` over private IPs, all with `--zone us-east-1c --network-group aws-use1c`. Every
job was submitted from machine 0 by `scripts/validate_nccl.py` (Trainer, `target="remote"`, one
GPU per machine), one job at a time.

## Across machines (scripts/validate_nccl.py)

Synthetic CIFAR-sized data, a 3-layer CNN with BatchNorm, batch 64 per GPU (each step trains on
64 x machines samples), SGD with momentum, 300 steps. ms/step is timed from step 50 to the end
with `torch.cuda.synchronize()` on rank 0.

| Case | Machines | Precision | ms/step | Loss (step 1 -> 300) | Checks |
|---|---|---|---|---|---|
| DDP (no process group: one process) | 1 | 16-mixed | 4.56 | 2.33 -> 1.52 | baseline |
| DDP + `sync_batchnorm` + `gradient_as_bucket_view` | 2 | 16-mixed | 12.95 | 2.29 -> 0.12 | nccl, SyncBatchNorm, identical weights on both ranks |
| same | 4 | 16-mixed | 19.51 | 2.30 -> 0.005 | nccl, SyncBatchNorm, identical weights on 4 ranks |
| same | 8 | 16-mixed | 25.67 | 2.27 -> 0.006 | nccl, SyncBatchNorm, identical weights on 8 ranks |
| manual optimization (`manual_backward`, `clip_gradients`, GradScaler) | 8 | 16-mixed | 17.69 | 2.27 -> 0.008 | nccl, identical weights on 8 ranks |
| FSDP (`strategy="fsdp"`) | 2 | 32-true | 24.87 | 2.29 -> 0.11 | nccl, each rank holds 50% of the parameters |
| FSDP | 8 | 32-true | 42.73 | 2.27 -> 0.007 | nccl, each rank holds 12.6% of the parameters |

Throughput: 1 machine 14,000 samples/s; 8 machines 19,900 samples/s (64 x 8 / 25.67 ms). Steps
are communication-bound on this network with this small model (TCP sockets, no EFA on
`g4dn.xlarge`); this is a validation, not a speed result.

**Bug found and fixed:** the first manual-optimization run on 8 machines did not learn (loss
2.27 -> 2.30). On 2 machines, only 16-mixed with `clip_gradients` failed (32-true, or no
clipping, learned). `clip_gradients` unscaled the wrapper returned by `self.optimizers()`, so
`GradScaler.step` unscaled the real optimizer's gradients again. Fixed in `8c32442` with a
regression test; the 8-machine row above is the rerun. Logs: `manual-*.log`.

## One GPU (scripts/validate_trainer_gpu.py, machine 0)

All checks passed (`trainer_gpu.log`): Inductor `torch.compile` (loss equal to eager, 0.3726),
16-mixed manual optimization (48 steps, 0.636 -> 0.373), `16-true` and `bf16-true`,
`PyTorchProfiler` with CUDA columns, `EMAWeightAveraging` with 16-mixed, `DeviceStatsMonitor`
CUDA memory.

## Environment note

Convolutions first failed with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`: the AMI sets
`LD_LIBRARY_PATH` to its system CUDA in `/etc/profile.d/dlami.sh`, which clashes with the PyTorch
wheel's cuDNN. Workers were restarted with `env -u LD_LIBRARY_PATH`; `aws-boot.sh` now clears it.
Workers run jobs from their own installed GPUlayer: after a fix, update every worker (a mixed
fleet gave ranks different code and different weights).
