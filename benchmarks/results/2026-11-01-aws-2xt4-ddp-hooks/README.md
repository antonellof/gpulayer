# DDP communication hooks and post-local SGD on Tesla T4s (AWS)

2026-11-01. 2x `g4dn.xlarge` Spot (1x Tesla T4 each), us-east-1c, NCCL over TCP (no EFA);
torch 2.15.0+cu132, GPUlayer at commit `bca5905`. One server (`gpulayer up`) on machine 0,
machine 1 joined. Harness: `scripts/validate_parallel.py --case ddp --hook ...`. Same model as
`2026-11-01-aws-2xt4-parallel` (8 residual MLP blocks, 67.2M parameters), batch 512 per GPU,
AdamW, 32-true, 200 steps; ms/step timed from the 10th step.

| Case | ms/step | Loss (first -> last logged) | Weight sums on the two ranks | Checks |
|---|---|---|---|---|
| plain DDP | 706.8 | 0.785 -> 0.016 | 8111.0410, 8111.0410 | 4/4 |
| `ddp_comm_hook=fp16_compress_hook` | 628.5 | 0.785 -> 0.017 | 8110.0171, 8110.0171 | 5/5 |
| post-local SGD (period 4 after 20 steps), 200 steps | 555.8 | 0.785 -> 0.0003 | 8126.8853, 8117.9888 | see below |
| post-local SGD, 197 steps (ends on an averaging step) | 563.6 | | 8126.3433, 8126.3433 | 6/6 |

- The fp16 hook is registered (DDP's logging data reports `fp16_compress_hook`), the replicas
  stay identical, the loss follows plain DDP, and steps are 11% faster: half the bytes cross
  the network.
- Post-local SGD: the optimizer becomes `PostLocalSGDOptimizer` and the hook
  `post_localSGD_hook`. After the warm-up each rank steps on its own and the models are
  averaged every 4 steps; steps are 21% faster than plain DDP. PyTorch's averager counts from
  0, so a 200-step run averages last at step 196 and ends with three local steps: the replicas
  differ by 0.11% (the harness's first check expected them identical and failed; it now checks
  that). Ending on an averaging step (197 steps) gives identical replicas.

These are speed differences of the communication strategies on this network, not a
comparison with another trainer. Logs: `ddp-*.log`; `nvidia-smi.txt`, `pip-freeze.txt`.
