# Environment variables

## Read by GPUlayer

| Variable | Used by | Meaning |
|---|---|---|
| `GPULAYER_HOME` | everything | local state directory (default `~/.gpulayer`) |
| `GPULAYER_URL` | SDK, CLI, Trainer, launch | server URL; unset means the embedded control plane |
| `GPULAYER_TOKEN` | SDK, CLI | client token |
| `GPULAYER_FINGERPRINT` | SDK, CLI, join | pinned server certificate (`sha256:...`) |
| `GPULAYER_JOIN_TOKEN` | `join`, `worker start` | worker join token |
| `GPULAYER_WORKER_PYTHON` | SDK in-process workers | interpreter for jobs |
| `GPULAYER_REGION`, `GPULAYER_ZONE` | `up`, `join`, `worker start` | default for `--region` and `--zone`: where this worker is, for [group placement](../architecture/scheduling.md#distributed-groups-and-network-locality); read from GCE metadata on Google Cloud when unset |

## Set for every job

| Variable | Meaning |
|---|---|
| `GPULAYER_JOB_ID`, `GPULAYER_TASK_ID`, `GPULAYER_ATTEMPT` | identity |
| `GPULAYER_NODE_RANK` | node rank (0 for independent jobs) |
| `GPULAYER_ACCELERATOR` | `cpu`, `cuda` or `mps`, as requested |
| `GPULAYER_INPUT_DIR` | staged input files |
| `GPULAYER_OUTPUT_DIR` | files written here come back as output artifacts |
| `GPULAYER_COMM_BACKEND` | distributed jobs: `gloo` or `nccl` |
| `CUDA_VISIBLE_DEVICES` | granted CUDA devices; empty for CPU jobs |
| `PYTHONUNBUFFERED=1` | so logs stream as they are written |

Distributed jobs also get torchrun's variables, with either launcher: `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and
`TORCHELASTIC_RUN_ID` (the job id with the direct launcher).

Jobs do not inherit the worker's environment, except `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`,
`TMPDIR`, `TERM`, the library and CUDA paths (`LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CUDA_HOME`,
`CUDA_PATH`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`) and the certificate paths
(`SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`). Pass anything else, including
proxies, with `env` or `secrets`.
