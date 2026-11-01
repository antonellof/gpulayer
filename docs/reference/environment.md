# Environment variables

## Read by GPUlayer

| Variable | Used by | Meaning |
|---|---|---|
| `GPULAYER_HOME` | everything | state directory (default `~/.gpulayer`) |
| `GPULAYER_URL` | SDK, CLI, Trainer, launch | server URL; unset means embedded |
| `GPULAYER_TOKEN` | SDK, CLI | client token |
| `GPULAYER_FINGERPRINT` | SDK, CLI, join | pinned server certificate (`sha256:...`) |
| `GPULAYER_JOIN_TOKEN` | `join`, `worker start` | worker join token |
| `GPULAYER_WORKER_PYTHON` | SDK in-process workers | interpreter for jobs |
| `GPULAYER_REGION`, `GPULAYER_ZONE` | `up`, `join`, `worker start` | default `--region` and `--zone` for [group placement](../architecture/scheduling.md#distributed-groups-and-network-locality); from GCE metadata on Google Cloud when unset |
| `GPULAYER_MAX_OUTPUT_BYTES` | workers | total output a task may return (default 4 GiB; one file at most 2 GiB). A file over it is skipped and named in the task's system log |

## Set for every job

| Variable | Meaning |
|---|---|
| `GPULAYER_JOB_ID`, `GPULAYER_TASK_ID`, `GPULAYER_ATTEMPT` | identity |
| `GPULAYER_NODE_RANK` | node rank (0 for independent jobs) |
| `GPULAYER_ACCELERATOR` | requested `cpu`, `cuda` or `mps` |
| `GPULAYER_INPUT_DIR` | staged inputs |
| `GPULAYER_OUTPUT_DIR` | files here become output artifacts; the job's Python's `bin` is first on `PATH` |
| `GPULAYER_COMM_BACKEND` | distributed: `gloo` or `nccl` |
| `CUDA_VISIBLE_DEVICES` | granted CUDA devices; empty for CPU jobs |
| `PYTHONUNBUFFERED=1` | logs stream as written |

Distributed jobs, with either launcher, also get `RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
`LOCAL_WORLD_SIZE`, `GROUP_RANK`, `MASTER_ADDR`, `MASTER_PORT` and `TORCHELASTIC_RUN_ID` (the
job id with the direct launcher).

Jobs inherit only these from the worker: `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TMPDIR`,
`TERM`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`, `NVIDIA_VISIBLE_DEVICES`,
`NVIDIA_DRIVER_CAPABILITIES`, `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`. Pass
anything else, including proxies, with `env` or `secrets`.
