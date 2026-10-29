# Getting started

## 1. Install

```bash
uv venv && uv pip install -e '.[dev]'
uv pip install torch            # optional; needed for MPS/CUDA detection, torchrun jobs, examples
gpulayer --version
```

State lives in `$GPULAYER_HOME` (default `~/.gpulayer`, mode 0700): `metadata.db` (SQLite),
`artifacts/` (content-addressed blobs) and `workers/<id>/` (task workspaces and artifact cache).

## 2. See what this machine offers

```bash
$ gpulayer resources detect
host      my-mac (darwin/arm64), 10 cores, 32.0 GB RAM, python 3.11.5
torch     2.5.1  distributed backends: gloo
device    [0] mps Apple arm64 GPU (MPS) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

Detection never guesses: unknown values are reported as unknown, and detected-but-unsupported
hardware (e.g. ROCm) is listed with `execution_supported=false`.

## 3. Run jobs locally

```bash
gpulayer submit -- python -c "print('hello')"
gpulayer submit --accelerator mps -- python train.py          # needs a worker with MPS
gpulayer submit --source ./proj --function pkg.mod:fn --arg '{"x": 1}'
gpulayer submit --timeout 60 --env MODE=fast --secret API_KEY=MY_API_KEY -- python job.py
gpulayer submit --input data.csv:raw/data.csv -- python job.py  # $GPULAYER_INPUT_DIR/raw/data.csv
```

Without `--detach`, the CLI starts an in-process local worker, streams logs, and exits with the
job's exit code. With `--detach` it prints the job id and the job waits for `gpulayer worker start`.

Inside a workload:

| Variable | Meaning |
|---|---|
| `GPULAYER_JOB_ID`, `GPULAYER_TASK_ID`, `GPULAYER_ATTEMPT` | identity |
| `GPULAYER_INPUT_DIR` / `GPULAYER_OUTPUT_DIR` | staged inputs / files returned as output artifacts |
| `GPULAYER_ACCELERATOR` | `cpu`, `cuda` or `mps` as requested |
| `GPULAYER_NODE_RANK`, `GPULAYER_COMM_BACKEND` | distributed jobs (plus torchrun's `RANK`, `WORLD_SIZE`, ...) |
| `CUDA_VISIBLE_DEVICES` | the granted CUDA devices; empty for CPU jobs |

A leading `python` in a command means the worker's configured interpreter. The workload does
**not** inherit the worker's environment beyond `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`,
`TMPDIR`, `TERM`; pass anything else with `--env` or `--secret`.

## 4. Use the SDK

```python
import gpulayer as gl

with gl.Client() as c:
    src = c.package("./proj")
    job = c.submit(gl.function("pkg.mod:fn", 1, 2, source=src))
    status = job.wait(timeout=300)
    print(status.state, job.result().return_value, job.logs_text())
    jobs = c.submit_batch([gl.command(["python", "-c", f"print({i})"]) for i in range(3)])
```

`Client(url=..., token=...)` talks to a self-hosted control plane with the same API.

## 5. Next

- Remote machines: [architecture/remote-workers.md](architecture/remote-workers.md)
- Multi-node PyTorch: [architecture/distributed.md](architecture/distributed.md)
- Examples: [../examples](../examples)
