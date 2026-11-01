# CLI

Global options go before the command:

| Option | Default | Meaning |
|---|---|---|
| `--home DIR` | `$GPULAYER_HOME` or `~/.gpulayer` | state directory |
| `--url URL` | `$GPULAYER_URL` | server; unset means embedded |
| `--token TOKEN` | `$GPULAYER_TOKEN` | client token |
| `--fingerprint FP` | `$GPULAYER_FINGERPRINT` | pinned server certificate |
| `--ca-file FILE` | | CA bundle for a CA-signed server |
| `-v LEVEL` | `WARNING` | log level |

## Servers and workers

| Command | Description |
|---|---|
| `gpulayer up [--host --port --advertise-address --no-worker --gpus 0,1\|none --name --label K=V --region --zone --network-group --python --tls --tls-cert --tls-key --insecure-http]` | server plus a local worker; prints join and connect commands |
| `gpulayer join URL --token T [--fingerprint FP --name --label K=V --region --zone --network-group --python --advertise-address --workdir --concurrency N --gpus 0,1\|none --max-tasks N --idle-exit S]` | join a server as a worker. `--concurrency`: tasks at once (default one per GPU). `--gpus`: GPUs offered. `--max-tasks`: exit after N tasks |
| `gpulayer dashboard [--port 8471 --no-open]` | web dashboard for local jobs |
| `gpulayer mcp [--read-only]` | MCP server on stdio ([guide](../guides/ai-agents.md)) |
| `gpulayer metrics` | Prometheus metrics (embedded, or `--url`) |
| `gpulayer gc --older-than AGE [--delete-jobs --dry-run]` | admin: drop logs and events of jobs finished before AGE (e.g. `7d`), optionally the jobs, and unreferenced artifacts |
| `gpulayer token create NAME [--role admin\|submitter\|join\|metrics --ttl 12h]` | named token, printed once (run on the server machine) |
| `gpulayer token list` | tokens with role and status, no values |
| `gpulayer token revoke NAME` | revoke; a running server applies it on the next request |
| `gpulayer server init [--rotate]` | client and join tokens for `server start`, printed once |
| `gpulayer server start [--host --port --tls-cert --tls-key --allow-insecure-remote --public-url]` | API only |
| `gpulayer worker start [--name --label --region --zone --network-group --join-token --python --advertise-address --workdir --concurrency --gpus --max-tasks --idle-exit]` | worker: embedded without `--url`, remote with it |
| `gpulayer worker status [--json]` | registered workers |
| `gpulayer worker revoke WORKER_ID` | revoke a worker's credentials |
| `gpulayer resources detect [--json --python]` | what this machine offers |
| `gpulayer resources list [--json]` | same as `worker status` |

## Jobs

```text
gpulayer submit [options] [-- COMMAND ...]
```

| Option | Meaning |
|---|---|
| `-f FILE` | `JobSpec` JSON file |
| `--name NAME` | job name |
| `--target local\|remote\|cloud`, `--fallback TARGET` | target and explicit fallbacks |
| `--source DIR` | upload a directory; the command runs inside it |
| `--input FILE[:DEST]` | stage a file into `$GPULAYER_INPUT_DIR` |
| `--function MODULE:FUNC --arg JSON ...` | function job |
| `--distributed --entrypoint SCRIPT --nodes N --procs-per-node P --backend gloo\|nccl [--launcher direct\|torchrun]` | distributed job |
| `--accelerator cpu\|cuda\|mps\|rocm --gpus N --gpu-memory GB --gpu-memory-kind dedicated\|unified --cpus N --memory GB` | resources |
| `--framework 'torch>=2.1'` | framework requirement |
| `--profile pytorch-cuda` | sets accelerator and frameworks ([profiles](job-spec.md#runtime-profiles)) |
| `--worker ID`, `--label K=V` | pin to workers |
| `--requirement PKG` (repeatable), `--requirements-file FILE` | extra packages, installed once per worker and reused |
| `--group-locality prefer\|network_group\|zone\|region\|any` | distributed: how close the machines must be (default `prefer`) |
| `--provider NAME --max-hourly-cost X` | allow provisioning through a provider |
| `--env K=V`, `--secret NAME=WORKER_ENV_KEY` | environment and secret references |
| `--timeout SECONDS`, `--max-attempts N` | limits and retries |
| `--detach` | print the job id and return |
| `--queue-if-unschedulable` | wait for a compatible worker instead of failing |

| Command | Description |
|---|---|
| `gpulayer job list [--limit N --json]` | recent jobs |
| `gpulayer job status ID [--json]` | state, placement explanation, tasks |
| `gpulayer job logs ID [-f --system --verbose]` | output |
| `gpulayer job events ID [--json]` | lifecycle events |
| `gpulayer job result ID [--download DIR]` | result JSON; optionally download outputs |
| `gpulayer job cancel ID` | cancel |

Without `--detach`, `gpulayer submit` exits with the job's exit code.
