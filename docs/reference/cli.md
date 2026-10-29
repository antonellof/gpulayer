# CLI

Global options go before the command:

| Option | Default | Meaning |
|---|---|---|
| `--home DIR` | `$GPULAYER_HOME` or `~/.gpulayer` | local state directory |
| `--url URL` | `$GPULAYER_URL` | server to talk to; unset means the embedded control plane |
| `--token TOKEN` | `$GPULAYER_TOKEN` | client token |
| `--fingerprint FP` | `$GPULAYER_FINGERPRINT` | pinned server certificate |
| `--ca-file FILE` | | CA bundle for a server with a CA-signed certificate |
| `-v LEVEL` | `WARNING` | log level |

## Servers and workers

| Command | Description |
|---|---|
| `gpulayer up [--host --port --advertise-address --no-worker --name --label K=V --python --tls --tls-cert --tls-key --insecure-http]` | start a server and use this machine as a worker; prints join and connect commands |
| `gpulayer join URL --token T [--fingerprint FP --name --label K=V --python --advertise-address --workdir --max-tasks N --idle-exit S]` | add this machine as a worker of a server |
| `gpulayer token create NAME [--role admin\|submitter\|join --ttl 12h]` | create a named token on the server machine (printed once) |
| `gpulayer token list` | list tokens with role and status, never their values |
| `gpulayer token revoke NAME` | revoke a token; a running server applies it on the next request |
| `gpulayer server init [--rotate]` | create client and join tokens for `server start` (printed once) |
| `gpulayer server start [--host --port --tls-cert --tls-key --allow-insecure-remote --public-url]` | serve the API only |
| `gpulayer worker start [--name --label --join-token --python --advertise-address --workdir --max-tasks --idle-exit]` | run a worker: embedded without `--url`, remote with it |
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
| `-f FILE` | submit a `JobSpec` JSON file |
| `--name NAME` | job name |
| `--target local\|remote\|cloud`, `--fallback TARGET` | placement target and explicit fallbacks |
| `--source DIR` | package and upload a directory; the command runs inside it |
| `--input FILE[:DEST]` | stage a file into `$GPULAYER_INPUT_DIR` |
| `--function MODULE:FUNC --arg JSON ...` | function job |
| `--distributed --entrypoint SCRIPT --nodes N --procs-per-node P --backend gloo\|nccl` | torchrun job |
| `--accelerator cpu\|cuda\|mps\|rocm --gpus N --gpu-memory GB --cpus N --memory GB` | resources |
| `--framework 'torch>=2.1'` | framework requirement |
| `--worker ID`, `--label K=V` | pin to workers |
| `--provider NAME --max-hourly-cost X` | allow provisioning through a provider adapter |
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

`gpulayer submit` without `--detach` exits with the job's exit code.
