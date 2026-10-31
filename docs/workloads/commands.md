# Commands and scripts

Any executable can be a GPUlayer job, such as a shell or Python script. The job runs in a fresh directory on the chosen worker, with
your source directory and input files staged next to it.

## Run a command

```bash
gpulayer submit -- python -c "print('hello')"
gpulayer submit --accelerator mps -- python train.py
gpulayer submit --source ./project --timeout 3600 --env MODE=fast -- python train.py --lr 0.01
gpulayer submit --input data.csv:raw/data.csv -- python prep.py   # file at $GPULAYER_INPUT_DIR/raw/data.csv
gpulayer submit --secret API_KEY=MY_API_KEY -- python call_api.py # read from the worker's environment
```

Without `--detach`, the CLI streams the output and exits with the job's exit code. With an
embedded control plane it also starts a worker in the same process for the duration of the
command.

A leading `python` means the worker's interpreter. Jobs do not inherit the worker's
environment; pass values with `--env` and `--secret` (see
[Environment variables](../reference/environment.md)).

## A complete example

[`examples/shell_job`](../../examples/shell_job) sums a CSV with `awk` and returns the result
file:

```bash
gpulayer submit --source examples/shell_job \
    --input examples/shell_job/sales.csv:sales.csv -- bash report.sh
gpulayer job result <job-id> --download ./out      # out/totals.csv
```

## Call a function

```bash
gpulayer submit --source ./project --function pkg.module:train --arg 0.01 --arg '{"epochs": 5}'
```

Arguments are JSON; the return value is printed as JSON.

## Distributed scripts

```bash
gpulayer submit --distributed --source ./project --entrypoint train_ddp.py \
    --nodes 2 --procs-per-node 4 --accelerator cuda --backend nccl
```

See [Distributed training](../pytorch/distributed.md).

## Queue now, run later

```bash
gpulayer submit --detach -- python long_job.py     # prints the job id
gpulayer worker start                              # a long-running worker picks it up
```

## Inspect and control

```bash
gpulayer job list
gpulayer job status <job-id>          # state, why it was placed (or not), tasks per node
gpulayer job logs <job-id> -f         # follow output
gpulayer job events <job-id>          # lifecycle events with timestamps
gpulayer job result <job-id> --download ./out
gpulayer job cancel <job-id>
gpulayer worker status
```

When a job cannot be placed, `job status` shows the reason for every worker:

```text
scheduling: no eligible worker
  wkr-10c5089a6044: needs 1 usable cuda device(s), worker has 0
```

## Against a server

Set `GPULAYER_URL`, `GPULAYER_TOKEN` and `GPULAYER_FINGERPRINT` (printed by
[`gpulayer up`](../guides/self-hosting.md)), or pass `--url`, `--token` and `--fingerprint` before the
command. The same commands then talk to the server instead of the embedded control plane.

Full option list: [CLI reference](../reference/cli.md).
