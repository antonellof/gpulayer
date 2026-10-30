# Run any script with launch()

`launch()` sends an existing training script to GPUlayer workers without changing the rest of
the script. Add two lines at the top:

```python
import gpulayer.pytorch as glt
glt.launch(nodes=1, devices=2, accelerator="cpu")

# ... your training script, unchanged ...
```

Run it as usual (`python train.py --lr 0.01`). In your terminal, `launch()`:

1. packages the script's directory and uploads it;
2. submits the same script with the same arguments: a single process for `nodes=1, devices=1`,
   otherwise a distributed job with `nodes x devices` processes;
3. streams the job's output to your terminal;
4. downloads everything the job wrote to `$GPULAYER_OUTPUT_DIR` into `gpulayer_runs/<job-id>/`;
5. exits with the job's exit code.

`launch(inputs={"data/train.bin": "/local/train.bin"})` uploads local files and places them
under `$GPULAYER_INPUT_DIR` on every node.

On the worker, `launch()` sees that it is already inside a job and returns at once, so the rest
of the script runs there with the usual torchrun environment (`RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`). Any training code that reads that environment
works, whichever library it uses.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `nodes` | `1` | number of workers |
| `devices` | `1` | processes per worker |
| `accelerator` | `"cpu"` | `"cpu"`, `"cuda"` or `"mps"` (no `"auto"`: the device must be known to place the job) |
| `target` | `"local"` | `"local"`, `"remote"` or `"cloud"` |
| `frameworks` | `{}` | version requirements, e.g. `{"torch": ">=2.2"}` |
| `env` | `{}` | environment variables for the job |
| `timeout_seconds` | `None` | kill the job after this long |
| `output_dir` | `"gpulayer_runs"` | where outputs are downloaded |
| `exit` | `True` | exit with the job's exit code; `False` returns a `LaunchResult` instead |
| `quiet` | `False` | do not stream logs |
| `client` | default client | a `gpulayer.Client` (embedded, or a remote control plane) |
| other keywords | | passed to `PlacementPolicy`, e.g. `worker_labels={"pool": "a100"}` |

With `exit=False`:

```python
result = glt.launch(nodes=2, devices=2, accelerator="cpu", exit=False)
if result is not None:              # None on the workers
    print(result.succeeded, result.job.id, result.outputs)
    raise SystemExit
```

## Writing outputs

Save anything you want back under `$GPULAYER_OUTPUT_DIR` from one process:

```python
import os, torch
if int(os.environ.get("RANK", "0")) == 0:
    torch.save(model.state_dict(), os.path.join(os.environ["GPULAYER_OUTPUT_DIR"], "model.pt"))
```

With more than one node, each node's files come back under `node<rank>/`.

## Helpers

| Function | Returns |
|---|---|
| `glt.is_launcher_process()` | `True` in the process that submits work, `False` on workers |
| `glt.in_gpulayer_job()` | `True` inside a GPUlayer job |
