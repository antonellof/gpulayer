# Run any script with launch()

`launch()` sends an existing training script to GPUlayer workers. Add two lines at the top:

```python
import gpulayer.pytorch as glt
glt.launch(nodes=1, devices=2, accelerator="cpu")

# ... your training script, unchanged ...
```

Run it as usual (`python train.py --lr 0.01`). `launch()` uploads the script's directory, runs
it with the same arguments on `nodes x devices` processes, streams the output, downloads
`$GPULAYER_OUTPUT_DIR` into `gpulayer_runs/<job-id>/` and exits with the job's exit code. On
the workers it returns at once, and the script runs with the torchrun environment (`RANK`,
`WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`).

`launch(inputs={"data/train.bin": "/local/train.bin"})` uploads files to `$GPULAYER_INPUT_DIR`
on every node.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `nodes` | `1` | number of workers |
| `devices` | `1` | processes per worker |
| `accelerator` | `"cpu"` | `"cpu"`, `"cuda"` or `"mps"` (no `"auto"`) |
| `target` | `"local"` | `"local"`, `"remote"` or `"cloud"` |
| `frameworks` | `{}` | e.g. `{"torch": ">=2.2"}` |
| `env` | `{}` | environment variables |
| `timeout_seconds` | `None` | kill the job after this long |
| `output_dir` | `"gpulayer_runs"` | where outputs are downloaded |
| `exit` | `True` | `False` returns a `LaunchResult` (`None` on workers) |
| `quiet` | `False` | do not stream logs |
| `client` | default client | a `gpulayer.Client` |
| other keywords | | passed to `PlacementPolicy`, e.g. `worker_labels={"pool": "a100"}` |

## Writing outputs

Save files under `$GPULAYER_OUTPUT_DIR` from one process. With several nodes, each node's files
come back under `node<rank>/`.

```python
import os, torch
if int(os.environ.get("RANK", "0")) == 0:
    torch.save(model.state_dict(), os.path.join(os.environ["GPULAYER_OUTPUT_DIR"], "model.pt"))
```

## Helpers

`glt.is_launcher_process()` is `True` in the process that submits work. `glt.in_gpulayer_job()`
is `True` inside a GPUlayer job.
