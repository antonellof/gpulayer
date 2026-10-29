# Installation

GpuLayer needs Python 3.10 or newer. PyTorch is optional for the core package and required for
everything in `gpulayer.pytorch`.

## From source

GpuLayer is not on PyPI yet.

```bash
git clone https://github.com/antonellof/gpulayer.git
cd gpulayer
uv venv && source .venv/bin/activate          # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'                    # or: pip install -e '.[dev]'
uv pip install torch                          # CPU, CUDA or MPS build for your platform
```

## Check the installation

```bash
gpulayer --version
gpulayer resources detect
```

`resources detect` prints what this machine can offer to jobs. On an Apple Silicon Mac it looks
like this:

```text
host      my-mac (darwin/arm64), 10 cores, 32.0 GB RAM, python 3.11.5
torch     2.15.0  distributed backends: gloo
device    [0] mps Apple arm64 GPU (MPS) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

On a machine without torch the `torch` line says `not installed` and only CPU jobs can run
there. That is fine for the core platform; GPU discovery uses torch in the interpreter that runs
your jobs.

## Where state is kept

Without a remote control plane, GpuLayer keeps its state in `$GPULAYER_HOME`
(default `~/.gpulayer`, created with mode 0700):

| Path | Contents |
|---|---|
| `metadata.db` | jobs, tasks, workers, events and logs (SQLite) |
| `artifacts/` | source bundles, inputs and outputs, addressed by SHA-256 |
| `workers/<id>/` | task workspaces and the artifact cache of local workers |

Delete the directory to start from scratch.

## Using a different Python for jobs

Jobs run with the interpreter of the worker. To run them in another environment, for example a
conda env with a CUDA build of torch:

```bash
gpulayer worker start --python /opt/conda/envs/torch/bin/python
export GPULAYER_WORKER_PYTHON=/opt/conda/envs/torch/bin/python   # for in-process workers
```
