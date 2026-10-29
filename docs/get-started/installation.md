# Installation

GpuLayer needs Python 3.10 or newer. PyTorch is optional for the core package and required for
everything in `gpulayer.pytorch`.

## From source

GpuLayer is not on PyPI yet.

```bash
git clone https://github.com/antonellof/gpulayer.git
cd gpulayer
uv venv && source .venv/bin/activate          # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e '.[dev,examples]'          # or: pip install -e '.[dev,examples]'
```

The `examples` extra installs what the examples need: PyTorch, NumPy, and MLX on Apple
Silicon. For the core alone, `uv pip install -e .` is enough; add the GPU frameworks your own
jobs use.

Jobs run in the Python of the worker that executes them. A framework a job asks for (for
example `frameworks={"numpy": ">=1.20"}`) must be installed there, or the job is rejected
with `framework 'numpy' not installed`. For local jobs, that Python is the one running your
script.

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
device    [0] mps Apple arm64 GPU (Metal) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

GPUs are detected through their driver or platform, so they show up with or without PyTorch.
The `torch` line, and the framework list in `gpulayer worker status --json`, show which GPU
frameworks are installed in the Python that runs jobs. Jobs that need a framework (for example
PyTorch with CUDA, or MLX) only go to workers that have it.

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
