# Installation

GPUlayer needs Python 3.10 or newer. PyTorch is optional for the core package and required for
everything in `gpulayer.pytorch`.

## From PyPI

```bash
pip install gpulayer                # the CLI and SDK
pip install 'gpulayer[torch]'       # with PyTorch, for gpulayer.pytorch and distributed jobs
pip install 'gpulayer[examples]'    # what the examples need: PyTorch, NumPy, MLX on Apple Silicon
uv tool install gpulayer            # the CLI in its own environment, with uv
```

Install it in the Python environment your jobs use: a worker runs `python` jobs with its own
interpreter, so that environment needs the frameworks your jobs import.

## With the install script

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh | sh
```

The script installs the CLI into its own virtual environment (default
`~/.local/share/gpulayer/venv`) and links `gpulayer` into `~/.local/bin`. It uses
[uv](https://docs.astral.sh/uv/) when it is installed, otherwise Python 3.10 or newer and git.
It never uses sudo. Read it first if you prefer: [`install.sh`](../../install.sh).

| Variable | Default | Meaning |
|---|---|---|
| `GPULAYER_REF` | `main` | branch, tag or commit to install |
| `GPULAYER_EXTRAS` | none | extras, for example `examples` (PyTorch, NumPy, MLX) |
| `GPULAYER_INSTALL_DIR` | `~/.local/share/gpulayer` | where the virtual environment goes |
| `GPULAYER_BIN_DIR` | `~/.local/bin` | where the `gpulayer` link goes |
| `GPULAYER_SOURCE` | GitHub | a local checkout or pip URL to install instead |

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh | GPULAYER_EXTRAS=examples sh
```

The one-line form needs the repository to be publicly readable. Until then, clone the
repository and run `GPULAYER_SOURCE=. sh install.sh`.

To uninstall, delete the install directory and the link:
`rm -rf ~/.local/share/gpulayer ~/.local/bin/gpulayer`. Job state in `~/.gpulayer` is separate.

## With Docker

The [`Dockerfile`](../../Dockerfile) builds an image with the CLI. It can run the server, a
worker, or any other command:

```bash
docker build -t gpulayer .                                    # CPU image (python:3.12-slim)
docker run -d --name gpulayer -p 8470:8470 -v gpulayer-data:/data gpulayer up --no-worker
docker logs gpulayer                                          # tokens and the join command
```

For NVIDIA workers, build on a CUDA base with PyTorch and run with the NVIDIA Container Toolkit:

```bash
docker build -t gpulayer:cuda \
  --build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 --build-arg EXTRAS=pytorch-cuda .
docker run -d --gpus all -v gpulayer-worker:/data gpulayer:cuda \
  join https://<server>:8470 --token <join token> --fingerprint <fingerprint>
```

[`deploy/docker/docker-compose.yml`](../../deploy/docker/docker-compose.yml) runs a server and a
worker together, with an optional NVIDIA worker profile. Images run as an unprivileged user
and keep their state in the `/data` volume.

Notes:

- A container is a packaging choice, not a sandbox for untrusted jobs; see
  [Security](../architecture/security.md).
- Apple GPUs are not available inside Docker on macOS. Run Mac workers natively.
- Distributed jobs need the workers to reach each other on the rendezvous port; use host
  networking or a shared Docker network, and set `--advertise-address`.
- Images are built locally for now. Published images (GitHub Container Registry) are planned.

## From source

For development on GPUlayer itself, install an editable checkout:

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

Without a remote control plane, GPUlayer keeps its state in `$GPULAYER_HOME`
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
