# Installation

GPUlayer needs Python 3.10 or newer. `gpulayer.pytorch` also needs PyTorch.

## From PyPI

```bash
pip install gpulayer                # the CLI and SDK
pip install 'gpulayer[torch]'       # with PyTorch, for gpulayer.pytorch and distributed jobs
pip install 'gpulayer[examples]'    # what the examples need: PyTorch, NumPy, MLX on Apple Silicon
uv tool install gpulayer            # the CLI in its own environment, with uv
```

Jobs run in the worker's Python, so install GPUlayer where your jobs' frameworks are. A job
that needs a missing framework is rejected, for example with `framework 'numpy' not installed`.

## With the install script

```bash
curl -fsSL https://gpulayer.dev/install.sh | sh
```

It installs the CLI into its own virtual environment and links `gpulayer` into `~/.local/bin`,
using [uv](https://docs.astral.sh/uv/) if present, otherwise Python 3.10+ and git. It never
uses sudo ([source](../../install.sh)).

| Variable | Default | Meaning |
|---|---|---|
| `GPULAYER_REF` | `main` | branch, tag or commit to install |
| `GPULAYER_EXTRAS` | none | extras, for example `examples` (PyTorch, NumPy, MLX) |
| `GPULAYER_INSTALL_DIR` | `~/.local/share/gpulayer` | where the virtual environment goes |
| `GPULAYER_BIN_DIR` | `~/.local/bin` | where the `gpulayer` link goes |
| `GPULAYER_SOURCE` | GitHub | a local checkout or pip URL to install instead |

```bash
curl -fsSL https://gpulayer.dev/install.sh | GPULAYER_EXTRAS=examples sh
```

The one-line form needs a publicly readable repository. Until then, clone it and run
`GPULAYER_SOURCE=. sh install.sh`. To uninstall:
`rm -rf ~/.local/share/gpulayer ~/.local/bin/gpulayer` (job state in `~/.gpulayer` stays).

## With Docker

```bash
docker build -t gpulayer .                                    # CPU image (python:3.12-slim)
docker run -d --name gpulayer -p 8470:8470 -v gpulayer-data:/data gpulayer up --no-worker
docker logs gpulayer                                          # tokens and the join command
```

For NVIDIA workers, build on a CUDA base and run with the NVIDIA Container Toolkit:

```bash
docker build -t gpulayer:cuda \
  --build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 --build-arg EXTRAS=pytorch-cuda .
docker run -d --gpus all -v gpulayer-worker:/data gpulayer:cuda \
  join https://<server>:8470 --token <join token> --fingerprint <fingerprint>
```

- [`deploy/docker/docker-compose.yml`](../../deploy/docker/docker-compose.yml) runs a server and
  a worker, with an optional NVIDIA profile.
- Releases publish `ghcr.io/antonellof/gpulayer:<version>` and `:<version>-torch` (CPU
  PyTorch). Build CUDA images yourself.
- A container is not a sandbox for untrusted jobs ([Security](../architecture/security.md)).
- Apple GPUs are not available in Docker on macOS; run Mac workers natively.
- Distributed jobs need host networking or a shared network, and `--advertise-address`.

## On Kubernetes

```bash
git clone https://github.com/antonellof/gpulayer && cd gpulayer
helm install gpulayer deploy/kubernetes/helm/gpulayer --namespace gpulayer --create-namespace
```

See [Kubernetes](../guides/kubernetes.md).

## From source

```bash
git clone https://github.com/antonellof/gpulayer.git
cd gpulayer
uv venv && source .venv/bin/activate          # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e '.[dev,examples]'          # or: pip install -e '.[dev,examples]'
```

## Check the installation

```bash
gpulayer --version
gpulayer resources detect
```

On an Apple Silicon Mac:

```text
host      my-mac (darwin/arm64), 10 cores, 32.0 GB RAM, python 3.11.5
torch     2.15.0  distributed backends: gloo
device    [0] mps Apple arm64 GPU (Metal) (32.0 GB unified) independent=hardware-validated distributed=unsupported
```

GPUs are detected with or without PyTorch. The `torch` line shows the frameworks in the Python
that runs jobs; jobs that need a framework only go to workers that have it.

## Where state is kept

Without a remote control plane, state lives in `$GPULAYER_HOME` (default `~/.gpulayer`, mode
0700). Delete it to start from scratch.

| Path | Contents |
|---|---|
| `metadata.db` | jobs, tasks, workers, events and logs (SQLite) |
| `artifacts/` | source bundles, inputs and outputs, addressed by SHA-256 |
| `workers/<id>/` | task workspaces and the artifact cache of local workers |

## Using a different Python for jobs

```bash
gpulayer worker start --python /opt/conda/envs/torch/bin/python
export GPULAYER_WORKER_PYTHON=/opt/conda/envs/torch/bin/python   # for in-process workers
```
