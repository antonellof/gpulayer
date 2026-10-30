# ADR-0017: Distribution: install script, Docker images, then packages

- **Status:** Accepted (milestone 2)
- **Date:** 2026-10-29

## Context

Installing GPUlayer meant cloning the repository and creating a virtual environment by hand.
That is fine for contributors but slows down the people GPUlayer is for: someone adding a GPU
server or a cloud VM as a worker wants one command. Servers and workers on Linux are often run
as containers. GPUlayer is not on PyPI, and its licence is still undecided (ADR-0012), which
blocks publishing packages and images for others to use.

## Decision

1. **Install script** (`install.sh`): POSIX `sh`, no sudo. It creates a private virtual
   environment (default `~/.local/share/gpulayer/venv`, with uv when available, otherwise
   `python3 -m venv`), installs GPUlayer from GitHub (`GPULAYER_REF`, `GPULAYER_EXTRAS`) and
   links `gpulayer` into `~/.local/bin`. Served as
   `curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh | sh`.
2. **Docker image** (`Dockerfile`): one image for the CLI, server and worker, built on
   `python:3.12-slim` by default, or on an NVIDIA CUDA base with PyTorch
   (`--build-arg BASE=... EXTRAS=pytorch-cuda`). It runs as an unprivileged user with state
   in `/data`. `deploy/docker/docker-compose.yml` runs a server and workers, with an NVIDIA
   worker profile.
3. **PyPI (done, 2026-10-30, after ADR-0012 chose Apache-2.0 and the repository went
   public):** `.github/workflows/publish.yml` builds the sdist and wheel and publishes them
   to PyPI with trusted publishing (OpenID Connect, no API token) when a GitHub release is
   published, from the `pypi` environment. `pip install gpulayer` and
   `uv tool install gpulayer` work.
4. **Later:**
   - publish images to GitHub Container Registry from CI (`ghcr.io/antonellof/gpulayer`,
     tags per release, `:cuda` variant), with SBOM and signatures;
   - switch the install script to the released package;
   - consider a Homebrew formula for macOS workers.

## Alternatives considered

- **PyPI first.** The most standard path, but it needs a licence and a release process, and
  it does not give operators a server image.
- **A single static binary** (PyInstaller, Nuitka). Easy to download, but jobs run in a
  Python environment with GPU frameworks anyway, so a frozen binary would still need one.
- **Run Mac workers in Docker.** Docker on macOS cannot reach the Apple GPU, so Mac workers
  stay native; the image targets Linux.

## Consequences and trade-offs

- The one-line installer only works for others once the repository is publicly readable.
  Until then it works from a clone (`GPULAYER_SOURCE=. sh install.sh`).
- Piping a script into `sh` asks users to trust it; the docs link to the script so it can be
  read first, and it never escalates privileges.
- The Docker image has not been build-tested in CI yet; CI is part of step 3.

## Security and operational implications

- Containers are packaging, not isolation for untrusted jobs (ADR-0009).
- The image runs as UID 10001 and keeps tokens, certificates and the database in `/data`;
  operators must protect that volume like the server's home directory.

## Follow-up

- Add CI that builds both images and runs the test suite in the CPU image.
- Revisit the CUDA base version when PyTorch drops support for it.
