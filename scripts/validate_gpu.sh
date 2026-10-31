#!/usr/bin/env bash
# Validate GPUlayer on a real NVIDIA machine. Run from an unpacked copy of the repository:
#
#   bash scripts/validate_gpu.sh 2>&1 | tee validate.log
#
# Needs an NVIDIA driver (nvidia-smi) and network access to PyPI. Installs uv and a venv in
# .venv-validate, then runs: hardware details, the full test suite, the CUDA hardware tests,
# scripts/validate_cuda.py, scripts/validate_multigpu.py (with 2 or more GPUs), and the benchmarks.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
status=0
step() { printf '\n==== %s\n' "$*"; }
run() { "$@" || { echo "FAILED: $*"; status=1; }; }

step "hardware and software"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi topo -m || true
uname -srm
grep PRETTY_NAME /etc/os-release
nproc && free -g | head -2

step "install"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --python 3.12 .venv-validate
# shellcheck disable=SC1091
. .venv-validate/bin/activate
uv pip install -q -e '.[dev,examples]'
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
gpulayer --version || true

step "gpulayer resources detect"
run gpulayer resources detect

step "test suite"
run pytest -q -p no:cacheprovider

step "CUDA hardware tests"
run pytest -v -p no:cacheprovider -m hardware_cuda tests/hardware

step "CUDA checks"
run python scripts/validate_cuda.py

if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  step "several GPUs: interconnect, GPU choice, NCCL"
  run python scripts/validate_multigpu.py
fi

step "benchmarks"
run python benchmarks/dispatch.py
run python benchmarks/distributed_startup.py --repeat 3

step "done"
[ "$status" = 0 ] && echo "ALL STEPS PASSED" || echo "SOME STEPS FAILED"
exit "$status"
