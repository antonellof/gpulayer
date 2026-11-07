#!/usr/bin/env bash
# Validate GPUlayer on a real AMD machine. Run from an unpacked copy of the repository, on a host
# or in a rocm/pytorch container (needs /dev/kfd and /dev/dri), with a ROCm build of torch:
#
#   bash scripts/validate_rocm.sh 2>&1 | tee validate.log
#
# Records what AMD's tools print in rocm-output/, since AMD publishes no sample of their JSON;
# keep that folder with the results. Set TORCH_INDEX to install torch into a new venv, for
# example https://download.pytorch.org/whl/nightly/rocm7.2; without it the current Python is
# used. Sets nothing up outside this folder and .venv-validate.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
status=0
step() { printf '\n==== %s\n' "$*"; }
run() { "$@" || { echo "FAILED: $*"; status=1; }; }
keep() { # name, command...: save a tool's output, whether or not the tool exists
  name=$1; shift
  if command -v "$1" >/dev/null 2>&1; then "$@" >"rocm-output/$name" 2>&1 || true; fi
}

step "hardware and software"
uname -srm
grep PRETTY_NAME /etc/os-release
nproc && free -g | head -2
ls -l /dev/kfd /dev/dri 2>&1
id

step "what AMD's tools print"
mkdir -p rocm-output
keep amd-smi-list.json amd-smi list --json
keep amd-smi-list-e.txt amd-smi list -e
keep amd-smi-static.json amd-smi static --json
keep amd-smi-static.txt amd-smi static
keep amd-smi-topology.json amd-smi topology --json
keep amd-smi-metric.json amd-smi metric --json
keep rocm-smi-productname.json rocm-smi --showproductname --json
keep rocm-smi-meminfo.json rocm-smi --showmeminfo vram --json
keep rocminfo.txt rocminfo
ls -l rocm-output

if [ -n "${TORCH_INDEX:-}" ]; then
  step "install"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv -q --python 3.12 .venv-validate
  # shellcheck disable=SC1091
  . .venv-validate/bin/activate
  uv pip install -q --pre torch --index-url "$TORCH_INDEX"
  uv pip install -q -e '.[dev,examples]'
else
  python -m pip install -q -e '.[dev,examples]'
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
PY

step "gpulayer resources detect"
run gpulayer resources detect

step "test suite"
run pytest -q -p no:cacheprovider

step "AMD GPU checks"
run python scripts/validate_rocm.py

printf '\n==== finished, status %s\n' "$status"
exit "$status"
