#!/usr/bin/env bash
# Run scripts/validate_multigpu.py on Kaggle's free "GPU T4 x2" (two T4s over PCIe).
#
#   benchmarks/cloud/kaggle.sh run [DEST]     # push, wait, download the log (default DEST:
#                                             # benchmarks/results/<date>-kaggle-t4x2/)
#
# Needs the kaggle CLI (`uv tool install kaggle`), an API token in ~/.kaggle/kaggle.json, and a
# phone-verified account (GPUs and internet). The kernel is private and named gpulayer-multigpu.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${2:-$ROOT/benchmarks/results/$(date +%F)-kaggle-t4x2}
user=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
work=$(mktemp -d)
cp "$ROOT/benchmarks/cloud/kaggle/validate.py" "$work/"
cat > "$work/kernel-metadata.json" <<META
{"id": "$user/gpulayer-multigpu", "title": "gpulayer-multigpu", "code_file": "validate.py",
 "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
 "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
META
case "${1:-}" in
  run)
    kaggle kernels push -p "$work" --accelerator NvidiaTeslaT4
    for _ in $(seq 1 90); do
      s=$(kaggle kernels status "$user/gpulayer-multigpu" 2>&1 | tail -1)
      echo "$s"
      case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
      sleep 20
    done
    mkdir -p "$DEST"
    kaggle kernels output "$user/gpulayer-multigpu" -p "$DEST"
    grep -E "^CHECK|SUMMARY" "$DEST"/*.log || tail -40 "$DEST"/*.log ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
