#!/usr/bin/env bash
# Render frames with Blender Cycles on the job's GPU. Blender must be installed on the worker
# (on PATH, or BLENDER=/path/to/blender in the job's environment).
#
#   render.sh [OPTIX|CUDA|CPU] [FRAMES] [SAMPLES] [FIRST_FRAME]
set -euo pipefail
blender=${BLENDER:-blender}
command -v "$blender" >/dev/null || { echo "blender not found on $(hostname)" >&2; exit 127; }
cd "$(dirname "$0")"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
exec "$blender" -b --factory-startup -P render.py -- "${1:-OPTIX}" "${2:-4}" "${3:-64}" "${4:-1}"
