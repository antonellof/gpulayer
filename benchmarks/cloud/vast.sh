#!/usr/bin/env bash
# Run GPUlayer's GPU validation on Vast.ai (rented GPU machines; needs the `vastai` CLI, logged
# in, with credit, and an SSH key registered on the account).
#
#   benchmarks/cloud/vast.sh offers [2]                # multi-GPU offers, cheapest first
#   benchmarks/cloud/vast.sh offers 2 A100_SXM4        # a given GPU model (SXM = NVLink)
#   benchmarks/cloud/vast.sh up gl-gpu OFFER_ID        # rent that offer
#   benchmarks/cloud/vast.sh push gl-gpu               # copy this checkout
#   benchmarks/cloud/vast.sh validate gl-gpu           # scripts/validate_gpu.sh (runs
#                                                      # validate_multigpu.py with 2+ GPUs)
#   benchmarks/cloud/vast.sh fetch gl-gpu benchmarks/results/<date>-vast/
#   benchmarks/cloud/vast.sh down gl-gpu               # destroy the instance (billing stops)
#
# Instances run the nvidia/cuda Ubuntu image; validate_gpu.sh installs Python and PyTorch with
# uv. PyTorch's default wheels need a host driver for CUDA 13 and a Turing or newer GPU
# (compute capability 7.5+), so `offers` filters on both.
# Vast.ai containers sit behind NAT with few open ports: NCCL across two instances is not
# expected to work here; validate it across machines on AWS or Google Cloud instead.
set -euo pipefail

KEY=${GL_VAST_KEY:-$HOME/.ssh/id_rsa}
IMAGE=${GL_VAST_IMAGE:-nvidia/cuda:13.0.1-runtime-ubuntu22.04}

id_of() { vastai show instances --raw | python3 -c "
import json,sys
ids=[str(i['id']) for i in json.load(sys.stdin) if i.get('label')=='$1']
print(ids[0] if ids else '')"; }
ssh_target() { vastai ssh-url "$(id_of "$1")" | sed -E 's#ssh://##'; }
on() {
  local vm=$1 t; shift
  t=$(ssh_target "$vm")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p "${t##*:}" "${t%:*}" "$@"
}

cmd=${1:-}; shift || true
case "$cmd" in
  offers)
    n=${1:-2} gpu=${2:-}
    q="num_gpus>=$n reliability>0.97 cuda_vers>=13.0 compute_cap>=750 rentable=true verified=true direct_port_count>=1"
    [ -n "$gpu" ] && q="$q gpu_name=$gpu"
    vastai search offers "$q" -o dph --raw | python3 -c "
import json,sys
for o in json.load(sys.stdin)[:15]:
    print(f\"{o['id']:>10}  {o['num_gpus']}x {o['gpu_name']:<16} \${o['dph_total']:.2f}/h  \"
          f\"cuda {o.get('cuda_max_good')}  cpu {o.get('cpu_name','')[:28]:<28} {o.get('geolocation','')}\")" ;;
  up)
    name=$1 offer=$2
    vastai create instance "$offer" --image "$IMAGE" --disk 60 --ssh --direct --label "$name" --raw \
      --onstart-cmd 'apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null' |
      python3 -c "import json,sys; print('instance', json.load(sys.stdin).get('new_contract'))"  # no API key
    for _ in $(seq 1 60); do
      state=$(vastai show instances --raw | python3 -c "
import json,sys
print(next((i.get('actual_status') or '' for i in json.load(sys.stdin) if i.get('label')=='$name'), ''))")
      [ "$state" = running ] && break; sleep 10
    done
    for _ in $(seq 1 30); do on "$name" 'command -v curl >/dev/null && nvidia-smi -L' 2>/dev/null && exit 0; sleep 10; done
    echo "instance $name not reachable over ssh" >&2; exit 1 ;;
  push)
    vm=$1 tgz=$(mktemp -t gpulayer-src).tgz
    root=$(cd "$(dirname "$0")/../.." && pwd)
    (cd "$root" && git ls-files | grep -v '^site/\|^docs/assets/dashboard/' ; echo site/src/lib/version.ts) \
      | (cd "$root" && COPYFILE_DISABLE=1 tar czf "$tgz" -T -)
    t=$(ssh_target "$vm")
    scp -i "$KEY" -o StrictHostKeyChecking=accept-new -P "${t##*:}" "$tgz" "${t%:*}:gpulayer-src.tgz"
    on "$vm" 'rm -rf gpulayer && mkdir gpulayer && tar xzf gpulayer-src.tgz -C gpulayer 2>/dev/null; ls gpulayer | head -3' ;;
  validate)
    on "$1" 'cd gpulayer && n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l);
      [ "$n" = 0 ] || { echo "GPU busy; not starting a timed run"; exit 1; };
      bash scripts/validate_gpu.sh > validate_gpu.log 2>&1; grep -E "^CHECK|SUMMARY|ALL STEPS|FAILED" validate_gpu.log | tail -30' ;;
  fetch)
    vm=$1 dest=$2; mkdir -p "$dest"
    t=$(ssh_target "$vm")
    scp -i "$KEY" -P "${t##*:}" "${t%:*}:gpulayer/validate_gpu.log" "$dest/" || true
    on "$vm" 'nvidia-smi; nvidia-smi topo -m; lscpu | head -20' > "$dest/nvidia-smi.txt" || true
    on "$vm" 'cd gpulayer && $HOME/.local/bin/uv pip freeze --python .venv-validate/bin/python' > "$dest/pip-freeze.txt" || true
    ls -la "$dest" ;;
  down)
    for vm in "$@"; do id=$(id_of "$vm"); [ -n "$id" ] && vastai destroy instance "$id" -y; done ;;
  *)
    sed -n '2,18p' "$0"; exit 1 ;;
esac
