#!/usr/bin/env bash
# Single-quoted commands below expand on the VM, not here.
# shellcheck disable=SC2016
# Run GPUlayer's validation and benchmarks on Google Cloud VMs. See benchmarks/README.md.
#
#   export GCP_PROJECT=my-project            # required
#   export GCP_ZONE=europe-west1-b           # optional; GPU creation also tries GPU_ZONES
#
#   benchmarks/cloud/gcp.sh gpu-up gl-gpu [nvidia-tesla-t4] [n1-standard-8]
#   benchmarks/cloud/gcp.sh cpu-up gl-cpu [e2-standard-8]
#   benchmarks/cloud/gcp.sh push gl-gpu                  # copy this checkout, install
#   benchmarks/cloud/gcp.sh validate gl-gpu              # scripts/validate_gpu.sh
#   benchmarks/cloud/gcp.sh compare gl-gpu               # GPUlayer vs PyTorch Lightning
#   benchmarks/cloud/gcp.sh cluster gl-gpu gl-cpu        # server on the first, workers on all
#   benchmarks/cloud/gcp.sh multi gl-gpu                 # benchmarks/multi_machine.py
#   benchmarks/cloud/gcp.sh fetch gl-gpu benchmarks/results/<date>-gcp/
#   benchmarks/cloud/gcp.sh down gl-gpu gl-cpu           # delete the VMs
#
# Many machines (no external IP on workers; they join at boot from metadata):
#   benchmarks/cloud/gcp.sh image gl-img                 # build an image with this checkout
#   benchmarks/cloud/gcp.sh server-up gl-server gl-img [n1-standard-4]
#   benchmarks/cloud/gcp.sh fleet-up gl-w 23 europe-west1-b gl-img gl-server [n1-standard-4]
#   benchmarks/cloud/gcp.sh scale gl-server              # benchmarks/scale_test.py
#   benchmarks/cloud/gcp.sh fleet-down gl-w              # delete every VM named gl-w*
#
# Every VM gets the label purpose=gpulayer-bench. Results stay on the VM until `fetch`;
# fetch before `down`. Timed runs (validate, compare) refuse to start while anything else
# uses the GPU. Join and client tokens stay on the server VM and are never printed.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
ZONE=${GCP_ZONE:-europe-west1-b}
GPU_ZONES=${GPU_ZONES:-"$ZONE europe-west1-c europe-west1-d europe-west4-a europe-west4-b europe-west4-c us-central1-a us-central1-b"}
IMAGE_GPU=(--image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release)
IMAGE_CPU=(--image-family ubuntu-2204-lts --image-project ubuntu-os-cloud)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=(--project "$GCP_PROJECT")

zone_of() { gcloud compute instances list "${G[@]}" --filter="name=$1" --format="value(zone.basename())"; }
on() { local vm=$1; shift; gcloud compute ssh "$vm" "${G[@]}" --zone "$(zone_of "$vm")" --quiet --command "$*"; }
ip_of() { gcloud compute instances describe "$1" "${G[@]}" --zone "$(zone_of "$1")" --format="value(networkInterfaces[0].networkIP)"; }

create() {  # create NAME ZONE MACHINE SPOT|GPU|STANDARD EXTRA...
  local name=$1 zone=$2 mt=$3 model=$4; shift 4
  local prov=()
  [ "$model" = GPU ] && prov=(--maintenance-policy TERMINATE)  # GPU VMs cannot live-migrate
  [ "$model" = SPOT ] && prov=(--provisioning-model SPOT --instance-termination-action DELETE)
  gcloud compute instances create "$name" "${G[@]}" --zone "$zone" --machine-type "$mt" \
    "${prov[@]}" --boot-disk-size "${DISK_SIZE:-100GB}" --boot-disk-type "${DISK_TYPE:-pd-balanced}" \
    --labels purpose=gpulayer-bench "$@" --format="value(name,zone.basename(),status)"
}

wait_ssh() { for _ in $(seq 1 30); do on "$1" true 2>/dev/null && return; sleep 6; done; echo "no ssh to $1" >&2; return 1; }

cmd=${1:-}; shift || true
case "$cmd" in
  gpu-up)
    name=$1 accel=${2:-nvidia-tesla-t4} mt=${3:-n1-standard-8}
    for z in $GPU_ZONES; do
      for model in SPOT GPU; do
        if create "$name" "$z" "$mt" "$model" --accelerator "type=$accel,count=1" \
          "${IMAGE_GPU[@]}" --metadata install-nvidia-driver=True 2>/tmp/gl-create.err; then
          wait_ssh "$name"
          for _ in $(seq 1 40); do on "$name" nvidia-smi -L 2>/dev/null && exit 0; sleep 10; done
          echo "driver not ready on $name" >&2; exit 1
        fi
        echo "$z $model: $(grep -o -m1 'does not have enough resources\|Quota.*' /tmp/gl-create.err || tail -1 /tmp/gl-create.err)" >&2
      done
    done
    echo "no $accel capacity in: $GPU_ZONES" >&2; exit 1 ;;
  cpu-up)
    create "$1" "$ZONE" "${2:-e2-standard-8}" STANDARD "${IMAGE_CPU[@]}"
    wait_ssh "$1" ;;
  push)
    vm=$1 tgz=$(mktemp -t gpulayer-src).tgz
    (cd "$ROOT" && git ls-files | grep -v '^site/\|^docs/assets/dashboard/' ; echo site/src/lib/version.ts) \
      | (cd "$ROOT" && COPYFILE_DISABLE=1 tar czf "$tgz" -T -)
    gcloud compute scp "$tgz" "$vm:gpulayer-src.tgz" "${G[@]}" --zone "$(zone_of "$vm")" --quiet
    on "$vm" 'set -e; rm -rf gpulayer; mkdir gpulayer; tar xzf gpulayer-src.tgz -C gpulayer 2>/dev/null
      command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
      export PATH=$HOME/.local/bin:$PATH; cd gpulayer; uv venv -q --python 3.12 .venv-validate
      if ! command -v nvidia-smi >/dev/null; then
        uv pip install -q --python .venv-validate/bin/python torch --index-url https://download.pytorch.org/whl/cpu
      fi
      uv pip install -q --python .venv-validate/bin/python -e ".[dev,examples]"
      .venv-validate/bin/gpulayer resources detect | tail -3' ;;
  validate)
    on "$1" 'cd gpulayer && n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l);
      [ "$n" = 0 ] || { echo "GPU busy; not starting a timed run"; exit 1; };
      bash scripts/validate_gpu.sh > validate_gpu.log 2>&1; tail -3 validate_gpu.log' ;;
  compare)
    on "$1" 'cd gpulayer && n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l);
      [ "$n" = 0 ] || { echo "GPU busy; not starting a timed run"; exit 1; };
      export PATH=$HOME/.local/bin:$PATH; uv pip install -q --python .venv-validate/bin/python "lightning==2.6.6";
      cd benchmarks/lightning_comparison && ../../.venv-validate/bin/python run.py > ../../lightning-comparison.log 2>&1;
      cat ../../lightning-comparison.log' ;;
  cluster)
    server=$1; shift
    sip=$(ip_of "$server")
    on "$server" "pkill -f '[b]in/gpulayer up' || true"
    on "$server" "mkdir -p glhome && GPULAYER_HOME=\$HOME/glhome setsid nohup gpulayer/.venv-validate/bin/gpulayer up \
      --host $sip --advertise-address $sip --name $server > up.log 2>&1 < /dev/null &
      for i in \$(seq 1 30); do grep -q 'gpulayer join' up.log && break; sleep 1; done
      fp=\$(grep -o -m1 'sha256:[0-9a-f]*' up.log)
      python3 gpulayer/benchmarks/cloud/write_env.py glhome https://$sip:8470 \$fp > env.sh
      chmod 600 env.sh; echo server up at https://$sip:8470"
    for w in "$@"; do
      [ "$w" = "$server" ] && continue
      wip=$(ip_of "$w")
      # The join token goes server -> worker over ssh stdin; it is never printed or stored here.
      on "$server" 'cat env.sh' | on "$w" "cat > env.sh; chmod 600 env.sh; . ./env.sh; pkill -f '[b]in/gpulayer join' || true
        mkdir -p glhome && GPULAYER_HOME=\$HOME/glhome setsid nohup gpulayer/.venv-validate/bin/gpulayer join \
        \$GPULAYER_URL --token \$GPULAYER_JOIN_TOKEN --fingerprint \$GPULAYER_FINGERPRINT --name $w \
        --advertise-address $wip --concurrency \$(nproc) > join.log 2>&1 < /dev/null & sleep 4; head -1 join.log"
    done ;;
  multi)
    on "$1" 'cd gpulayer && . ../env.sh && .venv-validate/bin/python benchmarks/multi_machine.py > multi_machine.log 2>&1; grep -E "^(WORKER|CHECK|RESULT|SUMMARY)" multi_machine.log' ;;
  fetch)
    vm=$1 dest=$2; mkdir -p "$dest"
    for f in validate_gpu.log lightning-comparison.log multi_machine.log benchmarks/multi_machine_results.json; do
      gcloud compute scp "$vm:gpulayer/$f" "$dest/" "${G[@]}" --zone "$(zone_of "$vm")" --quiet 2>/dev/null || true
    done
    on "$vm" 'nvidia-smi 2>/dev/null; true' > "$dest/nvidia-smi.txt" || true
    on "$vm" 'cd gpulayer && $HOME/.local/bin/uv pip freeze --python .venv-validate/bin/python' > "$dest/pip-freeze.txt" || true
    # Remove the VM user's home path from the saved files.
    for f in "$dest"/*; do [ -f "$f" ] && sed -i.bak "s#/home/[A-Za-z0-9_.-]*#/home/user#g" "$f" && rm -f "$f.bak"; done
    ls -la "$dest" ;;
  down)
    for vm in "$@"; do gcloud compute instances delete "$vm" "${G[@]}" --zone "$(zone_of "$vm")" --quiet; done
    gcloud compute instances list "${G[@]}" --filter="labels.purpose=gpulayer-bench" --format="value(name,status)" ;;
  image)
    img=$1 b="$1-builder" tgz=$(mktemp -t gpulayer-src).tgz
    DISK_SIZE=20GB DISK_TYPE=pd-standard create "$b" "$ZONE" e2-standard-4 STANDARD "${IMAGE_CPU[@]}" >/dev/null
    gcloud compute instances set-disk-auto-delete "$b" "${G[@]}" --zone "$ZONE" --auto-delete >/dev/null 2>&1 || true
    wait_ssh "$b"
    (cd "$ROOT" && git ls-files | grep -v '^site/\|^docs/assets/dashboard/' ; echo site/src/lib/version.ts) \
      | (cd "$ROOT" && COPYFILE_DISABLE=1 tar czf "$tgz" -T -)
    gcloud compute scp "$tgz" "$b:/tmp/gpulayer-src.tgz" "${G[@]}" --zone "$ZONE" --quiet
    on "$b" 'set -e; sudo mkdir -p /opt/gpulayer; sudo tar xzf /tmp/gpulayer-src.tgz -C /opt/gpulayer 2>/dev/null
      curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null 2>&1
      cd /opt/gpulayer; export UV_PYTHON_INSTALL_DIR=/opt/uv-python
      sudo -E uv venv -q --python 3.12 .venv
      sudo -E uv pip install -q --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
      sudo -E uv pip install -q --python .venv/bin/python -e ".[dev,examples]" "lightning==2.6.6"
      sudo ln -sf /opt/gpulayer/.venv/bin/gpulayer /usr/local/bin/gpulayer; gpulayer --version
      sudo chmod -R a+rwX /opt/gpulayer   # benchmark runs write their outputs next to the code'
    gcloud compute instances stop "$b" "${G[@]}" --zone "$ZONE" --quiet
    gcloud compute images create "$img" "${G[@]}" --source-disk "$b" --source-disk-zone "$ZONE" \
      --labels purpose=gpulayer-bench
    gcloud compute instances delete "$b" "${G[@]}" --zone "$ZONE" --quiet ;;
  server-up)
    name=$1 img=$2 mt=${3:-n1-standard-4}
    gcloud compute instances create "$name" "${G[@]}" --zone "$ZONE" --machine-type "$mt" \
      --image "$img" --boot-disk-type pd-standard --labels purpose=gpulayer-bench \
      --format="value(name,status)"
    wait_ssh "$name"
    sip=$(ip_of "$name")
    on "$name" "mkdir -p glhome; GPULAYER_HOME=\$HOME/glhome setsid nohup gpulayer up --no-worker \
      --host $sip --advertise-address $sip > up.log 2>&1 < /dev/null &
      for i in \$(seq 1 30); do grep -q 'gpulayer join' up.log && break; sleep 1; done
      fp=\$(grep -o -m1 'sha256:[0-9a-f]*' up.log)
      /opt/gpulayer/.venv/bin/python /opt/gpulayer/benchmarks/cloud/write_env.py glhome https://$sip:8470 \$fp > env.sh
      chmod 600 env.sh; mkdir -p .ssh; [ -f .ssh/fleet ] || ssh-keygen -q -t ed25519 -N '' -f .ssh/fleet -C gl@$name
      echo server up at https://$sip:8470" ;;
  fleet-up)
    prefix=$1 count=$2 zone=$3 img=$4 server=$5 mt=${6:-n1-standard-4}
    # Read the server's connection settings without printing them.
    env_sh=$(on "$server" 'cat env.sh')
    pub=$(on "$server" 'cat .ssh/fleet.pub')
    url=$(sed -n 's/^export GPULAYER_URL=//p' <<<"$env_sh")
    fp=$(sed -n 's/^export GPULAYER_FINGERPRINT=//p' <<<"$env_sh")
    jt=$(sed -n 's/^export GPULAYER_JOIN_TOKEN=//p' <<<"$env_sh")
    first=$(gcloud compute instances list "${G[@]}" --filter="name~^$prefix" --format="value(name)" | wc -l)
    names=(); for i in $(seq $((first + 1)) $((first + count))); do names+=("$(printf '%s%02d' "$prefix" "$i")"); done
    gcloud compute instances create "${names[@]}" "${G[@]}" --zone "$zone" --machine-type "$mt" \
      --image "$img" --boot-disk-type pd-standard --no-address --labels purpose=gpulayer-bench \
      --metadata "gl-url=$url,gl-fingerprint=$fp,gl-join-token=$jt,ssh-keys=gl:$pub" \
      --metadata-from-file "startup-script=$ROOT/benchmarks/cloud/worker-startup.sh" \
      --format="value(name,zone.basename(),status)" ;;
  scale)
    on "$1" 'cd /opt/gpulayer && . ~/env.sh && .venv/bin/python benchmarks/scale_test.py > ~/scale_test.log 2>&1; grep -E "^(WORKERS|CHECK|RESULT|SUMMARY)" ~/scale_test.log' ;;
  fleet-down)
    gcloud compute instances list "${G[@]}" --filter="name~^$1" --format="value(name,zone.basename())" |
      while read -r vm z; do echo "$vm $z"; done | awk '{z[$2]=z[$2]" "$1} END{for (k in z) print k z[k]}' |
      while read -r z vms; do
        # shellcheck disable=SC2086
        gcloud compute instances delete $vms "${G[@]}" --zone "$z" --quiet
      done ;;
  *)
    sed -n '4,28p' "$0"; exit 1 ;;
esac
