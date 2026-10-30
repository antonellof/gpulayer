#!/bin/sh
# Check a GPUlayer install on Kubernetes: workers joined, a job runs on a worker pod, and a
# distributed job spans two worker pods.
#
#   sh deploy/kubernetes/smoke-test.sh NAMESPACE RELEASE
set -eu
ns=${1:-gpulayer}
release=${2:-gpulayer}
server="${release}-gpulayer-server-0"
run() { kubectl -n "$ns" exec "$server" -- "$@"; }

echo "== waiting for 2 workers"
i=0
until [ "$(run gpulayer resources list 2>/dev/null | grep -c ' idle ')" -ge 2 ]; do
  i=$((i + 1)); [ "$i" -gt 60 ] && { run gpulayer resources list; exit 1; }
  sleep 2
done
run gpulayer resources list

echo "== a job on a worker pod"
run gpulayer submit --target remote -- python -c "import socket; print('hello from', socket.gethostname())"

echo "== a distributed job across two worker pods"
run sh -c 'mkdir -p /tmp/d && cat > /tmp/d/d.py <<PY
import socket, torch, torch.distributed as dist
dist.init_process_group("gloo")
r, w = dist.get_rank(), dist.get_world_size()
t = torch.tensor([float(r + 1)]); dist.all_reduce(t)
print(f"rank={r} world={w} sum={t.item()} host={socket.gethostname()}", flush=True)
assert t.item() == w * (w + 1) / 2
dist.destroy_process_group()
PY
gpulayer submit --target remote --distributed --source /tmp/d --entrypoint d.py --nodes 2 --procs-per-node 1'
echo "== ok"
