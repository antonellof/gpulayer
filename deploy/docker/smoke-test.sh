#!/bin/sh
# End-to-end check of the Docker image: a server container, a worker container that joins it
# over TLS with a pinned certificate, a job submitted from a third container, and /metrics.
#
#   sh deploy/docker/smoke-test.sh [image]        (default image: gpulayer:latest)
set -eu
IMAGE="${1:-gpulayer:latest}"
NET="gpulayer-smoke-$$"
cleanup() {
  docker rm -f "gl-server-$$" "gl-worker-$$" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT
fail() { echo "smoke-test: FAIL: $*" >&2; docker logs "gl-server-$$" >&2 2>&1 || true; exit 1; }

docker run --rm "$IMAGE" --version
docker network create "$NET" >/dev/null
docker run -d --name "gl-server-$$" --network "$NET" --network-alias gl-server "$IMAGE" \
  up --no-worker --advertise-address gl-server >/dev/null

i=0
until docker logs "gl-server-$$" 2>&1 | grep -q "GpuLayer is up"; do
  i=$((i + 1)); [ "$i" -lt 60 ] || fail "server did not start"; sleep 1
done
LOGS="$(docker logs "gl-server-$$" 2>&1)"
JOIN="$(printf '%s\n' "$LOGS" | grep -o 'glj_[A-Za-z0-9_-]*' | head -n 1)"
CLIENT="$(printf '%s\n' "$LOGS" | grep -o 'GPULAYER_TOKEN=glc_[A-Za-z0-9_-]*' | head -n 1 | cut -d= -f2)"
FP="$(printf '%s\n' "$LOGS" | grep -o 'GPULAYER_FINGERPRINT=[^ ]*' | head -n 1 | cut -d= -f2)"
[ -n "$JOIN" ] && [ -n "$CLIENT" ] && [ -n "$FP" ] || fail "could not read tokens from the banner"

docker run -d --name "gl-worker-$$" --network "$NET" --network-alias gl-worker "$IMAGE" \
  join https://gl-server:8470 --token "$JOIN" --fingerprint "$FP" \
  --name docker-worker --advertise-address gl-worker >/dev/null

client() {
  docker run --rm --network "$NET" -e GPULAYER_URL=https://gl-server:8470 \
    -e GPULAYER_TOKEN="$CLIENT" -e GPULAYER_FINGERPRINT="$FP" "$IMAGE" "$@"
}
i=0
until client worker status 2>/dev/null | grep -q docker-worker; do
  i=$((i + 1)); [ "$i" -lt 60 ] || fail "worker did not join"; sleep 1
done

OUT="$(client submit --target remote -- python -c "print('hello from a docker worker')")" \
  || fail "job failed: $OUT"
printf '%s\n' "$OUT" | grep -q "hello from a docker worker" || fail "job output missing: $OUT"
client metrics | grep -q 'gpulayer_jobs_finished_total{state="succeeded"} 1' \
  || fail "metrics do not show the finished job"
echo "smoke-test: OK ($IMAGE)"
