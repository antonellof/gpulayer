#!/bin/bash
# Boot script for fleet workers (gcp.sh fleet-up): join the server named in the instance
# metadata. Runs as root on every boot of a VM built from the `gcp.sh image` image.
# The join token is in instance metadata, readable by project members and by processes on the
# VM; fleets are short-lived test machines, deleted with `gcp.sh fleet-down`.
md() { curl -sf -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
url=$(md attributes/gl-url)
fp=$(md attributes/gl-fingerprint)
token=$(md attributes/gl-join-token)
ip=$(md network-interfaces/0/ip)
zone=$(md zone | awk -F/ '{print $NF}')
mkdir -p /var/lib/gpulayer
GPULAYER_HOME=/var/lib/gpulayer setsid nohup /opt/gpulayer/.venv/bin/gpulayer join "$url" \
  --token "$token" --fingerprint "$fp" --name "$(hostname)" --advertise-address "$ip" --label "zone=$zone" \
  --concurrency "$(nproc)" > /var/log/gpulayer-join.log 2>&1 < /dev/null &
