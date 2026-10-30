# Kubernetes

The Helm chart in `deploy/kubernetes/helm/gpulayer` runs a GPUlayer server and pools of
workers in a Kubernetes cluster:

- **Server:** a StatefulSet with one pod and a persistent volume for its database and
  artifacts, behind a ClusterIP Service on port 8470 (HTTPS).
- **CPU workers:** a Deployment, two pods by default, using the `-torch` image so PyTorch and
  distributed jobs work.
- **GPU workers:** an optional Deployment, each pod requesting `nvidia.com/gpu`.
- **Secrets:** the client and join tokens and a TLS certificate for the server, generated on
  install and kept on upgrade.

## Install

```bash
git clone https://github.com/antonellof/gpulayer && cd gpulayer
helm install gpulayer deploy/kubernetes/helm/gpulayer --namespace gpulayer --create-namespace
kubectl -n gpulayer get pods
```

Images come from `ghcr.io/antonellof/gpulayer`, published with each release: `<version>` (slim)
and `<version>-torch` (with CPU PyTorch). Workers start before the server is ready and wait for
it.

Run commands inside the server pod, where the URL, token and CA are already set:

```bash
kubectl -n gpulayer exec -it gpulayer-gpulayer-server-0 -- gpulayer resources list
kubectl -n gpulayer exec -it gpulayer-gpulayer-server-0 -- \
  gpulayer submit --target remote -- python -c "print('hello')"
```

## Use it from your laptop

```bash
kubectl -n gpulayer port-forward svc/gpulayer-gpulayer-server 8470
kubectl -n gpulayer get secret gpulayer-gpulayer-tls -o jsonpath='{.data.ca\.crt}' | base64 -d > gpulayer-ca.crt
export GPULAYER_URL=https://localhost:8470 GPULAYER_CA_FILE=$PWD/gpulayer-ca.crt
export GPULAYER_TOKEN=$(kubectl -n gpulayer get secret gpulayer-gpulayer-tokens \
  -o jsonpath='{.data.client-token}' | base64 -d)
gpulayer resources list
python train.py --accelerator cpu --nodes 2 --devices 2   # Trainer jobs go to the cluster
```

The dashboard is at `https://localhost:8470/ui/`; sign in with `$GPULAYER_TOKEN`.

## GPU workers

Build a CUDA image and push it to a registry your cluster can pull from:

```bash
docker build -t registry.example.com/gpulayer:cuda \
  --build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 --build-arg EXTRAS=pytorch-cuda .
docker push registry.example.com/gpulayer:cuda
```

Then enable the GPU pool. The cluster needs the NVIDIA device plugin (or GPU Operator), so
pods can request `nvidia.com/gpu`:

```bash
helm upgrade gpulayer deploy/kubernetes/helm/gpulayer -n gpulayer --reuse-values \
  --set workers.gpu.enabled=true --set workers.gpu.replicas=2 --set workers.gpu.gpusPerWorker=4 \
  --set workers.gpu.image.repository=registry.example.com/gpulayer --set workers.gpu.image.tag=cuda
```

Each GPU worker pod gets `gpusPerWorker` GPUs and runs one single-GPU job per GPU at a time.

## Settings

| Value | Default | Meaning |
|---|---|---|
| `workers.cpu.replicas`, `workers.cpu.concurrency` | `2`, `2` | CPU worker pods, and jobs at once per pod |
| `workers.gpu.enabled`, `replicas`, `gpusPerWorker` | `false`, `1`, `1` | GPU worker pool |
| `workers.*.nodeSelector`, `tolerations`, `resources` | | usual pod scheduling settings |
| `server.service.type` | `ClusterIP` | `LoadBalancer` to let machines outside the cluster join |
| `server.extraTLSNames` | `[]` | extra DNS names on the server certificate, such as a load balancer name |
| `server.persistence.size`, `storageClass` | `10Gi`, default class | the server's volume |
| `existingTokenSecret` | | a Secret with `client-token` and `join-token` instead of generated ones |
| `image.repository`, `image.tag` | `ghcr.io/antonellof/gpulayer`, chart `appVersion` | image to run |

## Machines outside the cluster

With `server.service.type=LoadBalancer` and the load balancer's name in
`server.extraTLSNames`, other machines can join the same server:

```bash
gpulayer join https://gpulayer.example.com:8470 --token <join-token> --ca-file gpulayer-ca.crt
```

## How it works and its limits

- One server pod (SQLite on its volume). Restarting it keeps jobs and history, and workers
  reconnect.
- Workers register under their pod name. A replaced pod registers as a new worker, and the
  old one shows as offline until `gpulayer gc` removes it.
- Distributed jobs rendezvous on pod IPs, so pods must reach each other on any TCP port (the
  Kubernetes default; check NetworkPolicies).
- Workers do not sandbox jobs (see [Security](../architecture/security.md)). Use namespaces and
  node pools as you would for any trusted batch workload.
- Tested in CI on a kind cluster (`deploy/kubernetes/smoke-test.sh`): install, a job on a worker
  pod, a distributed job across two worker pods, `helm upgrade`, and a server pod restart. GPU
  workers on Kubernetes have not been tested on GPU nodes yet.
