# Kubernetes

The Helm chart in `deploy/kubernetes/helm/gpulayer` installs a server (one-pod StatefulSet with
a persistent volume, ClusterIP Service on port 8470, HTTPS), two CPU worker pods using the
`-torch` image, an optional GPU worker pool, and generated tokens and TLS certificate (kept on
upgrade).

## Install

```bash
git clone https://github.com/antonellof/gpulayer && cd gpulayer
helm install gpulayer deploy/kubernetes/helm/gpulayer --namespace gpulayer --create-namespace
kubectl -n gpulayer get pods
```

Images: `ghcr.io/antonellof/gpulayer:<version>` and `<version>-torch` (CPU PyTorch). In the
server pod, URL, token and CA are already set:

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

Dashboard: `https://localhost:8470/ui/`, signed in with `$GPULAYER_TOKEN`.

## GPU workers

Requires the NVIDIA device plugin (or GPU Operator). Build and push a CUDA image, then enable
the pool:

```bash
docker build -t registry.example.com/gpulayer:cuda \
  --build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 --build-arg EXTRAS=pytorch-cuda .
docker push registry.example.com/gpulayer:cuda

helm upgrade gpulayer deploy/kubernetes/helm/gpulayer -n gpulayer --reuse-values \
  --set workers.gpu.enabled=true --set workers.gpu.replicas=2 --set workers.gpu.gpusPerWorker=4 \
  --set workers.gpu.image.repository=registry.example.com/gpulayer --set workers.gpu.image.tag=cuda
```

Each pod gets `gpusPerWorker` GPUs and runs one single-GPU job per GPU at a time.

## Settings

| Value | Default | Meaning |
|---|---|---|
| `workers.cpu.replicas`, `workers.cpu.concurrency` | `2`, `2` | CPU worker pods, and jobs at once per pod |
| `workers.gpu.enabled`, `replicas`, `gpusPerWorker` | `false`, `1`, `1` | GPU worker pool |
| `workers.*.nodeSelector`, `tolerations`, `resources` | | pod scheduling |
| `server.service.type` | `ClusterIP` | `LoadBalancer` to let outside machines join |
| `server.extraTLSNames` | `[]` | extra DNS names on the server certificate |
| `server.persistence.size`, `storageClass` | `10Gi`, default class | the server's volume |
| `existingTokenSecret` | | a Secret with `client-token` and `join-token` |
| `image.repository`, `image.tag` | `ghcr.io/antonellof/gpulayer`, chart `appVersion` | image to run |

## Machines outside the cluster

With `server.service.type=LoadBalancer` and the load balancer's name in `server.extraTLSNames`:

```bash
gpulayer join https://gpulayer.example.com:8470 --token <join-token> --ca-file gpulayer-ca.crt
```

## How it works and its limits

- One server pod with SQLite on its volume; restarts keep jobs and workers reconnect.
- A replaced worker pod registers as a new worker; the old one stays offline until `gpulayer gc`.
- Distributed jobs rendezvous on pod IPs, so pods must reach each other on any TCP port.
- Workers do not sandbox jobs (see [Security](../architecture/security.md)).
- Tested in CI on kind (`deploy/kubernetes/smoke-test.sh`): install, a job, a distributed job
  across two worker pods, `helm upgrade`, and a server pod restart. GPU workers on Kubernetes
  have not been tested on GPU nodes yet.
