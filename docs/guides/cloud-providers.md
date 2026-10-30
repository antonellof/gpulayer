# Cloud execution and provider adapters

GPUlayer can create the machines a job needs and delete them afterwards. The first real adapter
is Google Cloud (Compute Engine).

## Google Cloud

Install the extra on the server and give it credentials that may create and delete VMs in the
project (`gcloud auth application-default login`, a service account, or the VM's own service
account with the `cloud-platform` scope):

```bash
pip install 'gpulayer[gcp]'
```

Describe the machines GPUlayer may create in a JSON file:

```json
{
  "gcp": {
    "type": "gcp",
    "project": "my-project",
    "zone": "europe-west1-b",
    "offers": [
      {"id": "cpu4", "machine_type": "e2-standard-4"},
      {"id": "t4", "machine_type": "n1-standard-8", "gpu_type": "nvidia-tesla-t4",
       "gpu_count": 1, "spot": true, "packages": ["torch"],
       "frameworks": {"torch": "2.15", "torch.cuda": "2.15"}, "price_per_hour": 0.4}
    ]
  }
}
```

Start the server with it, and submit jobs that may use it:

```bash
gpulayer up --no-worker --providers providers.json
gpulayer submit --target cloud --provider gcp --accelerator cuda --gpus 1 -- python train.py
```

When no worker fits, GPUlayer picks the cheapest compatible offer (offers without a price come
last), creates the VM and waits for it. The VM's startup script installs the same GPUlayer
version from PyPI plus the offer's `packages`, and joins with a single-use token bound to that
instance. The job runs there, and the VM is deleted when the job ends.

| Offer field | Meaning |
|---|---|
| `machine_type` | a Compute Engine machine type |
| `gpu_type`, `gpu_count` | an accelerator type GPUlayer knows (T4, L4, V100, P100, P4, A100 40/80 GB, H100); others are rejected, not guessed |
| `spot` | a Spot VM (cheaper; Google may reclaim it) |
| `image`, `disk_gb` | default: a CUDA image for GPU offers, Ubuntu 22.04 otherwise; 100 GB |
| `packages`, `frameworks` | extra pip packages for jobs, and the frameworks the scheduler should expect from them |
| `price_per_hour`, `currency` | your price, used for `--max-hourly-cost`; GPUlayer has no price list of its own |

Other settings: `network`, `external_ip` (on by default, so a VM can reach PyPI unless you have
Cloud NAT), `labels`, `gpulayer_version` and `gpulayer_package`.

New VMs connect to the server's `public_url` (`--advertise-address`), so they must reach it,
for example on the same VPC network. Their metadata holds the single-use instance token; it is
useless once the worker has registered.

Validated on GCP (2026-10-30): a job submitted with `--provider gcp` created an
`e2-standard-2` VM in 12 s; it booted, installed GPUlayer and joined 46 s later, ran the job,
and was deleted 1 s after the job ended (`benchmarks/results/2026-10-30-gcp-provider/`). GPU
offers use the same path but have not been run through it yet.

## Interface

`providers/base.py` defines `ProviderAdapter`:

| Method | Notes |
|---|---|
| `features()` | which operations the adapter really supports |
| `list_offers()` | `InstanceOffer`s with a `WorkerCapabilities` template and an optional `PriceQuote` |
| `provision(offer_id, bootstrap)` | starts an instance; `WorkerBootstrap` carries the control-plane URL, an instance reference and a single-use join token |
| `get_instance(id)` / `terminate(id)` | state inspection; idempotent termination |

Errors: `ProviderError`, `CapacityUnavailable` (retryable), `QuotaExceeded`. Provider-specific
instance names are metadata only; the scheduler sees capabilities.

## Lifecycle (`control_plane/provisioning.py`)

1. A cloud job with `placement.provisioning` finds no eligible provider worker.
2. Offers are filtered with the same `compatibility_reasons()` function the scheduler uses,
   plus `max_hourly_cost`. When a cost cap is set, an offer without a reliable price is
   rejected. The cheapest known price wins, and ties break on offer id.
3. One instance per node is requested. Each gets a fresh single-use token, bound to that
   instance record; only its SHA-256 is stored.
4. The job moves to `provisioning`; an estimated cost record is attached
   (`kind="estimate"`, currency, unit, source, timestamp).
5. The booted worker registers with origin `provider` and its token. Identity fields
   (provider, region, reservation label) come from GPUlayer's record, not the worker's claims.
6. The worker is reserved for that job. The scheduler places the job only after the worker
   reports compatible capabilities, since a successfully provisioned machine may still lack a
   working GPU runtime.
7. On a terminal job state, `release="terminate_on_completion"` instances are terminated and
   their workers revoked. Failed terminations are logged as possible orphans.
8. Provider failure or `provisioning_timeout_seconds` fails the job and releases instances.

## Status

- `GCPProvider` (`providers/gcp.py`) is the first real adapter. It is unit-tested against a
  fake Compute Engine API and validated once on GCP with a CPU VM.
- `FakeProvider` is an in-memory, deterministic adapter with failure injection, simulated, used
  by the lifecycle tests.
- Other clouds (AWS, Azure, Lambda, RunPod) have no adapter yet. GPUlayer does not ship
  invented prices or capacities.

## Existing vs managed workers

A private GPU box that joins with a join token is an existing worker (`target=remote`). A
machine created by an adapter is a managed worker (`target=cloud`). GPUlayer schedules work on
both kinds and provisions only managed workers.
