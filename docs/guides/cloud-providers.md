# Cloud execution and provider adapters

GPUlayer can create the machines a job needs and delete them afterwards. The first real adapter
is Google Cloud (Compute Engine).

## Google Cloud

The server needs the extra and credentials that may create and delete VMs
(`gcloud auth application-default login`, a service account, or a VM service account with the
`cloud-platform` scope). List the machines it may create in a providers file:

```bash
pip install 'gpulayer[gcp]'
```

```json
{
  "gcp": {
    "type": "gcp",
    "project": "my-project",
    "zone": "europe-west1-b",
    "offers": [
      {"id": "cpu4", "machine_type": "e2-standard-4"},
      {"id": "t4", "machine_type": "n1-standard-8", "gpu_type": "nvidia-tesla-t4",
       "gpu_count": 1, "spot": true, "packages": ["torch", "numpy"],
       "frameworks": {"torch": "2.15", "torch.cuda": "2.15"}, "price_per_hour": 0.4}
    ]
  }
}
```

```bash
gpulayer up --no-worker --providers providers.json
gpulayer submit --target cloud --provider gcp --accelerator cuda --gpus 1 -- python train.py
```

When no worker fits, GPUlayer creates a VM from the cheapest compatible offer (unpriced offers
last). The VM installs the same GPUlayer version from PyPI plus `packages`, joins with a
single-use token, runs the job and is deleted when it ends.

| Offer field | Meaning |
|---|---|
| `machine_type` | a Compute Engine machine type |
| `gpu_type`, `gpu_count` | T4, L4, V100, P100, P4, A100 40/80 GB or H100; other types are rejected |
| `spot` | a Spot VM (cheaper; Google may reclaim it) |
| `image`, `disk_gb` | default: a CUDA image for GPU offers, Ubuntu 22.04 otherwise; 100 GB |
| `packages`, `frameworks` | extra pip packages, and the frameworks the scheduler should expect |
| `price_per_hour`, `currency` | your price, used for `--max-hourly-cost`; GPUlayer has no price list |

Also: `network`, `external_ip` (default on, so VMs reach PyPI without Cloud NAT), `labels`,
`gpulayer_version`, `gpulayer_package`. The PyTorch wheel does not install `numpy`; add it to
`packages` if jobs need it. VMs connect to the server's `public_url` (`--advertise-address`),
so it must be reachable from them, for example on the same VPC.

Validated on GCP (2026-10-30), GPUlayer 0.4.0 from PyPI on the server:

| Offer | VM created | Worker joined | Job | VM deleted |
|---|---|---|---|---|
| `e2-standard-2` (CPU) | 12 s after submit | 46 s later | a command | 1 s after the job ended |
| `n1-standard-8` + 1 T4, Spot | 23 s after submit | 78 s later, 1 accelerator, `torch.cuda` | `Trainer` on CUDA, fp32 and 16-mixed, 17 s | 1 s after the job ended |

Logs: `benchmarks/results/2026-10-30-gcp-provider/` and
`benchmarks/results/2026-10-30-gcp-provider-t4/`.

## Interface

`ProviderAdapter` in `providers/base.py` has `features()` (operations it really supports),
`list_offers()` (`InstanceOffer`s with a `WorkerCapabilities` template and optional
`PriceQuote`), `provision(offer_id, bootstrap)` (`WorkerBootstrap` carries the control-plane
URL, an instance reference and a single-use join token), `get_instance(id)` and idempotent
`terminate(id)`. Errors: `ProviderError`, `CapacityUnavailable` (retryable), `QuotaExceeded`.

## Lifecycle (`control_plane/provisioning.py`)

1. A cloud job with `placement.provisioning` finds no eligible worker. Offers are filtered with
   the scheduler's `compatibility_reasons()` and `max_hourly_cost` (with a cap, unpriced offers
   are rejected); the cheapest wins, ties broken by offer id.
2. One instance per node is requested, each with a single-use token bound to it (only its
   SHA-256 is stored). The job moves to `provisioning` with an `estimate` cost record
   ([Cost accounting](../architecture/cost-accounting.md)).
3. The worker registers with origin `provider`; its identity fields come from GPUlayer's record.
   It is reserved for the job, which is placed only once it reports compatible capabilities.
4. When the job ends, `release="terminate_on_completion"` instances are terminated and their
   workers revoked; failed terminations are logged as possible orphans. A provider failure or
   `provisioning_timeout_seconds` fails the job and releases instances.

## Status

- `GCPProvider` (`providers/gcp.py`): unit-tested against a fake Compute Engine API, validated
  on GCP with a CPU VM and a Spot T4 VM.
- `FakeProvider`: simulated, deterministic, with failure injection; used by lifecycle tests.
- AWS, Azure, Lambda and RunPod have no adapter yet.

## Existing vs managed workers

A machine that joins with a join token is an existing worker (`target=remote`). A machine an
adapter creates is a managed worker (`target=cloud`). Only managed workers are provisioned.
