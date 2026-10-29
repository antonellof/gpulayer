# JobSpec

Every job is a `JobSpec` (API version `gpulayer.dev/v1alpha1`). The SDK builders, the Trainer
and `launch()` create them for you; you can also write one as JSON and run
`gpulayer submit -f job.json`. Unknown fields are rejected.

```json
{
  "name": "finetune",
  "workload": {"kind": "command", "command": ["python", "train.py"], "source": null},
  "resources": {"accelerator": "cuda", "gpu_count": 2, "gpu_memory_gb": 40,
                "frameworks": {"torch": ">=2.2"}},
  "placement": {"target": "remote", "worker_labels": {"pool": "a100"},
                "on_unschedulable": "fail"},
  "env": {"EPOCHS": "10"},
  "secrets": [{"name": "HF_TOKEN", "source": "env", "key": "HF_TOKEN"}],
  "timeout_seconds": 7200,
  "retry": {"max_attempts": 2, "retry_on": ["worker_lost"]}
}
```

## Top level

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | string | `"job"` | letters, digits, `_ . -`; up to 63 characters |
| `workload` | object | required | one of the three kinds below |
| `resources` | `ResourceRequest` | CPU, 1 core | per node |
| `placement` | `PlacementPolicy` | target `local` | |
| `env` | map | `{}` | not secret; `GPULAYER_*` and torchrun names are reserved |
| `secrets` | list of `SecretRef` | `[]` | resolved on the worker from its own environment |
| `inputs` | list of `{artifact, path}` | `[]` | staged under `$GPULAYER_INPUT_DIR/path` |
| `timeout_seconds` | number | none | |
| `retry` | `{max_attempts, retry_on}` | `1`, `["worker_lost"]` | not allowed for distributed jobs |
| `labels` | map | `{}` | free-form |

## Workloads

| `kind` | Fields |
|---|---|
| `command` | `command` (argv; a leading `python` means the worker's interpreter), `source` |
| `function` | `entrypoint` (`module:function`), `args`, `kwargs` (JSON), `source` |
| `distributed` | `entrypoint` (script path in the bundle) and `args`, **or** `function`, `function_args`, `function_kwargs`; `source`, `num_workers`, `procs_per_worker`, `comm_backend` (`gloo`/`nccl`), `rendezvous` (`port`, `join_timeout_seconds`) |

## ResourceRequest

| Field | Default | Notes |
|---|---|---|
| `accelerator` | `cpu` | `cpu`, `cuda`, `mps`, `rocm` (rocm is never scheduled yet) |
| `gpu_count` | `0` (`1` for GPU accelerators) | devices per node |
| `gpu_memory_gb` | none | minimum per device |
| `gpu_memory_kind` | none | `dedicated` or `unified` |
| `min_cuda_compute_capability` | none | e.g. `"8.0"` |
| `cpu_cores`, `memory_gb`, `disk_gb` | 1, none, none | placement constraints, not enforced limits |
| `os`, `cpu_arch` | `[]` | e.g. `["linux"]`, `["x86_64"]` |
| `frameworks` | `{}` | PEP 440 specifiers, e.g. `{"torch": ">=2.2"}` |
| `python` | none | PEP 440 specifier for the worker's Python |

## PlacementPolicy

| Field | Default | Notes |
|---|---|---|
| `target` | `local` | `local`, `remote`, `cloud` |
| `fallback_targets` | `[]` | tried in order after `target` |
| `worker_ids`, `worker_labels` | `[]`, `{}` | pin to specific workers |
| `providers`, `regions` | `[]` | restrict provider-backed workers |
| `strategy` | `first_fit` | or `most_free_memory` |
| `on_unschedulable` | `queue` | `fail` rejects jobs no registered worker could ever run |
| `max_hourly_cost`, `currency` | none, `USD` | offers without a price are rejected when set |
| `provisioning` | none | `{provider, release, provisioning_timeout_seconds}` |
