# JobSpec

Every job is a `JobSpec` (API version `gpulayer.dev/v1alpha1`). The SDK, the Trainer and
`launch()` build them; you can also write JSON and run `gpulayer submit -f job.json`. Unknown
fields are rejected.

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
| `name` | string | `"job"` | letters, digits, `_ . -`; max 63 characters |
| `workload` | object | required | one of the kinds below |
| `resources` | `ResourceRequest` | CPU, 1 core | per node |
| `placement` | `PlacementPolicy` | target `local` | |
| `env` | map | `{}` | not secret; `GPULAYER_*` and torchrun names reserved |
| `secrets` | list of `SecretRef` | `[]` | resolved from the worker's environment |
| `inputs` | list of `{artifact, path}` | `[]` | staged at `$GPULAYER_INPUT_DIR/path` |
| `timeout_seconds` | number | none | |
| `retry` | `{max_attempts, retry_on}` | `1`, `["worker_lost"]` | not for distributed jobs |
| `labels` | map | `{}` | free-form |
| `environment` | object | none | `{requirements: [...], requirements_file: "path in bundle"}`, installed once per worker ([details](../workloads/frameworks.md#extra-packages-for-one-job)) |

## Workloads

| `kind` | Fields |
|---|---|
| `command` | `command` (argv; leading `python` means the worker's interpreter), `source` |
| `function` | `entrypoint` (`module:function`), `args`, `kwargs` (JSON), `source` |
| `distributed` | `entrypoint` (script in the bundle) and `args`, **or** `function`, `function_args`, `function_kwargs`; plus `source`, `num_workers`, `procs_per_worker`, `comm_backend` (`gloo`/`nccl`), `rendezvous` (`port`, `join_timeout_seconds`) |

## ResourceRequest

| Field | Default | Notes |
|---|---|---|
| `profile` | none | sets `accelerator` and `frameworks` (below) |
| `accelerator` | `cpu` | `cpu`, `cuda`, `mps`, `rocm` (rocm never scheduled yet) |
| `gpu_count` | `0` (`1` for GPU accelerators) | per node |
| `gpu_memory_gb` | none | minimum per device |
| `gpu_memory_kind` | none | `dedicated` or `unified` |
| `min_cuda_compute_capability` | none | e.g. `"8.0"` |
| `cpu_cores`, `memory_gb`, `disk_gb` | 1, none, none | placement constraints, not enforced limits |
| `os`, `cpu_arch` | `[]` | e.g. `["linux"]`, `["x86_64"]` |
| `frameworks` | `{}` | PEP 440, e.g. `{"torch": ">=2.2"}` |
| `python` | none | PEP 440 specifier for the worker's Python |

### Runtime profiles

Shorthand for common requirements. Explicit `frameworks` entries override the profile's; an
explicit `accelerator` must match. Workers are still matched on what they report, so a profile
never makes an incompatible worker eligible.

| Profile | Accelerator | Frameworks |
|---|---|---|
| `pytorch-cpu` | `cpu` | `torch>=2.1` |
| `pytorch-cuda` | `cuda` | `torch>=2.1`, `torch.cuda` |
| `pytorch-mps` | `mps` | `torch>=2.1`, `torch.mps` |
| `jax-cuda` | `cuda` | `jax>=0.4`, `jaxlib>=0.4` (deprecated, removed in 0.9.0; never run on a GPU) |
| `cupy-cuda` | `cuda` | `cupy>=12` (deprecated, removed in 0.9.0; never run on a GPU) |
| `mlx` | `mps` | `mlx>=0.10` |

```python
@gl.remote(profile="pytorch-cuda", gpus=2)
def finetune(...): ...
```

```bash
gpulayer submit --profile mlx -- python bench.py
```

## PlacementPolicy

| Field | Default | Notes |
|---|---|---|
| `target` | `local` | `local`, `remote`, `cloud` |
| `fallback_targets` | `[]` | tried in order after `target` |
| `worker_ids`, `worker_labels` | `[]`, `{}` | pin to workers |
| `providers`, `regions` | `[]` | restrict provider-backed workers; a `regions` entry may name a zone |
| `group_locality` | `prefer` | distributed: `prefer` the closest group (network group, then zone, then region); require one with `network_group`, `zone` or `region`; or `any` ([Scheduling](../architecture/scheduling.md#distributed-groups-and-network-locality)) |
| `strategy` | `spread` | `spread`: least-loaded first; `first_fit`: lowest worker id with a free slot (packs); `most_free_memory` |
| `on_unschedulable` | `queue` | `fail` rejects jobs no registered worker could ever run |
| `max_hourly_cost`, `currency` | none, `USD` | when set, unpriced offers are rejected |
| `provisioning` | none | `{provider, release, provisioning_timeout_seconds}` |
