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
| `environment` | object | none | `{requirements: [...], requirements_file: "path in bundle"}`: extra packages, installed once per worker and reused ([details](../workloads/frameworks.md#extra-packages-for-one-job)) |

## Workloads

| `kind` | Fields |
|---|---|
| `command` | `command` (argv; a leading `python` means the worker's interpreter), `source` |
| `function` | `entrypoint` (`module:function`), `args`, `kwargs` (JSON), `source` |
| `distributed` | `entrypoint` (script path in the bundle) and `args`, **or** `function`, `function_args`, `function_kwargs`; `source`, `num_workers`, `procs_per_worker`, `comm_backend` (`gloo`/`nccl`), `rendezvous` (`port`, `join_timeout_seconds`) |

## ResourceRequest

| Field | Default | Notes |
|---|---|---|
| `profile` | none | runtime profile (see below); fills in `accelerator` and `frameworks` |
| `accelerator` | `cpu` | `cpu`, `cuda`, `mps`, `rocm` (rocm is never scheduled yet) |
| `gpu_count` | `0` (`1` for GPU accelerators) | devices per node |
| `gpu_memory_gb` | none | minimum per device |
| `gpu_memory_kind` | none | `dedicated` or `unified` |
| `min_cuda_compute_capability` | none | e.g. `"8.0"` |
| `cpu_cores`, `memory_gb`, `disk_gb` | 1, none, none | placement constraints, not enforced limits |
| `os`, `cpu_arch` | `[]` | e.g. `["linux"]`, `["x86_64"]` |
| `frameworks` | `{}` | PEP 440 specifiers, e.g. `{"torch": ">=2.2"}` |
| `python` | none | PEP 440 specifier for the worker's Python |

### Runtime profiles

A profile is a shorthand for common requirements. Explicit `frameworks` entries override the
profile's, and an explicit `accelerator` must match it. Workers are still matched on what they
report, so a profile never makes an incompatible worker eligible.

| Profile | Accelerator | Frameworks |
|---|---|---|
| `pytorch-cpu` | `cpu` | `torch>=2.1` |
| `pytorch-cuda` | `cuda` | `torch>=2.1`, `torch.cuda` |
| `pytorch-mps` | `mps` | `torch>=2.1`, `torch.mps` |
| `jax-cuda` | `cuda` | `jax>=0.4`, `jaxlib>=0.4` |
| `cupy-cuda` | `cuda` | `cupy>=12` |
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
| `worker_ids`, `worker_labels` | `[]`, `{}` | pin to specific workers |
| `providers`, `regions` | `[]` | restrict provider-backed workers; a `regions` entry may also name a zone |
| `group_locality` | `prefer` | distributed jobs: `prefer` the closest group (one network group, zone, then region), require one with `network_group`, `zone` or `region`, or `any`; see [Scheduling](../architecture/scheduling.md#distributed-groups-and-network-locality) |
| `strategy` | `spread` | `spread`: least-loaded worker first; `first_fit`: lowest worker id with a free slot (packs work); `most_free_memory` |
| `on_unschedulable` | `queue` | `fail` rejects jobs no registered worker could ever run |
| `max_hourly_cost`, `currency` | none, `USD` | offers without a price are rejected when set |
| `provisioning` | none | `{provider, release, provisioning_timeout_seconds}` |
