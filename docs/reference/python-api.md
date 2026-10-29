# Python API

## `gpulayer`

```python
import gpulayer as gl
```

| Name | Description |
|---|---|
| `gl.connect(url, token, *, fingerprint=None, ca_file=None) -> Client` | use a GpuLayer server for this process; checks the connection immediately |
| `gl.default_client() -> Client` | the client set by `connect()`, else one from `$GPULAYER_URL`, else the embedded control plane |
| `gl.remote(fn=None, **options)` | decorator returning a `RemoteFunction`; see [Python functions](../workloads/functions.md) |
| `gl.Client(...)` | explicit client, described below |
| `gl.command(argv, **kw)`, `gl.function(entrypoint, *args, **kw)`, `gl.distributed(entrypoint, *, source, num_workers, procs_per_worker, **kw)` | build a `JobSpec` |
| `gl.JobSpec`, `gl.ResourceRequest`, `gl.PlacementPolicy`, `gl.Accelerator`, `gl.ExecutionTarget`, `gl.JobState` | contracts; see [JobSpec](job-spec.md) |
| `gl.RemoteError` | raised when a remote run does not succeed; `.job` holds the `Job` |

### `Client`

```python
gl.Client(url=None, token=None, *, home=None, ca_file=None, fingerprint=None,
          local_worker=True, worker_python=None)
```

Without `url` (and without `$GPULAYER_URL`) the client runs an embedded control plane in
`$GPULAYER_HOME` and starts in-process workers when jobs need them.

| Method | Description |
|---|---|
| `submit(spec) -> Job` | submit a `JobSpec` |
| `submit_batch(specs) -> list[Job]` | independent jobs |
| `run(spec, timeout=None) -> JobResult` | submit and wait |
| `job(job_id) -> Job`, `jobs(limit=50)`, `workers()` | look up |
| `package(directory) -> ArtifactReference` | upload a source bundle |
| `upload(path_or_bytes, name="")`, `download(ref) -> bytes` | artifacts |
| `ensure_local_workers(n)` | embedded mode: run at least `n` in-process workers |
| `close()` | stop in-process workers (also on `with` exit) |

### `Job`

| Method | Description |
|---|---|
| `status() -> JobStatus` | state, placement decision, tasks |
| `wait(timeout=None) -> JobStatus` | block until the job is finished |
| `result() -> JobResult` | exit code, return value, outputs, costs |
| `cancel()` | request cancellation (idempotent) |
| `logs(after_seq=0)`, `logs_text()`, `follow_logs()` | output lines |
| `events()` | lifecycle events |

## `gpulayer.pytorch`

```python
import gpulayer.pytorch as glt
```

| Name | Description | Page |
|---|---|---|
| `glt.Module` | `nn.Module` with `training_step`, `validation_step`, `configure_optimizers`, `log`, `load_from_checkpoint` | [Trainer and Module](../pytorch/trainer.md) |
| `glt.Trainer(...)` | `fit(model, train, val=None, ckpt_path=None)`, `validate(model, loader)` | [Trainer and Module](../pytorch/trainer.md) |
| `glt.launch(...) -> LaunchResult or None` | run this script on GpuLayer workers | [launch()](../pytorch/launch.md) |
| `glt.Runtime(accelerator, precision)` | device, DDP, sharding and precision helpers | [Runtime helpers](../pytorch/runtime.md) |
| `glt.is_launcher_process()`, `glt.in_gpulayer_job()` | where the code is running | |
| `glt.resolve_accelerator(name)`, `glt.seed_everything(seed)` | utilities | |

`gpulayer.pytorch` imports torch; `import gpulayer` does not.
