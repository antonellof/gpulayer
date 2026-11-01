# Python API

## `gpulayer`

```python
import gpulayer as gl
```

| Name | Description |
|---|---|
| `gl.connect(url, token, *, fingerprint=None, ca_file=None) -> Client` | use a server for this process; checks the connection now |
| `gl.default_client() -> Client` | the `connect()` client, else one from `$GPULAYER_URL`, else embedded |
| `gl.remote(fn=None, **options)` | decorator returning a `RemoteFunction` ([Python functions](../workloads/functions.md)) |
| `gl.Client(...)` | explicit client, below |
| `gl.command(argv, **kw)`, `gl.function(entrypoint, *args, **kw)`, `gl.distributed(entrypoint, *, source, num_workers, procs_per_worker, **kw)` | build a `JobSpec` |
| `gl.JobSpec`, `gl.ResourceRequest`, `gl.PlacementPolicy`, `gl.Accelerator`, `gl.ExecutionTarget`, `gl.JobState` | contracts ([JobSpec](job-spec.md)) |
| `gl.RemoteError` | a remote run failed; `.job` holds the `Job` |

### `Client`

```python
gl.Client(url=None, token=None, *, home=None, ca_file=None, fingerprint=None,
          local_worker=True, worker_python=None)
```

With no `url` and no `$GPULAYER_URL`, it runs an embedded control plane in `$GPULAYER_HOME`
and starts in-process workers on demand.

| Method | Description |
|---|---|
| `submit(spec) -> Job` | submit a `JobSpec` |
| `submit_batch(specs) -> list[Job]` | independent jobs, 500 per request, one scheduling pass per request |
| `run(spec, timeout=None) -> JobResult` | submit and wait |
| `job(job_id) -> Job`, `jobs(limit=50)`, `workers()` | look up |
| `package(directory) -> ArtifactReference` | upload a source bundle |
| `upload(path_or_bytes, name="")`, `download(ref) -> bytes` | artifacts |
| `ensure_local_workers(n)` | embedded: at least `n` in-process workers |
| `close()` | stop in-process workers (also on `with` exit) |

### `Job`

| Method | Description |
|---|---|
| `status() -> JobStatus` | state, placement decision, tasks |
| `wait(timeout=None) -> JobStatus` | block until finished |
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
| `glt.Module` | `nn.Module` with Lightning's steps and hooks, `log`, `save_hyperparameters`, manual optimization, `load_from_checkpoint` | [Trainer and Module](../pytorch/trainer.md) |
| `glt.DataModule` | `prepare_data`, `setup`, the four loaders | [Trainer and Module](../pytorch/trainer.md#datamodule) |
| `glt.Trainer(...)` | `fit(model, train, val, datamodule=, ckpt_path=)`, `validate`, `test`, `predict`, `save_checkpoint` | [Trainer and Module](../pytorch/trainer.md) |
| Callbacks, loggers, profilers, `glt.Tuner` | `ModelCheckpoint`, `EarlyStopping`, `EMAWeightAveraging`, `CSVLogger`, `SimpleProfiler`, ... | [Callbacks, loggers and profilers](../pytorch/callbacks.md) |
| `glt.launch(...) -> LaunchResult or None` | run this script on workers | [launch()](../pytorch/launch.md) |
| `glt.Runtime(accelerator, precision)` | device, DDP, sharding and precision helpers | [Runtime helpers](../pytorch/runtime.md) |
| `glt.is_launcher_process()`, `glt.in_gpulayer_job()` | where the code runs | |
| `glt.resolve_accelerator(name)`, `glt.seed_everything(seed)` | utilities | |

Full signatures: [PyTorch API reference](../pytorch/api.md). `gpulayer.pytorch` imports torch;
`import gpulayer` does not.
