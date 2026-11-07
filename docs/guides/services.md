# Long-running services

A service is a command that keeps running, such as an inference server. GPUlayer places it like
a job, restarts it, and reports where it listens (design: ADR-0019).

## Start one

```bash
gpulayer submit --target remote --service --accelerator cuda --gpus 1 -- \
  sh -c 'exec python -m vllm.entrypoints.openai.api_server --model my-model --port "$PORT"'
```

Against a server, this waits until every replica accepts connections, prints the job id and a
`host:port` per replica, and returns.

```python
import gpulayer as gl

client = gl.connect()
job = client.submit(gl.service(["python", "serve.py"], replicas=2, accelerator="cuda", gpus=1))
print(job.endpoints(wait=120))   # ['10.0.0.5:41237', '10.0.0.6:38811']
job.cancel()                     # stops every replica
```

On an AMD Ryzen AI NPU, `gl.flm_service("llama3.2:1b")` does this for you: see
[AMD Ryzen AI NPUs](amd-npu.md).

## Behaviour

| Setting | Default | Meaning |
|---|---|---|
| `replicas` (`--replicas`) | 1 | copies, each on a different worker with its own GPUs |
| `port` (`--no-port` to turn off) | on | the worker sets `$PORT` to a free port; the service must listen on it |
| `restart` (`--restart`) | `always` | restart on exit; `on_failure` only after a non-zero exit |
| `max_restarts` (`--max-restarts`) | no limit | restarts per replica before the service ends |
| `retry.max_attempts` | 10 | re-placements of the whole service after a worker is lost |
| `timeout_seconds` | none | runs until cancelled unless set |

- **Endpoints** are `<worker advertise address>:<port>`, reported once the port accepts TCP
  connections (`gpulayer job status JOB` shows `endpoint=`). Each restart picks a new port.
- **Restarts** back off from 1 s, doubling to 30 s, and reset after a minute of uptime.
- **Ending:** when one replica ends for good, the others are stopped and the job takes its
  outcome. If a worker is lost, all replicas stop and the service is placed again.
- **Logs:** `gpulayer job logs JOB` has every replica's output and a line per restart.

## Limits

- No load balancer: clients must reach the workers' advertise addresses.
- Health checking is only "the port accepts connections". No HTTP health checks, rolling
  updates or autoscaling yet.
- Isolation from other jobs is limited to process groups and reserved GPUs
  ([Security](../architecture/security.md)).

## Tests

`tests/integration/test_services.py` runs a real HTTP service on a local worker and checks
restart on a new port and cancel. Control plane tests cover distinct-worker placement, endpoints
and re-placement after worker loss (simulated workers). The Kubernetes smoke test requests a
service on a worker pod from the server pod.
