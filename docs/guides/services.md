# Long-running services

A service is a command that keeps running, such as an inference server, a notebook or a small
API next to a GPU. GPUlayer places it with the same capability matching as a job, keeps it
running, and tells you where it listens. See ADR-0019 for the design.

## Start one

```bash
gpulayer submit --target remote --service --accelerator cuda --gpus 1 -- \
  sh -c 'exec python -m vllm.entrypoints.openai.api_server --model my-model --port "$PORT"'
```

Against a server (`--url` or `$GPULAYER_URL`), `submit --service` waits until every replica
accepts connections, prints the job id and one `host:port` per replica, and returns. The
service keeps running on the workers.

From Python:

```python
import gpulayer as gl

client = gl.connect()
job = client.submit(gl.service(["python", "serve.py"], replicas=2, accelerator="cuda", gpus=1))
print(job.endpoints(wait=120))   # ['10.0.0.5:41237', '10.0.0.6:38811']
job.cancel()                     # stops every replica
```

## Behaviour

| Setting | Default | Meaning |
|---|---|---|
| `replicas` (`--replicas`) | 1 | copies of the command, each on a different worker, each with its own GPUs |
| `port` (`--no-port` to turn off) | on | the worker picks a free port and sets `$PORT`; the service must listen on it |
| `restart` (`--restart`) | `always` | restart when the process exits; `on_failure` restarts only after a non-zero exit |
| `max_restarts` (`--max-restarts`) | no limit | restarts per replica before the service ends |
| `retry.max_attempts` | 10 | times the whole service is placed again after a worker is lost |
| `timeout_seconds` | none | a service runs until cancelled unless you set one |

- **Endpoints.** A replica's endpoint is `<worker advertise address>:<port>`. The worker
  reports it once the port accepts a TCP connection, and clears it while the replica
  restarts. `gpulayer job status JOB` shows `endpoint=` per task.
- **Restarts.** The worker waits 1 s before the first restart and doubles the wait up to 30 s.
  A replica that stayed up for more than a minute starts again from 1 s. Each restart gets a
  new port, so the endpoint can change; read it again after a restart.
- **Ending.** When one replica ends for good (its restarts are used up, or `on_failure` saw a
  clean exit), the other replicas are stopped and the job takes that replica's outcome.
- **Worker loss.** If a worker stops sending heartbeats, its replica is marked lost, the
  others are stopped, and the service is queued to be placed again.
- **Logs.** `gpulayer job logs JOB` has the output of every replica and a line for each
  restart.

## Limits

- There is no load balancer or public entry point. Clients connect to the reported endpoints,
  so they must reach the workers' advertise addresses (same cluster, VPN or private network).
- Health checking is only "the port accepts connections". There are no HTTP health checks,
  rolling updates or autoscaling yet.
- After a worker loss the whole service moves, including healthy replicas.
- The worker does not isolate the service from other jobs on the same machine beyond process
  groups and the GPUs it reserved; see the security model.

## Tests

`tests/integration/test_services.py` starts a real HTTP service on a local worker, makes it
exit, checks it restarts on a new port, and cancels it. The control plane tests cover
placement on distinct workers, endpoint reporting and re-placement after worker loss
(simulated workers). The Kubernetes smoke test starts a service on a worker pod and requests
it from the server pod.
