# AMD Ryzen AI NPUs

**Status: implemented, not run on an NPU.** Tested with a stand-in `flm` that answers as
FastFlowLM's source says the real one does. To train, use a GPU: an NPU only serves models.

GPUlayer can place an inference server on the NPU of a Ryzen AI machine, through
[FastFlowLM](https://github.com/ROCm/FastFlowLM), which gives you an OpenAI-compatible API. It adds
placement, restarts and endpoints; FastFlowLM does the work on the NPU
([ADR-0026](../decisions/ADR-0026-npu-inference-services.md)).

## Requirements

A Linux machine with an XDNA2 NPU (Strix, Strix Halo, Kraken or Gorgon Point) and FastFlowLM, with
its driver and firmware, on the worker. GPUlayer does not install it. Check with:

```bash
flm validate
gpulayer resources detect       # lists an npu device, or says why not
```

The worker and the server need GPUlayer 0.11 or later: an older server does not accept a worker
that reports an NPU.

## Serving a model

```python
import gpulayer as gl

client = gl.connect()
job = client.submit(gl.flm_service("llama3.2:1b", target="remote"))
print(job.endpoints(wait=600))   # ['10.0.0.5:41237']; the first start downloads the model
job.cancel()
```

- `gl.flm_service` takes `args=[...]` for `flm serve` (`["--pmode", "turbo"]`), and the options of
  [services](services.md). A model FastFlowLM does not know ends the job after `max_restarts`.
- An endpoint is reported once FastFlowLM listens, which is after the model is downloaded and
  loaded: it is ready when it appears, and the wait can be long the first time.
- Each replica takes one NPU for itself, so a second service waits. FastFlowLM queues requests
  to its one NPU (10 by default, `--q-len`) and answers 503 past that.
- A machine with a GPU can train while its NPU serves: each is a separate slot.

From the command line, with `$PORT` from the worker:

```bash
gpulayer submit --target remote --service --accelerator npu -- \
  sh -c 'exec flm serve llama3.2:1b --host 127.0.0.1 --port "$PORT"'
```

## Security

FastFlowLM has no authentication. Treat access to its port as access to the model.

- By default a replica listens on the worker's loopback, so the endpoint reported, with the
  worker's address, works only from the worker itself. Reach it from there, or through a tunnel.
- `gl.flm_service(..., expose=True)` binds every interface, for anyone who can reach the worker.
  Use it only on a network you trust.
- Its responses allow any web origin, also with `--cors 0` as read in its source, so on a laptop
  a web page you open can call a server on loopback.

## What does not work

- Training: `Trainer(accelerator="npu")` and `launch(accelerator="npu")` stop with an explanation.
  FastFlowLM does inference only, and nothing we read connects it to PyTorch.
- Distributed jobs on an NPU.
- First-generation XDNA NPUs, which FastFlowLM does not list, and your own models: FastFlowLM's
  pages describe no way to convert one.
- NPU utilisation in worker metrics and the dashboard.
