# ADR-0026: Ryzen AI NPUs as a place to serve models

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

AMD Ryzen AI machines have an NPU (XDNA) next to their GPU, and many laptops have nothing else
that accelerates models. The product focus (ADR-0023) is training on every GPU you can reach.

[FastFlowLM](https://github.com/ROCm/FastFlowLM), now part of ROCm, runs large language models on
these NPUs. As read on 2026-11-07 (v1.0.12), it does inference only: no page read mentions training,
fine-tuning or PyTorch. It supports XDNA2 chips (Strix, Strix Halo, Kraken, Gorgon Point), needs
recent drivers, downloads models from Hugging Face, and `flm serve` starts an OpenAI-compatible
server on port 52625. Its code is MIT; its NPU kernels are free for any use and redistributed only
unmodified. The pages read do not say whether you can convert your own model.

So an NPU is neither a GPU nor a place to run the Trainer, and must not be reported as one
(ADR-0005: never infer support from a backend's name).

FastFlowLM's source settles what its pages left open: `flm validate --json` exits 0 and sets
`ready` when driver, firmware, columns and memlock are all right; `flm version --json` gives the
version; `flm serve` takes `--host` (default `127.0.0.1`) and `--port`, and starts listening only
after the model is loaded. It has no authentication, runs one NPU request at a time, and its lock
is per process, so two servers would fight over one NPU.

## Decision

1. **An NPU is its own kind of device:** `Accelerator.NPU`, vendor AMD, with memory unknown, a
   `flm` framework version and no `torch.*` capability. Jobs ask for it by accelerator, so a job
   asking for a GPU is never placed on it, and it is reserved like a GPU, one job at a time. It adds
   a task slot, so a machine can serve on its NPU and train on its GPU.
2. **It runs services, through `ServiceJob`** (ADR-0019), which already has replicas, `$PORT`,
   endpoints and restarts: `gl.flm_service(model)` builds one, restarting on failure, a limited
   number of times, because an unknown model fails on every start. No new job type. Command jobs
   may use it too; distributed jobs are refused.
3. **Detection comes from FastFlowLM itself** (`flm validate --json`), as for torch. An NPU is
   reported only when `ready` is true; otherwise the worker says why in a note. No NPU is guessed
   from `/dev/accel`, because GPUlayer cannot drive one without FastFlowLM.
4. **The Trainer and `launch` refuse it** with a message that says why, instead of falling back to
   the CPU.
5. **GPUlayer does not ship FastFlowLM**: it is software on the worker, like torch.
6. **A service listens on loopback unless told otherwise** (`expose=True`), because FastFlowLM has
   no authentication.

## Consequences

An enum value that older components do not know must not reach them. It appears only in NPU jobs
and in the capabilities of a worker that has a ready NPU, so such a worker needs a 0.11 server.
Machines without FastFlowLM are unaffected, beyond one failed `flm` lookup when a worker starts.
The endpoint of a loopback service names the worker's address but works only on the worker.
This widens the product past training, which ADR-0023 narrowed, so it stays a small addition.

## Alternatives

- **Train on the NPU.** Nothing read says this is possible through PyTorch. Needs its own
  investigation before any promise.
- **A FastFlowLM provider or built-in inference API.** More to build and keep, for what a service
  job already does.
- **Treat it as a ROCm device.** Wrong: ROCm's PyTorch path drives GPUs. AMD's compatibility
  matrix lists Ryzen integrated GPUs for ROCm, which are a separate question from the NPU.

## Security and operational implications

`flm serve` has no authentication, hence point 6, and the source sends permissive CORS headers on
every response, also with `--cors 0`, so a web page opened on the same machine can call a loopback
server. FastFlowLM downloads models from Hugging Face on the worker, checked against digests, so
a service needs network access there. The tag passed to `flm_service` is validated, and passed as
an argument, not through a shell string. `flm validate` may raise its own memlock limit.

## Follow-up

Run it on an XDNA2 machine: the reports here come from reading the source. An HTTP readiness
check is not needed while FastFlowLM listens only after loading a model; if that changes, a
service health check will be.
