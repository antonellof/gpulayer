# ADR-0026: Ryzen AI NPUs as a place to serve models

- **Status:** Proposed
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

## Decision proposed

1. **An NPU is its own kind of device.** A new `Accelerator.NPU` with vendor AMD, a capability such
   as `npu.xdna2`, and no `torch.*` pseudo-framework. A job asking for a GPU is never placed on it,
   and an NPU never counts towards `gpu_count`.
2. **It runs services, through `ServiceJob`** (ADR-0019), which already has replicas, `$PORT`,
   endpoints and restarts: `flm serve` is one command. No new job type.
3. **Detection comes from FastFlowLM itself** (`flm validate`), in a subprocess, as for torch; if
   its output cannot be read, the worker says so in a note and reports no NPU.
4. **The Trainer rejects it** with a message that says why, instead of falling back to CPU.
5. **GPUlayer does not ship FastFlowLM**: it is a package on the worker, like torch.

## Consequences

An enum value that older workers do not know must not reach them; as with `resume`, it appears
only in jobs that ask for it. It widens the product past training, which ADR-0023 narrowed, so
it should be a small addition and take no work from ROCm validation.

## Alternatives

- **Train on the NPU.** Nothing read says this is possible through PyTorch. Needs its own
  investigation before any promise.
- **A FastFlowLM provider or built-in inference API.** More to build and keep, for what a service
  job already does.
- **Treat it as a ROCm device.** Wrong: ROCm's PyTorch path drives GPUs. AMD's compatibility
  matrix lists Ryzen integrated GPUs for ROCm, which are a separate question from the NPU.

## Security and operational implications

`flm serve` is an unauthenticated local server as far as the pages read show, so a service must not
expose it beyond what GPUlayer's endpoints already allow. To be reviewed before implementation,
along with model downloads on workers.

## Follow-up

Needs a decision from the maintainer, and an XDNA2 machine to run `flm validate` on: its output is
not documented on the pages read.
