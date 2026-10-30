# Benchmarks

Scripts that measure GPUlayer, and the results they produced. The write-up with charts is
[Performance and comparison](../docs/get-started/performance-and-comparison.md); internals are
in [Performance](../docs/architecture/performance.md).

| Script | Measures | Needs |
|---|---|---|
| `dispatch.py` | time GPUlayer adds around a job, burst throughput | nothing |
| `distributed_startup.py` | where a distributed job's start-up goes, direct launcher vs torchrun | torch |
| `http_requests.py` | HTTP API latency and throughput over TLS | nothing |
| `lightning_comparison/` | GPUlayer `Trainer` vs PyTorch Lightning: ms per step and start-up | torch, `lightning`, a CUDA GPU for the GPU cases |
| `multi_machine.py` | placement, distributed jobs and failure across machines; hundreds of jobs | a server with workers on 2+ machines |
| `../scripts/validate_gpu.sh` | full validation on an NVIDIA machine (tests, CUDA checks, benchmarks) | an NVIDIA GPU |

Rules for timed runs: run one measurement at a time, with nothing else on the GPU; report the
hardware, versions and method with every number; keep the raw logs in `results/`.

## Run locally

```bash
uv pip install -e '.[dev,examples]'
python benchmarks/dispatch.py --jobs 40 --workers 4
python benchmarks/distributed_startup.py --repeat 4            # direct launcher and torchrun
python benchmarks/http_requests.py

uv pip install lightning==2.6.6
cd benchmarks/lightning_comparison && python run.py mlp-fp32 cnn-fp16   # CUDA cases
python run.py ddp-cpu-2                                                   # CPU only
```

## Run on Google Cloud

`cloud/gcp.sh` creates VMs, copies this checkout, runs the benchmarks and brings the results
back. It needs the `gcloud` CLI, logged in, and a project with Compute Engine enabled and GPU
quota. VMs are labelled `purpose=gpulayer-bench`; delete them with `down` when finished.

```bash
export GCP_PROJECT=my-project GCP_ZONE=europe-west1-b
C=benchmarks/cloud/gcp.sh

# One NVIDIA machine: validation and the Lightning comparison
$C gpu-up gl-gpu nvidia-tesla-t4          # tries GPU_ZONES, Spot first, then standard
$C push gl-gpu                             # copy this checkout and install it
$C validate gl-gpu                         # scripts/validate_gpu.sh
$C compare gl-gpu                          # benchmarks/lightning_comparison (about 10 min)

# Add a CPU machine and test across both
$C cpu-up gl-cpu
$C push gl-cpu
$C cluster gl-gpu gl-cpu                   # server on gl-gpu, a worker on each machine
$C multi gl-gpu                            # benchmarks/multi_machine.py

# Save everything, then delete the VMs
$C fetch gl-gpu benchmarks/results/$(date +%F)-gcp/
$C down gl-gpu gl-cpu
```

Notes:

- GPU capacity varies. `gpu-up` tries each zone in `GPU_ZONES`, Spot then standard; a new
  project's GPU quota is often 0 or 1 until Google approves a request.
- `cluster` binds the server to the VM's internal address. The default network's
  `default-allow-internal` firewall rule lets the machines reach each other; nothing is
  exposed to the internet. Tokens stay on the VMs.
- Run the script with `bash`, not `zsh`: zsh does not split unquoted variables into words.
- `gcp.sh` repeats the commands used for the 2026-10-30 runs, collected into one script. The
  script as a whole has not yet been run end to end.

## Results

| Directory | Contents |
|---|---|
| `results/2026-10-30-gcp/summary.json` | the numbers behind the docs and charts |
| `results/2026-10-30-gcp/t4/` | Tesla T4: validation logs, Lightning comparison, `nvidia-smi`, package versions |
| `results/2026-10-30-gcp/multi-machine/` | two VMs: multi-machine log and results, HTTP before and after |

Regenerate the charts after adding results:

```bash
python scripts/make_perf_charts.py benchmarks/results/2026-10-30-gcp/summary.json
```
