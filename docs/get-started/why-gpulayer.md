# Why GpuLayer

Most GPU code assumes the GPU is on the machine where you run it. GpuLayer removes that
assumption. You describe what a workload needs (an accelerator, memory, frameworks), and
GpuLayer runs it on a machine that has it, wherever that machine is.

## Where it sits

GPU frameworks, from PyTorch and its training libraries to JAX, MLX and CuPy, decide how a
computation uses the hardware they are given. They work inside the machine, or the cluster,
they were started on. Choosing that machine, shipping code and data to it, starting the work,
watching it and collecting the results is left to you.

GpuLayer is that missing layer. It does not replace your framework, and it is not limited to
training: rendering, inference batches, simulations, data processing and benchmarks are all
workloads. PyTorch gets an extra integration (a Trainer, `launch()`, runtime helpers) because
distributed training needs more coordination than most workloads.

## When it helps

| Situation | Without GpuLayer | With GpuLayer |
|---|---|---|
| Code on a laptop, GPU somewhere else | ssh, copy code, rebuild the environment, run in tmux, copy results back | `gpulayer submit --target remote -- <cmd>`, or `python train.py --target remote` |
| Several GPU machines (team, lab, rented VMs) | track free machines by hand; jobs crash on the wrong GPU, driver or library version | one queue; each job lands where its accelerator and frameworks are available, or is rejected with the reason |
| Mixed hardware (NVIDIA boxes and Apple Silicon Macs) | separate scripts and setup per machine | one interface; CUDA jobs go to NVIDIA workers, Metal jobs to Macs |
| A batch of independent tasks (sweeps, rendering, data processing) | loops and ad-hoc scripts | `fn.map([...])` in parallel across free workers |
| Multi-node training | `torchrun` on every node with matching ranks, addresses and ports | `Trainer(num_nodes=2, devices=4)` or `launch(nodes=2, ...)` |

## A concrete example

You have a MacBook and a Linux box with two NVIDIA GPUs. Once, on the Linux box:

```bash
gpulayer up
```

Then, from the laptop, any of these runs on the box and streams its output back:

```bash
gpulayer submit --accelerator cuda --source ./video -- ./transcode.sh
python sweep.py                                  # @gl.remote(...).map([...]) over both GPUs
python train.py --accelerator cuda --devices 2   # PyTorch DDP on both GPUs
```

Outputs land in your current directory (`gpulayer_runs/<job-id>/`, or `gpulayer job result
--download`). The Mac itself can run MLX or PyTorch MPS jobs from the same queue.

## When not to use it

- **One machine, one GPU, interactive work.** Running your code directly is simpler. GpuLayer
  adds a control plane, workers and tokens you do not need.
- **Long-running services.** An inference server that must stay up and accept traffic is not
  supported yet; GpuLayer runs jobs that finish.
- **Untrusted users.** Workloads run as normal processes on workers, not in a sandbox. Use
  GpuLayer among people who trust each other.
- **Validated scale today.** GpuLayer is pre-alpha. NVIDIA GPUs and multi-machine runs are
  implemented but not validated yet, distributed jobs use torchrun only, and automatic cloud
  provisioning is simulated. Cloud VMs you start yourself can join with `gpulayer join`.

Two constraints: function arguments and results must be JSON, and code ships as files (the
directory of your script), not as pickled Python objects.
