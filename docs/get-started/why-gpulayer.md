# Why GpuLayer

PyTorch decides how a model trains. GpuLayer decides where it trains, and handles everything
between the machine you type on and the machine that runs the job.

## When it helps

On one machine with one GPU, plain `python train.py` is already as simple as it gets, and
GpuLayer adds little. It pays off as soon as the GPU you want is not the one in front of you,
or there is more than one.

| Situation | Without GpuLayer | With GpuLayer |
|---|---|---|
| Code on a laptop, GPU in another machine (office server, cloud VM) | ssh in, copy the code, rebuild the environment, run in tmux, copy checkpoints back | `python train.py --target remote`: code shipped, logs streamed, checkpoint downloaded, trained weights back in your local `model` |
| Several GPU machines (team, lab, rented VMs) | keep track by hand of which box is free; jobs crash on the wrong GPU, CUDA build or memory size | one queue; each job goes to a machine that fits, or is rejected before it starts with the reason |
| Multi-node training | on every node: same code and environment, `torchrun` with matching node rank, master address and port; start all together; clean up when one dies | `Trainer(num_nodes=2, devices=4)` |
| Hyperparameter sweep | a sequential loop, or your own scripts to spread runs | `train.map([...])` runs in parallel on free workers |
| Same code on a Mac (MPS), CPU and CUDA | device selection, DDP and sampler boilerplate, different launch commands | `accelerator="auto"`; DDP, data sharding and precision handled |

## A concrete example

You have a MacBook and a Linux box with two NVIDIA GPUs.

Without GpuLayer: push the code, ssh to the box, `git pull`, check that the environment
matches, run `torchrun --nproc-per-node 2 train.py` inside `tmux`, check back later,
`scp` the checkpoint to the laptop, load it.

With GpuLayer, once:

```bash
# on the GPU box
gpulayer up
```

Then, from the laptop, every time:

```bash
python train.py --accelerator cuda --devices 2
```

The laptop prints the training logs as they happen. When the job ends, the checkpoint is in
`gpulayer_runs/<job-id>/` and the `model` object in your script holds the trained weights.

## When not to use it

- **One machine, one GPU, interactive work.** Plain PyTorch is simpler. GpuLayer adds a
  control plane, workers and tokens you do not need.
- **You need a hardened multi-tenant service.** Workloads run as normal processes on workers,
  not in a sandbox. Use GpuLayer among people who trust each other.
- **You need what it has not validated yet.** GpuLayer is pre-alpha. Multi-machine training
  and NVIDIA GPUs are implemented but not validated on real hardware, and automatic cloud
  provisioning is simulated. Cloud machines you start yourself can join with `gpulayer join`.
- **You need a large training framework.** The Trainer is small: no callbacks, early stopping
  or FSDP yet. You can still keep another framework and use [`launch()`](../pytorch/launch.md)
  to run its scripts on GpuLayer workers.

Two constraints to know about: `@gl.remote` arguments and results must be JSON, and code
ships as files (the directory of your script), not as pickled Python objects.
