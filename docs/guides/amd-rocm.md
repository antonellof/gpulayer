# AMD ROCm

**Status: implemented, not run on AMD hardware.** Everything below comes from the code, tests with
simulated devices, and AMD's documentation. A real ROCm wheel (torch 2.15.0 nightly, ROCm 7.2) was
tested on a machine with no AMD GPU: it reports `torch.version.hip` and the `nccl` backend, and the
Trainer's loop tests pass on its CPU. If you run it on a GPU, the
[validation](#validating-a-machine) tells us what works.

## Requirements

An AMD GPU that ROCm supports, the `amdgpu` driver, and a ROCm build of PyTorch in the worker's
Python. AMD's [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
lists the GPUs. AMD's [PyTorch instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
list the install commands and the `rocm/pytorch` image. Your user must be in the `video` and
`render` groups.

```bash
python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"
gpulayer resources detect
```

A ROCm build sets `torch.version.hip` and drives AMD GPUs through the `torch.cuda` API, so model
code that uses `torch.cuda` or `.cuda()` runs unchanged. GPUlayer reports them as `rocm` with vendor
AMD, never as CUDA, and takes their names and memory from torch. Without a ROCm build, it prints a
note and PyTorch jobs are not placed there.

## Training

```python
trainer = glt.Trainer(accelerator="rocm", devices=2, max_epochs=10)
```

`accelerator="gpu"` and `"auto"` also pick AMD GPUs on a ROCm build. To send a run from a machine
without one, say `accelerator="rocm"`: there `"gpu"` means CUDA. Jobs get their GPUs through
`HIP_VISIBLE_DEVICES`.

## Containers

```bash
docker build -t gpulayer:rocm --build-arg BASE=rocm/pytorch:latest --build-arg EXTRAS=base-torch .
docker run --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G gpulayer:rocm ...
```

Built on x86 with `rocm/pytorch:latest` (torch 2.14.0, ROCm 10.1) and started without a GPU; not run
on one.

## What is different from NVIDIA

- Several GPUs use the `nccl` backend, which is RCCL on a ROCm build. The ROCm wheel includes that
  backend, and AMD's pages name RCCL, but do not say that `backend="nccl"` works: the validation
  checks it, and gloo as well.
- GPU utilisation and memory in worker metrics and the dashboard are NVIDIA only.
- GPUs in a machine are not ranked by interconnect, as they are on NVIDIA, because AMD does not
  publish the output of `amd-smi topology`.
- cuDNN autotuning is not an AMD feature, and neither are `TransformerEnginePrecision` and
  `to_tensorrt`.

## Validating a machine

From a copy of the repository, on the machine or in a `rocm/pytorch` container:

```bash
bash scripts/validate_rocm.sh 2>&1 | tee validate.log
```

It trains in each precision, checks checkpoints and `HIP_VISIBLE_DEVICES`, and with two GPUs runs
DDP on both backends. It also saves what `amd-smi` and `rocm-smi` print in `rocm-output/`, which is
how GPUlayer can later read memory, topology and live metrics without guessing. Keep that folder.
