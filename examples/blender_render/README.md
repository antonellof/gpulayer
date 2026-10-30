# Blender rendering on a remote GPU

`render.sh` renders frames of a scene that `render.py` builds in Blender, so the job needs no
input files. Frames come back as outputs.

Install Blender on the GPU worker, and give the worker a label so that render jobs only go to
machines that have it:

```bash
# on the worker
gpulayer join https://SERVER:8470 --fingerprint sha256:... --label blender=4.5
```

Submit from anywhere that can reach the server:

```bash
gpulayer submit --target remote --accelerator cuda --gpus 1 --label blender=4.5 \
  --source examples/blender_render -- ./render.sh OPTIX 8 64
gpulayer job result JOB_ID --download frames/
```

`OPTIX` uses the RT cores of RTX and newer cards and also runs on a T4; `CUDA` works on any
CUDA GPU; `METAL` is the Apple Silicon GPU (`--accelerator mps`); `CPU` renders without the GPU,
for comparison. The first Metal or OptiX render compiles kernels (over a minute once). If Blender cannot use the requested
GPU device the job fails and says which devices it found; it does not fall back to the CPU.

To spread a long animation over several GPUs, submit one job per frame range
(`./render.sh OPTIX 25 64 1`, `./render.sh OPTIX 25 64 26`, ...): GPUlayer places each on a
free GPU.
