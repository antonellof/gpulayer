# Examples

Each example is one file. Data is generated or embedded, so nothing is downloaded. All of them
have been run on an Apple Silicon Mac.

## PyTorch training

These need `torch`.

| Example | What it shows | Try |
|---|---|---|
| [`trainer_regression/`](trainer_regression/train.py) | the smallest `Module` + `Trainer` | `python train.py`, then `--accelerator cpu --devices 2`, then `--nodes 2` |
| [`trainer_image_classifier/`](trainer_image_classifier/train.py) | a CNN on generated shapes, accuracy metrics, predictions after training | `python train.py`, `--accelerator cpu --devices 2` |
| [`trainer_char_lm/`](trainer_char_lm/train.py) | a small character-level transformer: cosine LR schedule per step, gradient clipping, mixed precision, resume, text generation | `python train.py`, then `--resume gpulayer_runs/trainer_char_lm-train/checkpoints/last.ckpt --epochs 6` |
| [`runtime_custom_loop/`](runtime_custom_loop/train.py) | your own training loop; `Runtime` handles device, DDP, sharding and precision | `python train.py`, `--devices 2` |
| [`launch_ddp_script/`](launch_ddp_script/train_ddp.py) | a plain PyTorch DDP script made portable with two `launch()` lines | `python train_ddp.py`, `--nodes 2 --devices 2` |
| [`remote_sweep/`](remote_sweep/sweep.py) | `@gl.remote`: a local call, then four learning rates in parallel with `.map()` | `python sweep.py --accelerator mps` |

Every training example accepts `--accelerator`, `--devices`, `--nodes` and `--target`.
`--target remote` sends the run to the workers of a server started with `gpulayer up`
(set `GPULAYER_URL`, `GPULAYER_TOKEN` and `GPULAYER_FINGERPRINT` first).

Outputs (checkpoints, metrics) land in `gpulayer_runs/` in the directory you run from.

## Not only PyTorch

| Example | What it shows | Try |
|---|---|---|
| [`shell_job/`](shell_job) | a shell script as a job, with an input file and an output artifact | see its README |
| [`numpy_batch/`](numpy_batch/estimate.py) | a NumPy function split into eight parallel jobs with `@gl.remote` and `.map()` | `python estimate.py` |
| [`mlx_metal/`](mlx_metal/matmul.py) | MLX on the Apple GPU (no PyTorch); the job requires `mps` and `mlx` | `pip install mlx`, then `python matmul.py` |
| [`gpu_service/`](gpu_service/serve.py) | a PyTorch model served over HTTP from a GPU as a long-running service; run on an Apple M2 Pro (about 1.1 ms per request) | see the file's docstring |
