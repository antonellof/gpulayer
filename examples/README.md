# Examples

Every example is one file and needs only `torch`. Data is generated or embedded, so nothing is
downloaded. All of them have been run on an Apple Silicon Mac (CPU and MPS).

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
