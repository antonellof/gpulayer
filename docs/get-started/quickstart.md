# Quickstart

This walkthrough takes about fifteen minutes. You run a shell command and a Python function as
jobs, then train one PyTorch model in several places: your current process, two processes with
DDP, two nodes and a GPU job. Every command was run on an Apple Silicon Mac; the CPU steps work
on any machine.

## 1. Install

The examples live in the repository, so clone it and install with the examples extra:

```bash
git clone https://github.com/antonellof/gpulayer.git && cd gpulayer
uv venv && source .venv/bin/activate
uv pip install -e '.[dev,examples]'
gpulayer resources detect
```

To use GPUlayer without the examples, the
[install script](installation.md#with-the-install-script) or the
[Docker image](installation.md#with-docker) is enough.

## 2. Run a command as a job

```bash
gpulayer submit --source examples/shell_job \
    --input examples/shell_job/sales.csv:sales.csv -- bash report.sh
```

```text
submitted job-88eeb3960503 (independent, target local)
running on my-mac, accelerator=cpu
east,99.99
north,162.75
south,90.00
job-88eeb3960503 succeeded
```

The CLI started a local worker, shipped the directory and the input file to it, ran the script
and streamed its output. Any executable works the same way. See
[Commands and scripts](../workloads/commands.md).

## 3. Run Python functions in parallel

[`examples/numpy_batch/estimate.py`](../../examples/numpy_batch/estimate.py) splits a NumPy job
into eight chunks with `@gl.remote` and `.map()`:

```bash
python examples/numpy_batch/estimate.py
```

```text
pi ~= 3.14160 from 40,000,000 samples in 8 jobs
```

On an Apple Silicon Mac (the `examples` extra installs MLX there), the same pattern runs on the
GPU:

```bash
python examples/mlx_metal/matmul.py
```

```text
{'n': 4096, 'cpu_s': 0.0637, 'gpu_s': 0.0316, 'gpu_gflops': 4349.9}
```

## 4. Look at the training file

[`examples/trainer_regression/train.py`](../../examples/trainer_regression/train.py) is a small
regression model written against `gpulayer.pytorch`:

```python
import gpulayer.pytorch as glt

class Regressor(glt.Module):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

trainer = glt.Trainer(max_epochs=args.epochs, accelerator=args.accelerator,
                      devices=args.devices, num_nodes=args.nodes, target=args.target)
trainer.fit(model, train_loader, val_loader)
```

The command-line flags only change the `Trainer` arguments.

## 5. Train in this process

```bash
python examples/trainer_regression/train.py
```

```text
[gpulayer] training on mps, precision 32-true, 5 epoch(s)
epoch 0 done in 0.21s: train_loss=74.4570 val_loss=2.3970
...
epoch 4 done in 0.057s: train_loss=0.1070 val_loss=0.0798
final metrics: {'train_loss': 0.107, 'val_loss': 0.0798, 'epoch': 4, 'epoch_seconds': 0.057}
checkpoint: gpulayer_runs/trainer_regression-train/checkpoints/last.ckpt
```

`accelerator="auto"` picked MPS on a Mac. It picks CUDA on an NVIDIA machine and CPU otherwise.
No job was submitted: with one device and no target, the Trainer runs in your process like any
training loop.

## 6. Same file, two processes with DDP

```bash
python examples/trainer_regression/train.py --accelerator cpu --devices 2
```

```text
[gpulayer] job-5cc477d4ffad: train.py on 1 node(s) x 2 cpu process(es), target local
epoch 0 done in 0.05s: train_loss=... val_loss=...
...
[gpulayer] job-5cc477d4ffad succeeded; 2 output file(s) in gpulayer_runs/job-5cc477d4ffad
final metrics: {'train_loss': 0.5561, 'val_loss': 0.4698, 'epoch': 4, 'epoch_seconds': 0.047}
checkpoint: gpulayer_runs/job-5cc477d4ffad/node0/checkpoints/last.ckpt
```

This time the Trainer submitted a job. A local worker ran the script under `torchrun` with two
processes, the Trainer wrapped the model in DDP and sharded the data, and the checkpoint came
back to `gpulayer_runs/`. Your local `model` object now holds the trained weights.

## 7. Two nodes

```bash
python examples/trainer_regression/train.py --accelerator cpu --nodes 2 --devices 2
```

GPUlayer starts two local workers, places one node on each, publishes a rendezvous address
from node 0, and launches torchrun on both. World size is 4. On real hardware the workers would
be different machines; see [Remote workers](../guides/remote-workers.md).

## 8. Use the GPU through a job

```bash
python examples/trainer_regression/train.py --accelerator mps --target local     # Apple Silicon
python examples/trainer_regression/train.py --accelerator cuda --target local    # NVIDIA
```

Ask for hardware that no worker has and the job is rejected before it starts, with a reason per
worker:

```text
[gpulayer] job-e8061b7e6ccc failed: unschedulable: no eligible worker
  (wkr-f50e620bfaef: torch on this worker cannot use CUDA, needs 1 usable cuda device(s),
   worker has 0)
```

## 9. A PyTorch sweep

[`examples/remote_sweep/sweep.py`](../../examples/remote_sweep/sweep.py) decorates a training
function with `@gl.remote` and runs four learning rates as four jobs:

```bash
python examples/remote_sweep/sweep.py --accelerator mps
```

```text
plain local call: {'lr': 0.1, 'loss': 0.06246, 'device': 'cpu'}
{'lr': 0.3, 'loss': 12.91103, 'device': 'mps'}
{'lr': 0.1, 'loss': 0.00383, 'device': 'mps'}
{'lr': 0.03, 'loss': 0.02718, 'device': 'mps'}
{'lr': 0.01, 'loss': 0.06529, 'device': 'mps'}
best: {'lr': 0.1, 'loss': 0.00383, 'device': 'mps'}
```

## 10. Inspect what ran

```bash
gpulayer job list
gpulayer job status <job-id>     # state, why it was placed where it was, per-node tasks
gpulayer job logs <job-id>
gpulayer job events <job-id>
gpulayer worker status
```

If a job is rejected with `framework 'numpy' not installed` (or `mlx`, `torch`), the Python
that runs the job does not have that package. For local jobs that is the Python running your
script, so install it there (`uv pip install -e '.[dev,examples]'` covers every example), then
run the command again.

`gpulayer worker status` lists workers that are online. Each script run starts in-process
workers that go offline when it ends; `--all` shows those too, and `gpulayer gc --older-than 1d`
removes old ones.

## 11. Optional: a server and a second machine

Start a server on one machine. It also runs jobs on its own hardware:

```bash
gpulayer up
```

It prints three things: a `gpulayer join ...` line, a `gl.connect(...)` line and an
`export ...` line. Paste the join line on a second machine (a GPU box or a cloud VM):

```bash
gpulayer join https://192.168.1.56:8470 --token glj_... --fingerprint sha256:...
```

Then, on your laptop, paste the export line and run the same training file. Jobs now go to the
server's workers:

```bash
export GPULAYER_URL=https://192.168.1.56:8470 GPULAYER_TOKEN=glc_... GPULAYER_FINGERPRINT=sha256:...
python examples/trainer_regression/train.py --accelerator cpu --nodes 2   # one node per machine
gpulayer worker status
```

More in [Run a server](../guides/self-hosting.md).

## Next

- [Any GPU framework](../workloads/frameworks.md) for JAX, MLX, CuPy and others
- [Trainer and Module](../pytorch/trainer.md) for the full training API
- [Run any script with launch()](../pytorch/launch.md) if you already have a training script
- [Core concepts](concepts.md) for how placement works
