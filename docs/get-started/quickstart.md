# Quickstart

Train one PyTorch model in your process, with DDP, on two nodes, on a GPU, as a sweep, and on a
server other machines join. About ten minutes; every command was run on an Apple Silicon Mac,
and the CPU steps work on any machine.

## 1. Install

```bash
pip install 'gpulayer[examples]'          # GPUlayer plus PyTorch and NumPy for the examples
git clone https://github.com/antonellof/gpulayer.git && cd gpulayer
gpulayer resources detect
```

The install script, Docker and source installs are in [Installation](installation.md).

## 2. The training file

[`examples/trainer_regression/train.py`](../../examples/trainer_regression/train.py) is a small
regression model:

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

The command-line flags below only change the `Trainer` arguments.

## 3. Train in this process

```bash
python examples/trainer_regression/train.py
```

```text
[gpulayer] training on mps, precision 32-true, 5 epoch(s)
epoch 0 done in 0.21s: train_loss=74.4570 val_loss=2.3970
...
epoch 4 done in 0.057s: train_loss=0.1070 val_loss=0.0798
checkpoint: gpulayer_runs/trainer_regression-train/checkpoints/last.ckpt
```

`accelerator="auto"` picked MPS (CUDA on an NVIDIA machine, else CPU). With one device and no
target, no job is submitted.

## 4. Two processes with DDP

```bash
python examples/trainer_regression/train.py --accelerator cpu --devices 2
```

```text
[gpulayer] job-5cc477d4ffad: train.py on 1 node(s) x 2 cpu process(es), target local
...
[gpulayer] job-5cc477d4ffad succeeded; 2 output file(s) in gpulayer_runs/job-5cc477d4ffad
checkpoint: gpulayer_runs/job-5cc477d4ffad/node0/checkpoints/last.ckpt
```

Now the Trainer submitted a job: a local worker ran two processes with torchrun's environment,
the model trained under DDP on sharded data, and the checkpoint and trained weights came back.

## 5. Two nodes

```bash
python examples/trainer_regression/train.py --accelerator cpu --nodes 2 --devices 2
```

Two local workers each run one node of two processes (world size 4). On real hardware the
workers are different machines ([Remote workers](../guides/remote-workers.md)).

## 6. A GPU through a job

```bash
python examples/trainer_regression/train.py --accelerator mps --target local     # Apple Silicon
python examples/trainer_regression/train.py --accelerator cuda --target local    # NVIDIA
```

Hardware no worker has is rejected before the job starts, with a reason per worker:

```text
[gpulayer] job-e8061b7e6ccc failed: unschedulable: no eligible worker
  (wkr-f50e620bfaef: torch on this worker cannot use CUDA, needs 1 usable cuda device(s),
   worker has 0)
```

## 7. A sweep

[`examples/remote_sweep/sweep.py`](../../examples/remote_sweep/sweep.py) runs a `@gl.remote`
training function for four learning rates as four jobs:

```bash
python examples/remote_sweep/sweep.py --accelerator mps
```

```text
{'lr': 0.3, 'loss': 12.91103, 'device': 'mps'}
{'lr': 0.1, 'loss': 0.00383, 'device': 'mps'}
...
best: {'lr': 0.1, 'loss': 0.00383, 'device': 'mps'}
```

## 8. Inspect what ran

```bash
gpulayer job list
gpulayer job status <job-id>     # state, why it was placed there, per-node tasks
gpulayer job logs <job-id>
gpulayer worker status           # --all includes the finished in-process workers
```

`framework 'torch' not installed` means the Python running the job lacks the package; for
local jobs that is your script's Python.

## 9. A server and a second machine

`gpulayer up` starts a server that keeps running; other machines join it and your laptop sends
it jobs. Each step can run in its own terminal on one machine.

```bash
gpulayer up
```

```text
GPUlayer is up at https://192.168.1.56:8470  (TLS, self-signed certificate pinned by fingerprint)
Add a machine (GPU server, workstation, cloud VM) by running on it:
  gpulayer join https://192.168.1.56:8470 --token glj_... --fingerprint sha256:...
Or from a shell:
  export GPULAYER_URL=https://192.168.1.56:8470 GPULAYER_TOKEN=glc_... GPULAYER_FINGERPRINT=sha256:...
```

Run the join line on a second machine, and the export line where you submit from. Then the same
file trains on both:

```bash
python examples/trainer_regression/train.py --accelerator cpu --nodes 2
```

```text
[gpulayer] job-77b0944e7dba: train.py on 2 node(s) x 1 cpu process(es), target remote
...
[gpulayer] job-77b0944e7dba succeeded; 3 output file(s) in gpulayer_runs/job-77b0944e7dba
```

The nodes connect to each other for the PyTorch rendezvous, so each worker must reach the
others' addresses. More in [Run a server](../guides/self-hosting.md).

## 10. Not only PyTorch

```bash
gpulayer submit --source examples/shell_job \
    --input examples/shell_job/sales.csv:sales.csv -- bash report.sh
python examples/numpy_batch/estimate.py      # a NumPy function as eight parallel jobs
```

See [Commands and scripts](../workloads/commands.md) and [Python functions](../workloads/functions.md).

## Next

- [Trainer and Module](../pytorch/trainer.md)
- [Run an existing DDP script with launch()](../pytorch/launch.md)
- [Performance and comparison](performance-and-comparison.md)
