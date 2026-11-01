# Command line

`GPULayerCLI` (also `glt.LightningCLI`) turns a training script into a command line, like
PyTorch Lightning's `LightningCLI`: the arguments come from the `__init__` signatures of the
Trainer, your model and your DataModule.

```python
# train.py
import gpulayer.pytorch as glt
from my_project import MyModel, MyData

if __name__ == "__main__":
    glt.LightningCLI(MyModel, MyData, trainer_defaults={"accelerator": "auto"})
```

```bash
python train.py fit --trainer.max_epochs 10 --model.lr 0.01 --data.batch_size 64
python train.py fit --trainer.devices 4 --trainer.num_nodes 2 --trainer.accelerator cuda
python train.py fit --print_config > experiment.yaml     # the full configuration
python train.py fit --config experiment.yaml --trainer.max_epochs 20
python train.py test --ckpt_path gpulayer_runs/my_project-train/checkpoints/best.ckpt
```

- **Subcommands**: `fit`, `validate`, `test`, `predict`; each calls that Trainer method with
  the model, the DataModule and `--ckpt_path`.
- **Values** are converted with the parameter's type annotation: numbers, `true`/`false`,
  `none`/`null`, and lists or dicts as JSON.
- **Config files** are JSON, or YAML when PyYAML is installed, with the sections `trainer`,
  `model`, `data`, `seed_everything` and `ckpt_path`. Several `--config` files merge in order;
  command-line values override them.
- **Objects** are written as `{"class_path": "module.Class", "init_args": {...}}`, for example
  callbacks and loggers:

```yaml
trainer:
  max_epochs: 30
  callbacks:
    - class_path: gpulayer.pytorch.EarlyStopping
      init_args: {monitor: val_loss, patience: 3}
model:
  lr: 0.003
```

- `seed_everything` defaults to `true` (a seed is picked and recorded in the configuration);
  give a number to fix it or `false` to skip seeding.
- With `run=False` there are no subcommands: the CLI builds `cli.model`, `cli.datamodule` and
  `cli.trainer`, and the script calls the Trainer itself.

It uses only the standard library (Lightning's uses `jsonargparse`), so nested `class_path`
objects are supported but not Lightning's linking of arguments between sections.
