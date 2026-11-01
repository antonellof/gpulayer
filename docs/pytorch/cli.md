# Command line

`GPULayerCLI` (also `glt.LightningCLI`) builds a command line from the `__init__` signatures of
the Trainer, your model and your DataModule, like Lightning's `LightningCLI`.

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

- Subcommands: `fit`, `validate`, `test`, `predict`.
- Values follow the type annotation; lists and dicts are JSON.
- Config files are JSON, or YAML with PyYAML, with sections `trainer`, `model`, `data`,
  `seed_everything` and `ckpt_path`. Several `--config` files merge in order; flags win.
- Objects such as callbacks use `class_path` and `init_args`:

```yaml
trainer:
  max_epochs: 30
  callbacks:
    - class_path: gpulayer.pytorch.EarlyStopping
      init_args: {monitor: val_loss, patience: 3}
model:
  lr: 0.003
```

- `seed_everything` defaults to `true` (a recorded random seed); give a number or `false`.
- `run=False` skips subcommands and builds `cli.model`, `cli.datamodule` and `cli.trainer`.

Unlike Lightning's, it does not use `jsonargparse` and cannot link arguments between sections.
