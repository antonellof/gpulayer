# Command line

`GPULayerCLI` builds a command line from the `__init__` signatures of
the Trainer, your model and your DataModule, like Lightning's `LightningCLI`.

```python
# train.py
import gpulayer.pytorch as glt
from my_project import MyModel, MyData

if __name__ == "__main__":
    glt.GPULayerCLI(MyModel, MyData, trainer_defaults={"accelerator": "auto"})
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

- `--optimizer AdamW --optimizer.lr 1e-3` and `--lr_scheduler` define `configure_optimizers`
  from the command line; `--model MyModel` picks a subclass (`subclass_mode_model=True`).
- The resolved configuration is saved as `config.yaml` in the run's log folder
  (`save_config_callback`).
- Subclass the CLI and override Lightning's hooks (`add_arguments_to_parser`,
  `before_instantiate_classes`, ...); `parser.link_arguments("data.num_classes",
  "model.num_classes", apply_on="instantiate")` links sections.

It uses the standard library's `argparse`, not `jsonargparse`: values are converted from the
type annotations, and there are no environment-variable configs. `cli.config` is flat, not
nested under the subcommand.
