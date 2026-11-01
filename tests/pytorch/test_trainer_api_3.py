"""Lightning parity, part 3: the rest of Lightning's public API found by comparing the two
code bases (utilities, accelerator classes, Module hooks and loading, Trainer members,
SIGTERM checkpoints, init_module, plugins, strategy options).

CPU. The distributed cases start real processes through a launched job (gloo). The
bitsandbytes and Transformer Engine plugins are tested with fake packages (SIMULATED): the
real ones need NVIDIA GPUs."""

import os
import signal
import subprocess
import sys
import textwrap
import types

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.errors import ValidationFailed  # noqa: E402
from gpulayer.pytorch import utilities as U  # noqa: E402


def ds(n=32):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    return TensorDataset(x, x.sum(1, keepdim=True))


class Lin(glt.Module):
    def __init__(self, lr: float = 0.05):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.log("test_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("num_sanity_val_steps", 0)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


# ------------------------------------------------------------------ utilities


def test_grad_norm_and_flops():
    m = nn.Linear(3, 2)
    m(torch.ones(1, 3)).sum().backward()
    norms = U.grad_norm(m, 2)
    assert set(norms) == {"grad_2.0_norm/weight", "grad_2.0_norm/bias", "grad_2.0_norm_total"}
    total = torch.cat([m.weight.grad.flatten(), m.bias.grad]).norm(2)
    assert float(norms["grad_2.0_norm_total"]) == pytest.approx(float(total))
    with pytest.raises(ValueError):
        U.grad_norm(m, 0)
    with torch.device("meta"):
        big = nn.Linear(64, 32)
        x = torch.randn(8, 64)
    assert U.measure_flops(big, lambda: big(x)) == 2 * 8 * 64 * 32


def test_rank_zero_and_data_helpers(monkeypatch):
    calls = []
    fn = U.rank_zero_only(lambda: calls.append(1) or "ran")
    monkeypatch.setenv("RANK", "1")
    assert fn() is None and calls == []
    monkeypatch.setenv("RANK", "0")
    assert fn() == "ran"
    batch = {"x": torch.ones(2), "y": [torch.zeros(1), (torch.ones(1),)], "n": 3}
    moved = U.move_data_to_device(batch, "cpu")
    assert moved["n"] == 3 and isinstance(moved["y"][1], tuple)
    assert U.suggested_max_num_workers(1) >= 1
    with pytest.raises(ValueError):
        U.suggested_max_num_workers(0)
    assert U.is_picklable(1) and not U.is_picklable(lambda: 0)
    assert U.GradClipAlgorithmType.NORM == "NORM" and U.GradClipAlgorithmType.from_str("value")
    assert U.is_shared_filesystem(None)


def test_shared_parameters_are_found_and_tied_again():
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    m[1].weight = m[0].weight
    groups = U.find_shared_parameters(m)
    assert groups == [["0.weight", "1.weight"]]
    m[1].weight = nn.Parameter(m[0].weight.detach().clone())
    U.set_shared_parameters(m, groups)
    assert m[1].weight is m[0].weight


# ---------------------------------------------------------------- accelerators


def test_accelerator_objects_and_gpu_alias(tmp_path):
    t = trainer(tmp_path, accelerator=glt.CPUAccelerator(), max_epochs=1)
    t.fit(Lin(), DataLoader(ds(), batch_size=8))
    assert t.global_step == 4 and t.accelerator == "cpu"
    assert glt.Trainer(accelerator="gpu").accelerator == "cuda"
    assert glt.CPUAccelerator.is_available() and glt.find_usable_cuda_devices(0) == []
    assert glt.CUDAAccelerator.name() == "cuda" and glt.XLAAccelerator.name() == "tpu"


# ------------------------------------------------------------------ Module


class Frozen(Lin):
    """A frozen submodule stays in eval mode; dropout can be kept on in validation."""

    def __init__(self):
        super().__init__()
        self.head = nn.Dropout(0.5)
        self.head.eval()
        self.zeroed = 0

    def on_validation_model_zero_grad(self):
        self.zeroed += 1
        super().on_validation_model_zero_grad()

    def validation_step(self, batch, batch_idx):
        assert not self.training and not self.head.training
        super().validation_step(batch, batch_idx)


def test_eval_and_train_hooks_restore_each_submodule(tmp_path):
    m = Frozen()
    t = trainer(tmp_path, max_epochs=2)
    t.fit(m, DataLoader(ds(), batch_size=8), DataLoader(ds(8), batch_size=8))
    assert m.zeroed == 2  # once per validation inside fit
    assert m.training and not m.head.training  # each submodule back to its own mode


def test_strict_loading_and_model_loaders(tmp_path):
    t = trainer(tmp_path, max_epochs=1)
    t.fit(Lin(), DataLoader(ds(), batch_size=8))

    class Bigger(Lin):
        def __init__(self, lr: float = 0.05):
            super().__init__(lr)
            self.extra = nn.Linear(1, 1)

    with pytest.raises(RuntimeError):
        Bigger.load_from_checkpoint(t.checkpoint_path)

    class Lenient(Bigger):
        def __init__(self, lr: float = 0.05):
            super().__init__(lr)
            self.strict_loading = False

    assert Lenient.load_from_checkpoint(t.checkpoint_path).hparams.lr == 0.05
    with pytest.raises(ValidationFailed, match="train_dataloader"):
        Lin().train_dataloader()
    with pytest.raises(ValidationFailed, match="fit needs training data"):
        trainer(tmp_path).fit(Lin())


def test_configure_sharded_model_is_called(tmp_path):
    class Late(Lin):
        def configure_sharded_model(self):
            self.built = True

    m = Late()
    trainer(tmp_path, max_epochs=1).fit(m, DataLoader(ds(), batch_size=8))
    assert m.built


# ----------------------------------------------------------------- Trainer


class Recording(glt.Logger):
    def __init__(self):
        self.metrics, self.hparams = [], []

    def log_metrics(self, metrics, step=None):
        self.metrics.append(dict(metrics))

    def log_hyperparams(self, params):
        self.hparams.append(dict(params))


def test_log_key_prefix_and_autolog_hparams(tmp_path):
    lg = Recording()
    t = trainer(tmp_path, max_epochs=1, logger=lg, log_key_prefix="trainer/",
                enable_autolog_hparams=False)  # fmt: skip
    t.fit(Lin(), DataLoader(ds(), batch_size=8))
    last = lg.metrics[-1]
    assert "trainer/epoch" in last and "epoch" not in last and "train_loss" in last
    assert lg.hparams == []
    lg2 = Recording()
    trainer(tmp_path, max_epochs=1, logger=lg2).fit(Lin(), DataLoader(ds(), batch_size=8))
    assert lg2.hparams == [{"lr": 0.05}]
    with pytest.raises(ValidationFailed, match="model_registry"):
        glt.Trainer(model_registry="my-model")


def test_trainer_members(tmp_path, capsys):
    t = trainer(tmp_path, max_epochs=1, num_sanity_val_steps=1)
    t.fit(Lin(), DataLoader(ds(), batch_size=8), DataLoader(ds(16), batch_size=8))
    assert t.num_sanity_val_batches == [1]
    assert t.distributed_sampler_kwargs is None and t.device_ids == []
    assert t.enable_validation and not t.evaluating
    assert t.precision_plugin.precision == "32-true" and t.scaler is None
    assert t.progress_bar_callback is None
    t.test(Lin(), DataLoader(ds(16), batch_size=8))
    t.predict(Lin(), DataLoader(ds(24), batch_size=8))
    assert t.num_test_batches == 2 and t.num_predict_batches == [3]
    t.print("from rank zero")
    assert "from rank zero" in capsys.readouterr().out


def test_init_module_dtype_and_empty_init(tmp_path):
    t = glt.Trainer(accelerator="cpu", precision="bf16-true", default_root_dir=tmp_path)
    with t.init_module():
        m = nn.Linear(4, 4)
    assert m.weight.dtype == torch.bfloat16 and torch.get_default_dtype() == torch.float32

    class Five(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3))
            nn.init.constant_(self.w, 5.0)

    with trainer(tmp_path).init_module(empty_init=True):
        skipped = Five()
    assert float(skipped.w.sum()) == 0.0  # nn.init skipped
    assert float(Five().w.sum()) == 15.0


class SigtermAt(glt.Callback):
    def __init__(self, step):
        self.step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == self.step:
            os.kill(os.getpid(), signal.SIGTERM)


def test_sigterm_saves_a_checkpoint_and_resumes_mid_epoch(tmp_path):
    t = trainer(tmp_path, max_epochs=2, callbacks=[SigtermAt(3)])
    with pytest.raises(glt.SIGTERMException):
        t.fit(Lin(), DataLoader(ds(), batch_size=8))
    assert t.received_sigterm and t.state.status == "interrupted"
    ckpt = torch.load(t.checkpoint_path, weights_only=False)
    assert ckpt["global_step"] == 3 and ckpt["batches_done"] == 3
    assert signal.getsignal(signal.SIGTERM) is not t._sigterm_handler  # handler restored
    t2 = trainer(tmp_path, max_epochs=2)
    t2.fit(Lin(), DataLoader(ds(), batch_size=8), ckpt_path=t.checkpoint_path)
    assert t2.global_step == 8  # 1 batch left in epoch 0, then 4 in epoch 1


# ------------------------------------------------------------------ plugins


def test_layer_sync_revert_and_xla_precision():
    m = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    synced = glt.TorchSyncBatchNorm().apply(m)
    assert isinstance(synced[1], nn.SyncBatchNorm)
    back = glt.TorchSyncBatchNorm().revert(synced)
    assert not isinstance(back[1], nn.SyncBatchNorm)
    back.eval()
    assert back(torch.ones(3, 2)).shape == (3, 2)
    assert isinstance(glt.TorchSyncBatchNorm(), glt.LayerSync)
    with pytest.raises(ValidationFailed):
        glt.XLAPrecision("16-mixed")
    assert glt.FSDPPrecision("bf16-mixed").precision == "bf16-mixed"


def fake_bnb(monkeypatch):
    nn_mod = types.SimpleNamespace()

    class Linear4bit(nn.Linear):
        def __init__(self, i, o, bias=True, compute_dtype=None, compress_statistics=False,
                     quant_type="nf4"):  # fmt: skip
            super().__init__(i, o, bias=bias)
            self.quant_type, self.compress = quant_type, compress_statistics

    def Params4bit(data, requires_grad=False, **kw):  # noqa: N802
        return nn.Parameter(data, requires_grad=requires_grad)

    nn_mod.Linear4bit, nn_mod.Params4bit = Linear4bit, Params4bit
    nn_mod.Linear8bitLt = nn.Linear
    nn_mod.Int8Params = Params4bit
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.SimpleNamespace(nn=nn_mod))
    return Linear4bit


def test_bitsandbytes_replaces_linear_layers(tmp_path, monkeypatch):
    with pytest.raises(ValidationFailed, match="mode"):
        glt.BitsandbytesPrecision("int4")
    Q = fake_bnb(monkeypatch)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    plugin = glt.BitsandbytesPrecision("nf4-dq", ignore_modules={"2"})
    plugin.convert_module(model)
    assert isinstance(model[0], Q) and model[0].quant_type == "nf4" and model[0].compress
    assert type(model[2]) is nn.Linear and model[2].weight.dtype == torch.bfloat16
    assert glt.Trainer(plugins=[plugin]).precision == "bf16-true"


def test_transformer_engine_context_wraps_every_forward(tmp_path, monkeypatch):
    entered = []

    class Ctx:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *a):
            return False

    te = types.SimpleNamespace(
        Linear=lambda i, o, bias=True, params_dtype=None: nn.Linear(i, o, bias=bias),
        LayerNorm=lambda n, eps=1e-5, params_dtype=None: nn.LayerNorm(n, eps=eps),
        fp8_autocast=lambda enabled, fp8_recipe: Ctx(),
    )
    recipe_mod = types.SimpleNamespace(DelayedScaling=lambda **kw: ("recipe", kw))
    monkeypatch.setitem(sys.modules, "transformer_engine", types.SimpleNamespace(pytorch=te))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setitem(
        sys.modules, "transformer_engine.common", types.SimpleNamespace(recipe=recipe_mod)
    )
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", recipe_mod)
    plugin = glt.TransformerEnginePrecision(weights_dtype=torch.float32)
    t = trainer(tmp_path, max_epochs=1, plugins=[plugin])
    t.fit(Lin(), DataLoader(ds(), batch_size=8))
    assert len(entered) == 4  # one per training step


# ---------------------------------------------------------------- strategies


def test_strategy_options():
    s = glt.DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True,
                              offload_params_device="nvme", nvme_path="/nvme", pin_memory=True,
                              overlap_comm=False, partition_activations=True)  # fmt: skip
    cfg = s.deepspeed_config("16-mixed", 2, 1.0, 8)
    z = cfg["zero_optimization"]
    assert z["offload_param"]["device"] == "nvme" and z["offload_param"]["nvme_path"] == "/nvme"
    assert z["overlap_comm"] is False and cfg["activation_checkpointing"]["partition_activations"]
    assert cfg["fp16"]["initial_scale_power"] == 16 and cfg["train_micro_batch_size_per_gpu"] == 8
    with pytest.raises(ValidationFailed, match="accumulate_grad_batches"):
        glt.DeepSpeedStrategy(config={"gradient_accumulation_steps": 4}).deepspeed_config(
            "32-true", 1, None, None
        )
    with pytest.raises(ValidationFailed):
        glt.FSDPStrategy(sharding_strategy="EVERYTHING")
    assert glt.FSDPStrategy(auto_wrap_policy={nn.Linear}).should_shard(nn.Linear(1, 1))
    assert glt.ModelParallelStrategy().mesh_shape(8, 4) == (2, 4)
    assert glt.ModelParallelStrategy(tensor_parallel_size=2).mesh_shape(8, 4) == (4, 2)
    with pytest.raises(ValidationFailed):
        glt.ModelParallelStrategy(data_parallel_size=3).mesh_shape(8, 4)
    assert isinstance(glt.Trainer(strategy="fsdp_cpu_offload").strategy, glt.FSDPStrategy)
    assert isinstance(glt.DDPStrategy(), glt.ParallelStrategy)


def run_script(tmp_path, body, timeout=300, name="script.py"):
    script = tmp_path / name
    script.write_text(textwrap.dedent(body))
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    p = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=timeout)  # fmt: skip
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p.stdout


COMMON = """
import json, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import gpulayer.pytorch as glt

def data():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=8)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(8, 16), nn.Linear(16, 8)
    def forward(self, x):
        return self.b(torch.relu(self.a(x)))

class Net(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.block, self.head = Block(), nn.Linear(8, 1)
    def forward(self, x):
        return self.head(self.block(x))
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
"""


@pytest.mark.distributed
def test_fsdp_options_and_a_sharded_checkpoint_resume(tmp_path):
    import json

    out = run_script(
        tmp_path,
        COMMON
        + """
if __name__ == "__main__":
    s = glt.FSDPStrategy(sharding_strategy="SHARD_GRAD_OP", auto_wrap_policy={Block},
                         activation_checkpointing_policy={Block}, state_dict_type="sharded")
    t = glt.Trainer(accelerator="cpu", devices=2, strategy=s, max_epochs=1,
                    log_every_n_steps=0, num_sanity_val_steps=0)
    model = Net()
    t.fit(model, data)
    if glt.is_launcher_process():
        ckpt = t.checkpoint_path
        print("RESULT", json.dumps({"files": sorted(p.name for p in ckpt.iterdir()),
              "ckpt": str(ckpt.resolve()), "step": t.global_step,
              "w": model.head.weight.sum().item()}))
""",
    )
    r = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert "meta.pt" in r["files"] and any(f.endswith(".distcp") for f in r["files"])
    assert r["step"] == 4  # 64 samples / 2 ranks / batch 8
    assert r["w"] != 0.0  # trained weights came back from the sharded folder
    out = run_script(
        tmp_path,
        COMMON
        + f"""
if __name__ == "__main__":
    s = glt.FSDPStrategy(sharding_strategy="HYBRID_SHARD", state_dict_type="sharded")
    t = glt.Trainer(accelerator="cpu", devices=2, strategy=s, max_epochs=2,
                    log_every_n_steps=0, num_sanity_val_steps=0)
    t.fit(Net(), data, ckpt_path={r["ckpt"]!r})
    if glt.is_launcher_process():
        print("RESULT", json.dumps({{"step": t.global_step, "epochs": len(t.history)}}))
""",
        name="resume.py",
    )
    resumed = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert resumed == {"step": 8, "epochs": 2}  # epoch 0 from the folder, then epoch 1


@pytest.mark.distributed
def test_model_parallel_tensor_parallel_on_two_processes(tmp_path):
    out = run_script(
        tmp_path,
        COMMON
        + """
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

class TP(Net):
    def configure_model(self):
        tp = self.device_mesh["tensor_parallel"]
        parallelize_module(self.block, tp, {"a": ColwiseParallel(), "b": RowwiseParallel()})

if __name__ == "__main__":
    s = glt.ModelParallelStrategy(save_distributed_checkpoint=False)
    t = glt.Trainer(accelerator="cpu", devices=2, strategy=s, max_epochs=2,
                    log_every_n_steps=0, num_sanity_val_steps=0)
    model = TP()
    t.fit(model, data)
    h = t.history
    print("RESULT", json.dumps({"first": h[0]["train_loss"], "last": h[-1]["train_loss"],
          "step": t.global_step, "shape": list(model.block.a.weight.shape)}))
""",
    )
    import json

    r = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert r["last"] < r["first"]
    assert r["step"] == 16  # tensor-parallel ranks read the same data: 64 / 8 per epoch
    assert r["shape"] == [16, 8]  # full weights gathered into the launching process


@pytest.mark.distributed
def test_ddp_comm_hook_and_post_local_sgd(tmp_path):
    out = run_script(
        tmp_path,
        COMMON
        + """
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState, post_localSGD_hook)

if __name__ == "__main__":
    s = glt.DDPStrategy(ddp_comm_hook=default_hooks.fp16_compress_hook, timeout=__import__(
        "datetime").timedelta(minutes=5), process_group_backend="gloo")
    t = glt.Trainer(accelerator="cpu", devices=2, strategy=s, max_epochs=2,
                    log_every_n_steps=0, num_sanity_val_steps=0)
    t.fit(Net(), data)
    a = t.history
    s2 = glt.DDPStrategy(ddp_comm_state=PostLocalSGDState(process_group=None, subgroup=None,
                         start_localSGD_iter=2), ddp_comm_hook=post_localSGD_hook,
                         model_averaging_period=2)
    t2 = glt.Trainer(accelerator="cpu", devices=2, strategy=s2, max_epochs=2,
                     log_every_n_steps=0, num_sanity_val_steps=0)
    t2.fit(Net(), data)
    b = t2.history
    print("RESULT", json.dumps([a[0]["train_loss"], a[-1]["train_loss"],
                                b[0]["train_loss"], b[-1]["train_loss"]]))
""",
    )
    import json

    a0, a1, b0, b1 = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert a1 < a0 and b1 < b0


def test_lit_logger_with_a_fake_litlogger(tmp_path, monkeypatch):
    """SIMULATED: litlogger is replaced by a fake that records what the logger sends."""
    sent = {}

    class Series(list):
        def append(self, v, step=None):
            super().append((step, v))

    class Experiment(dict):
        def __init__(self, **kw):
            super().__init__()
            sent["init"] = kw
            self.url = "https://lightning.ai/x"

        def __missing__(self, key):
            self[key] = Series()
            return self[key]

        def finalize(self, status):
            sent["status"] = status

    fake = types.SimpleNamespace(
        Experiment=Experiment,
        Model=lambda path, **kw: ("model", str(path), kw),
        File=lambda p: ("file", p),
    )
    monkeypatch.setitem(sys.modules, "litlogger", fake)
    lg = glt.LitLogger(root_dir=tmp_path, name="exp", teamspace="team", log_model=True)
    t = trainer(tmp_path, max_epochs=2, logger=lg)
    t.fit(Lin(), DataLoader(ds(), batch_size=8))
    exp = lg.experiment
    assert sent["init"]["teamspace"] == "team" and sent["init"]["name"] == "exp"
    assert [s for s, _ in exp["train_loss"]] == [4, 8] and exp["lr"] == "0.05"
    assert exp["exp-model"][0] == "model" and exp["exp-model"][1].endswith("last.ckpt")
    assert sent["status"] == "success" and lg.url.startswith("https://")


# --------------------------------------------------- Lightning's performance tricks


def test_sharding_keeps_every_loader_setting():
    from gpulayer.pytorch.runtime import Runtime

    rt = Runtime("cpu")
    rt.world_size, rt.global_rank = 2, 1  # as in a two-process run

    def init(_):
        return None

    loader = DataLoader(ds(), batch_size=4, shuffle=True, num_workers=2, persistent_workers=True,
                        prefetch_factor=4, worker_init_fn=init, timeout=5)  # fmt: skip
    out = rt.dataloader(loader)
    assert out.sampler.num_replicas == 2 and out.sampler.rank == 1
    assert out.persistent_workers and out.prefetch_factor == 4
    assert out.worker_init_fn is init and out.timeout == 5 and out.num_workers == 2


def test_dataloader_worker_hint_once_and_silenceable(tmp_path, monkeypatch):
    import warnings

    from gpulayer.pytorch import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "_HINTED", set())
    monkeypatch.setattr(U, "suggested_max_num_workers", lambda n: 7)
    with pytest.warns(UserWarning, match="num_workers=7"):
        trainer(tmp_path, max_epochs=1).fit(Lin(), DataLoader(ds(), batch_size=8))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        trainer(tmp_path, max_epochs=1).fit(Lin(), DataLoader(ds(), batch_size=8))
    assert not [w for w in seen if "num_workers" in str(w.message)]  # once per process
    monkeypatch.setattr(trainer_mod, "_HINTED", set())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        U.disable_possible_user_warnings()
        trainer(tmp_path, max_epochs=1).fit(Lin(), DataLoader(ds(), batch_size=8))
    assert not [w for w in seen if "num_workers" in str(w.message)]


def test_tensor_core_hint(monkeypatch):
    """SIMULATED: an Ampere GPU is faked through torch.cuda's query functions."""
    import warnings

    from gpulayer.pytorch import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "_HINTED", {"workers"})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "A100")
    rt = types.SimpleNamespace(device=torch.device("cuda", 0))
    with pytest.warns(UserWarning, match="A100 has Tensor Cores"):
        trainer_mod._performance_hints(rt, [], 1)
    monkeypatch.setattr(trainer_mod, "_HINTED", {"workers"})
    torch.set_float32_matmul_precision("high")
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            trainer_mod._performance_hints(rt, [], 1)
        assert not seen
    finally:
        torch.set_float32_matmul_precision("highest")


def test_deepspeed_in_one_process_gets_a_process_group_and_local_rank(tmp_path, monkeypatch):
    """SIMULATED deepspeed. Found on a Tesla T4: DeepSpeed needs a process group and
    LOCAL_RANK even in one process (else it tries MPI discovery). The Trainer provides both for
    the run and removes them afterwards."""
    import torch.distributed as dist

    seen = {}

    class Engine:
        def __init__(self, module, optimizer):
            self.module, self.optimizer = module, optimizer

        def __call__(self, *a):
            return self.module(*a)

        def backward(self, loss):
            loss.backward()

        def is_gradient_accumulation_boundary(self):
            return True

        def step(self):
            self.optimizer.step()
            self.optimizer.zero_grad()

    def initialize(model, optimizer, config):
        seen["group"] = dist.is_initialized() and dist.get_world_size() == 1
        seen["local_rank"] = os.environ.get("LOCAL_RANK")
        return Engine(model, optimizer), optimizer, None, None

    monkeypatch.setitem(sys.modules, "deepspeed", types.SimpleNamespace(initialize=initialize))
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    t = trainer(tmp_path, max_epochs=1, strategy="deepspeed_stage_2")
    t.fit(Lin(), DataLoader(ds(), batch_size=8))
    assert seen == {"group": True, "local_rank": "0"} and t.global_step == 4
    assert "LOCAL_RANK" not in os.environ and not dist.is_initialized()
    assert glt.is_launcher_process()


def test_sharded_checkpoint_folders_from_several_nodes_are_merged(tmp_path):
    """Each node uploads only its own shard files; the launching process puts them together."""
    out = tmp_path / "job"
    for node, files in (
        ("node0", [".metadata", "meta.pt", "__0_0.distcp"]),
        ("node1", ["__1_0.distcp"]),
    ):
        folder = out / node / "checkpoints" / "last.ckpt"
        folder.mkdir(parents=True)
        for f in files:
            (folder / f).write_text(node)
    t = trainer(tmp_path)
    t._load_outputs(out, Lin())  # fake shards do not load: metrics only, but merged
    merged = out / "last.ckpt"
    assert sorted(p.name for p in merged.iterdir()) == [
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
        "meta.pt",
    ]
    assert t.checkpoint_path == merged
