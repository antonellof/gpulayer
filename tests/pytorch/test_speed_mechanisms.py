"""One test per speed mechanism in the table in docs/pytorch/from-lightning.md ("Speed
features"): each checks the behaviour, not only that a name exists. CPU, one process, except
where noted; the GPU-side effects were measured on hardware (benchmarks/results/)."""

import types

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import gpulayer.pytorch as glt  # noqa: E402
from gpulayer.pytorch.runtime import Runtime  # noqa: E402


def data(n=64, batch=8):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=batch)


class Lin(glt.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.lin(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self.lin(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def trainer(tmp_path, **kw):
    kw.setdefault("accelerator", "cpu")
    kw.setdefault("log_every_n_steps", 0)
    kw.setdefault("num_sanity_val_steps", 0)
    kw.setdefault("enable_checkpointing", False)
    return glt.Trainer(default_root_dir=tmp_path, **kw)


def test_only_overridden_hooks_are_called(tmp_path):
    """Hooks nobody overrides cost nothing per step: they are not in the call lists."""

    class OneHook(glt.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            pass

    t = trainer(tmp_path, max_epochs=1, callbacks=[OneHook()])
    t.fit(Lin(), data())
    cache = t._hook_cache
    assert len(cache["on_train_batch_end"]) == 1
    assert cache["on_train_batch_start"] == [] and cache["on_before_backward"] == []


def test_logged_values_are_not_read_from_the_device_every_step(tmp_path, monkeypatch):
    """Reading a logged tensor (.item(), float()) forces a host-device sync; the count must not
    grow with the number of steps."""
    reads = {"n": 0}
    item, to_float = torch.Tensor.item, torch.Tensor.__float__

    def counting_item(self):
        reads["n"] += 1
        return item(self)

    def counting_float(self):
        reads["n"] += 1
        return to_float(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    monkeypatch.setattr(torch.Tensor, "__float__", counting_float)
    counts = []
    for n in (32, 256):  # 4 and 32 steps per epoch
        reads["n"] = 0
        trainer(tmp_path, max_epochs=1).fit(Lin(), data(n))
        counts.append(reads["n"])
    assert counts[0] == counts[1], counts


def test_evaluation_runs_under_inference_mode(tmp_path):
    seen = []

    class Check(Lin):
        def validation_step(self, batch, batch_idx):
            seen.append(torch.is_inference_mode_enabled())
            super().validation_step(batch, batch_idx)

    trainer(tmp_path).validate(Check(), data(16))
    assert seen and all(seen)


def test_gradients_are_freed_not_zeroed(tmp_path):
    """zero_grad(set_to_none=True): no memset per parameter, and memory is released."""
    seen = []

    class Check(Lin):
        def training_step(self, batch, batch_idx):
            if batch_idx > 0:
                seen.append(all(p.grad is None for p in self.parameters()))
            return super().training_step(batch, batch_idx)

    trainer(tmp_path, max_epochs=1).fit(Check(), data())
    assert seen and all(seen)


def test_gradient_exchange_only_on_optimizer_steps(tmp_path, monkeypatch):
    """With accumulation, DDP's all-reduce is skipped (no_sync) on the batches that do not step."""
    flags = []
    real = Runtime.no_sync

    def record(self, model, enabled=True):
        flags.append(enabled)
        return real(self, model, enabled)

    monkeypatch.setattr(Runtime, "no_sync", record)
    trainer(tmp_path, max_epochs=1, accumulate_grad_batches=2).fit(Lin(), data(32))
    assert flags == [True, False, True, False]  # skip, exchange, skip, exchange


def test_ddp_options_reach_distributed_data_parallel():
    kw = glt.DDPStrategy(static_graph=True, gradient_as_bucket_view=True).ddp_kwargs()
    assert kw == {"static_graph": True, "gradient_as_bucket_view": True}


def test_each_rank_gets_its_share_of_cpu_threads(monkeypatch):
    """GPU jobs: each process gets cpu_cores / processes threads; CPU jobs: one per rank
    (measured faster). Users can override it with env."""
    from gpulayer.contracts import Accelerator, DistributedJob, JobSpec, ResourceRequest
    from gpulayer.worker.agent import WorkerAgent

    agent = WorkerAgent.__new__(WorkerAgent)
    agent.python = "/usr/bin/python3"
    agent.caps = types.SimpleNamespace(cpu_cores=16)

    def env_for(accel, gpus, env=None):
        spec = JobSpec(
            workload=DistributedJob(entrypoint="t.py", num_workers=1, procs_per_worker=4),
            resources=ResourceRequest(accelerator=accel, gpu_count=gpus),
            env=env or {},
        )
        a = types.SimpleNamespace(spec=spec, job_id="j", task_id="t", attempt=1, node_rank=0,
                                  devices=[0, 1, 2, 3] if gpus else None)  # fmt: skip
        return agent._environment(a, __import__("pathlib").Path("/tmp"))[0]

    assert env_for(Accelerator.CUDA, 4)["OMP_NUM_THREADS"] == "4"
    assert env_for(Accelerator.CPU, 0)["OMP_NUM_THREADS"] == "1"
    assert env_for(Accelerator.CUDA, 4, {"OMP_NUM_THREADS": "2"})["OMP_NUM_THREADS"] == "2"


def test_the_autocast_context_is_built_once():
    """torch.autocast asks the driver for the device count when built (0.5 ms per step
    measured on an RTX 3070 host): GPUlayer reuses one instance."""
    rt = Runtime("cpu", "bf16-mixed")
    assert rt.autocast() is rt.autocast()
    with rt.autocast():
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_compiled_model_trains_compiled(tmp_path):
    model = Lin()
    compiled = torch.compile(model, backend="eager")
    t = trainer(tmp_path, max_epochs=1)
    t.fit(compiled, data())
    assert t.global_step == 8 and getattr(model, "_compiled_call_impl", None) is not None


def test_checkpoints_written_in_the_background(tmp_path):
    """AsyncCheckpointIO: training does not wait for the disk."""
    io = glt.AsyncCheckpointIO()
    t = trainer(tmp_path, max_epochs=2, enable_checkpointing=True, plugins=[io])
    t.fit(Lin(), data())
    assert t.checkpoint_path is not None and t.checkpoint_path.is_file()
