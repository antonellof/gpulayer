"""Automatic resume end to end: a real worker and real processes on this machine. The
preemption is a SIGTERM the script sends itself; no cloud VM is reclaimed. CPU, gloo for the
group of two machines (two workers on one host)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")

SCRIPT = """
    import json, os, signal, sys
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import gpulayer.pytorch as glt

    class Stop(glt.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            first = os.environ.get("GPULAYER_ATTEMPT", "1") == "1"
            if first and trainer.global_step == STOP:
                print("TERM at", trainer.global_step, flush=True)
                os.kill(os.getpid(), signal.SIGTERM)

    class Model(glt.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = nn.Linear(4, 1)
            self.first_step = None
        def training_step(self, batch, batch_idx):
            if self.first_step is None:
                self.first_step = self.trainer.global_step
                print("STARTED attempt", os.environ.get("GPULAYER_ATTEMPT"), "at step",
                      self.first_step, flush=True)
            return nn.functional.mse_loss(self.net(batch[0]), batch[1])
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    if __name__ == "__main__":
        x = torch.randn(40, 4)
        loader = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=4)
        t = glt.Trainer(max_epochs=3, accelerator="cpu", log_every_n_steps=0,
                        num_sanity_val_steps=0, callbacks=[Stop()], DEVICES,
                        launch_options={"max_attempts": ATTEMPTS})
        t.fit(Model(), loader)
        print("RESULT", json.dumps({"step": t.global_step, "attempts": t.launch_result.status.attempts
                                    if t.launch_result else None}), flush=True)
"""


def run(tmp_path, devices: str, stop: int, attempts: int = 3):
    script = tmp_path / "train.py"
    script.write_text(
        textwrap.dedent(SCRIPT)
        .replace("DEVICES", devices)
        .replace("ATTEMPTS", str(attempts))
        .replace("STOP", str(stop))
    )
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path / "home")}
    for k in ("GPULAYER_URL", "GPULAYER_JOB_ID", "LOCAL_RANK", "WORLD_SIZE", "RANK"):
        env.pop(k, None)
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.integration
def test_preempted_job_resumes_from_the_kept_checkpoint(tmp_path):
    p = run(tmp_path, "target='local'", 12)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "STARTED attempt 1 at step 0" in out
    assert "STARTED attempt 2 at step 12" in out, out[-3000:]
    result = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert result["step"] == 30
    assert "::gpulayer-checkpoint::" not in out


@pytest.mark.integration
def test_preempted_group_restarts_from_the_kept_checkpoint(tmp_path):
    # 5 steps per epoch on each of 2 ranks. Ranks agree to stop at the end of the epoch (no log
    # step in this run), so the checkpoint is the one after step 10.
    p = run(tmp_path, "devices=2", 7)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "STARTED attempt 2 at step 10" in out, out[-3000:]
    result = json.loads(out.split("RESULT ", 1)[1].splitlines()[0])
    assert result["step"] == 15


@pytest.mark.integration
def test_without_attempts_a_preempted_job_fails(tmp_path):
    p = run(tmp_path, "target='local'", 12, attempts=1)
    assert p.returncode != 0
