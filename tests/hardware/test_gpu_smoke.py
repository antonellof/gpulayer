"""Hardware-gated smoke tests. They SKIP unless the hardware and torch backend exist.

Run explicitly with:  pytest -m "hardware_mps or hardware_cuda" tests/hardware
"""

import pytest

import gpulayer as gl
from gpulayer.contracts import Accelerator, JobState
from gpulayer.gpu.discovery import detect_capabilities

CAPS = detect_capabilities()

KERNEL = """
import torch
dev = torch.device("{dev}")
a = torch.randn(512, 512, device=dev)
b = (a @ a.T).sum().item()
print("device", dev, "ok", b == b)
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GPULAYER_HOME", str(tmp_path / "home"))
    with gl.Client() as c:
        yield c


@pytest.mark.hardware_mps
@pytest.mark.skipif(not CAPS.devices(Accelerator.MPS), reason="no PyTorch MPS device")
def test_mps_independent_job(client):
    job = client.submit(gl.command(["python", "-c", KERNEL.format(dev="mps")], accelerator="mps"))
    st = job.wait(120)
    assert st.state == JobState.SUCCEEDED, (st.error, job.logs_text())
    assert "device mps ok True" in job.logs_text()


@pytest.mark.hardware_cuda
@pytest.mark.skipif(
    not [d for d in CAPS.devices(Accelerator.CUDA) if d.execution_supported],
    reason="no usable CUDA device",
)
def test_cuda_independent_job(client):
    job = client.submit(gl.command(["python", "-c", KERNEL.format(dev="cuda")], accelerator="cuda"))
    st = job.wait(120)
    assert st.state == JobState.SUCCEEDED, (st.error, job.logs_text())
    assert "device cuda ok True" in job.logs_text()
