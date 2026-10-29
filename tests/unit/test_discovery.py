"""Capability discovery with mocked probes (no GPU, torch or Apple hardware required)."""

import json

from gpulayer.contracts import Accelerator, MemoryKind
from gpulayer.gpu.discovery import TorchProbe, detect_capabilities, detect_nvidia, probe_torch


def runner_for(responses):
    def run(cmd, timeout):
        for key, value in responses.items():
            if key in cmd[0] or (len(cmd) > 1 and key in cmd[1]):
                return value
        return 127, ""

    return run


def test_nvidia_smi_parsing():
    row = "NVIDIA A100-SXM4-80GB, 81920, 8.0, 550.54"
    out = f"0, {row}\n1, {row}\n"
    devs, note = detect_nvidia(runner_for({"nvidia-smi": (0, out)}))
    assert note is None and len(devs) == 2
    assert devs[0].memory_gb == 80.0 and devs[0].compute_capability == "8.0"


def test_nvidia_smi_failure_is_reported_not_guessed():
    devs, note = detect_nvidia(runner_for({"nvidia-smi": (9, "")}))
    assert devs == [] and "failed" in note


def test_mps_available_reports_unified_memory():
    caps = detect_capabilities(
        runner=runner_for({"sysctl": (0, str(32 * 1024**3))}),
        torch_probe=lambda: TorchProbe(
            torch_version="2.4.0", mps_built=True, mps_available=True, gloo=True
        ),
    )
    mps = caps.devices(Accelerator.MPS)
    assert len(mps) == 1 and mps[0].memory_kind == MemoryKind.UNIFIED
    assert caps.frameworks == {"torch": "2.4.0"} and caps.launchers == ["torchrun"]


def test_mps_built_but_unavailable():
    caps = detect_capabilities(
        runner=runner_for({}),
        torch_probe=lambda: TorchProbe(torch_version="2.4.0", mps_built=True, mps_available=False),
    )
    assert caps.devices(Accelerator.MPS) == []
    assert any("MPS is not available" in n for n in caps.notes)


def test_missing_torch_is_graceful():
    caps = detect_capabilities(
        runner=runner_for({}), torch_probe=lambda: TorchProbe(error="No module named 'torch'")
    )
    assert caps.accelerators == [] and caps.frameworks == {} and caps.launchers == []
    assert any("torch unavailable" in n for n in caps.notes)


def test_probe_torch_parses_subprocess_output():
    payload = {
        "torch": "2.4.0",
        "cuda_available": False,
        "mps_built": True,
        "mps_available": True,
        "gloo": True,
        "nccl": False,
    }
    tp = probe_torch(runner=lambda cmd, t: (0, json.dumps(payload)))
    assert tp.mps_available and tp.gloo and tp.torch_version == "2.4.0"
    assert probe_torch(runner=lambda cmd, t: (1, "")).error
    assert probe_torch(runner=lambda cmd, t: (0, "garbage")).error


def test_nvidia_visible_without_cuda_runtime_is_not_executable():
    out = "0, NVIDIA L4, 23034, 8.9, 550.54\n"
    caps = detect_capabilities(
        runner=runner_for({"nvidia-smi": (0, out)}),
        torch_probe=lambda: TorchProbe(error="no torch"),
    )
    dev = caps.devices(Accelerator.CUDA)[0]
    assert dev.execution_supported is False


def test_rocm_is_detection_only():
    rocm = json.dumps({"card0": {"Card series": "MI300X"}})
    caps = detect_capabilities(
        runner=runner_for({"rocm-smi": (0, rocm)}), torch_probe=lambda: TorchProbe(error="none")
    )
    dev = caps.devices(Accelerator.ROCM)[0]
    assert dev.execution_supported is False
