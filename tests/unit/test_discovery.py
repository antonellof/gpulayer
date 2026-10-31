"""Capability discovery with mocked probes (no GPU, torch or Apple hardware required)."""

import json

from gpulayer.contracts import Accelerator, MemoryKind
from gpulayer.gpu.discovery import (
    TorchProbe,
    detect_capabilities,
    detect_location,
    detect_nvidia,
    gce_region,
    probe_torch,
)


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
        c_compiler=lambda: None,
    )
    mps = caps.devices(Accelerator.MPS)
    assert len(mps) == 1 and mps[0].memory_kind == MemoryKind.UNIFIED
    assert caps.frameworks == {"torch": "2.4.0", "torch.mps": "2.4.0"}
    assert caps.launchers == ["direct", "torchrun"]


def test_mps_built_but_unavailable(monkeypatch):
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    caps = detect_capabilities(
        runner=runner_for({}),
        torch_probe=lambda: TorchProbe(torch_version="2.4.0", mps_built=True, mps_available=False),
    )
    assert caps.devices(Accelerator.MPS) == []
    assert "torch.mps" not in caps.frameworks


def test_missing_torch_is_graceful(monkeypatch):
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
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


def test_nvidia_usable_without_torch_cuda(monkeypatch):
    """The driver alone makes CUDA usable; torch's CUDA support is a separate capability."""
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    out = "0, NVIDIA L4, 23034, 8.9, 550.54\n"
    caps = detect_capabilities(
        runner=runner_for({"nvidia-smi": (0, out)}),
        torch_probe=lambda: TorchProbe(torch_version="2.4.0", packages={"cupy-cuda12x": "13.3"}),
    )
    dev = caps.devices(Accelerator.CUDA)[0]
    assert dev.execution_supported is True
    assert caps.frameworks["cupy"] == "13.3" and "torch.cuda" not in caps.frameworks
    assert any("no CUDA support" in n for n in caps.notes)


def test_torch_cuda_capability_reported(monkeypatch):
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    out = "0, NVIDIA L4, 23034, 8.9, 550.54\n"
    caps = detect_capabilities(
        runner=runner_for({"nvidia-smi": (0, out)}),
        torch_probe=lambda: TorchProbe(torch_version="2.4.0", cuda_available=True),
    )
    assert caps.frameworks["torch.cuda"] == "2.4.0"


def test_apple_gpu_without_torch(monkeypatch):
    """An Apple Silicon Mac offers its GPU (Metal) to MLX etc. even without torch."""
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: True)
    caps = detect_capabilities(
        runner=runner_for({"sysctl": (0, str(16 * 1024**3))}),
        torch_probe=lambda: TorchProbe(error="no torch", packages={"mlx": "0.20.0"}),
    )
    (gpu,) = caps.devices(Accelerator.MPS)
    assert gpu.memory_kind == MemoryKind.UNIFIED and "Metal" in gpu.name
    assert caps.frameworks == {"mlx": "0.20.0"}


def test_rocm_is_detection_only():
    rocm = json.dumps({"card0": {"Card series": "MI300X"}})
    caps = detect_capabilities(
        runner=runner_for({"rocm-smi": (0, rocm)}), torch_probe=lambda: TorchProbe(error="none")
    )
    dev = caps.devices(Accelerator.ROCM)[0]
    assert dev.execution_supported is False


def _four_gpus(monkeypatch, **kw):
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    row = "NVIDIA L4, 23034, 8.9, 550.54"
    out = "".join(f"{i}, {row}\n" for i in range(4))
    return detect_capabilities(
        runner=runner_for({"nvidia-smi": (0, out)}),
        torch_probe=lambda: TorchProbe(torch_version="2.4.0", cuda_available=True),
        **kw,
    )


def test_gpus_splits_a_machine_between_workers(monkeypatch):
    """Two workers on one machine offer disjoint GPUs and get one slot per GPU they offer."""
    a = _four_gpus(monkeypatch, gpus=[0, 1])
    b = _four_gpus(monkeypatch, gpus=[2, 3])
    assert [d.index for d in a.devices(Accelerator.CUDA)] == [0, 1]
    assert [d.index for d in b.devices(Accelerator.CUDA)] == [2, 3]
    assert a.max_concurrent_tasks == 2 and b.max_concurrent_tasks == 2
    assert _four_gpus(monkeypatch).max_concurrent_tasks == 4


def test_gpus_none_offers_cpu_only(monkeypatch):
    caps = _four_gpus(monkeypatch, gpus=[])
    assert caps.devices(Accelerator.CUDA) == [] and "GPUs not offered" in caps.notes


def test_gpus_rejects_unknown_index(monkeypatch):
    import pytest

    with pytest.raises(ValueError, match=r"GPU index \[7\] not found"):
        _four_gpus(monkeypatch, gpus=[0, 7])


def test_gpus_cli_parsing():
    import pytest

    from gpulayer.cli.main import _gpus

    assert _gpus(None) is None and _gpus("none") == [] and _gpus("1, 0,1") == [0, 1]
    with pytest.raises(SystemExit):
        _gpus("first")


def test_up_tokens_can_come_from_the_environment(tmp_path, monkeypatch):
    import json

    import pytest

    from gpulayer.cli.main import _up_tokens

    client, join = "glc_" + "a" * 40, "glj_" + "b" * 40
    monkeypatch.setenv("GPULAYER_UP_CLIENT_TOKEN", client)
    monkeypatch.setenv("GPULAYER_UP_JOIN_TOKEN", join)
    assert _up_tokens(tmp_path) == (client, join)
    assert json.loads((tmp_path / "up-tokens.json").read_text())["join_token"] == join
    monkeypatch.setenv("GPULAYER_UP_JOIN_TOKEN", "short")
    with pytest.raises(SystemExit, match="at least 32"):
        _up_tokens(tmp_path)
    monkeypatch.delenv("GPULAYER_UP_JOIN_TOKEN")
    with pytest.raises(SystemExit, match="set both"):
        _up_tokens(tmp_path)


def test_join_waits_for_an_unreachable_server_but_not_a_wrong_token(monkeypatch):
    import pytest

    from gpulayer.cli import main as cli
    from gpulayer.errors import Unauthorized

    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionRefusedError("refused")
        return "channel"

    assert cli._register_when_reachable(flaky, "https://x") == "channel" and len(calls) == 3

    def rejected():
        raise Unauthorized("bad token")

    with pytest.raises(Unauthorized):
        cli._register_when_reachable(rejected, "https://x")
    with pytest.raises(ConnectionRefusedError):
        cli._register_when_reachable(
            lambda: (_ for _ in ()).throw(ConnectionRefusedError()), "u", wait=0
        )


def test_jax_and_cupy_cuda_only_when_they_run_on_the_gpu(monkeypatch):
    """Installed is not enough: 'jax.cuda' / 'cupy.cuda' come from a probe that uses the GPU."""
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    smi = "0, Tesla T4, 15360, 7.5, 580.65\n"
    calls = []

    def run(cmd, timeout):
        if cmd[0] == "nvidia-smi":
            return 0, smi
        if "ACCEL" in cmd[-1]:
            calls.append(cmd)
            return 0, 'noise\nACCEL {"jax.cuda": "0.7.2"}\n'  # CuPy failed on the GPU
        return 127, ""

    pkgs = {"jax": "0.7.2", "jaxlib": "0.7.2", "cupy-cuda12x": "13.6.0"}
    caps = detect_capabilities(runner=run, torch_probe=lambda: TorchProbe(packages=pkgs))
    assert caps.frameworks["jax.cuda"] == "0.7.2" and "cupy.cuda" not in caps.frameworks
    assert caps.frameworks["cupy"] == "13.6.0"  # still installed, just not usable on the GPU
    assert len(calls) == 1

    calls.clear()  # neither installed: no probe at all
    caps = detect_capabilities(runner=run, torch_probe=lambda: TorchProbe(packages={}))
    assert "jax.cuda" not in caps.frameworks and calls == []
    # No NVIDIA driver: no probe either
    caps = detect_capabilities(runner=runner_for({}), torch_probe=lambda: TorchProbe(packages=pkgs))
    assert "jax.cuda" not in caps.frameworks


# ------------------------------------------------------------------ location


def test_gce_region_from_zone():
    assert gce_region("europe-west1-b") == "europe-west1"
    assert gce_region("us-central1-a") == "us-central1"
    assert gce_region("nozone") is None


def test_location_is_not_asked_off_gce(monkeypatch):
    monkeypatch.setattr("gpulayer.gpu.discovery._on_gce", lambda: False)

    def boom(*a, **k):
        raise AssertionError("metadata server must not be contacted")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert detect_location() == (None, None)


def test_location_from_gce_metadata(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"projects/123456/zones/europe-west4-a"

    seen = {}

    def urlopen(req, timeout):
        seen["header"] = req.get_header("Metadata-flavor")
        return Resp()

    monkeypatch.setattr("gpulayer.gpu.discovery._on_gce", lambda: True)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert detect_location() == ("europe-west4", "europe-west4-a")
    assert seen["header"] == "Google"


def test_location_metadata_failure_is_not_guessed(monkeypatch):
    def urlopen(req, timeout):
        raise OSError("unreachable")

    monkeypatch.setattr("gpulayer.gpu.discovery._on_gce", lambda: True)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert detect_location() == (None, None)


def test_explicit_location_skips_the_probe(monkeypatch):
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    probe = lambda: (_ for _ in ()).throw(AssertionError("probe called"))  # noqa: E731
    caps = detect_capabilities(
        runner=runner_for({}),
        torch_probe=lambda: TorchProbe(),
        zone="rack-zone",
        location_probe=probe,
    )
    assert caps.zone == "rack-zone" and caps.region is None
    caps = detect_capabilities(
        runner=runner_for({}),
        torch_probe=lambda: TorchProbe(),
        location_probe=lambda: ("r1", "r1-a"),
    )
    assert (caps.region, caps.zone) == ("r1", "r1-a")


def test_torch_compile_needs_a_c_compiler(monkeypatch):
    monkeypatch.setattr("gpulayer.gpu.discovery.apple_silicon", lambda: False)
    tp = lambda: TorchProbe(torch_version="2.15.0", gloo=True)  # noqa: E731
    caps = detect_capabilities(
        runner=runner_for({}),
        torch_probe=tp,
        location_probe=None,
        c_compiler=lambda: "/usr/bin/gcc",
    )
    assert caps.frameworks["torch.compile"] == "2.15.0"
    caps = detect_capabilities(
        runner=runner_for({}), torch_probe=tp, location_probe=None, c_compiler=lambda: None
    )
    assert "torch.compile" not in caps.frameworks
    assert any("C compiler" in n for n in caps.notes)
