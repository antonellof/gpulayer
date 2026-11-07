"""AMD Ryzen AI NPUs (ADR-0026). The reports below follow FastFlowLM's source
(``sanity_check_npu_stack`` in src/src/main.cpp); no NPU was involved."""

from __future__ import annotations

import json
import re

import pytest
from conftest import make_caps
from pydantic import ValidationError

import gpulayer as gl
from gpulayer.contracts import (
    Accelerator,
    AcceleratorDevice,
    CommandJob,
    DistributedJob,
    GpuVendor,
    JobSpec,
    JobState,
    MemoryKind,
    ResourceRequest,
)
from gpulayer.control_plane.scheduler import compatibility_reasons, default_max_tasks
from gpulayer.errors import ValidationFailed
from gpulayer.gpu.discovery import TorchProbe, detect_capabilities, detect_npu
from gpulayer.gpu.support import SUPPORT, unsupported_reason

READY = {
    "object": "npu_stack_validation",
    "platform": "linux",
    "kernel": "6.17.0",
    "kernel_ok": True,
    "amd_device_found": True,
    "drm_version": "0.6",
    "devices": [
        {
            "device": "/dev/accel/accel0",
            "fw_major": 1,
            "fw_minor": 1,
            "fw_patch": 0,
            "fw_build": 0,
            "fw_ok": True,
            "cols": 8,
        }
    ],
    "all_fw_ok": True,
    "enough_cols": True,
    "memlock_limit": "infinity",
    "memlock_ok": True,
    "npu_platform": "aie2p",
    "ready": True,
}


def runner(validate=None, version=(0, '{ "version": "1.0.12" }')):
    report = validate or (0, json.dumps(READY))

    def run(cmd, timeout):
        if cmd[0] != "flm":
            return 127, ""
        return report if cmd[1] == "validate" else version

    return run


def npu_worker(**kw) -> AcceleratorDevice:
    return AcceleratorDevice(
        index=0,
        accelerator=Accelerator.NPU,
        vendor=GpuVendor.AMD,
        name="AMD NPU (aie2p)",
        memory_kind=MemoryKind.UNIFIED,
        **kw,
    )


def test_ready_npu_is_reported_as_its_own_kind():
    devices, version, notes = detect_npu(runner())
    assert version == "1.0.12" and notes == []
    assert [(d.accelerator, d.vendor, d.name) for d in devices] == [
        (Accelerator.NPU, GpuVendor.AMD, "AMD NPU (aie2p)")
    ]
    assert devices[0].driver_version == "0.6"
    assert devices[0].memory_gb is None  # never guessed


def test_without_flm_there_is_nothing_to_say():
    assert detect_npu(lambda cmd, t: (127, "")) == ([], None, [])


def test_core_backend_reports_no_device_list():
    report = {"object": "npu_stack_validation", "npu_platform": "dev04", "ready": True}
    devices, _, _ = detect_npu(runner(validate=(0, json.dumps(report))))
    assert len(devices) == 1


@pytest.mark.parametrize(
    ("code", "report", "note"),
    [
        (1, {**READY, "ready": False, "all_fw_ok": False}, "all_fw_ok"),
        (1, {**READY, "ready": False, "memlock_ok": False}, "memlock_ok"),
        (0, {**READY, "ready": False}, "not ready"),
        (1, {**READY}, "not ready"),
    ],
)
def test_a_stack_that_is_not_ready_gives_a_note_and_no_device(code, report, note):
    devices, version, notes = detect_npu(runner(validate=(code, json.dumps(report))))
    assert devices == [] and version is None
    assert note in notes[0]


@pytest.mark.parametrize("out", ["", "garbage", "[1, 2]", '{"ready": tru'])
def test_unreadable_validation_gives_no_device(out):
    devices, _, notes = detect_npu(runner(validate=(0, out)))
    assert devices == [] and notes


def test_capabilities_report_the_npu_and_flm():
    caps = detect_capabilities(runner=runner(), torch_probe=lambda: TorchProbe(error="none"))
    assert [d.accelerator for d in caps.devices(Accelerator.NPU)] == [Accelerator.NPU]
    assert caps.frameworks["flm"] == "1.0.12"
    assert not caps.devices(Accelerator.CUDA)


def test_flm_service_listens_on_loopback_unless_told():
    spec = gl.flm_service("llama3.2:1b", args=["--pmode", "turbo"])
    assert spec.workload.command[3:7] == ["flm", "llama3.2:1b", "127.0.0.1", "--pmode"]
    assert spec.resources.accelerator == Accelerator.NPU and spec.resources.gpu_count == 1
    assert spec.resources.frameworks == {"flm": ">=1.0"}
    assert spec.workload.restart == "on_failure" and spec.workload.max_restarts == 5
    assert gl.flm_service("llama3.2:1b", expose=True).workload.command[5] == "0.0.0.0"


@pytest.mark.parametrize("tag", ["", "a b", "$(reboot)", "x;y", "-flag", "a:b:c"])
def test_flm_service_checks_the_model_tag(tag):
    with pytest.raises(ValidationFailed, match="model tag"):
        gl.flm_service(tag)


def test_an_npu_is_never_a_gpu():
    caps = make_caps(accelerators=[npu_worker()], frameworks={"flm": "1.0.12", "torch": "2.5"})
    gpu_job = JobSpec(
        workload=CommandJob(command=["x"]),
        resources=ResourceRequest(accelerator=Accelerator.CUDA, gpu_count=1),
    )
    assert any("cuda" in r for r in compatibility_reasons(gpu_job, caps))
    gpu_job = JobSpec(
        workload=CommandJob(command=["x"]),
        resources=ResourceRequest(accelerator=Accelerator.ROCM, gpu_count=1),
    )
    assert any("rocm" in r for r in compatibility_reasons(gpu_job, caps))


def test_an_npu_job_needs_an_npu():
    spec = gl.flm_service("llama3.2:1b")
    assert any("npu" in r for r in compatibility_reasons(spec, make_caps()))
    ready = make_caps(accelerators=[npu_worker()], frameworks={"flm": "1.0.12"})
    assert compatibility_reasons(spec, ready) == []
    old = make_caps(accelerators=[npu_worker()], frameworks={"flm": "0.9.9"})
    assert compatibility_reasons(spec, old)


def test_two_services_do_not_share_an_npu(cp):
    """FastFlowLM's lock is per process, so two servers would fight over the NPU."""
    caps = make_caps(accelerators=[npu_worker()], frameworks={"flm": "1.0.12"})
    assert default_max_tasks(caps) >= 1
    worker = cp.register_worker("laptop", caps).worker_id
    first = cp.submit(gl.flm_service("llama3.2:1b"))
    second = cp.submit(gl.flm_service("qwen3.5:2b"))
    assert first.tasks[0].worker_id == worker and first.tasks[0].node_rank == 0
    assert cp.get_job(second.job_id).state in (JobState.QUEUED, JobState.FAILED)


def test_a_gpu_job_and_an_npu_service_fit_on_one_machine():
    from conftest import cuda_device

    caps = make_caps(accelerators=[cuda_device(), npu_worker()], frameworks={"flm": "1.0.12"})
    assert default_max_tasks(caps) == 2


def test_distributed_npu_jobs_are_refused():
    with pytest.raises(ValidationError, match="does not train"):
        JobSpec(
            workload=DistributedJob(entrypoint="t.py", num_workers=2),
            resources=ResourceRequest(accelerator=Accelerator.NPU, gpu_count=1),
        )


def test_support_matrix_says_what_it_is():
    from gpulayer.contracts import WorkloadType

    assert unsupported_reason(Accelerator.NPU, WorkloadType.SERVICE) is None
    assert "unsupported" in unsupported_reason(Accelerator.NPU, WorkloadType.DISTRIBUTED)
    assert re.search("Not yet run on an NPU", SUPPORT[Accelerator.NPU].note)


def test_jobs_without_an_npu_serialize_as_before():
    """Workers reject unknown fields and values: nothing new reaches them without NPU jobs."""
    assert '"npu"' not in gl.command(["x"]).model_dump_json()
