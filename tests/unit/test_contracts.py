import pytest
from pydantic import ValidationError

from gpulayer.contracts import (
    ArtifactReference,
    CommandJob,
    DistributedJob,
    ExecutionTarget,
    FunctionJob,
    InputArtifact,
    JobSpec,
    PlacementPolicy,
    ProvisioningPolicy,
    ResourceRequest,
    RetryPolicy,
    SecretRef,
    TaskOutcome,
    TaskState,
    WorkloadType,
)

SHA = "a" * 64
REF = ArtifactReference(uri=f"artifact://sha256/{SHA}", sha256=SHA, size_bytes=1)


def test_roundtrip_preserves_workload_type():
    spec = JobSpec(
        workload=DistributedJob(
            entrypoint="train.py", num_workers=2, procs_per_worker=2, source=REF
        )
    )
    again = JobSpec.model_validate_json(spec.model_dump_json())
    assert again.workload_type == WorkloadType.DISTRIBUTED
    assert again.workload.world_size == 4
    assert JobSpec(workload=CommandJob(command=["x"])).workload_type == WorkloadType.INDEPENDENT


def test_target_and_workload_type_are_independent():
    spec = JobSpec(
        workload=CommandJob(command=["x"]), placement=PlacementPolicy(target=ExecutionTarget.CLOUD)
    )
    assert spec.workload_type == WorkloadType.INDEPENDENT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accelerator": "cpu", "gpu_count": 1},
        {"accelerator": "cpu", "gpu_memory_gb": 8},
        {"accelerator": "mps", "min_cuda_compute_capability": "8.0"},
    ],
)
def test_resource_request_rejects_contradictions(kwargs):
    with pytest.raises(ValidationError):
        ResourceRequest(**kwargs)


def test_gpu_accelerator_defaults_to_one_device():
    assert ResourceRequest(accelerator="cuda").gpu_count == 1


def test_distributed_mps_rejected():
    with pytest.raises(ValidationError, match="MPS"):
        JobSpec(
            workload=DistributedJob(entrypoint="t.py"), resources=ResourceRequest(accelerator="mps")
        )


def test_distributed_retry_rejected():
    with pytest.raises(ValidationError, match="retry"):
        JobSpec(workload=DistributedJob(entrypoint="t.py"), retry=RetryPolicy(max_attempts=2))


def test_nccl_requires_cuda():
    with pytest.raises(ValidationError, match="nccl"):
        JobSpec(workload=DistributedJob(entrypoint="t.py", comm_backend="nccl"))


@pytest.mark.parametrize("path", ["/etc/passwd", "../x", "a/../../b", "C:/x"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValidationError):
        InputArtifact(artifact=REF, path=path)
    with pytest.raises(ValidationError):
        DistributedJob(entrypoint=path)


def test_reserved_env_and_secret_collisions():
    with pytest.raises(ValidationError, match="reserved"):
        JobSpec(workload=CommandJob(command=["x"]), env={"GPULAYER_JOB_ID": "x"})
    with pytest.raises(ValidationError, match="reserved"):
        JobSpec(workload=CommandJob(command=["x"]), env={"RANK": "0"})
    with pytest.raises(ValidationError, match="collide"):
        JobSpec(
            workload=CommandJob(command=["x"]),
            env={"A": "1"},
            secrets=[SecretRef(name="A", key="B")],
        )


def test_function_entrypoint_format():
    FunctionJob(entrypoint="pkg.mod:fn")
    with pytest.raises(ValidationError):
        FunctionJob(entrypoint="lambda: 1")


def test_artifact_uri_must_match_digest():
    with pytest.raises(ValidationError):
        ArtifactReference(uri=f"artifact://sha256/{'b' * 64}", sha256=SHA, size_bytes=1)


def test_provisioning_requires_cloud_target():
    with pytest.raises(ValidationError):
        PlacementPolicy(target="local", provisioning=ProvisioningPolicy(provider="fake"))
    PlacementPolicy(
        target="local", fallback_targets=["cloud"], provisioning=ProvisioningPolicy(provider="fake")
    )


def test_worker_cannot_report_lost():
    with pytest.raises(ValidationError):
        TaskOutcome(state=TaskState.LOST)
    with pytest.raises(ValidationError):
        TaskOutcome(state=TaskState.RUNNING)


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        JobSpec.model_validate({"workload": {"kind": "command", "command": ["x"]}, "gpus": 1})
