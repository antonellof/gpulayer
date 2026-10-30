"""Framework-agnostic launcher: run *this script* on GPUlayer-placed workers.

    import gpulayer.pytorch as glt
    glt.launch(nodes=2, devices=2, accelerator="cuda", target="remote")
    # ... the rest of any PyTorch script, whatever framework it uses ...

Outside a GPUlayer job, ``launch`` packages the script's directory, submits the same
script with the same arguments, streams logs, downloads outputs and (by default) exits with
the job's exit code. Inside the job it returns ``None`` immediately and the script continues,
with the standard torchrun environment (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, ...).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpulayer.contracts import (
    Accelerator,
    CommandJob,
    DistributedJob,
    InputArtifact,
    JobSpec,
    JobState,
    JobStatus,
    PlacementPolicy,
    ResourceRequest,
)
from gpulayer.errors import ValidationFailed
from gpulayer.sdk.client import Client, Job


def in_gpulayer_job() -> bool:
    return "GPULAYER_JOB_ID" in os.environ


def in_torchrun() -> bool:
    return "TORCHELASTIC_RUN_ID" in os.environ or "LOCAL_RANK" in os.environ


def is_launcher_process() -> bool:
    """True in the process that submits work, False on workers (where the work runs)."""
    return not (in_gpulayer_job() or in_torchrun())


@dataclass
class LaunchResult:
    job: Job
    status: JobStatus
    output_dir: Path
    outputs: list[Path] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status.state == JobState.SUCCEEDED


def launch(
    *,
    nodes: int = 1,
    devices: int = 1,
    accelerator: str = "cpu",
    target: str | None = None,
    client: Client | None = None,
    script: str | Path | None = None,
    argv: list[str] | None = None,
    frameworks: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    inputs: dict[str, str | Path] | None = None,
    timeout_seconds: float | None = None,
    name: str | None = None,
    output_dir: str | Path | None = None,
    exit: bool = True,  # noqa: A002 - mirrors sys.exit semantics
    quiet: bool = False,
    **placement: Any,
) -> LaunchResult | None:
    """Submit this script as a GPUlayer job, or return ``None`` when already inside one.

    ``inputs`` maps a path under ``$GPULAYER_INPUT_DIR`` in the job to a local file, which is
    uploaded and placed there on every node (for example a checkpoint to resume from)."""
    if not is_launcher_process():
        return None
    if accelerator == "auto":
        raise ValidationFailed(
            "accelerator='auto' cannot be resolved before placement; pass 'cpu', 'cuda' or "
            "'mps' to launch() (inside the job, Runtime/Trainer resolve 'auto')"
        )
    script_path = Path(script or sys.argv[0]).resolve()
    if not script_path.is_file():
        raise ValidationFailed(f"cannot find the script to launch: {script_path}")
    args = list(sys.argv[1:] if argv is None else argv)
    from gpulayer.sdk.remote import default_client, default_target, explain_failure

    c = client or default_client()
    target = target or default_target(c)
    source = c.package(script_path.parent)
    accel = Accelerator(accelerator)
    # This is the PyTorch integration: the worker must have torch, and torch must be able to
    # drive the requested GPU (a CUDA box whose torch is CPU-only does not qualify).
    required = {"torch": ">=0"}
    if accel == Accelerator.CUDA:
        required["torch.cuda"] = ">=0"
    elif accel == Accelerator.MPS:
        required["torch.mps"] = ">=0"
    resources = ResourceRequest(
        accelerator=accel,
        gpu_count=devices if accel != Accelerator.CPU else 0,
        frameworks={**required, **(frameworks or {})},
    )
    if nodes * devices > 1:
        workload: Any = DistributedJob(
            entrypoint=script_path.name,
            args=args,
            source=source,
            num_workers=nodes,
            procs_per_worker=devices,
            comm_backend="nccl" if accel == Accelerator.CUDA else "gloo",
        )
    else:
        workload = CommandJob(command=["python", script_path.name, *args], source=source)
    spec = JobSpec(
        name=(name or script_path.stem.replace("_", "-"))[:63],
        workload=workload,
        resources=resources,
        placement=PlacementPolicy(
            target=target, on_unschedulable=placement.pop("on_unschedulable", "fail"), **placement
        ),
        env=env or {},
        inputs=[
            InputArtifact(artifact=c.upload(Path(local)), path=dest)
            for dest, local in (inputs or {}).items()
        ],
        timeout_seconds=timeout_seconds,
    )
    job = c.submit(spec)
    if not quiet:
        print(
            f"[gpulayer] {job.id}: {script_path.name} on {nodes} node(s) x {devices} "
            f"{accel.value} process(es), target {target}",
            file=sys.stderr,
            flush=True,
        )
    for line in job.follow_logs():
        if quiet or line.stream == "system":
            continue
        prefix = f"[{line.origin}] " if nodes * devices > 1 else ""
        print(
            prefix + line.line,
            file=sys.stderr if line.stream == "stderr" else sys.stdout,
            flush=True,
        )
    status = job.status()
    result = job.result()
    out = Path(output_dir or "gpulayer_runs") / job.id
    files = []
    for ref in result.outputs:
        dest = (out / ref.name).resolve()
        if not dest.is_relative_to(out.resolve()):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(c.download(ref))
        files.append(dest)
    if not quiet:
        msg = f"[gpulayer] {job.id} {status.state.value}"
        if status.state != JobState.SUCCEEDED:
            msg += f": {explain_failure(status)}"
        if files:
            msg += f"; {len(files)} output file(s) in {out}"
        print(msg, file=sys.stderr, flush=True)
    launched = LaunchResult(job=job, status=status, output_dir=out, outputs=files)
    if exit:
        sys.exit(0 if launched.succeeded else (result.exit_code or 1))
    return launched
