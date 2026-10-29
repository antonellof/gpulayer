"""Submit the training job and an inference FunctionJob through the GpuLayer SDK.

    python examples/pytorch_train/run.py                   # CPU
    python examples/pytorch_train/run.py --accelerator mps # Apple Silicon
    python examples/pytorch_train/run.py --accelerator cuda

Uses the embedded control plane in $GPULAYER_HOME with an in-process local worker;
pass --url/--token (or GPULAYER_URL/GPULAYER_TOKEN) to target a remote control plane.
"""

import argparse
import json
from pathlib import Path

import gpulayer as gl

HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--accelerator", choices=["cpu", "cuda", "mps"], default="cpu")
ap.add_argument("--target", choices=["local", "remote", "cloud"], default="local")
ap.add_argument("--url")
ap.add_argument("--token")
ap.add_argument("--epochs", default="200")
args = ap.parse_args()

with gl.Client(url=args.url, token=args.token) as client:
    source = client.package(HERE)  # deterministic, content-addressed bundle of this folder
    resources = gl.ResourceRequest(accelerator=args.accelerator, frameworks={"torch": ">=2.0"})
    placement = gl.PlacementPolicy(target=args.target, on_unschedulable="fail")

    train = client.submit(
        gl.command(
            ["python", "train.py"],
            name="train-mlp",
            source=source,
            resources=resources,
            placement=placement,
            env={"EPOCHS": args.epochs},
            timeout_seconds=600,
        )
    )
    for line in train.follow_logs():
        print(f"[train] {line.line}")
    result = train.result()
    print("train:", result.state.value, result.error or "")
    if result.state != gl.JobState.SUCCEEDED:
        status = train.status()
        if status.scheduling:
            print("scheduling:", status.scheduling.summary)
            for c in status.scheduling.candidates:
                print("  ", c.worker_id, "; ".join(c.reasons) or "eligible")
        raise SystemExit(1)

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    for ref in result.outputs:
        (out_dir / ref.name).write_bytes(client.download(ref))
        print(f"downloaded {ref.name} ({ref.size_bytes} bytes, sha256 {ref.sha256[:12]}...)")
    print("metrics:", json.loads((out_dir / "metrics.json").read_text()))

    infer = client.run(
        gl.function(
            "predict:predict",
            [[1.0, 1.0], [0.0, 2.0]],
            name="predict",
            source=source,
            resources=gl.ResourceRequest(frameworks={"torch": ">=2.0"}),
            placement=placement,
        ),
        timeout=300,
    )
    print("predict:", infer.state.value, infer.return_value or infer.error)
