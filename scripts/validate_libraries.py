"""Run each GPU library the site lists as a GPUlayer job on an NVIDIA GPU, and check it ran there.

    python scripts/validate_libraries.py [jax cupy tensorflow onnxruntime transformers ...]

Each job asks for one CUDA GPU and installs its library with per-job requirements (ADR-0022),
on top of the worker's Python (which has PyTorch). Each job must prove the GPU was used (a
device name from the library itself), not only that the import worked. Prints a CHECK line per
library with the environment preparation time, cold, then warm on a second run.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import gpulayer as gl
from gpulayer.contracts import JobState

JOBS: dict[str, tuple[list[str], str]] = {
    "jax": (
        ["jax[cuda12]"],
        """
import jax, jax.numpy as jnp
d = jax.devices("gpu")
x = jnp.ones((2048, 2048))
s = float((x @ x).sum())
print("RESULT", "jax", jax.__version__, d[0].device_kind, s == 2048.0**3)
""",
    ),
    "cupy": (
        ["cupy-cuda12x[ctk]"],
        """
import cupy as cp
x = cp.ones((2048, 2048), dtype=cp.float32)
s = float((x @ x).sum())
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print("RESULT", "cupy", cp.__version__, name, s == 2048.0**3)
""",
    ),
    "tensorflow": (
        ["tensorflow[and-cuda]"],
        """
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
with tf.device("/GPU:0"):
    x = tf.ones((2048, 2048))
    s = float(tf.reduce_sum(tf.matmul(x, x)))
name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "?")
print("RESULT", "tensorflow", tf.__version__, name, s == 2048.0**3)
""",
    ),
    "onnxruntime": (
        ["onnxruntime-gpu", "onnx"],
        """
import numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper
a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [512, 512])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [512, 512])
g = helper.make_graph([helper.make_node("MatMul", ["a", "a"], ["y"])], "mm", [a], [y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 9
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider"])
used = s.get_providers()[0]
out = s.run(None, {"a": np.ones((512, 512), np.float32)})[0]
print("RESULT", "onnxruntime", ort.__version__, used, used == "CUDAExecutionProvider"
      and float(out.sum()) == 512.0**3)
""",
    ),
    "transformers": (
        ["transformers"],
        """
import torch, transformers
from transformers import BertConfig, BertModel
cfg = BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                 intermediate_size=1024)
model = BertModel(cfg).cuda().eval()  # random weights: nothing is downloaded
ids = torch.randint(0, cfg.vocab_size, (8, 128), device="cuda")
with torch.no_grad():
    out = model(input_ids=ids).last_hidden_state
print("RESULT", "transformers", transformers.__version__, torch.cuda.get_device_name(0),
      out.is_cuda and out.shape == (8, 128, 256))
""",
    ),
}


def main() -> int:
    names = sys.argv[1:] or list(JOBS)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp, gl.Client(home=Path(tmp) / "home") as c:
        src = Path(tmp) / "src"
        src.mkdir()
        for name in names:
            (src / f"{name}_check.py").write_text(JOBS[name][1])
        bundle = c.package(src)
        c.ensure_local_worker()
        for name in names:
            reqs = JOBS[name][0]
            for attempt in ("cold", "warm"):
                t0 = time.perf_counter()
                job = c.submit(
                    gl.command(
                        ["python", f"{name}_check.py"],
                        source=bundle,
                        accelerator="cuda",
                        gpus=1,
                        requirements=reqs,
                        name=f"{name}-{attempt}",
                        timeout_seconds=3600,
                    )
                )
                st = job.wait(3600)
                logs = job.logs_text()
                result = next((ln for ln in logs.splitlines() if ln.startswith("RESULT")), "")
                done = [e for e in job.events() if e.type.startswith("task.")]
                prep = next((e.data.get("prepare_seconds") for e in done if e.data), None)
                ok = st.state == JobState.SUCCEEDED and result.endswith("True")
                failures += 0 if ok else 1
                print(
                    f"CHECK {name}-{attempt} {'PASS' if ok else 'FAIL'} {result[7:]} "
                    f"(environment {prep}s, total {time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
                if not ok:
                    print(logs[-2500:], st.error, flush=True)
                    break
    print(f"SUMMARY {'all passed' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
