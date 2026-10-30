"""JAX and CuPy on a CUDA GPU, without PyTorch. Submit with the frameworks the job needs:

    gpulayer submit --target remote --accelerator cuda --gpus 1 \
        --framework jax.cuda --framework cupy.cuda \
        --source examples/jax_cupy_cuda -- python matmul.py

`jax.cuda` and `cupy.cuda` are reported by workers whose JAX / CuPy actually ran on the GPU,
so the job is never placed where those libraries would fall back to the CPU or fail.
"""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cupy as cp  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N = 4096


def tflops(seconds: float) -> float:
    return 2 * N**3 / seconds / 1e12


gpus = jax.devices("cuda")
assert gpus, "JAX sees no CUDA device"
a = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.float32)
mm = jax.jit(lambda x: x @ x)
mm(a).block_until_ready()  # compile
t0 = time.perf_counter()
for _ in range(10):
    r = mm(a)
r.block_until_ready()
dt = (time.perf_counter() - t0) / 10
print(
    f"JAX {jax.__version__} on {gpus[0].device_kind}: {dt * 1000:.1f} ms, {tflops(dt):.2f} TFLOP/s",
    flush=True,
)

x = cp.random.standard_normal((N, N), dtype=cp.float32)
(x @ x).sum()
cp.cuda.Device().synchronize()
t0 = time.perf_counter()
for _ in range(10):
    y = x @ x
cp.cuda.Device().synchronize()
dt = (time.perf_counter() - t0) / 10
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"CuPy {cp.__version__} on {name}: {dt * 1000:.1f} ms, {tflops(dt):.2f} TFLOP/s")
print("OK")
