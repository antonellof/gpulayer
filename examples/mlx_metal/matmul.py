"""A non-PyTorch GPU workload: MLX on the Apple GPU (Metal), placed by GPUlayer.

The job asks for the Apple GPU and the `mlx` framework; GPUlayer only places it on a worker
that reports both. The same pattern works for CuPy, JAX or any other GPU library: name the
accelerator and the framework, and write ordinary code for that library.

    pip install mlx
    python matmul.py
"""

import time

import gpulayer as gl


@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})
def matmul_benchmark(n: int, repeats: int = 10) -> dict:
    import mlx.core as mx

    timings = {}
    for name, device in (("cpu", mx.cpu), ("gpu", mx.gpu)):
        mx.set_default_device(device)
        a = mx.random.normal((n, n))
        b = mx.random.normal((n, n))
        mx.eval(a @ b)  # warm up
        start = time.perf_counter()
        for _ in range(repeats):
            mx.eval(a @ b)
        timings[name] = (time.perf_counter() - start) / repeats
    return {
        "n": n,
        "cpu_s": round(timings["cpu"], 4),
        "gpu_s": round(timings["gpu"], 4),
        "gpu_gflops": round(2 * n**3 / timings["gpu"] / 1e9, 1),
    }


if __name__ == "__main__":
    for result in matmul_benchmark.map([1024, 2048, 4096], parallelism=1):
        print(result)
