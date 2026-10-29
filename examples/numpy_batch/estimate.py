"""Any Python library, any workload: a NumPy Monte Carlo job split into parallel chunks.

GPUlayer is not tied to a training framework. A function that uses NumPy, SciPy, pandas,
OpenCV, CuPy or anything else runs the same way: decorate it, then .run() or .map() it.

    python estimate.py
"""

import gpulayer as gl


@gl.remote(frameworks={"numpy": ">=1.20"})
def estimate_pi(seed: int, samples: int = 5_000_000) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    xy = rng.random((samples, 2))
    inside = int(((xy**2).sum(axis=1) <= 1.0).sum())
    return {"seed": seed, "inside": inside, "samples": samples}


if __name__ == "__main__":
    chunks = estimate_pi.map(range(8))  # eight jobs, spread over the available workers
    inside = sum(c["inside"] for c in chunks)
    samples = sum(c["samples"] for c in chunks)
    print(f"pi ~= {4 * inside / samples:.5f} from {samples:,} samples in {len(chunks)} jobs")
