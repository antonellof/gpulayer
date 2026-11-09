"""Validate long-running services (ADR-0019) on an NVIDIA GPU, with a local worker:

    python scripts/validate_service_gpu.py [--vllm Qwen/Qwen2.5-0.5B-Instruct]

Prints `CHECK name: PASS|FAIL detail` lines:

1. serve     examples/gpu_service as a service with one GPU: the endpoint answers, the model
             runs on the GPU; request latency over 200 sequential requests.
2. restart   its process is killed: the worker starts it again on a new port, which answers.
3. cancel    the service ends and its port stops answering.
4. vllm      (with --vllm) vLLM installed with per-job requirements serves the model with an
             OpenAI-compatible API on $PORT, and a completion comes back.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import gpulayer as gl

ap = argparse.ArgumentParser()
ap.add_argument("--vllm", default=None, help="a model for vLLM to serve")
ap.add_argument("--accelerator", default="cuda", help="mps to try it on Apple Silicon")
args = ap.parse_args()
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    failed += not ok
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)


def call(endpoint: str, path: str = "/", body: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://{endpoint}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def answers(endpoint: str) -> bool:
    try:
        call(endpoint, timeout=2)
        return True
    except Exception:
        return False


def wait_for(fn, timeout: float, step: float = 0.5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(step)
    return None


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp, gl.Client(home=Path(tmp) / "home") as c:
        c.ensure_local_worker()
        bundle = c.package(repo / "examples" / "gpu_service")
        try:
            t0 = time.perf_counter()
            job = c.submit(
                gl.service(["python", "serve.py"], source=bundle, accelerator=args.accelerator,
                           gpus=1, name="gpu-service")
            )  # fmt: skip
            first = job.endpoints(wait=300)
            up = time.perf_counter() - t0
            endpoint = first[0] if first else ""
            ready = wait_for(lambda: answers(endpoint), 120) if endpoint else None
            device = call(endpoint)["device"] if ready else "?"
            times = []
            out: dict = {}
            for i in range(200):
                s = time.perf_counter()
                out = call(endpoint, "/predict", [float(i % 7)] * 64)
                times.append((time.perf_counter() - s) * 1000)
            times.sort()
            on_gpu = "NVIDIA" in device or args.accelerator != "cuda"
            p50, p95 = statistics.median(times), times[189]
            check(
                "serve", bool(ready) and on_gpu and len(out.get("output", [])) == 10,
                f"{device} at {endpoint}, up in {up:.1f}s; request p50 {p50:.2f} ms, "
                f"p95 {p95:.2f} ms, GPU time {out.get('ms', 0):.3f} ms",
            )  # fmt: skip

            subprocess.run(["pkill", "-f", "python serve.py"], check=False)
            again = wait_for(
                lambda: [e for e in job.endpoints() if e != endpoint and answers(e)], 180
            )
            restarted = again[0] if again else ""
            # The worker ships logs in the background: wait for its line about the restart.
            line = wait_for(
                lambda: next((ln.line for ln in job.logs() if "restart 1" in ln.line), ""),
                30,
            )
            check(
                "restart", bool(restarted) and call(restarted)["device"] == device and bool(line),
                f"{endpoint} -> {restarted}; log: {line.strip()!r}",
            )  # fmt: skip

            job.cancel()
            st = job.wait(120)
            gone = wait_for(lambda: not answers(restarted), 60)
            check("cancel", bool(gone), f"job {st.state.value}, port closed {bool(gone)}")
        except Exception:
            check("service", False, traceback.format_exc()[-2500:])

        if args.vllm:
            try:
                t0 = time.perf_counter()
                cmd = (
                    f'exec vllm serve {args.vllm} --host 0.0.0.0 --port "$PORT" '
                    "--max-model-len 2048 --gpu-memory-utilization 0.8"
                )
                job = c.submit(
                    gl.service(["sh", "-c", cmd], accelerator="cuda", gpus=1, name="vllm",
                               requirements=["vllm"])
                )  # fmt: skip
                endpoints = job.endpoints(wait=2400)
                endpoint = endpoints[0] if endpoints else ""

                def models() -> bool:
                    try:
                        return bool(call(endpoint, "/v1/models", timeout=5)["data"])
                    except Exception:
                        return False

                ready = wait_for(models, 1200, step=2) if endpoint else None
                up = time.perf_counter() - t0
                text = ""
                if ready:
                    s = time.perf_counter()
                    r = call(endpoint, "/v1/completions",
                             {"model": args.vllm, "prompt": "The capital of France is",
                              "max_tokens": 16, "temperature": 0}, timeout=120)  # fmt: skip
                    text = r["choices"][0]["text"]
                    took = (time.perf_counter() - s) * 1000
                check(
                    "vllm", bool(text.strip()),
                    f"{args.vllm} up in {up:.0f}s (install, download, load); "
                    + (f"16 tokens in {took:.0f} ms: {text.strip()[:60]!r}" if text else
                       job.logs_text()[-1500:]),
                )  # fmt: skip
                job.cancel()
            except Exception:
                check("vllm", False, traceback.format_exc()[-2500:])
    print(f"SUMMARY {'all passed' if not failed else f'{failed} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
