"""Serving on an NPU through a real worker and real processes. SIMULATED: ``flm`` is a stand-in
that answers as FastFlowLM's source says the real one does. No NPU or FastFlowLM is involved."""

from __future__ import annotations

import json
import os
import stat
import urllib.request

import pytest

pytest.importorskip("torch")

import gpulayer as gl  # noqa: E402
from gpulayer.contracts import JobState  # noqa: E402

pytestmark = pytest.mark.integration

FLM = r"""#!PYTHON
import http.server, json, os, sys

command = sys.argv[1]
if command == "validate":
    print(json.dumps({"object": "npu_stack_validation", "drm_version": "0.6",
                      "devices": [{"device": "/dev/accel/accel0", "cols": 8}],
                      "npu_platform": "aie2p", "ready": True}, indent=4))
elif command == "version":
    print('{ "version": "1.0.12" }')
elif command == "serve":
    model = sys.argv[2]
    if model.startswith("unknown"):
        print("Model not found:", model)
        sys.exit(1)
    flags = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    seen = {"model": model, "flags": flags, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hip": os.environ.get("HIP_VISIBLE_DEVICES")}
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(seen).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *a):
            pass
    http.server.HTTPServer((flags["--host"], int(flags["--port"])), H).serve_forever()
"""


@pytest.fixture
def flm(tmp_path, monkeypatch):
    import sys

    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "flm"
    script.write_text(FLM.replace("PYTHON", sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GPULAYER_HOME", str(tmp_path / "home"))


def ask(endpoint: str) -> dict:
    port = endpoint.rsplit(":", 1)[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
        return json.loads(r.read())


def test_model_is_served_on_loopback_with_no_gpu(flm):
    with gl.Client() as c:
        job = c.submit(gl.flm_service("llama3.2:1b", args=["--pmode", "turbo"]))
        (endpoint,) = job.endpoints(wait=60)
        seen = ask(endpoint)
        job.cancel()
    assert seen["model"] == "llama3.2:1b"
    assert seen["flags"]["--host"] == "127.0.0.1" and seen["flags"]["--pmode"] == "turbo"
    assert seen["cuda"] == "" and seen["hip"] == ""


def test_exposing_is_a_choice(flm):
    with gl.Client() as c:
        job = c.submit(gl.flm_service("llama3.2:1b", expose=True))
        (endpoint,) = job.endpoints(wait=60)
        assert ask(endpoint)["flags"]["--host"] == "0.0.0.0"
        job.cancel()


def test_unknown_model_stops_instead_of_retrying_for_ever(flm):
    with gl.Client() as c:
        job = c.submit(gl.flm_service("unknown:1b", max_restarts=1))
        status = job.wait(timeout=60)
    assert status.state == JobState.FAILED
