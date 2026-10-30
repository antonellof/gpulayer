"""Long-running services (ADR-0019) with real processes on a local worker."""

import time
import urllib.request

import pytest

import gpulayer as gl
from gpulayer.contracts import JobSpec, JobState, ServiceJob, WorkloadType

pytestmark = pytest.mark.integration

# A tiny HTTP service: answers with its pid; /exit makes it exit with code 3.
SERVER = """
import http.server, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/exit":
            os._exit(3)
        body = f"pid {os.getpid()}".encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass
print("listening on", os.environ["PORT"], flush=True)
http.server.HTTPServer(("0.0.0.0", int(os.environ["PORT"])), H).serve_forever()
"""


def get(endpoint, path="/"):
    host, port = endpoint.rsplit(":", 1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.read().decode()


def test_service_contract_defaults():
    spec = gl.service(["python", "-c", "pass"], replicas=3)
    assert isinstance(spec.workload, ServiceJob) and spec.workload_type == WorkloadType.SERVICE
    assert spec.num_nodes == 3 and spec.retry.max_attempts == 10 and spec.timeout_seconds is None
    with pytest.raises(ValueError, match="PORT"):
        JobSpec(workload=ServiceJob(command=["x"]), env={"PORT": "80"})


def test_service_runs_restarts_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("GPULAYER_HOME", str(tmp_path / "home"))
    with gl.Client() as c:
        job = c.submit(gl.service(["python", "-c", SERVER]))
        (endpoint,) = job.endpoints(wait=60)
        first = get(endpoint)
        assert first.startswith("pid ")
        assert job.status().state == JobState.RUNNING

        # The process exits: the worker starts it again, on a new port, and reports it.
        with pytest.raises(OSError):
            get(endpoint, "/exit")
        deadline = time.time() + 60
        second = None
        while time.time() < deadline:
            found = job.endpoints()
            if found and found[0] != endpoint:
                second = found[0]
                break
            time.sleep(0.2)
        assert second, job.logs_text()[-2000:]
        assert get(second) != first  # a new process
        # The worker notes the restart in the job's log (system lines, shipped asynchronously).
        deadline = time.time() + 10
        while time.time() < deadline and not any("restart 1" in x.line for x in job.logs()):
            time.sleep(0.2)
        assert any("restart 1" in x.line for x in job.logs())

        st = job.cancel()
        st = job.wait(60)
        assert st.state == JobState.CANCELLED
        assert job.endpoints() == []
