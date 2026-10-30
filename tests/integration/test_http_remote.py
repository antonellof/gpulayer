"""Remote-worker path over authenticated HTTP on one host.

The worker runs in a separate OS process that shares NO state with the control plane
except the HTTP API (separate $GPULAYER_HOME, artifacts fetched over the wire). This
validates the protocol, auth and artifact transfer; it does not prove cross-machine
networking (see docs/architecture/remote-workers.md).
"""

import os
import subprocess
import sys
import time

import pytest

import gpulayer as gl
from gpulayer.api.auth import generate_credentials
from gpulayer.api.client import HttpTransport, HttpWorkerChannel
from gpulayer.api.server import ControlPlaneServer
from gpulayer.contracts import JobState, WorkerOrigin
from gpulayer.errors import ProtocolMismatch, Unauthorized

pytestmark = pytest.mark.integration


@pytest.fixture
def server(real_cp):
    creds, client_token, join_token = generate_credentials()
    srv = ControlPlaneServer(real_cp, creds, host="127.0.0.1", port=0, reconcile_interval=0.2)
    srv.start()
    yield srv, client_token, join_token
    srv.shutdown()


def test_plaintext_on_public_interface_refused(real_cp):
    creds, _, _ = generate_credentials()
    with pytest.raises(ValueError, match="plaintext"):
        ControlPlaneServer(real_cp, creds, host="0.0.0.0", port=0)


def test_auth_boundaries(server):
    srv, client_token, join_token = server
    from gpulayer.contracts import WorkerCapabilities

    caps = WorkerCapabilities(
        hostname="h",
        os="linux",
        cpu_arch="x86_64",
        cpu_cores=1,
        python_version="3.11.0",
        origin="existing",
    )
    # Wrong / swapped credentials are rejected.
    with pytest.raises(Unauthorized):
        HttpTransport(srv.url, "bogus").request("GET", "/v1/jobs")
    with pytest.raises(Unauthorized):
        HttpTransport(srv.url, join_token).request("GET", "/v1/jobs")
    with pytest.raises(Unauthorized):
        HttpWorkerChannel.register(srv.url, client_token, "w", caps)
    with pytest.raises(Unauthorized):  # a static join token cannot impersonate a cloud worker
        HttpWorkerChannel.register(
            srv.url,
            join_token,
            "w",
            caps.model_copy(update={"origin": WorkerOrigin.PROVIDER, "instance_id": "inst-x"}),
        )
    ch = HttpWorkerChannel.register(srv.url, join_token, "w", caps)
    with pytest.raises(Unauthorized):  # worker session cannot use client endpoints
        ch.t.request("GET", "/v1/jobs")
    with pytest.raises(ProtocolMismatch):
        HttpTransport(srv.url, client_token).request(
            "GET", "/v1/jobs", headers={"X-GpuLayer-Protocol": "999"}
        )
    health = HttpTransport(srv.url, "").request("GET", "/v1/health")
    assert health["status"] == "ok"


def test_remote_worker_process_executes_job(server, tmp_path):
    srv, client_token, join_token = server
    env = {
        **os.environ,
        "GPULAYER_HOME": str(tmp_path / "worker-home"),
        "GPULAYER_JOIN_TOKEN": join_token,
    }
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpulayer.cli.main",
            "--url",
            srv.url,
            "worker",
            "start",
            "--name",
            "remote-1",
            "--idle-exit",
            "60",
            "--max-tasks",
            "1",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        c = gl.Client(url=srv.url, token=client_token)
        src = tmp_path / "proj"
        src.mkdir()
        (src / "job.py").write_text("import platform\nprint('ran on', platform.node())\n")
        job = c.submit(
            gl.command(
                ["python", "job.py"],
                source=c.package(src),
                target="remote",
                placement=gl.PlacementPolicy(target="remote"),
            )
        )
        st = job.wait(90)
        assert st.state == JobState.SUCCEEDED, (st.error, st.scheduling)
        assert job.logs_text().startswith("ran on")
        (w,) = c.workers()
        assert w.name == "remote-1" and w.capabilities.origin.value == "existing"
        # local-target jobs never land on the remote worker
        local = c.submit(gl.command(["true"]))
        time.sleep(0.5)
        assert c.job(local.id).status().state == JobState.QUEUED
    finally:
        out = worker.communicate(timeout=90)[0]
    assert worker.returncode == 0, out


def test_client_reuses_one_connection_per_thread(server, monkeypatch):
    from gpulayer.api import server as server_mod

    srv, client_token, _ = server
    accepted = []
    setup = server_mod._RequestHandler.setup

    def counting_setup(self):
        accepted.append(self.client_address)
        setup(self)

    monkeypatch.setattr(server_mod._RequestHandler, "setup", counting_setup)
    t = HttpTransport(srv.url, client_token)
    for _ in range(20):
        t.request("GET", "/v1/jobs?limit=1")
    assert len(accepted) == 1


def test_client_replaces_a_connection_the_server_closed(server, monkeypatch):
    from gpulayer.api import server as server_mod

    srv, client_token, _ = server
    monkeypatch.setattr(server_mod._RequestHandler, "timeout", 0.2)  # idle connections close
    t = HttpTransport(srv.url, client_token)
    t.request("GET", "/v1/jobs?limit=1")
    time.sleep(0.6)  # the server has closed the kept-open connection by now
    assert t.request("GET", "/v1/jobs?limit=1") == []
    sub = t.request("POST", "/v1/jobs", gl.command(["true"]).model_dump(mode="json"))
    time.sleep(0.6)
    jobs = t.request("GET", "/v1/jobs?limit=10")
    assert [j["job_id"] for j in jobs] == [sub["job_id"]]  # the retried request ran once
