"""Token roles, expiry, revocation, rate limiting and worker artifact scoping over HTTP."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

import gpulayer as gl
from gpulayer.api.auth import ServerCredentials, token_hash
from gpulayer.api.client import HttpControlPlaneClient, HttpTransport, HttpWorkerChannel
from gpulayer.api.server import ControlPlaneServer
from gpulayer.contracts import WorkerCapabilities
from gpulayer.errors import Forbidden, RateLimited, Unauthorized

pytestmark = pytest.mark.integration

CAPS = WorkerCapabilities(
    hostname="h",
    os="linux",
    cpu_arch="x86_64",
    cpu_cores=4,
    python_version="3.11.0",
    origin="existing",
)


@pytest.fixture
def server(real_cp, tmp_path):
    creds = ServerCredentials(path=tmp_path / "creds.json")
    tokens = {
        "admin": creds.create("admin", "admin"),
        "alice": creds.create("alice", "submitter"),
        "bob": creds.create("bob", "submitter"),
        "join": creds.create("join", "join"),
    }
    creds.save()
    srv = ControlPlaneServer(real_cp, creds, host="127.0.0.1", port=0, reconcile_interval=0.2)
    srv.start()
    yield srv, creds, tokens
    srv.shutdown()


def client(srv, token):
    return HttpControlPlaneClient(HttpTransport(srv.url, token))


def spec():
    return gl.command(["true"], placement=gl.PlacementPolicy(target="remote"))


def test_submitters_cancel_only_their_own_jobs(server):
    srv, _, t = server
    job = client(srv, t["alice"]).submit(spec())
    assert job.submitted_by == "client:alice"
    with pytest.raises(Forbidden, match="submitter"):
        client(srv, t["bob"]).cancel_job(job.job_id)
    assert client(srv, t["bob"]).get_job(job.job_id).job_id == job.job_id  # reads are shared
    assert client(srv, t["alice"]).cancel_job(job.job_id).state.value == "cancelled"
    other = client(srv, t["bob"]).submit(spec())
    assert client(srv, t["admin"]).cancel_job(other.job_id).state.value == "cancelled"


def test_only_admins_revoke_workers(server):
    srv, _, t = server
    ch = HttpWorkerChannel.register(srv.url, t["join"], "w", CAPS)
    with pytest.raises(Forbidden, match="admin"):
        client(srv, t["alice"]).revoke_worker(ch.worker_id)
    client(srv, t["admin"]).revoke_worker(ch.worker_id)
    with pytest.raises(Forbidden, match="revoked"):
        ch.heartbeat()


def test_revoked_remote_worker_agent_stops(server, tmp_path):
    from gpulayer.worker.agent import WorkerAgent

    srv, _, t = server
    ch = HttpWorkerChannel.register(srv.url, t["join"], "w", CAPS)
    ch.heartbeat_interval = 0.1
    agent = WorkerAgent(ch, CAPS, tmp_path / "w", poll_interval=0.05)
    import threading

    th = threading.Thread(target=agent.run, daemon=True)
    th.start()
    client(srv, t["admin"]).revoke_worker(ch.worker_id)
    th.join(10)
    assert not th.is_alive() and agent.revoked


def test_join_token_cannot_act_as_client_and_vice_versa(server):
    srv, _, t = server
    with pytest.raises(Unauthorized):
        client(srv, t["join"]).list_jobs()
    with pytest.raises(Unauthorized):
        HttpWorkerChannel.register(srv.url, t["alice"], "w", CAPS)


def test_expired_and_revoked_tokens_stop_working_without_restart(server):
    srv, creds, t = server
    assert client(srv, t["alice"]).list_jobs() == []
    on_disk = ServerCredentials.load(creds.path)
    on_disk.revoke("alice")
    for e in on_disk.tokens:
        if e.name == "bob":
            e.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    on_disk.save()
    os.utime(creds.path, (1, 1))  # make sure the mtime changes even on coarse clocks
    with pytest.raises(Unauthorized):
        client(srv, t["alice"]).list_jobs()
    with pytest.raises(Unauthorized):
        client(srv, t["bob"]).list_jobs()
    assert client(srv, t["admin"]).list_jobs() == []


def test_failed_logins_are_rate_limited(server):
    srv, _, t = server
    for _ in range(srv.limiter.limit):
        with pytest.raises(Unauthorized):
            client(srv, "glc_wrong").list_jobs()
    with pytest.raises(RateLimited):
        client(srv, t["admin"]).list_jobs()  # even a valid token waits out the window


def test_workers_read_only_artifacts_of_their_tasks(server):
    srv, _, t = server
    admin = client(srv, t["admin"])
    used = admin.put_artifact(b"bundle-for-this-job", "a")
    unrelated = admin.put_artifact(b"someone else's data", "b")
    ch = HttpWorkerChannel.register(srv.url, t["join"], "w", CAPS)
    admin.submit(gl.command(["true"], source=used, placement=gl.PlacementPolicy(target="remote")))
    assert ch.acquire() is not None
    assert ch.get_artifact(used.sha256) == b"bundle-for-this-job"
    with pytest.raises(Forbidden, match="not used by any task"):
        ch.get_artifact(unrelated.sha256)
    assert admin.get_artifact(unrelated.sha256) == b"someone else's data"


def test_gc_is_admin_only(server):
    srv, _, t = server
    with pytest.raises(Forbidden, match="admin"):
        client(srv, t["alice"]).gc(0, dry_run=True)
    report = client(srv, t["admin"]).gc(0, dry_run=True)
    assert set(report) >= {"jobs", "logs", "events", "workers", "artifacts"}


def test_legacy_credentials_file_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "client_token_hashes": [token_hash("glc_old")],
                "join_token_hashes": [token_hash("glj_old")],
            }
        )
    )
    creds = ServerCredentials.load(path)
    assert creds.authenticate("glc_old", ("admin",)).role == "admin"
    assert creds.is_join("glj_old") and not creds.is_client("glj_old")


def test_token_cli(tmp_path):
    env = {**os.environ, "GPULAYER_HOME": str(tmp_path)}

    def gpl(*args):
        return subprocess.run(
            [sys.executable, "-m", "gpulayer.cli.main", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    p = gpl("token", "create", "ci", "--role", "join", "--ttl", "2h")
    assert p.returncode == 0 and p.stdout.strip().startswith("glj_")
    listed = gpl("token", "list").stdout
    assert "ci" in listed and "join" in listed and "active" in listed and "expires" in listed
    assert p.stdout.strip() not in listed  # values are never listed
    assert gpl("token", "revoke", "ci").returncode == 0
    assert "revoked" in gpl("token", "list").stdout
