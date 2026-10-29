"""Prometheus /metrics, the read-only metrics role, the dashboard, and the Grafana JSON."""

import json
import re
import urllib.request
from pathlib import Path

import pytest

import gpulayer as gl
from gpulayer.api.auth import ServerCredentials
from gpulayer.api.client import HttpControlPlaneClient, HttpTransport, HttpWorkerChannel
from gpulayer.api.server import ControlPlaneServer
from gpulayer.contracts import TaskOutcome, TaskState, WorkerCapabilities
from gpulayer.errors import Unauthorized

pytestmark = pytest.mark.integration
ROOT = Path(__file__).parents[2]
CAPS = WorkerCapabilities(
    hostname="h",
    os="linux",
    cpu_arch="x86_64",
    cpu_cores=4,
    python_version="3.11.0",
    origin="existing",
)


@pytest.fixture
def server(real_cp):
    creds = ServerCredentials()
    t = {r: creds.create(r, r) for r in ("admin", "submitter", "join", "metrics")}
    srv = ControlPlaneServer(real_cp, creds, host="127.0.0.1", port=0, reconcile_interval=0.2)
    srv.start()
    yield srv, t
    srv.shutdown()


def scrape(url, token):
    req = urllib.request.Request(url + "/metrics", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as r:  # no protocol header, like Prometheus
        return r.headers["Content-Type"], r.read().decode()


def run_one_job(srv, t):
    admin = HttpControlPlaneClient(HttpTransport(srv.url, t["admin"]))
    ch = HttpWorkerChannel.register(srv.url, t["join"], "w", CAPS)
    job = admin.submit(gl.command(["true"], placement=gl.PlacementPolicy(target="remote")))
    a = ch.acquire()
    ch.start(a.task_id)
    ch.finish(a.task_id, TaskOutcome(state=TaskState.SUCCEEDED, exit_code=0))
    return admin, job


def test_metrics_are_valid_prometheus_text(server):
    parser = pytest.importorskip("prometheus_client.parser")
    srv, t = server
    run_one_job(srv, t)
    ctype, text = scrape(srv.url, t["metrics"])
    assert ctype.startswith("text/plain; version=0.0.4")
    families = {f.name: f for f in parser.text_string_to_metric_families(text)}
    assert families["gpulayer_jobs"].type == "gauge"
    assert families["gpulayer_job_run_seconds"].type == "histogram"
    # prometheus_client strips the _total suffix from counter family names
    assert families["gpulayer_jobs_finished"].type == "counter"
    finished = {
        s.labels["state"]: s.value
        for s in families["gpulayer_jobs_finished"].samples
        if s.name.endswith("_total")
    }
    assert finished == {"succeeded": 1.0}
    buckets = [
        s for s in families["gpulayer_job_run_seconds"].samples if s.name.endswith("_bucket")
    ]
    bounds = [float(s.labels["le"]) for s in buckets]
    assert bounds == sorted(bounds) and bounds[-1] == float("inf")
    values = [s.value for s in buckets]
    assert values == sorted(values)  # cumulative


def test_metrics_role_is_read_only(server):
    srv, t = server
    assert "gpulayer_info" in scrape(srv.url, t["submitter"])[1]  # client roles can scrape
    with pytest.raises(Unauthorized):
        HttpControlPlaneClient(HttpTransport(srv.url, t["metrics"])).list_jobs()
    with pytest.raises(urllib.error.HTTPError) as e:
        scrape(srv.url, t["join"])
    assert e.value.code == 401


def test_counters_survive_garbage_collection(server):
    srv, t = server
    admin, _ = run_one_job(srv, t)
    admin.gc(0, delete_jobs=True)
    assert admin.list_jobs() == []
    text = admin.metrics_text()
    assert 'gpulayer_jobs_finished_total{state="succeeded"} 1' in text
    assert 'gpulayer_jobs{state="succeeded"} 0' in text


def test_dashboard_is_served_with_strict_headers(server):
    srv, t = server
    with urllib.request.urlopen(srv.url + "/ui/", timeout=10) as r:
        assert r.headers["Content-Type"].startswith("text/html")
        csp = r.headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert b"/ui/app.js" in r.read()
    for name in ("app.js", "app.css"):
        with urllib.request.urlopen(f"{srv.url}/ui/{name}", timeout=10) as r:
            assert r.status == 200
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(srv.url + "/ui/server.py", timeout=10)
    assert e.value.code == 404
    who = HttpControlPlaneClient(HttpTransport(srv.url, t["submitter"])).whoami()
    assert who == {"principal": "client:submitter", "role": "submitter"}


def test_dashboard_never_builds_html_from_strings():
    js = (ROOT / "src/gpulayer/api/ui/app.js").read_text()
    for risky in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert risky not in js
    assert "style=" not in js  # inline styles would be blocked by the CSP


def test_grafana_dashboard_uses_only_exported_metrics(real_cp):
    dash = json.loads((ROOT / "deploy/monitoring/grafana/gpulayer-dashboard.json").read_text())
    exprs = [t["expr"] for p in dash["panels"] for t in p["targets"]]
    used = set(re.findall(r"\bgpulayer_[a-z_]+", " ".join(exprs)))
    text = real_cp.metrics_text()
    exported = set(re.findall(r"^# TYPE (gpulayer_[a-z_]+)", text, re.M))
    counters_and_histograms = {
        "gpulayer_jobs_submitted_total",
        "gpulayer_jobs_finished_total",
        "gpulayer_tasks_lost_total",
        "gpulayer_job_run_seconds",
        "gpulayer_job_queue_wait_seconds",
    }
    known = exported | counters_and_histograms
    known |= (
        {f"{m}_bucket" for m in known} | {f"{m}_sum" for m in known} | {f"{m}_count" for m in known}
    )
    assert used and used <= known, used - known
