"""Real local execution: SDK -> embedded control plane -> worker agent -> subprocess."""

import json
import sys
import threading
import time

import pytest

import gpulayer as gl
from gpulayer.contracts import JobState, WorkerOrigin
from gpulayer.gpu.discovery import TorchProbe, detect_capabilities
from gpulayer.worker.agent import WorkerAgent
from gpulayer.worker.channel import EmbeddedChannel

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GPULAYER_HOME", str(tmp_path / "home"))
    with gl.Client() as c:
        yield c


def test_command_job_end_to_end(client):
    script = "import os; print('hi', os.environ['GPULAYER_NODE_RANK'])"
    job = client.submit(gl.command(["python", "-c", script]))
    st = job.wait(60)
    assert st.state == JobState.SUCCEEDED, st.error
    assert job.logs_text() == "hi 0"
    assert job.result().exit_code == 0


def test_failing_command(client):
    script = "import sys; print('boom', file=sys.stderr); sys.exit(7)"
    job = client.submit(gl.command(["python", "-c", script]))
    st = job.wait(60)
    assert st.state == JobState.FAILED and job.result().exit_code == 7
    assert [line.stream for line in job.logs() if line.line == "boom"] == ["stderr"]


def test_function_job_from_source_bundle(client, tmp_path):
    src = tmp_path / "proj"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "__init__.py").write_text("")
    (src / "pkg" / "math_ops.py").write_text(
        "def scale(xs, factor=1):\n    return {'out': [x * factor for x in xs]}\n"
        "def bad():\n    raise ValueError('nope')\n"
        "def unserializable():\n    return object()\n"
    )
    ref = client.package(src)
    res = client.run(
        gl.function("pkg.math_ops:scale", [1, 2], kwargs={"factor": 3}, source=ref), timeout=60
    )
    assert res.state == JobState.SUCCEEDED and res.return_value == {"out": [3, 6]}
    res = client.run(gl.function("pkg.math_ops:bad", source=ref), timeout=60)
    assert res.state == JobState.FAILED and "ValueError: nope" in res.error
    res = client.run(gl.function("pkg.math_ops:unserializable", source=ref), timeout=60)
    assert res.state == JobState.FAILED and "JSON-serializable" in res.error


def test_timeout_kills_process_group(client):
    res = client.run(
        gl.command(["python", "-c", "import time; time.sleep(60)"], timeout_seconds=1), timeout=60
    )
    assert res.state == JobState.TIMED_OUT


def test_cancel_running_job(client):
    job = client.submit(
        gl.command(["python", "-c", "import time\nprint('started', flush=True)\ntime.sleep(60)"])
    )
    deadline = time.time() + 30
    while "started" not in job.logs_text() and time.time() < deadline:
        time.sleep(0.1)
    job.cancel()
    assert job.wait(30).state == JobState.CANCELLED


def test_inputs_outputs_env_and_secret_redaction(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "s3cr3t-value")
    data = tmp_path / "in.txt"
    data.write_text("payload")
    from gpulayer.contracts import InputArtifact

    inp = InputArtifact(artifact=client.upload(data), path="data/in.txt")
    script = (
        "import os, pathlib\n"
        "d = pathlib.Path(os.environ['GPULAYER_INPUT_DIR'])\n"
        "o = pathlib.Path(os.environ['GPULAYER_OUTPUT_DIR'])\n"
        "(o / 'sub').mkdir()\n"
        "(o / 'sub' / 'out.txt').write_text((d / 'data/in.txt').read_text().upper())\n"
        "print('key is', os.environ['API_KEY'], os.environ['MODE'])\n"
        "print('leak' if 'MY_API_KEY' in os.environ else 'isolated')\n"
    )
    job = client.submit(
        gl.command(
            ["python", "-c", script],
            inputs=[inp],
            env={"MODE": "x"},
            secrets=[{"name": "API_KEY", "key": "MY_API_KEY"}],
        )
    )
    assert job.wait(60).state == JobState.SUCCEEDED
    text = job.logs_text()
    assert "s3cr3t-value" not in text and "key is *** x" in text and "isolated" in text
    assert "s3cr3t" not in job.status().model_dump_json()
    (out,) = job.result().outputs
    assert out.name == "sub/out.txt" and client.download(out) == b"PAYLOAD"


def test_missing_secret_fails_task(client):
    job = client.submit(
        gl.command(["python", "-c", "pass"], secrets=[{"name": "T", "key": "GPULAYER_TEST_ABSENT"}])
    )
    st = job.wait(60)
    assert st.state == JobState.FAILED and "not available on this worker" in st.error


def test_incompatible_local_job_fails_with_explanation(client):
    job = client.submit(
        gl.command(
            ["python", "-c", "pass"],
            accelerator="rocm",
            placement=gl.PlacementPolicy(on_unschedulable="fail"),
        )
    )
    st = job.wait(30)
    assert st.state == JobState.FAILED and "unschedulable" in st.error
    assert any("rocm" in r for c in st.scheduling.candidates for r in c.reasons)


def test_batch_of_independent_jobs(client):
    jobs = client.submit_batch([gl.command(["python", "-c", f"print({i})"]) for i in range(3)])
    assert [j.wait(60).state for j in jobs] == [JobState.SUCCEEDED] * 3
    assert sorted(j.logs_text() for j in jobs) == ["0", "1", "2"]


def test_two_workers_share_queue(real_cp, tmp_path):
    """Two independent worker agents drain one queue; each job runs exactly once here."""
    caps = detect_capabilities(
        origin=WorkerOrigin.LOCAL, torch_probe=lambda: TorchProbe(error="skipped")
    )
    agents = []
    for i in range(2):
        ch = EmbeddedChannel.register(real_cp, f"w{i}", caps)
        agents.append(
            WorkerAgent(ch, caps, tmp_path / f"w{i}", python=sys.executable, poll_interval=0.05)
        )
    threads = [threading.Thread(target=a.run, daemon=True) for a in agents]
    for t in threads:
        t.start()
    jobs = [
        real_cp.submit(gl.command(["python", "-c", "import time; time.sleep(0.3)"]))
        for _ in range(4)
    ]
    deadline = time.time() + 60
    while time.time() < deadline and not all(
        real_cp.get_job(j.job_id).state.terminal for j in jobs
    ):
        time.sleep(0.1)
    for a in agents:
        a.stop.set()
    for t in threads:
        t.join(10)
    workers_used = {real_cp.get_job(j.job_id).tasks[0].worker_id for j in jobs}
    assert all(real_cp.get_job(j.job_id).state == JobState.SUCCEEDED for j in jobs)
    assert len(workers_used) == 2
    assert json.loads(real_cp.get_job(jobs[0].job_id).model_dump_json())["attempts"] == 1


def test_library_paths_reach_jobs_but_credentials_do_not(client, monkeypatch):
    # Found on Kaggle: the GPU driver is located through LD_LIBRARY_PATH; without it the
    # worker saw two GPUs and the job's ranks saw none.
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/driver/lib64")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:token@proxy:3128")
    script = (
        "import os; print(os.environ.get('LD_LIBRARY_PATH'), "
        "'proxy' if 'HTTPS_PROXY' in os.environ else 'no-proxy')"
    )
    job = client.submit(gl.command(["python", "-c", script]))
    assert job.wait(60).state == JobState.SUCCEEDED
    assert job.logs_text() == "/opt/driver/lib64 no-proxy"
