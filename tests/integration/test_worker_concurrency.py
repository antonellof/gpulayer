"""A worker with several task slots runs independent jobs at the same time (real processes)."""

import sys
import threading
import time

import pytest
from conftest import make_caps

import gpulayer as gl
from gpulayer.worker.agent import WorkerAgent
from gpulayer.worker.channel import EmbeddedChannel

pytestmark = pytest.mark.integration


def test_worker_runs_jobs_concurrently_up_to_its_slots(real_cp, tmp_path):
    caps = make_caps(max_concurrent_tasks=3)
    ch = EmbeddedChannel.register(real_cp, "w", caps)
    agent = WorkerAgent(ch, caps, tmp_path / "w")
    th = threading.Thread(target=agent.run, daemon=True)
    th.start()
    try:
        sleep = [sys.executable, "-c", "import time; time.sleep(1)"]
        t0 = time.monotonic()
        jobs = [real_cp.submit(gl.command(sleep)).job_id for _ in range(3)]
        states = [real_cp.wait_job(j, 30).state.value for j in jobs]
        elapsed = time.monotonic() - t0
    finally:
        agent.stop.set()
        real_cp.signal()
        th.join(30)
    assert states == ["succeeded"] * 3
    assert elapsed < 2.5, f"3 x 1 s jobs took {elapsed:.1f}s; expected them to overlap"
    starts = sorted(t.started_at for j in jobs for t in real_cp.get_job(j).tasks if t.started_at)
    assert (starts[-1] - starts[0]).total_seconds() < 0.9  # all started before any finished
