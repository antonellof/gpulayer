"""REAL distributed execution: two worker agents (two "nodes") on one host, each
launching torchrun, forming one gloo process group of world size 4 and verifying an
all_reduce. Scope: multi-process, multi-node-topology on a single machine over
loopback. It does NOT establish multi-machine support.
"""

import threading
import time

import pytest

import gpulayer as gl
from gpulayer.contracts import JobState, WorkerOrigin
from gpulayer.gpu.discovery import detect_capabilities
from gpulayer.worker.agent import WorkerAgent
from gpulayer.worker.channel import EmbeddedChannel

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.distributed

TRAIN = """
import os, sys, torch, torch.distributed as dist
dist.init_process_group(os.environ["GPULAYER_COMM_BACKEND"])
rank, world = dist.get_rank(), dist.get_world_size()
t = torch.tensor([float(rank + 1)])
dist.all_reduce(t)
expected = world * (world + 1) / 2
# Text and newline in separate writes, the case that used to interleave between ranks
# before each rank's output was captured separately (torchrun --tee).
print(f"rank={rank} world={world} sum={t.item()} node={os.environ['GPULAYER_NODE_RANK']}",
      end="", flush=True)
print("", flush=True)
assert t.item() == expected, (t.item(), expected)
if rank == 0:
    out = os.path.join(os.environ["GPULAYER_OUTPUT_DIR"], "result.txt")
    open(out, "w").write(str(t.item()))
if os.environ.get("FAIL_RANK") == str(rank):
    raise SystemExit(3)
dist.barrier()
dist.destroy_process_group()
"""


@pytest.fixture
def group(real_cp, tmp_path):
    caps = detect_capabilities(origin=WorkerOrigin.LOCAL, advertise_address="127.0.0.1")
    if "torchrun" not in caps.launchers:
        pytest.skip("torch.distributed with gloo not available")
    agents = []
    for i in range(2):
        ch = EmbeddedChannel.register(real_cp, f"node{i}", caps)
        agents.append(WorkerAgent(ch, caps, tmp_path / f"node{i}", poll_interval=0.05))
    threads = [threading.Thread(target=a.run, daemon=True) for a in agents]
    for t in threads:
        t.start()
    yield real_cp
    for a in agents:
        a.stop.set()
    for t in threads:
        t.join(30)


def _wait(cp, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = cp.get_job(job_id)
        if st.state.terminal:
            return st
        time.sleep(0.2)
    raise TimeoutError(cp.get_job(job_id))


def _source(cp, tmp_path):
    from gpulayer.artifacts.store import build_source_bundle

    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(TRAIN)
    return cp.put_artifact(build_source_bundle(src), "src", "source")


def test_two_node_allreduce(group, tmp_path):
    cp = group
    spec = gl.distributed(
        "train.py",
        source=_source(cp, tmp_path),
        num_workers=2,
        procs_per_worker=2,
        timeout_seconds=150,
    )
    st = _wait(cp, cp.submit(spec).job_id)
    logs = [line.line for line in cp.job_logs(st.job_id)]
    assert st.state == JobState.SUCCEEDED, (st.error, logs[-20:])
    sums = [line for line in logs if line.startswith("rank=")]
    assert len(sums) == 4 and all("world=4 sum=10.0" in s for s in sums)
    # Every line from the training script is tagged with the global rank that wrote it.
    tagged = {
        line.rank: line.line for line in cp.job_logs(st.job_id) if line.line.startswith("rank=")
    }
    assert sorted(tagged) == [0, 1, 2, 3]
    assert all(text.startswith(f"rank={r} ") for r, text in tagged.items())
    assert {t.node_rank for t in st.tasks} == {0, 1}
    assert len({t.worker_id for t in st.tasks}) == 2
    res = cp.job_result(st.job_id)
    assert [o.name for o in res.outputs] == ["node0/result.txt"]


def test_rank_failure_fails_whole_job(group, tmp_path):
    cp = group
    spec = gl.distributed(
        "train.py",
        source=_source(cp, tmp_path),
        num_workers=2,
        procs_per_worker=1,
        env={"FAIL_RANK": "1"},
        timeout_seconds=150,
    )
    st = _wait(cp, cp.submit(spec).job_id)
    assert st.state == JobState.FAILED  # fail_fast: decided on the first failed node
    # The surviving node is then stopped (heartbeat cancel or torchrun peer failure).
    deadline = time.time() + 60
    while time.time() < deadline and not all(t.state.terminal for t in cp.get_job(st.job_id).tasks):
        time.sleep(0.2)
    states = {t.state.value for t in cp.get_job(st.job_id).tasks}
    assert states <= {"failed", "cancelled"} and "failed" in states
