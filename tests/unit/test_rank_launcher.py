"""rank_launcher.py on its own: real subprocesses, no torch needed."""

import subprocess
import sys
import time

from gpulayer.runtime import rank_launcher

ECHO = """
import os, sys, time
r = os.environ["RANK"]
print(f"rank={r} local={os.environ['LOCAL_RANK']} role={os.environ['ROLE_RANK']}", end="")
sys.stdout.flush()
print(flush=True)
print(f"err {r}", file=sys.stderr, flush=True)
if os.environ.get("FAIL_RANK") == r:
    sys.exit(7)
if os.environ.get("HANG"):
    time.sleep(60)
"""


def run(tmp_path, nproc, first, **env):
    script = tmp_path / "s.py"
    script.write_text(ECHO)
    return subprocess.run(
        [sys.executable, rank_launcher.__file__, str(nproc), str(first)]
        + [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", **env},
    )


def test_ranks_get_their_environment_and_prefixed_lines(tmp_path):
    p = run(tmp_path, 3, 3)
    assert p.returncode == 0, p.stderr
    out = sorted(p.stdout.splitlines())
    assert out == [
        f"[default{i}]:rank={3 + i} local={i} role={3 + i}" for i in range(3)
    ]  # whole lines, never split between ranks
    assert sorted(p.stderr.splitlines()) == [f"[default{i}]:err {3 + i}" for i in range(3)]


def test_first_failure_stops_the_others_and_sets_the_exit_code(tmp_path):
    t0 = time.time()
    p = run(tmp_path, 2, 0, FAIL_RANK="1", HANG="1")
    assert p.returncode == 7
    assert time.time() - t0 < 30  # rank 0 would sleep 60 s
    assert "rank 1 exited with code 7; stopped the others" in p.stderr
