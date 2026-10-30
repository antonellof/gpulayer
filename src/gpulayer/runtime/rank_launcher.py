"""Direct rank launcher for DistributedJob (``launcher="direct"``). Standard library only.

    python rank_launcher.py <nproc> <first_rank> <python> <script> [args...]

Starts ``nproc`` copies of ``python script args`` with the per-rank torchrun environment
(RANK, LOCAL_RANK; the worker sets the rest), and exits when they have all exited. It replaces
torchrun's elastic agent, which costs about 1.2 s per job on start-up and shutdown
(benchmarks/distributed_startup.py), with a plain parent process:

- Rank 0 hosts the c10d TCPStore itself (TORCHELASTIC_USE_AGENT_STORE stays unset).
- Each rank's lines are re-emitted with the "[default<local rank>]:" prefix of torchrun
  ``--tee``, so the worker attributes them to ranks the same way.
- fail_fast: the first rank that exits non-zero stops the others; the exit code is its code.

The worker copies this file into the job workspace, like function_runner.py.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading

KILL_GRACE = 5.0  # seconds between SIGTERM and SIGKILL for the surviving ranks


def _pump(pipe: object, out: object, prefix: bytes, lock: threading.Lock) -> None:
    for raw in iter(pipe.readline, b""):  # type: ignore[attr-defined]
        if not raw.endswith(b"\n"):
            raw += b"\n"
        with lock:
            out.write(prefix + raw)  # type: ignore[attr-defined]
            out.flush()  # type: ignore[attr-defined]


def _stop(procs: list[subprocess.Popen[bytes]]) -> None:
    alive = [p for p in procs if p.poll() is None]
    for p in alive:
        p.send_signal(signal.SIGTERM)
    for p in alive:
        try:
            p.wait(timeout=KILL_GRACE)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main(argv: list[str]) -> int:
    nproc, first_rank = int(argv[0]), int(argv[1])
    cmd = argv[2:]
    lock = threading.Lock()
    procs: list[subprocess.Popen[bytes]] = []
    pumps: list[threading.Thread] = []
    for local in range(nproc):
        env = dict(os.environ)
        rank = str(first_rank + local)
        env.update({"RANK": rank, "LOCAL_RANK": str(local), "ROLE_RANK": rank})
        p = subprocess.Popen(
            cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        procs.append(p)
        prefix = f"[default{local}]:".encode()
        for pipe, out in ((p.stdout, sys.stdout.buffer), (p.stderr, sys.stderr.buffer)):
            t = threading.Thread(target=_pump, args=(pipe, out, prefix, lock), daemon=True)
            t.start()
            pumps.append(t)

    by_pid = {p.pid: p for p in procs}
    failed: tuple[int, int] | None = None  # (local rank, exit code)
    remaining = set(by_pid)
    while remaining:
        pid, status = os.wait()  # blocks until any rank exits; no polling
        if pid not in remaining:
            continue
        remaining.discard(pid)
        p = by_pid[pid]
        p.returncode = os.waitstatus_to_exitcode(status)
        if p.returncode != 0 and failed is None:
            failed = (procs.index(p), p.returncode)
            _stop(procs)
            remaining.clear()
    for t in pumps:
        t.join(timeout=5)
    if failed is None:
        return 0
    local, code = failed
    sys.stderr.write(f"rank {first_rank + local} exited with code {code}; stopped the others\n")
    return code if code > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
