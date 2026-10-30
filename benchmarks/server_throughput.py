"""Jobs per second through one server with remote workers: the path a fleet uses.

    python benchmarks/server_throughput.py [--workers 6 --slots 4 --jobs 1000] [--batch]

Starts a TLS control-plane server in this process, `--workers` worker processes
(`gpulayer join`, separate homes) and submits `--jobs` tiny function jobs from 16 client
threads, then waits for all of them. Reports jobs/s and this process's CPU time per job (the
server's work; the client threads are light). `--profile` writes a cProfile of the server.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gpulayer as gl
from gpulayer.api.auth import generate_credentials
from gpulayer.api.server import ControlPlaneServer
from gpulayer.api.tls import certificate_fingerprint, ensure_self_signed
from gpulayer.artifacts.store import LocalArtifactStore
from gpulayer.contracts import PlacementPolicy
from gpulayer.control_plane.service import ControlPlane
from gpulayer.storage.sqlite import SqliteStore


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1000)
    ap.add_argument("--profile")
    ap.add_argument("--batch", action="store_true", help="submit with Client.submit_batch")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    cp = ControlPlane(SqliteStore(tmp / "db.sqlite"), LocalArtifactStore(tmp / "a"))
    cert, key = ensure_self_signed(tmp / "tls")
    creds, token, join = generate_credentials()
    srv = ControlPlaneServer(
        cp, creds, host="127.0.0.1", port=0, tls_cert=str(cert), tls_key=str(key)
    )
    srv.start()
    fp = certificate_fingerprint(cert)
    procs = []
    for i in range(args.workers):
        env = {**os.environ, "GPULAYER_HOME": str(tmp / f"w{i}")}
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gpulayer.cli.main",
                    "join",
                    srv.url,
                    "--token",
                    join,
                    "--fingerprint",
                    fp,
                    "--name",
                    f"w{i}",
                    "--concurrency",
                    str(args.slots),
                    "--advertise-address",
                    "127.0.0.1",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    try:
        c = gl.Client(url=srv.url, token=token, fingerprint=fp)
        deadline = time.time() + 60
        while len(c.workers()) < args.workers and time.time() < deadline:
            time.sleep(0.2)
        d = tmp / "src"
        d.mkdir()
        (d / "tasks.py").write_text("def tiny(i):\n    return i\n")
        src = c.package(d)
        specs = [
            gl.function("tasks:tiny", i, source=src, placement=PlacementPolicy(target="remote"))
            for i in range(args.jobs)
        ]
        prof = cProfile.Profile() if args.profile else None
        if prof:
            prof.enable()
        c0, t0 = cpu(), time.perf_counter()
        with ThreadPoolExecutor(16) as ex:
            jobs = c.submit_batch(specs) if args.batch else list(ex.map(c.submit, specs))
            states = list(ex.map(lambda j: j.wait(600).state.value, jobs))
        wall, used = time.perf_counter() - t0, cpu() - c0
        if prof:
            prof.disable()
            prof.dump_stats(args.profile)
        ok = states.count("succeeded")
        print(
            f"{'batch ' if args.batch else ''}{args.jobs} jobs on {args.workers} workers x "
            f"{args.slots} slots: {ok} succeeded, "
            f"{wall:.1f}s, {args.jobs / wall:.1f} jobs/s, server+client CPU "
            f"{used / args.jobs * 1000:.1f} ms per job"
        )
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(10)
        srv.shutdown()


if __name__ == "__main__":
    main()
