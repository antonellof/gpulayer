"""Request latency and throughput of the HTTP API (the path remote workers and clients use).

    python benchmarks/http_requests.py [--requests 300 --threads 8]

Starts a server with a self-signed certificate on 127.0.0.1 (the same TLS setup as
`gpulayer up`) and times GET /v1/jobs?limit=1 from one thread, then from several threads at
once, through gpulayer's HTTP client. Over a real network add the round-trip time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

from gpulayer.api.auth import generate_credentials
from gpulayer.api.client import HttpTransport
from gpulayer.api.server import ControlPlaneServer
from gpulayer.api.tls import certificate_fingerprint, ensure_self_signed
from gpulayer.artifacts.store import LocalArtifactStore
from gpulayer.control_plane.service import ControlPlane
from gpulayer.storage.sqlite import SqliteStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        cp = ControlPlane(SqliteStore(Path(tmp, "db.sqlite")), LocalArtifactStore(Path(tmp, "a")))
        cert, key = ensure_self_signed(Path(tmp, "tls"))
        creds, token, _ = generate_credentials()
        srv = ControlPlaneServer(
            cp, creds, host="127.0.0.1", port=0, tls_cert=str(cert), tls_key=str(key)
        )
        srv.start()
        try:
            t = HttpTransport(srv.url, token, fingerprint=certificate_fingerprint(cert))
            for _ in range(5):
                t.request("GET", "/v1/jobs?limit=1")  # warm up
            lat = []
            for _ in range(args.requests):
                t0 = time.perf_counter()
                t.request("GET", "/v1/jobs?limit=1")
                lat.append(time.perf_counter() - t0)
            lat.sort()
            print(
                f"one thread    p50 {statistics.median(lat) * 1000:6.2f} ms   "
                f"p95 {lat[int(len(lat) * 0.95)] * 1000:6.2f} ms   "
                f"{len(lat) / sum(lat):7.0f} req/s"
            )

            per = args.requests // args.threads

            def burst() -> None:
                for _ in range(per):
                    t.request("GET", "/v1/jobs?limit=1")

            ths = [threading.Thread(target=burst) for _ in range(args.threads)]
            t0 = time.perf_counter()
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            took = time.perf_counter() - t0
            print(f"{args.threads} threads    {per * args.threads / took:7.0f} req/s")
        finally:
            srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
