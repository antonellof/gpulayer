"""``gpulayer`` command-line interface.

Without ``--url`` (or $GPULAYER_URL) commands use the embedded control plane in
$GPULAYER_HOME. With it, they talk to a self-hosted control plane over HTTP(S).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from gpulayer import __version__
from gpulayer.contracts import (
    RUNTIME_PROFILES,
    Accelerator,
    CommandJob,
    DistributedJob,
    ExecutionTarget,
    FunctionJob,
    InputArtifact,
    JobSpec,
    JobState,
    PlacementPolicy,
    ProvisioningPolicy,
    ResourceRequest,
    RetryPolicy,
    SecretRef,
    WorkerOrigin,
)
from gpulayer.errors import GpuLayerError
from gpulayer.sdk.client import Client

EXIT_JOB_FAILED = 1
EXIT_USAGE = 2


def _kv(items: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"error: {what} must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        if isinstance(obj, list):
            print(
                json.dumps(
                    [o.model_dump(mode="json") if isinstance(o, BaseModel) else o for o in obj],
                    indent=2,
                )
            )
        elif isinstance(obj, BaseModel):
            print(obj.model_dump_json(indent=2))
        else:
            print(json.dumps(obj, indent=2, default=str))


def _client(args: argparse.Namespace, *, local_worker: bool = True) -> Client:
    return Client(
        fingerprint=args.fingerprint,
        url=args.url,
        token=args.token,
        home=args.home,
        ca_file=args.ca_file,
        local_worker=local_worker,
        worker_python=getattr(args, "python", None),
    )


# ----------------------------------------------------------------------- server


def cmd_server_init(args: argparse.Namespace) -> int:
    from gpulayer.api.auth import generate_credentials
    from gpulayer.sdk.home import gpulayer_home

    path = gpulayer_home(args.home) / "server-credentials.json"
    if path.exists() and not args.rotate:
        print(f"credentials already exist at {path}; use --rotate to replace them", file=sys.stderr)
        return EXIT_USAGE
    creds, client, join = generate_credentials()
    creds.save(path)
    print(f"Wrote token hashes to {path}. The tokens below are shown ONCE; store them safely.")
    print(f"client token (GPULAYER_TOKEN):       {client}")
    print(f"worker join token (GPULAYER_JOIN_TOKEN): {join}")
    return 0


def cmd_server_start(args: argparse.Namespace) -> int:
    from gpulayer.api.auth import ServerCredentials
    from gpulayer.api.server import ControlPlaneServer
    from gpulayer.control_plane.service import ControlPlaneConfig
    from gpulayer.sdk.home import gpulayer_home, open_control_plane

    home = gpulayer_home(args.home)
    cred_path = home / "server-credentials.json"
    if not cred_path.exists():
        print("no credentials: run `gpulayer server init` first", file=sys.stderr)
        return EXIT_USAGE
    cp = open_control_plane(home, config=ControlPlaneConfig(public_url=args.public_url))
    server = ControlPlaneServer(
        cp,
        ServerCredentials.load(cred_path),
        host=args.host,
        port=args.port,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        allow_insecure_remote=args.allow_insecure_remote,
    )
    print(f"GPUlayer control plane listening on {server.url} (state: {home})", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


# ------------------------------------------------------------------------ up/join


def _up_tokens(home: Path) -> tuple[str, str]:
    """Client and join tokens for `gpulayer up`, kept in a 0600 file so restarts reuse them."""
    import secrets

    path = home / "up-tokens.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["client_token"], data["join_token"]
    client, join = "glc_" + secrets.token_urlsafe(32), "glj_" + secrets.token_urlsafe(32)
    path.write_text(json.dumps({"client_token": client, "join_token": join}))
    os.chmod(path, 0o600)
    return client, join


def _up_credentials(home: Path, client_token: str, join_token: str) -> Any:
    """Make sure the `up` tokens are in the credentials file (`gpulayer token` edits it)."""
    from datetime import datetime, timezone

    from gpulayer.api.auth import ServerCredentials, TokenEntry, token_hash

    path = home / "server-credentials.json"
    creds = ServerCredentials.load(path) if path.exists() else ServerCredentials(path=path)
    now = datetime.now(timezone.utc).isoformat()
    for name, role, token in (("up-admin", "admin", client_token), ("up-join", "join", join_token)):
        if not any(t.hash == token_hash(token) for t in creds.tokens):
            creds.tokens.append(
                TokenEntry(name=name, role=role, hash=token_hash(token), created_at=now)
            )
    creds.save(path)
    return creds


def cmd_up(args: argparse.Namespace) -> int:
    """One command: control plane + (optionally) this machine as a worker."""
    from gpulayer.api.server import ControlPlaneServer, _is_loopback
    from gpulayer.api.tls import certificate_fingerprint, ensure_self_signed
    from gpulayer.control_plane.service import ControlPlaneConfig
    from gpulayer.gpu.discovery import default_advertise_address, detect_capabilities
    from gpulayer.sdk.client import _python_runner
    from gpulayer.sdk.home import gpulayer_home, open_control_plane
    from gpulayer.worker.agent import WorkerAgent
    from gpulayer.worker.channel import EmbeddedChannel

    home = gpulayer_home(args.home)
    client_token, join_token = _up_tokens(home)
    creds = _up_credentials(home, client_token, join_token)
    loopback = _is_loopback(args.host)
    cert, key, fingerprint = args.tls_cert, args.tls_key, None
    if not cert and (args.tls or (not loopback and not args.insecure_http)):
        cert_path, key_path = ensure_self_signed(home / "tls")
        cert, key = str(cert_path), str(key_path)
        fingerprint = certificate_fingerprint(cert_path)
    advertise = args.advertise_address or ("127.0.0.1" if loopback else default_advertise_address())
    cp = open_control_plane(home)
    server = ControlPlaneServer(
        cp,
        creds,
        host=args.host,
        port=args.port,
        tls_cert=cert,
        tls_key=key,
        allow_insecure_remote=args.insecure_http,
    )
    port = server.httpd.server_address[1]
    url = f"{'https' if cert else 'http'}://{advertise}:{port}"
    cp.config = ControlPlaneConfig(public_url=url)
    server.start()

    agent = None
    worker_desc = "disabled (--no-worker)"
    if not args.no_worker:
        python = args.python or sys.executable
        caps = detect_capabilities(
            origin=WorkerOrigin.EXISTING,
            advertise_address=advertise,
            labels=_kv(args.label, "--label"),
            runner=_python_runner(python),
        )
        channel = EmbeddedChannel.register(cp, args.name or caps.hostname, caps)
        agent = WorkerAgent(channel, caps, home / "workers" / channel.worker_id, python=python)
        threading.Thread(target=agent.run, daemon=True).start()
        devices = ", ".join(d.accelerator.value for d in caps.accelerators) or "cpu"
        worker_desc = f"{caps.hostname} ({devices}, {caps.cpu_cores} CPU cores)"

    fp_flag = f" --fingerprint {fingerprint}" if fingerprint else ""
    fp_arg = f', fingerprint="{fingerprint}"' if fingerprint else ""
    fp_env = f" GPULAYER_FINGERPRINT={fingerprint}" if fingerprint else ""
    tls_note = (
        "TLS, self-signed certificate pinned by fingerprint"
        if fingerprint
        else ("TLS" if cert else "plain HTTP: use only on a trusted network")
    )
    print(
        f"""
GPUlayer is up at {url}  ({tls_note})
Worker on this machine: {worker_desc}

Add a machine (GPU server, workstation, cloud VM) by running on it:
  gpulayer join {url} --token {join_token}{fp_flag}

Dashboard: {url}/ui/  (sign in with the client token below)
Prometheus metrics: {url}/metrics

Use it from Python:
  import gpulayer as gl
  gl.connect("{url}", token="{client_token}"{fp_arg})

Or from a shell:
  export GPULAYER_URL={url} GPULAYER_TOKEN={client_token}{fp_env}

Tokens are kept in {home / "up-tokens.json"}. Press Ctrl+C to stop.
""",
        flush=True,
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    if agent is not None:
        agent.stop.set()
    server.shutdown()
    print("stopped")
    return 0


def cmd_join(args: argparse.Namespace) -> int:
    """Register this machine as a worker of a GPUlayer server (`gpulayer up`)."""
    args.url = args.server_url
    args.join_token = args.token
    return cmd_worker_start(args)


# --------------------------------------------------------------------- maintenance


def cmd_gc(args: argparse.Namespace) -> int:
    from gpulayer.api.auth import parse_ttl

    age = parse_ttl(args.older_than).total_seconds()
    with _client(args, local_worker=False) as c:
        report = c._api.gc(age, delete_jobs=args.delete_jobs, dry_run=args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    mib = report.get("artifact_bytes", 0) / 1024**2
    print(
        f"{verb}: {report['logs']} log lines, {report['events']} events, "
        f"{report['jobs']} jobs, {report['workers']} workers, "
        f"{report['artifacts']} artifacts ({mib:.1f} MiB)"
    )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Serve the dashboard for the local (embedded) control plane on this machine."""
    import webbrowser

    from gpulayer.api.auth import ServerCredentials
    from gpulayer.api.server import ControlPlaneServer
    from gpulayer.sdk.home import open_control_plane

    if args.url:
        print(f"open {args.url.rstrip('/')}/ui/ and sign in with a client token", file=sys.stderr)
        return 0
    creds = ServerCredentials()  # in memory only: a one-time admin token for this session
    token = creds.create("dashboard", "admin")
    server = ControlPlaneServer(
        open_control_plane(args.home), creds, host="127.0.0.1", port=args.port
    )
    port = server.httpd.server_address[1]
    url = f"http://127.0.0.1:{port}/ui/#token={token}"
    server.start()
    print(f"GPUlayer dashboard: {url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    server.shutdown()
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        sys.stdout.write(c._api.metrics_text())
    return 0


# ------------------------------------------------------------------------ tokens


def _creds_path(args: argparse.Namespace) -> Path:
    from gpulayer.sdk.home import gpulayer_home

    return gpulayer_home(args.home) / "server-credentials.json"


def cmd_token_create(args: argparse.Namespace) -> int:
    from gpulayer.api.auth import ServerCredentials, parse_ttl

    path = _creds_path(args)
    creds = ServerCredentials.load(path) if path.exists() else ServerCredentials(path=path)
    token = creds.create(args.name, args.role, parse_ttl(args.ttl) if args.ttl else None)
    creds.save(path)
    entry = creds.tokens[-1]
    expiry = f", expires {entry.expires_at}" if entry.expires_at else ""
    print(f"created {args.role} token '{args.name}'{expiry}. It is shown once:", file=sys.stderr)
    print(token)
    return 0


def cmd_token_list(args: argparse.Namespace) -> int:
    from gpulayer.api.auth import ServerCredentials

    path = _creds_path(args)
    if not path.exists():
        print("no credentials yet: run `gpulayer up` or `gpulayer server init`", file=sys.stderr)
        return EXIT_USAGE
    for t in ServerCredentials.load(path).tokens:
        status = "active" if t.active() else ("revoked" if t.revoked else "expired")
        print(
            f"{t.name:20} {t.role:10} {status:8} created {t.created_at[:19]}"
            + (f"  expires {t.expires_at[:19]}" if t.expires_at else "")
        )
    return 0


def cmd_token_revoke(args: argparse.Namespace) -> int:
    from gpulayer.api.auth import ServerCredentials

    path = _creds_path(args)
    creds = ServerCredentials.load(path)
    n = creds.revoke(args.name)
    if not n:
        print(f"no active token named {args.name!r}", file=sys.stderr)
        return EXIT_USAGE
    creds.save(path)
    print(f"revoked {args.name}; a running server applies this on its next request")
    return 0


# ----------------------------------------------------------------------- worker


def cmd_worker_start(args: argparse.Namespace) -> int:
    from gpulayer.gpu.discovery import default_advertise_address, detect_capabilities
    from gpulayer.sdk.client import _python_runner
    from gpulayer.sdk.home import gpulayer_home, open_control_plane
    from gpulayer.worker.agent import WorkerAgent

    python = args.python or sys.executable
    labels = _kv(args.label, "--label")
    if args.url:
        from gpulayer.api.client import HttpWorkerChannel

        join = args.join_token or os.environ.get("GPULAYER_JOIN_TOKEN")
        if not join:
            print(
                "a join token is required (--join-token or $GPULAYER_JOIN_TOKEN)", file=sys.stderr
            )
            return EXIT_USAGE
        caps = detect_capabilities(
            origin=WorkerOrigin.EXISTING,
            labels=labels,
            advertise_address=args.advertise_address or default_advertise_address(),
            runner=_python_runner(python),
            max_concurrent_tasks=getattr(args, "concurrency", None),
        )
        channel: Any = HttpWorkerChannel.register(
            args.url,
            join,
            args.name or caps.hostname,
            caps,
            ca_file=args.ca_file,
            fingerprint=args.fingerprint,
        )
    else:
        from gpulayer.worker.channel import EmbeddedChannel

        caps = detect_capabilities(
            origin=WorkerOrigin.LOCAL,
            labels=labels,
            advertise_address=args.advertise_address or "127.0.0.1",
            runner=_python_runner(python),
            max_concurrent_tasks=getattr(args, "concurrency", None),
        )
        channel = EmbeddedChannel.register(
            open_control_plane(args.home), args.name or f"local-{os.getpid()}", caps
        )
    workdir = (
        Path(args.workdir)
        if args.workdir
        else gpulayer_home(args.home) / "workers" / channel.worker_id
    )
    agent = WorkerAgent(channel, caps, workdir, python=python)
    accel = ", ".join(f"{d.accelerator.value}:{d.name}" for d in caps.accelerators) or "cpu only"
    print(f"worker {channel.worker_id} registered ({caps.origin.value}; {accel})", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: agent.stop.set())
    try:
        n = agent.run(max_tasks=args.max_tasks, idle_exit_seconds=args.idle_exit)
    except KeyboardInterrupt:
        agent.stop.set()
        return 130
    print(f"worker {channel.worker_id} stopped after {n} task(s)")
    return 0


def cmd_worker_status(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        workers = c.workers()
    hidden = 0
    if not getattr(args, "all", False):
        shown = [w for w in workers if w.state.value != "offline"]
        hidden, workers = len(workers) - len(shown), shown
    if args.json:
        _print(workers, True)
        return 0
    if not workers:
        print("no workers online")
    for w in workers:
        caps = w.capabilities
        accel = ",".join(f"{d.accelerator.value}" for d in caps.accelerators) or "cpu"
        hb = w.last_heartbeat_at.strftime("%H:%M:%S") if w.last_heartbeat_at else "-"
        print(
            f"{w.worker_id:18} {w.name:22} {w.state.value:9} {caps.target.value:6} "
            f"{accel:10} tasks={len(w.running_task_ids)} heartbeat={hb}"
        )
    if hidden:
        print(
            f"({hidden} offline worker(s) hidden; --all shows them, `gpulayer gc` removes old ones)"
        )
    return 0


def cmd_worker_revoke(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        c._api.revoke_worker(args.worker_id)
    print(f"revoked {args.worker_id}")
    return 0


# -------------------------------------------------------------------- resources


def cmd_resources_detect(args: argparse.Namespace) -> int:
    from gpulayer.gpu.discovery import detect_capabilities
    from gpulayer.gpu.support import SUPPORT
    from gpulayer.sdk.client import _python_runner

    caps = detect_capabilities(runner=_python_runner(args.python or sys.executable))
    if args.json:
        _print(caps, True)
        return 0
    print(
        f"host      {caps.hostname} ({caps.os}/{caps.cpu_arch}), {caps.cpu_cores} cores, "
        f"{caps.memory_gb} GB RAM, python {caps.python_version}"
    )
    print(
        f"torch     {caps.frameworks.get('torch', 'not installed')}"
        f"  distributed backends: {','.join(caps.distributed_backends) or 'none'}"
    )
    for d in caps.accelerators:
        mem = f"{d.memory_gb} GB {d.memory_kind.value}" if d.memory_gb else "memory unknown"
        sup = SUPPORT[d.accelerator]
        print(
            f"device    [{d.index}] {d.accelerator.value} {d.name} ({mem}) "
            f"independent={sup.independent.value} distributed={sup.distributed.value}"
        )
    if not caps.accelerators:
        print("device    none detected (CPU execution only)")
    for n in caps.notes:
        print(f"note      {n}")
    return 0


# ------------------------------------------------------------------------ submit


def build_spec(args: argparse.Namespace, client: Client) -> JobSpec:
    if args.file:
        spec = JobSpec.model_validate_json(Path(args.file).read_text())
        return spec
    source = client.package(args.source) if args.source else None
    if args.distributed:
        if not args.entrypoint:
            raise SystemExit("error: --distributed requires --entrypoint SCRIPT")
        if source is None:
            raise SystemExit("error: --distributed requires --source DIR")
        workload: Any = DistributedJob(
            entrypoint=args.entrypoint,
            args=args.command,
            source=source,
            num_workers=args.nodes,
            procs_per_worker=args.procs_per_node,
            comm_backend=args.backend,
        )
    elif args.function:
        workload = FunctionJob(
            entrypoint=args.function, args=[json.loads(a) for a in args.arg or []], source=source
        )
    else:
        if not args.command:
            raise SystemExit("error: give a command after --, or --function, --distributed, -f")
        workload = CommandJob(command=args.command, source=source)

    inputs = []
    for item in args.input or []:
        local, _, dest = item.partition(":")
        inputs.append(InputArtifact(artifact=client.upload(local), path=dest or Path(local).name))
    frameworks: dict[str, str] = {}
    for f in args.framework or []:
        name = f.split(">")[0].split("<")[0].split("=")[0].split("~")[0].split("!")[0]
        frameworks[name] = f[len(name) :] or ">=0"
    accel_name = args.accelerator or (
        RUNTIME_PROFILES[args.profile]["accelerator"] if args.profile in RUNTIME_PROFILES else "cpu"
    )
    accelerator = Accelerator(accel_name)
    resources = ResourceRequest(
        profile=args.profile,
        accelerator=accelerator,
        gpu_count=args.gpus or (1 if accelerator != Accelerator.CPU else 0),
        gpu_memory_gb=args.gpu_memory,
        gpu_memory_kind=args.gpu_memory_kind,
        cpu_cores=args.cpus,
        memory_gb=args.memory,
        frameworks=frameworks,
    )
    provisioning = ProvisioningPolicy(provider=args.provider) if args.provider else None
    placement = PlacementPolicy(
        target=ExecutionTarget(args.target),
        fallback_targets=[ExecutionTarget(t) for t in args.fallback or []],
        worker_ids=args.worker or [],
        worker_labels=_kv(args.label, "--label"),
        on_unschedulable="queue" if args.queue_if_unschedulable else "fail",
        max_hourly_cost=args.max_hourly_cost,
        provisioning=provisioning,
    )
    secrets = [SecretRef(name=k, key=v) for k, v in _kv(args.secret, "--secret").items()]
    return JobSpec(
        name=args.name,
        workload=workload,
        resources=resources,
        placement=placement,
        env=_kv(args.env, "--env"),
        secrets=secrets,
        inputs=inputs,
        timeout_seconds=args.timeout,
        retry=RetryPolicy(max_attempts=args.max_attempts),
    )


def cmd_submit(args: argparse.Namespace) -> int:
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    # A detached embedded submission cannot use an in-process worker (it would die with
    # this process), so it waits in the queue for `gpulayer worker start`.
    in_process = not args.detach and not args.url
    if args.detach:
        args.queue_if_unschedulable = True
    with _client(args, local_worker=in_process) as c:
        spec = build_spec(args, c)
        job = c.submit(spec)
        print(
            f"submitted {job.id} ({spec.workload_type.value}, target "
            f"{spec.placement.target.value})",
            file=sys.stderr,
        )
        if args.detach:
            print(job.id)
            return 0
        try:
            for line in job.follow_logs():
                stream = sys.stderr if line.stream != "stdout" else sys.stdout
                prefix = f"[{line.origin}] " if spec.workload_type.value == "distributed" else ""
                print(prefix + line.line, file=stream, flush=True)
        except KeyboardInterrupt:
            print(f"\ninterrupted; cancelling {job.id}", file=sys.stderr)
            job.cancel()
            job.wait(60)
        st = job.status()
        res = job.result()
        print(f"{job.id} {st.state.value}" + (f": {st.error}" if st.error else ""), file=sys.stderr)
        if st.state == JobState.FAILED and st.scheduling and not st.scheduling.placed:
            _explain(st.scheduling, sys.stderr)
        if res.return_value is not None:
            print(json.dumps(res.return_value))
        return 0 if st.state == JobState.SUCCEEDED else (res.exit_code or EXIT_JOB_FAILED)


def _explain(decision: Any, out: Any) -> None:
    print(f"scheduling: {decision.summary}", file=out)
    for cand in decision.candidates:
        print(
            f"  {cand.worker_id}: {'eligible' if cand.eligible else '; '.join(cand.reasons)}",
            file=out,
        )


# ------------------------------------------------------------------------- jobs


def cmd_job_status(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        st = c.job(args.job_id).status()
    if args.json:
        _print(st, True)
        return 0
    print(f"job       {st.job_id} ({st.name})")
    print(f"state     {st.state.value}" + (f"  error: {st.error}" if st.error else ""))
    print(
        f"workload  {st.workload_type.value}, target {st.spec.placement.target.value}, "
        f"accelerator {st.spec.resources.accelerator.value}, attempts {st.attempts}"
    )
    if st.scheduling:
        _explain(st.scheduling, sys.stdout)
    for t in st.tasks:
        print(
            f"task      {t.task_id} attempt={t.attempt} node={t.node_rank} worker={t.worker_id} "
            f"{t.state.value}"
            + (f" exit={t.exit_code}" if t.exit_code is not None else "")
            + (f" ({t.error})" if t.error else "")
        )
    return 0


def cmd_job_list(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        jobs = c.jobs(limit=args.limit)
    if args.json:
        _print(jobs, True)
        return 0
    for j in jobs:
        print(
            f"{j.job_id:18} {j.name:20} {j.state.value:12} {j.workload_type.value:12} "
            f"{j.spec.placement.target.value:7} {j.created_at:%Y-%m-%d %H:%M:%S}"
        )
    return 0


def cmd_job_logs(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        job = c.job(args.job_id)
        lines = job.follow_logs() if args.follow else iter(job.logs())
        for line in lines:
            if line.stream == "system" and not args.system:
                continue
            print(f"[{line.origin} {line.stream}] {line.line}" if args.verbose else line.line)
    return 0


def cmd_job_result(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        res = c.job(args.job_id).result()
        if args.download:
            out = Path(args.download)
            for ref in res.outputs:
                dest = (out / ref.name).resolve()
                if not dest.is_relative_to(out.resolve()):
                    raise GpuLayerError(f"refusing unsafe output name {ref.name!r}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(c.download(ref))
                print(f"downloaded {ref.name} -> {dest}", file=sys.stderr)
    _print(res, True)
    return 0


def cmd_job_cancel(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        st = c.job(args.job_id).cancel()
    print(
        f"{st.job_id} {st.state.value}"
        + (" (cancellation requested)" if not st.state.terminal else "")
    )
    return 0


def cmd_job_events(args: argparse.Namespace) -> int:
    with _client(args, local_worker=False) as c:
        events = c.job(args.job_id).events()
    if args.json:
        _print(events, True)
        return 0
    for e in events:
        print(f"{e.timestamp:%H:%M:%S.%f}"[:-3] + f"  {e.type:22} {e.message}")
    return 0


# ------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpulayer", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"gpulayer {__version__}")
    p.add_argument(
        "--home", default=None, help="state directory (default $GPULAYER_HOME or ~/.gpulayer)"
    )
    p.add_argument(
        "--url",
        default=os.environ.get("GPULAYER_URL"),
        help="remote control plane URL (default $GPULAYER_URL; unset = embedded)",
    )
    p.add_argument("--token", default=None, help="client token (default $GPULAYER_TOKEN)")
    p.add_argument("--ca-file", default=None, help="CA bundle to verify a TLS control plane")
    p.add_argument(
        "--fingerprint",
        default=os.environ.get("GPULAYER_FINGERPRINT"),
        help="pinned server certificate, as printed by `gpulayer up` ($GPULAYER_FINGERPRINT)",
    )
    p.add_argument("-v", "--log-level", default="WARNING")
    sub = p.add_subparsers(dest="group", required=True)

    s = sub.add_parser("up", help="start a server (and use this machine as a worker)")
    s.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    s.add_argument("--port", type=int, default=8470)
    s.add_argument("--advertise-address", help="address other machines use to reach this one")
    s.add_argument("--no-worker", action="store_true", help="do not run jobs on this machine")
    s.add_argument("--name", help="worker name for this machine")
    s.add_argument("--label", action="append", metavar="KEY=VALUE")
    s.add_argument("--python", help="interpreter for jobs on this machine")
    s.add_argument("--tls", action="store_true", help="use TLS even on a loopback address")
    s.add_argument("--tls-cert", help="use this certificate instead of a self-signed one")
    s.add_argument("--tls-key")
    s.add_argument(
        "--insecure-http",
        action="store_true",
        help="plain HTTP on a non-loopback address (trusted networks only)",
    )
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("join", help="add this machine as a worker of a GPUlayer server")
    s.add_argument("server_url", help="URL printed by `gpulayer up`")
    s.add_argument(
        "--token",
        default=os.environ.get("GPULAYER_JOIN_TOKEN"),
        help="join token printed by `gpulayer up`",
    )
    # Accept these after `join` too (as printed by `gpulayer up`). SUPPRESS keeps the
    # global value when they are given before the subcommand instead.
    s.add_argument(
        "--fingerprint",
        default=argparse.SUPPRESS,
        help="server certificate fingerprint printed by `gpulayer up`",
    )
    s.add_argument("--ca-file", default=argparse.SUPPRESS)
    s.add_argument("--name")
    s.add_argument("--label", action="append", metavar="KEY=VALUE")
    s.add_argument("--python", help="interpreter used for jobs (default: this one)")
    s.add_argument("--advertise-address", help="address peers use for distributed rendezvous")
    s.add_argument("--concurrency", type=int, help="tasks to run at once (default: one per GPU)")
    s.add_argument("--workdir")
    s.add_argument("--max-tasks", type=int)
    s.add_argument("--idle-exit", type=float, metavar="SECONDS")
    s.set_defaults(fn=cmd_join)

    s = sub.add_parser("gc", help="remove old logs, events, jobs and unused artifacts")
    s.add_argument("--older-than", required=True, metavar="AGE", help="e.g. 7d, 12h, 90m")
    s.add_argument(
        "--delete-jobs",
        action="store_true",
        help="also delete finished jobs older than AGE (not only their logs)",
    )
    s.add_argument("--dry-run", action="store_true", help="report without deleting")
    s.set_defaults(fn=cmd_gc)

    s = sub.add_parser("dashboard", help="open the web dashboard for local jobs")
    s.add_argument("--port", type=int, default=8471)
    s.add_argument("--no-open", action="store_true", help="print the URL without opening it")
    s.set_defaults(fn=cmd_dashboard)

    s = sub.add_parser("metrics", help="print Prometheus metrics (embedded or server)")
    s.set_defaults(fn=cmd_metrics)

    tk = sub.add_parser("token", help="manage server tokens (on the server machine)")
    tks = tk.add_subparsers(dest="cmd", required=True)
    s = tks.add_parser("create", help="create a named token (printed once)")
    s.add_argument("name")
    s.add_argument("--role", choices=["admin", "submitter", "join", "metrics"], default="submitter")
    s.add_argument("--ttl", help="expire after e.g. 90m, 12h or 7d")
    s.set_defaults(fn=cmd_token_create)
    s = tks.add_parser("list", help="list tokens (never their values)")
    s.set_defaults(fn=cmd_token_list)
    s = tks.add_parser("revoke", help="revoke a token by name")
    s.add_argument("name")
    s.set_defaults(fn=cmd_token_revoke)

    srv = sub.add_parser("server", help="self-hosted control plane").add_subparsers(
        dest="cmd", required=True
    )
    s = srv.add_parser("init", help="generate client and worker-join tokens")
    s.add_argument("--rotate", action="store_true")
    s.set_defaults(fn=cmd_server_init)
    s = srv.add_parser("start", help="serve the control-plane API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8470)
    s.add_argument("--tls-cert")
    s.add_argument("--tls-key")
    s.add_argument(
        "--allow-insecure-remote",
        action="store_true",
        help="allow plaintext HTTP on a non-loopback address (not recommended)",
    )
    s.add_argument("--public-url", help="URL that provisioned workers use to reach this server")
    s.set_defaults(fn=cmd_server_start)

    wk = sub.add_parser("worker", help="run or inspect workers").add_subparsers(
        dest="cmd", required=True
    )
    s = wk.add_parser("start", help="register this machine and execute assignments")
    s.add_argument("--name")
    s.add_argument("--label", action="append", metavar="KEY=VALUE")
    s.add_argument("--join-token", help="default $GPULAYER_JOIN_TOKEN (remote mode)")
    s.add_argument("--python", help="interpreter used for workloads (default: this one)")
    s.add_argument("--advertise-address", help="address peers use for distributed rendezvous")
    s.add_argument("--concurrency", type=int, help="tasks to run at once (default: one per GPU)")
    s.add_argument("--workdir")
    s.add_argument("--max-tasks", type=int)
    s.add_argument(
        "--idle-exit", type=float, metavar="SECONDS", help="exit after this long without work"
    )
    s.set_defaults(fn=cmd_worker_start)
    s = wk.add_parser("status", help="list workers (online ones unless --all)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--all", action="store_true", help="include offline workers")
    s.set_defaults(fn=cmd_worker_status)
    s = wk.add_parser("revoke", help="revoke a worker's credentials")
    s.add_argument("worker_id")
    s.set_defaults(fn=cmd_worker_revoke)

    rs = sub.add_parser("resources", help="inspect compute resources").add_subparsers(
        dest="cmd", required=True
    )
    s = rs.add_parser("list", help="workers and their capabilities (online unless --all)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--all", action="store_true", help="include offline workers")
    s.set_defaults(fn=cmd_worker_status)
    s = rs.add_parser("detect", help="detect this machine's capabilities")
    s.add_argument("--json", action="store_true")
    s.add_argument("--python")
    s.set_defaults(fn=cmd_resources_detect)

    s = sub.add_parser(
        "submit", help="submit a job", usage="gpulayer submit [options] [-- COMMAND ...]"
    )
    s.add_argument("-f", "--file", help="JobSpec JSON file")
    s.add_argument("--name", default="job")
    s.add_argument("--target", choices=[t.value for t in ExecutionTarget], default="local")
    s.add_argument("--fallback", action="append", choices=[t.value for t in ExecutionTarget])
    s.add_argument("--source", metavar="DIR", help="package and upload this directory")
    s.add_argument("--input", action="append", metavar="FILE[:DEST]")
    s.add_argument("--function", metavar="MODULE:FUNC")
    s.add_argument("--arg", action="append", metavar="JSON", help="function argument (JSON)")
    s.add_argument(
        "--distributed",
        action="store_true",
        help="one coordinated torchrun workload across --nodes workers",
    )
    s.add_argument("--entrypoint", metavar="SCRIPT")
    s.add_argument("--nodes", type=int, default=1)
    s.add_argument("--procs-per-node", type=int, default=1)
    s.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    s.add_argument(
        "--accelerator",
        choices=[a.value for a in Accelerator],
        default=None,
        help="default: the profile's accelerator, else cpu",
    )
    s.add_argument(
        "--profile",
        choices=sorted(RUNTIME_PROFILES),
        help="runtime profile, e.g. pytorch-cuda (fills in accelerator and frameworks)",
    )
    s.add_argument("--gpus", type=int)
    s.add_argument("--gpu-memory", type=float, metavar="GB")
    s.add_argument(
        "--gpu-memory-kind",
        choices=["dedicated", "unified"],
        help="require discrete VRAM or unified memory (Apple Silicon)",
    )
    s.add_argument("--cpus", type=float, default=1)
    s.add_argument("--memory", type=float, metavar="GB")
    s.add_argument("--framework", action="append", metavar="SPEC", help="e.g. 'torch>=2.1'")
    s.add_argument("--worker", action="append", metavar="WORKER_ID")
    s.add_argument("--label", action="append", metavar="KEY=VALUE")
    s.add_argument("--provider", help="allow provisioning through this provider adapter")
    s.add_argument("--max-hourly-cost", type=float)
    s.add_argument("--env", action="append", metavar="KEY=VALUE")
    s.add_argument("--secret", action="append", metavar="NAME=WORKER_ENV_KEY")
    s.add_argument("--timeout", type=float, metavar="SECONDS")
    s.add_argument("--max-attempts", type=int, default=1)
    s.add_argument("--detach", action="store_true", help="print the job id and return")
    s.add_argument(
        "--queue-if-unschedulable",
        action="store_true",
        help="wait for a compatible worker instead of failing",
    )
    s.add_argument("--python", help="interpreter for the in-process local worker")
    s.add_argument("command", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_submit)

    jb = sub.add_parser("job", help="inspect and control jobs").add_subparsers(
        dest="cmd", required=True
    )
    for name, fn, help_ in (
        ("status", cmd_job_status, "job state and placement"),
        ("result", cmd_job_result, "result JSON"),
        ("cancel", cmd_job_cancel, "cancel a job"),
        ("events", cmd_job_events, "lifecycle events"),
        ("logs", cmd_job_logs, "job output"),
    ):
        s = jb.add_parser(name, help=help_)
        s.add_argument("job_id")
        s.set_defaults(fn=fn)
        if name in ("status", "events"):
            s.add_argument("--json", action="store_true")
        if name == "logs":
            s.add_argument("-f", "--follow", action="store_true")
            s.add_argument("--system", action="store_true", help="include system lines")
            s.add_argument("--verbose", action="store_true", help="prefix node and stream")
        if name == "result":
            s.add_argument("--download", metavar="DIR", help="download output artifacts")
    s = jb.add_parser("list", help="recent jobs")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_job_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.fn(args))
    except (GpuLayerError, ValidationError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
