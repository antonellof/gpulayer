"""Worker agent: heartbeat, acquire, prepare, execute, report, clean up.

The same agent serves local, existing-remote and provider-provisioned machines; only
the ``WorkerChannel`` differs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from gpulayer.artifacts.store import extract_bundle
from gpulayer.contracts import (
    Accelerator,
    ArtifactKind,
    ArtifactReference,
    CommandJob,
    DistributedJob,
    FunctionJob,
    JobSpec,
    RendezvousInfo,
    TaskAssignment,
    TaskOutcome,
    TaskState,
    WorkerCapabilities,
)
from gpulayer.errors import Conflict, Forbidden, GpuLayerError, Unauthorized
from gpulayer.runtime import dist_function_runner, function_runner
from gpulayer.runtime.distributed import free_port, torchrun_argv
from gpulayer.runtime.process import run_process
from gpulayer.worker.channel import WorkerChannel

log = logging.getLogger("gpulayer.worker")

# Only these variables leak from the worker's environment into workloads; everything
# else (including credentials in the worker's env) must be passed explicitly.
_ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "TERM")
MAX_OUTPUT_FILES = 1000
MAX_OUTPUT_BYTES = 512 * 1024**2


class _LogBuffer:
    """Batches log lines to the control plane and redacts secret values."""

    def __init__(self, channel: WorkerChannel, task_id: str, secrets: list[str]) -> None:
        self.channel, self.task_id = channel, task_id
        self.secrets = sorted((s for s in secrets if s), key=len, reverse=True)
        self.lines: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def __call__(self, stream: str, line: str) -> None:
        for s in self.secrets:
            line = line.replace(s, "***")
        with self.lock:
            self.lines.append((stream, line))
            if len(self.lines) >= 200 or time.monotonic() - self.last > 0.5:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self.lines:
            batch, self.lines = self.lines, []
            try:
                self.channel.logs(self.task_id, batch)
            except GpuLayerError as e:  # logs are best-effort; never kill the task for them
                log.warning("dropping %d log lines: %s", len(batch), e)
        self.last = time.monotonic()

    def flush(self) -> None:
        with self.lock:
            self._flush_locked()


class WorkerAgent:
    def __init__(
        self,
        channel: WorkerChannel,
        capabilities: WorkerCapabilities,
        workdir: str | Path,
        *,
        python: str = sys.executable,
        poll_interval: float = 0.5,
        keep_workspaces: bool = False,
    ) -> None:
        self.channel = channel
        self.caps = capabilities
        self.workdir = Path(workdir)
        self.python = python
        self.poll_interval = poll_interval
        self.keep_workspaces = keep_workspaces
        self.stop = threading.Event()
        self.revoked = False
        self._cancel: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        self._hb_thread: threading.Thread | None = None

    # ------------------------------------------------------------- lifecycle

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.channel.heartbeat_interval):
            try:
                resp = self.channel.heartbeat()
                revoked = resp.revoked
            except (Forbidden, Unauthorized) as e:
                # Over HTTP a revoked session token is rejected outright.
                log.error("control plane rejected this worker's credentials: %s", e)
                revoked = True
            except GpuLayerError as e:
                log.warning("heartbeat failed: %s", e)
                continue
            except OSError as e:
                log.warning("control plane unreachable: %s", e)
                continue
            if revoked:
                log.error("worker revoked by control plane; stopping")
                self.revoked = True
                with self._cancel_lock:
                    for ev in self._cancel.values():
                        ev.set()
                self.stop.set()
                return
            with self._cancel_lock:
                for tid in resp.cancel_task_ids:
                    if tid in self._cancel:
                        self._cancel[tid].set()

    def start_heartbeats(self) -> None:
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="gpulayer-heartbeat"
        )
        self._hb_thread.start()

    def run(
        self,
        *,
        max_tasks: int | None = None,
        idle_exit_seconds: float | None = None,
        deregister: bool = True,
    ) -> int:
        """Process assignments until stopped. Returns the number of tasks executed."""
        self.start_heartbeats()
        done = 0
        idle_since = time.monotonic()
        try:
            while not self.stop.is_set():
                try:
                    assignment = self.channel.acquire()
                except (GpuLayerError, OSError) as e:
                    log.warning("acquire failed: %s", e)
                    assignment = None
                if assignment is None:
                    if (
                        idle_exit_seconds is not None
                        and time.monotonic() - idle_since > idle_exit_seconds
                    ):
                        break
                    self.stop.wait(self.poll_interval)
                    continue
                self.execute(assignment)
                done += 1
                idle_since = time.monotonic()
                if max_tasks is not None and done >= max_tasks:
                    break
        finally:
            self.stop.set()
            if deregister and not self.revoked:
                try:
                    self.channel.deregister()
                except (GpuLayerError, OSError) as e:
                    log.warning("deregister failed: %s", e)
        return done

    # ------------------------------------------------------------- execution

    def execute(self, a: TaskAssignment) -> None:
        try:
            self.channel.start(a.task_id)
        except Conflict as e:
            log.info("skipping task %s: %s", a.task_id, e)
            return
        cancel = threading.Event()
        with self._cancel_lock:
            self._cancel[a.task_id] = cancel
        ws = self.workdir / "tasks" / a.task_id
        try:
            outcome = self._run(a, ws, cancel)
        except Exception as e:  # preparation failures are task failures, not worker crashes
            log.exception("task %s failed during preparation", a.task_id)
            outcome = TaskOutcome(state=TaskState.FAILED, error=f"{type(e).__name__}: {e}")
        finally:
            with self._cancel_lock:
                self._cancel.pop(a.task_id, None)
        try:
            self.channel.finish(a.task_id, outcome)
        except GpuLayerError as e:
            log.error("could not report outcome of %s: %s", a.task_id, e)
        if not self.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)

    def _fetch(self, ref: ArtifactReference) -> bytes:
        cache = self.workdir / "cache" / ref.sha256
        if cache.exists():
            return cache.read_bytes()
        data = self.channel.get_artifact(ref.sha256)
        import hashlib

        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise GpuLayerError(f"artifact {ref.uri} failed integrity verification")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    def _environment(self, a: TaskAssignment, ws: Path) -> tuple[dict[str, str], list[str]]:
        spec = a.spec
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env.update(spec.env)
        secret_values = []
        for s in spec.secrets:
            value = os.environ.get(s.key)
            if value is None:
                raise GpuLayerError(f"secret {s.key!r} is not available on this worker")
            env[s.name] = value
            secret_values.append(value)
        env.update(
            {
                "GPULAYER_JOB_ID": a.job_id,
                "GPULAYER_TASK_ID": a.task_id,
                "GPULAYER_ATTEMPT": str(a.attempt),
                "GPULAYER_NODE_RANK": str(a.node_rank),
                "GPULAYER_INPUT_DIR": str(ws / "inputs"),
                "GPULAYER_OUTPUT_DIR": str(ws / "outputs"),
                "GPULAYER_ACCELERATOR": spec.resources.accelerator.value,
                "PYTHONUNBUFFERED": "1",
            }
        )
        if spec.resources.accelerator == Accelerator.CUDA:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(d.index)
                for d in self.caps.devices(Accelerator.CUDA)[: spec.resources.gpu_count]
            )
        elif spec.resources.accelerator == Accelerator.CPU:
            env["CUDA_VISIBLE_DEVICES"] = ""  # CPU jobs must not grab a GPU by accident
        if isinstance(spec.workload, DistributedJob):
            env["GPULAYER_COMM_BACKEND"] = spec.workload.comm_backend
        return env, secret_values

    def _run(self, a: TaskAssignment, ws: Path, cancel: threading.Event) -> TaskOutcome:
        spec: JobSpec = a.spec
        w = spec.workload
        if ws.exists():
            shutil.rmtree(ws)
        (ws / "inputs").mkdir(parents=True)
        (ws / "outputs").mkdir()
        src = ws / "src"
        source = getattr(w, "source", None)
        if source is not None:
            extract_bundle(self._fetch(source), src)
        for inp in spec.inputs:
            dest = (ws / "inputs" / inp.path).resolve()
            if not dest.is_relative_to((ws / "inputs").resolve()):
                raise GpuLayerError(f"unsafe input path {inp.path!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self._fetch(inp.artifact))

        env, secret_values = self._environment(a, ws)
        sink = _LogBuffer(self.channel, a.task_id, secret_values)
        cwd = str(src if source is not None else ws)
        result_file: Path | None = None

        if isinstance(w, CommandJob):
            argv = list(w.command)
            if argv[0] == "python":  # documented: 'python' means the worker's interpreter
                argv[0] = self.python
        elif isinstance(w, FunctionJob):
            runner = ws / "function_runner.py"
            shutil.copyfile(function_runner.__file__, runner)
            call = ws / "call.json"
            call.write_text(
                json.dumps({"entrypoint": w.entrypoint, "args": w.args, "kwargs": w.kwargs})
            )
            result_file = ws / "result.json"
            if source is not None:
                env["PYTHONPATH"] = str(src)
            argv = [self.python, str(runner), str(call), str(result_file)]
        elif isinstance(w, DistributedJob):
            rdzv = self._rendezvous(a)
            if _is_loopback(rdzv.master_addr):
                # Every node is on this host: keep gloo on the loopback interface instead of
                # letting it pick one from the hostname (which may resolve to a LAN address).
                env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
            sink(
                "system",
                f"torchrun node {rdzv.node_rank}/{rdzv.num_nodes} rendezvous "
                f"{rdzv.master_addr}:{rdzv.master_port} world_size {rdzv.world_size}",
            )
            if w.function is not None:
                # Every rank calls module:function; rank 0 writes the JSON result.
                entry = ws / "dist_function_runner.py"
                shutil.copyfile(dist_function_runner.__file__, entry)
                call = ws / "call.json"
                call.write_text(
                    json.dumps(
                        {
                            "entrypoint": w.function,
                            "args": w.function_args,
                            "kwargs": w.function_kwargs,
                        }
                    )
                )
                if a.node_rank == 0:
                    result_file = ws / "result.json"
                if source is not None:
                    env["PYTHONPATH"] = str(src)
                argv = torchrun_argv(self.python, w, rdzv, str(entry))
                argv += [str(call), str(ws / "result.json")]
            else:
                assert w.entrypoint is not None
                entry = (src / w.entrypoint).resolve()
                if not entry.is_relative_to(src.resolve()) or not entry.is_file():
                    raise GpuLayerError(f"entrypoint {w.entrypoint!r} not found in source bundle")
                argv = torchrun_argv(self.python, w, rdzv, str(entry))
        else:  # pragma: no cover - exhaustiveness
            raise GpuLayerError(f"unsupported workload {type(w).__name__}")

        out = run_process(
            argv, env=env, cwd=cwd, on_line=sink, timeout=spec.timeout_seconds, cancel=cancel
        )
        sink.flush()
        outputs = self._upload_outputs(ws / "outputs", a)
        if out.cancelled:
            return TaskOutcome(
                state=TaskState.CANCELLED,
                exit_code=out.exit_code,
                error="cancelled",
                outputs=outputs,
            )
        if out.timed_out:
            return TaskOutcome(
                state=TaskState.TIMED_OUT,
                exit_code=out.exit_code,
                error=f"exceeded timeout of {spec.timeout_seconds}s",
                outputs=outputs,
            )
        if out.error:
            return TaskOutcome(state=TaskState.FAILED, error=out.error)
        if result_file is not None:
            if not result_file.exists():
                return TaskOutcome(
                    state=TaskState.FAILED,
                    exit_code=out.exit_code,
                    error="function runner produced no result",
                    outputs=outputs,
                )
            res = json.loads(result_file.read_text())
            if not res["ok"]:
                err = res["error"]
                for s in secret_values:
                    err = err.replace(s, "***")
                return TaskOutcome(
                    state=TaskState.FAILED,
                    exit_code=out.exit_code,
                    error=err.strip().splitlines()[-1] if err.strip() else err,
                    outputs=outputs,
                )
            return TaskOutcome(
                state=TaskState.SUCCEEDED,
                exit_code=out.exit_code,
                return_value=res["value"],
                outputs=outputs,
            )
        state = TaskState.SUCCEEDED if out.exit_code == 0 else TaskState.FAILED
        return TaskOutcome(
            state=state,
            exit_code=out.exit_code,
            error=None if out.exit_code == 0 else f"exit code {out.exit_code}",
            outputs=outputs,
        )

    def _rendezvous(self, a: TaskAssignment) -> RendezvousInfo:
        if a.rendezvous is not None:
            return a.rendezvous
        if a.node_rank != 0:
            raise GpuLayerError("rank > 0 assignment delivered without rendezvous info")
        w = a.spec.workload
        assert isinstance(w, DistributedJob)
        addr = self.caps.advertise_address or "127.0.0.1"
        port = w.rendezvous.port or free_port()
        return self.channel.publish_rendezvous(a.task_id, addr, port)

    def _upload_outputs(self, out_dir: Path, a: TaskAssignment) -> list[ArtifactReference]:
        refs: list[ArtifactReference] = []
        total = 0
        prefix = f"node{a.node_rank}/" if isinstance(a.spec.workload, DistributedJob) else ""
        for p in sorted(out_dir.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            if len(refs) >= MAX_OUTPUT_FILES:
                break
            total += p.stat().st_size
            if total > MAX_OUTPUT_BYTES:
                break
            name = prefix + p.relative_to(out_dir).as_posix()
            refs.append(self.channel.put_artifact(p.read_bytes(), name, ArtifactKind.OUTPUT))
        return refs


def _is_loopback(addr: str) -> bool:
    import ipaddress

    try:
        return addr == "localhost" or ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return False
