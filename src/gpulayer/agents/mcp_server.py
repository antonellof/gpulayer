"""GPUlayer as a Model Context Protocol (MCP) server, for AI and coding agents.

    gpulayer mcp                 # stdio transport; embedded mode runs jobs on this machine
    GPULAYER_URL=... GPULAYER_TOKEN=... gpulayer mcp      # use a GPUlayer server instead
    gpulayer mcp --read-only     # inspect only: no submit, cancel or download

The protocol is JSON-RPC 2.0, one message per line on stdin/stdout. Only the parts a tool
server needs are implemented: initialize, ping, tools/list and tools/call. Tool failures are
returned as tool results with ``isError: true`` so the agent can read and react to them.

Agents act with the permissions of the token in use (or of the local user in embedded mode):
submitting a job runs code on a worker. Use a submitter token, never an admin token.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import gpulayer
from gpulayer.contracts import (
    Accelerator,
    CommandJob,
    ExecutionTarget,
    InputArtifact,
    JobSpec,
    JobState,
    JobStatus,
    PlacementPolicy,
    ResourceRequest,
)
from gpulayer.errors import GpuLayerError

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_WAIT = 600.0  # seconds a single tool call may block
MAX_LOG_LINES = 2000

INSTRUCTIONS = """\
GPUlayer runs commands on GPU and CPU machines (this computer and any servers that joined).
Typical flow: gpulayer_resources to see what is available; gpulayer_submit with the command
(and source_dir to ship a project directory); gpulayer_wait or gpulayer_job_status; then
gpulayer_job_logs and gpulayer_download_outputs. Ask for accelerator "cuda" (NVIDIA) or
"mps" (Apple GPU) only when the work needs a GPU. If a job cannot be placed, the status
explains why for every worker; fix the request instead of retrying blindly. The first
element "python" in a command means the worker's own Python. Files written to
$GPULAYER_OUTPUT_DIR are returned as outputs."""


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


JOB_ID = {"type": "string", "description": "Job id, e.g. job-1a2b3c4d5e6f"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "gpulayer_resources",
        "description": "List the workers that can run jobs now: their GPUs, GPU memory, "
        "frameworks (torch, jax, mlx, ...), CPU, memory and free task slots.",
        "inputSchema": _schema({}),
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "gpulayer_submit",
        "description": "Run a command as a GPUlayer job on a machine that has what it needs. "
        "Returns the job id; with wait_seconds > 0 also waits and returns the outcome and the "
        "last log lines. Use source_dir to ship a local project directory (the command runs "
        "inside it).",
        "inputSchema": _schema(
            {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": 'argv, e.g. ["python", "train.py", "--epochs", "3"]',
                },
                "name": {"type": "string", "description": "Short job name"},
                "accelerator": {
                    "type": "string",
                    "enum": ["cpu", "cuda", "mps"],
                    "description": "cuda = NVIDIA GPU, mps = Apple GPU; default cpu",
                },
                "gpus": {"type": "integer", "minimum": 0, "description": "Number of GPUs"},
                "gpu_memory_gb": {"type": "number", "minimum": 0},
                "frameworks": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": 'Required packages on the worker, e.g. {"torch": ">=2.1"}',
                },
                "source_dir": {
                    "type": "string",
                    "description": "Local directory to package and run in (relative to the "
                    "server's working directory)",
                },
                "inputs": {
                    "type": "array",
                    "items": _schema(
                        {"path": {"type": "string"}, "dest": {"type": "string"}}, ["path"]
                    ),
                    "description": "Local files staged into $GPULAYER_INPUT_DIR",
                },
                "env": {"type": "object", "additionalProperties": {"type": "string"}},
                "target": {"type": "string", "enum": ["local", "remote", "cloud"]},
                "timeout_seconds": {"type": "number", "minimum": 1},
                "wait_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": MAX_WAIT,
                    "description": "Wait up to this long for the job to finish (default 0)",
                },
            },
            ["command"],
        ),
    },
    {
        "name": "gpulayer_job_status",
        "description": "State of a job, where it was placed, exit codes and errors. When it "
        "could not be placed, gives the reason for every worker.",
        "inputSchema": _schema({"job_id": JOB_ID}, ["job_id"]),
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "gpulayer_wait",
        "description": "Wait until a job finishes (or the timeout passes) and return its "
        "status with the last log lines. Call again if it is still running.",
        "inputSchema": _schema(
            {
                "job_id": JOB_ID,
                "timeout_seconds": {"type": "number", "minimum": 0, "maximum": MAX_WAIT},
            },
            ["job_id"],
        ),
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "gpulayer_job_logs",
        "description": "Output of a job. Distributed jobs prefix each line with its rank.",
        "inputSchema": _schema(
            {
                "job_id": JOB_ID,
                "tail": {"type": "integer", "minimum": 1, "maximum": MAX_LOG_LINES},
                "after_seq": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Only lines after this sequence number (for following)",
                },
            },
            ["job_id"],
        ),
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "gpulayer_list_jobs",
        "description": "Recent jobs, newest first.",
        "inputSchema": _schema(
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "state": {"type": "string", "enum": [s.value for s in JobState]},
            }
        ),
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "gpulayer_cancel",
        "description": "Cancel a job that has not finished.",
        "inputSchema": _schema({"job_id": JOB_ID}, ["job_id"]),
        "annotations": {"destructiveHint": True},
    },
    {
        "name": "gpulayer_download_outputs",
        "description": "Download the files a finished job wrote to $GPULAYER_OUTPUT_DIR into "
        "a local directory.",
        "inputSchema": _schema(
            {"job_id": JOB_ID, "dest_dir": {"type": "string"}}, ["job_id", "dest_dir"]
        ),
    },
]
WRITE_TOOLS = {"gpulayer_submit", "gpulayer_cancel", "gpulayer_download_outputs"}


class ToolError(Exception):
    """Reported to the agent as a tool result with isError: true."""


def _status(st: JobStatus) -> dict[str, Any]:
    out: dict[str, Any] = {
        "job_id": st.job_id,
        "name": st.name,
        "state": st.state.value,
        "terminal": st.state.terminal,
        "workload": st.workload_type.value,
        "error": st.error,
        "tasks": [
            {
                "node_rank": t.node_rank,
                "worker_id": t.worker_id,
                "state": t.state.value,
                "exit_code": t.exit_code,
                "error": t.error,
            }
            for t in st.tasks
        ],
    }
    if st.scheduling is not None:
        out["placement"] = st.scheduling.summary
        if not st.scheduling.placed:
            out["why_not_placed"] = {
                c.worker_id: c.reasons for c in st.scheduling.candidates if c.reasons
            }
    return out


class GpuLayerMcp:
    def __init__(self, client: gpulayer.Client, *, read_only: bool = False) -> None:
        self.c = client
        self.read_only = read_only
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "gpulayer_resources": self.resources,
            "gpulayer_submit": self.submit,
            "gpulayer_job_status": lambda a: _status(self.c.job(a["job_id"]).status()),
            "gpulayer_wait": self.wait,
            "gpulayer_job_logs": self.logs,
            "gpulayer_list_jobs": self.list_jobs,
            "gpulayer_cancel": lambda a: _status(self.c.job(a["job_id"]).cancel()),
            "gpulayer_download_outputs": self.download,
        }

    # ------------------------------------------------------------------ tools

    def tools(self) -> list[dict[str, Any]]:
        return [t for t in TOOLS if not (self.read_only and t["name"] in WRITE_TOOLS)]

    def resources(self, _: dict[str, Any]) -> Any:
        if self.c.mode == "embedded":
            self.c.ensure_local_worker()
        workers = []
        for w in self.c.workers():
            if w.state.value == "offline":
                continue
            cap = w.capabilities
            workers.append(
                {
                    "name": w.name,
                    "worker_id": w.worker_id,
                    "state": w.state.value,
                    "target": cap.target.value,
                    "accelerators": [
                        {
                            "type": d.accelerator.value,
                            "name": d.name,
                            "memory_gb": d.memory_gb,
                            "usable": d.execution_supported,
                        }
                        for d in cap.accelerators
                    ],
                    "frameworks": cap.frameworks,
                    "cpu_cores": cap.cpu_cores,
                    "memory_gb": cap.memory_gb,
                    "running_tasks": len(w.running_task_ids),
                    "task_slots": cap.max_concurrent_tasks,
                }
            )
        return {"mode": self.c.mode, "workers": workers}

    def submit(self, a: dict[str, Any]) -> Any:
        accel = Accelerator(a.get("accelerator", "cpu"))
        gpus = a.get("gpus")
        source = None
        if a.get("source_dir"):
            d = Path(a["source_dir"]).expanduser()
            if not d.is_dir():
                raise ToolError(f"source_dir {str(d)!r} is not a directory")
            source = self.c.package(d)
        inputs = []
        for item in a.get("inputs", []):
            p = Path(item["path"]).expanduser()
            if not p.is_file():
                raise ToolError(f"input {str(p)!r} is not a file")
            inputs.append(InputArtifact(artifact=self.c.upload(p), path=item.get("dest") or p.name))
        target = ExecutionTarget(
            a.get("target") or ("remote" if self.c.mode == "remote" else "local")
        )
        spec = JobSpec(
            name=a.get("name") or "agent-job",
            workload=CommandJob(command=list(a["command"]), source=source),
            resources=ResourceRequest(
                accelerator=accel,
                gpu_count=gpus if gpus is not None else (0 if accel == Accelerator.CPU else 1),
                gpu_memory_gb=a.get("gpu_memory_gb"),
                frameworks=a.get("frameworks") or {},
            ),
            placement=PlacementPolicy(target=target, on_unschedulable="fail"),
            inputs=inputs,
            env=a.get("env") or {},
            timeout_seconds=a.get("timeout_seconds"),
        )
        job = self.c.submit(spec)
        wait = float(a.get("wait_seconds") or 0)
        if wait > 0:
            return self.wait({"job_id": job.id, "timeout_seconds": wait})
        return _status(job.status())

    def wait(self, a: dict[str, Any]) -> Any:
        job = self.c.job(a["job_id"])
        timeout = min(float(a.get("timeout_seconds", 60)), MAX_WAIT)
        try:
            st = job.wait(timeout)
        except TimeoutError:
            st = job.status()
        out = _status(st)
        out["log_tail"] = [self._line(x) for x in job.logs()[-50:]]
        if st.state.terminal:
            res = job.result()
            out["exit_code"] = res.exit_code
            out["outputs"] = [o.name for o in res.outputs]
            if res.return_value is not None:
                out["return_value"] = res.return_value
        return out

    @staticmethod
    def _line(x: Any) -> str:
        prefix = f"[{x.origin}] " if x.rank is not None else ""
        return prefix + (f"[{x.stream}] " if x.stream != "stdout" else "") + x.line

    def logs(self, a: dict[str, Any]) -> Any:
        lines = self.c.job(a["job_id"]).logs(after_seq=int(a.get("after_seq", 0)))
        tail = int(a.get("tail", 200))
        shown = lines[-tail:]
        return {
            "lines": [self._line(x) for x in shown],
            "omitted": len(lines) - len(shown),
            "last_seq": lines[-1].seq if lines else int(a.get("after_seq", 0)),
        }

    def list_jobs(self, a: dict[str, Any]) -> Any:
        jobs = self.c.jobs(limit=int(a.get("limit", 20)))
        if a.get("state"):
            jobs = [j for j in jobs if j.state.value == a["state"]]
        return [
            {
                "job_id": j.job_id,
                "name": j.name,
                "state": j.state.value,
                "created_at": j.created_at.isoformat(),
            }
            for j in jobs
        ]

    def download(self, a: dict[str, Any]) -> Any:
        res = self.c.job(a["job_id"]).result()
        out = Path(a["dest_dir"]).expanduser().resolve()
        written = []
        for ref in res.outputs:
            dest = (out / ref.name).resolve()
            if not dest.is_relative_to(out):
                raise ToolError(f"refusing unsafe output name {ref.name!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.c.download(ref))
            written.append(str(dest))
        return {"files": written}

    # --------------------------------------------------------------- protocol

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = self.handlers.get(name)
        if handler is None or (self.read_only and name in WRITE_TOOLS):
            return _text(f"unknown or disabled tool: {name}", error=True)
        try:
            result = handler(args or {})
        except (GpuLayerError, ToolError, ValueError, KeyError) as e:
            return _text(f"{type(e).__name__}: {e}", error=True)
        body = _text(json.dumps(result, indent=2, default=str))
        if isinstance(result, dict):
            body["structuredContent"] = json.loads(json.dumps(result, default=str))
        return body

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            return None  # notification (e.g. notifications/initialized)
        params = msg.get("params") or {}
        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            result: Any = {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "gpulayer", "version": gpulayer.__version__},
                "instructions": INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools()}
        elif method == "tools/call":
            result = self.call_tool(params.get("name", ""), params.get("arguments") or {})
        else:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": mid, "result": result}


def _text(text: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def serve(
    client: gpulayer.Client,
    *,
    read_only: bool = False,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> None:
    """Serve MCP over stdio until stdin closes."""
    proto_in = stdin or sys.stdin
    proto_out = stdout or sys.stdout
    # Anything else that prints must not corrupt the protocol stream.
    sys.stdout = sys.stderr
    server = GpuLayerMcp(client, read_only=read_only)
    for raw in proto_in:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            reply: dict[str, Any] | None = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(e)},
            }
        else:
            reply = server.handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            proto_out.write(json.dumps(reply) + "\n")
            proto_out.flush()
