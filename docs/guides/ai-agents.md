# AI and coding agents

AI and coding agents (Claude Code, Claude Desktop, Cursor, Codex, VS Code agents, or your own)
can use GPUlayer to run work on GPUs, for example training a model on an NVIDIA server or
running a sweep across several machines, and then read the logs and fetch the results. There
are three ways to connect them, from most to least structured:

| Way | Best for | Setup |
|---|---|---|
| [MCP server](#mcp-server) | agents that support the Model Context Protocol | `gpulayer mcp` |
| [Agent Skill](#agent-skill) | agents that run shell commands and read `SKILL.md` files | copy one folder |
| [HTTP API](#http-api) | your own agents and services | a token |

All three go through the same API, placement rules and permissions as people do, so an agent
can do only what its token allows.

## MCP server

`gpulayer mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server on stdio.
It runs as long as the agent session. Without a server configured it works on this machine
and keeps a local worker alive for the session; with `GPULAYER_URL` and `GPULAYER_TOKEN` it
uses a GPUlayer server and its workers.

### Tools

| Tool | What it does |
|---|---|
| `gpulayer_resources` | workers available now: GPUs and GPU memory, frameworks, CPU, memory, free slots |
| `gpulayer_submit` | run a command; optional `source_dir` (ship a project), `inputs`, `accelerator` (`cpu`, `cuda`, `mps`), `gpus`, `gpu_memory_gb`, `frameworks`, `env`, `timeout_seconds`, and `wait_seconds` to wait for the result |
| `gpulayer_wait` | wait for a job (up to 10 minutes per call) and return its status, exit code, outputs and last log lines |
| `gpulayer_job_status` | state, placement and exit codes; for a job no worker could run, the reason for every worker |
| `gpulayer_job_logs` | output, with `tail` and `after_seq` for following; distributed jobs prefix each line with its rank |
| `gpulayer_list_jobs` | recent jobs, optionally by state |
| `gpulayer_cancel` | cancel a job |
| `gpulayer_download_outputs` | copy the files a job wrote to `$GPULAYER_OUTPUT_DIR` into a local directory |

`gpulayer mcp --read-only` offers only the tools that read state: no submit, cancel or
download. Tool failures (a missing directory, an unknown job, a request no worker can satisfy)
come back as tool results the agent can read, not as protocol errors.

The server speaks MCP protocol versions 2025-06-18, 2025-03-26 and 2024-11-05, and has been
tested with the official MCP Python SDK client (version 2.3).

### Connect an agent

Use the full path to `gpulayer` (`which gpulayer`) for desktop apps, which often do not see
your shell's `PATH`. Leave out the `env` part to run jobs on this machine.

**Claude Code**

```bash
claude mcp add gpulayer -- gpulayer mcp
# against a server:
claude mcp add gpulayer -e GPULAYER_URL=https://gpu-server:8470 -e GPULAYER_TOKEN=glc_... \
  -e GPULAYER_FINGERPRINT=sha256:... -- gpulayer mcp
```

**Claude Desktop** (`claude_desktop_config.json`), **Cursor** (`.cursor/mcp.json`) and other
clients that use the `mcpServers` format:

```json
{
  "mcpServers": {
    "gpulayer": {
      "command": "/Users/you/.local/bin/gpulayer",
      "args": ["mcp"],
      "env": {
        "GPULAYER_URL": "https://gpu-server:8470",
        "GPULAYER_TOKEN": "glc_...",
        "GPULAYER_FINGERPRINT": "sha256:..."
      }
    }
  }
}
```

**VS Code** (`.vscode/mcp.json`):

```json
{
  "servers": {
    "gpulayer": { "type": "stdio", "command": "gpulayer", "args": ["mcp"] }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`):

```toml
[mcp_servers.gpulayer]
command = "gpulayer"
args = ["mcp"]
env = { GPULAYER_URL = "https://gpu-server:8470", GPULAYER_TOKEN = "glc_..." }
```

Check the configuration format of your agent's version; these change between releases.

### What an agent session looks like

> Train `train.py` in this repository on an NVIDIA GPU for 3 epochs and tell me the final
> validation loss.

The agent calls `gpulayer_resources`, sees a worker with CUDA and torch, calls
`gpulayer_submit` with `command: ["python", "train.py", "--epochs", "3"]`,
`source_dir: "."`, `accelerator: "cuda"`, then `gpulayer_wait` until the job finishes and
reads the loss from the log tail. If no worker has CUDA, the submit result gives the reason
for each worker, and the agent can report it.

## Agent Skill

[`integrations/skills/gpulayer/SKILL.md`](../../integrations/skills/gpulayer/SKILL.md) teaches an
agent to use the `gpulayer` CLI: checking resources, submitting with the right accelerator and
frameworks, following jobs, reading rejection reasons, and the rules for using it safely.

For Claude Code, copy the folder into your skills directory:

```bash
cp -r integrations/skills/gpulayer ~/.claude/skills/            # all projects
cp -r integrations/skills/gpulayer .claude/skills/              # this project only
```

Agents that read `AGENTS.md` or similar instruction files can include the body of
`SKILL.md` there instead. The skill needs the `gpulayer` command on the agent's `PATH`.

## HTTP API

Agents that make HTTP requests can use the [HTTP API](../reference/http-api.md) directly:
submit a `JobSpec`, long-poll `GET /v1/jobs/{id}?wait=30` until it is terminal, read
`GET /v1/jobs/{id}/logs`, and fetch outputs. Send `Authorization: Bearer <token>` and
`X-GpuLayer-Protocol: 1` with every request. The Python SDK (`gpulayer.Client`) wraps the same
calls.

## Security

- Submitting a job runs code on a worker, so give an agent the narrowest access that works:
  `gpulayer mcp --read-only` for monitoring, and otherwise a `submitter` token
  (`gpulayer token create my-agent --role submitter --ttl 12h`), never an admin token.
  Submitters can cancel only their own jobs.
- Tokens given to an agent can be listed and revoked on the server (`gpulayer token list`,
  `gpulayer token revoke my-agent`); expiring tokens limit the damage of a leaked one.
- In local mode the MCP server acts as your user on this machine, and `source_dir`, `inputs`
  and `download_outputs` read and write local paths the agent chooses. Run agents you trust,
  in a project directory, and review their actions as your agent host allows.
- Workers do not sandbox jobs (see [Security](../architecture/security.md)). Jobs an agent
  submits run with the same isolation as any other job.
- Every job records who submitted it (`submitted_by`), so agent jobs are visible on the
  dashboard and in `gpulayer job list`.
