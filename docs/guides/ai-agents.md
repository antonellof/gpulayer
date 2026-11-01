# AI and coding agents

Agents such as Claude Code, Cursor or Codex can run jobs on GPUs through GPUlayer, read the logs
and fetch results, with only the permissions their token grants.

| Way | Best for | Setup |
|---|---|---|
| [MCP server](#mcp-server) | agents that support the Model Context Protocol | `gpulayer mcp` |
| [Agent Skill](#agent-skill) | agents that run shell commands and read `SKILL.md` files | copy one folder |
| [HTTP API](#http-api) | your own agents and services | a token |

## MCP server

`gpulayer mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server on stdio.
Without `GPULAYER_URL` and `GPULAYER_TOKEN` it runs jobs on this machine; with them, on a
GPUlayer server.

| Tool | What it does |
|---|---|
| `gpulayer_resources` | workers available now: GPUs and GPU memory, frameworks, CPU, memory, free slots |
| `gpulayer_submit` | run a command; optional `source_dir` (ship a project), `inputs`, `accelerator` (`cpu`, `cuda`, `mps`), `gpus`, `gpu_memory_gb`, `frameworks`, `env`, `timeout_seconds`, `wait_seconds` |
| `gpulayer_wait` | wait for a job (up to 10 minutes per call); returns status, exit code, outputs and last log lines |
| `gpulayer_job_status` | state, placement and exit codes, or each worker's reason for rejecting the job |
| `gpulayer_job_logs` | output, with `tail` and `after_seq`; distributed jobs prefix lines with the rank |
| `gpulayer_list_jobs` | recent jobs, optionally by state |
| `gpulayer_cancel` | cancel a job |
| `gpulayer_download_outputs` | copy the files a job wrote to `$GPULAYER_OUTPUT_DIR` into a local directory |

`--read-only` offers only the read tools. Errors come back as tool results, not protocol
errors. Supports MCP protocol versions 2025-06-18, 2025-03-26 and 2024-11-05; tested with the
official MCP Python SDK client 2.3.

### Connect an agent

Leave out `env` to run jobs on this machine. Desktop apps may not see your `PATH`, so use the
full path from `which gpulayer`. Config formats change between agent releases.

```bash
# Claude Code
claude mcp add gpulayer -- gpulayer mcp
claude mcp add gpulayer -e GPULAYER_URL=https://gpu-server:8470 -e GPULAYER_TOKEN=glc_... \
  -e GPULAYER_FINGERPRINT=sha256:... -- gpulayer mcp
```

Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`) and other
`mcpServers` clients:

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

VS Code (`.vscode/mcp.json`):

```json
{ "servers": { "gpulayer": { "type": "stdio", "command": "gpulayer", "args": ["mcp"] } } }
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.gpulayer]
command = "gpulayer"
args = ["mcp"]
env = { GPULAYER_URL = "https://gpu-server:8470", GPULAYER_TOKEN = "glc_..." }
```

## Agent Skill

[`integrations/skills/gpulayer/SKILL.md`](../../integrations/skills/gpulayer/SKILL.md) teaches an
agent to use the `gpulayer` CLI, which must be on its `PATH`. For Claude Code:

```bash
cp -r integrations/skills/gpulayer ~/.claude/skills/            # all projects
cp -r integrations/skills/gpulayer .claude/skills/              # this project only
```

Other agents can paste the body of `SKILL.md` into `AGENTS.md` or similar.

## HTTP API

Submit a `JobSpec`, long-poll `GET /v1/jobs/{id}?wait=30`, then read `GET /v1/jobs/{id}/logs`
and fetch outputs. See the [HTTP API](../reference/http-api.md); `gpulayer.Client` wraps it.

## Security

- Use `gpulayer mcp --read-only` for monitoring, otherwise a `submitter` token
  (`gpulayer token create my-agent --role submitter --ttl 12h`), never an admin token. Revoke
  with `gpulayer token revoke my-agent`.
- In local mode the MCP server acts as your user and reads and writes the paths the agent
  picks. Run agents you trust, in a project directory.
- Workers do not sandbox jobs (see [Security](../architecture/security.md)). Each job records
  `submitted_by`.
