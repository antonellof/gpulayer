"""`gpulayer mcp`: the MCP server over stdio, driven like an agent host would."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


class Session:
    def __init__(self, home, *args):
        env = {**os.environ, "GPULAYER_HOME": str(home)}
        env.pop("GPULAYER_URL", None)
        self.p = subprocess.Popen(
            [sys.executable, "-m", "gpulayer.cli.main", "mcp", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.next_id = 0

    def request(self, method, params=None):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        assert reply["id"] == self.next_id
        return reply

    def notify(self, method):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.p.stdin.flush()

    def call(self, tool, **arguments):
        result = self.request("tools/call", {"name": tool, "arguments": arguments})["result"]
        return result, json.loads(result["content"][0]["text"]) if not result["isError"] else None

    def close(self):
        self.p.stdin.close()
        assert self.p.wait(timeout=60) == 0


@pytest.fixture
def session(tmp_path):
    s = Session(tmp_path / "home")
    init = s.request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    )
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "gpulayer"
    s.notify("notifications/initialized")
    yield s
    s.close()


def test_tools_are_listed_with_schemas(session):
    tools = {t["name"]: t for t in session.request("tools/list")["result"]["tools"]}
    assert {
        "gpulayer_resources",
        "gpulayer_submit",
        "gpulayer_wait",
        "gpulayer_job_logs",
        "gpulayer_job_status",
        "gpulayer_cancel",
        "gpulayer_download_outputs",
    } <= set(tools)
    assert tools["gpulayer_submit"]["inputSchema"]["required"] == ["command"]
    _, res = session.call("gpulayer_resources")
    assert res["mode"] == "embedded" and res["workers"]


def test_agent_runs_a_project_and_gets_its_outputs(session, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "work.py").write_text(
        "import os, pathlib\n"
        "data = pathlib.Path(os.environ['GPULAYER_INPUT_DIR'], 'numbers.txt').read_text()\n"
        "total = sum(int(x) for x in data.split())\n"
        "print('total', total)\n"
        "pathlib.Path(os.environ['GPULAYER_OUTPUT_DIR'], 'total.txt').write_text(str(total))\n"
    )
    (tmp_path / "numbers.txt").write_text("1 2 3 4")
    raw, res = session.call(
        "gpulayer_submit",
        command=["python", "work.py"],
        source_dir=str(proj),
        inputs=[{"path": str(tmp_path / "numbers.txt")}],
        name="agent-sum",
        wait_seconds=120,
    )
    assert res["state"] == "succeeded", res
    assert res["exit_code"] == 0 and res["outputs"] == ["total.txt"]
    assert "total 10" in res["log_tail"]
    assert raw["structuredContent"]["state"] == "succeeded"
    _, logs = session.call("gpulayer_job_logs", job_id=res["job_id"], tail=5)
    assert logs["lines"] == ["total 10"]
    _, dl = session.call(
        "gpulayer_download_outputs", job_id=res["job_id"], dest_dir=str(tmp_path / "out")
    )
    assert (tmp_path / "out" / "total.txt").read_text() == "10"
    assert dl["files"]


def test_unplaceable_job_and_bad_input_come_back_as_tool_errors(session):
    _, res = session.call("gpulayer_submit", command=["true"], accelerator="cuda", gpus=8)
    assert res["state"] == "failed" and res["why_not_placed"]  # every worker says why
    raw, _ = session.call("gpulayer_job_status", job_id="job-000000000000")
    assert raw["isError"] and "not found" in raw["content"][0]["text"]
    raw, _ = session.call("gpulayer_submit", command=["true"], source_dir="/no/such/dir")
    assert raw["isError"]


def test_read_only_mode_hides_write_tools(tmp_path):
    s = Session(tmp_path / "home", "--read-only")
    s.request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"},
        },
    )
    names = {t["name"] for t in s.request("tools/list")["result"]["tools"]}
    assert "gpulayer_submit" not in names and "gpulayer_cancel" not in names
    raw, _ = s.call("gpulayer_submit", command=["true"])
    assert raw["isError"]
    assert s.request("bogus/method")["error"]["code"] == -32601
    s.close()
