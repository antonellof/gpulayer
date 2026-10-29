import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def gpl(home, *args, check=True):
    env = {**os.environ, "GPULAYER_HOME": str(home)}
    env.pop("GPULAYER_URL", None)
    p = subprocess.run(
        [sys.executable, "-m", "gpulayer.cli.main", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check and p.returncode != 0:
        raise AssertionError(f"exit {p.returncode}\n{p.stdout}\n{p.stderr}")
    return p


def test_submit_inspect_result(tmp_path):
    p = gpl(tmp_path, "submit", "--name", "hello", "--", "python", "-c", "print('hello cli')")
    assert "hello cli" in p.stdout
    job_id = p.stderr.split()[1]
    assert "succeeded" in gpl(tmp_path, "job", "status", job_id).stdout
    assert gpl(tmp_path, "job", "logs", job_id).stdout.strip() == "hello cli"
    assert json.loads(gpl(tmp_path, "job", "result", job_id).stdout)["state"] == "succeeded"
    assert job_id in gpl(tmp_path, "job", "list").stdout
    assert "job.succeeded" in gpl(tmp_path, "job", "events", job_id).stdout


def test_failed_job_exit_code(tmp_path):
    p = gpl(tmp_path, "submit", "--", "python", "-c", "raise SystemExit(4)", check=False)
    assert p.returncode == 4


def test_unschedulable_explains(tmp_path):
    p = gpl(tmp_path, "submit", "--accelerator", "rocm", "--", "true", check=False)
    assert p.returncode != 0 and "rocm" in p.stderr and "unschedulable" in p.stderr


def test_detach_then_worker_then_cancel(tmp_path):
    job_id = gpl(tmp_path, "submit", "--detach", "--", "python", "-c", "print(1)").stdout.strip()
    assert "queued" in gpl(tmp_path, "job", "status", job_id).stdout
    gpl(tmp_path, "worker", "start", "--max-tasks", "1", "--idle-exit", "30")
    assert "succeeded" in gpl(tmp_path, "job", "status", job_id).stdout
    other = gpl(tmp_path, "submit", "--detach", "--", "true").stdout.strip()
    assert "cancelled" in gpl(tmp_path, "job", "cancel", other).stdout


def test_function_and_resources_detect(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text("def add(a, b):\n    return a + b\n")
    p = gpl(
        tmp_path, "submit", "--source", str(src), "--function", "m:add", "--arg", "2", "--arg", "40"
    )
    assert p.stdout.strip() == "42"
    caps = json.loads(gpl(tmp_path, "resources", "detect", "--json").stdout)
    assert caps["cpu_cores"] >= 1


def test_gc_command(tmp_path):
    gpl(tmp_path, "submit", "--", "python", "-c", "print('to be collected')")
    out = gpl(tmp_path, "gc", "--older-than", "0m", "--dry-run").stdout
    assert out.startswith("would remove:") and "log lines" in out
    out = gpl(tmp_path, "gc", "--older-than", "0m", "--delete-jobs").stdout
    assert out.startswith("removed:") and "1 jobs" in out
    assert gpl(tmp_path, "job", "list").stdout.strip() == ""


def test_gpu_memory_kind_flag(tmp_path):
    p = gpl(
        tmp_path,
        "submit",
        "--accelerator",
        "mps",
        "--gpu-memory-kind",
        "dedicated",
        "--",
        "true",
        check=False,
    )
    assert p.returncode != 0
    assert "requires dedicated GPU memory" in p.stderr or "usable mps device" in p.stderr


def test_profile_flag(tmp_path):
    ok = gpl(
        tmp_path,
        "submit",
        "--profile",
        "pytorch-cpu",
        "--",
        "python",
        "-c",
        "print('ok')",
        check=False,
    )
    try:
        import torch  # noqa: F401
    except ImportError:
        assert ok.returncode != 0 and "framework 'torch' not installed" in ok.stderr
    else:
        assert ok.returncode == 0 and "ok" in ok.stdout
    p = gpl(tmp_path, "submit", "--profile", "jax-cuda", "--", "true", check=False)
    assert p.returncode != 0 and "unschedulable" in p.stderr
