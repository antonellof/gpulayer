"""Per-job environments (ADR-0022): built once per worker, reused, reported apart.

Offline: the "package" is a wheel built here, installed from its file path.
"""

import base64
import hashlib
import zipfile
from pathlib import Path

import pytest

import gpulayer as gl
from gpulayer.contracts import EnvironmentSpec, JobState
from gpulayer.errors import GpuLayerError
from gpulayer.worker.environments import EnvironmentCache

pytestmark = pytest.mark.integration


def make_wheel(directory: Path, name: str = "gltestpkg", value: str = "42") -> Path:
    """A minimal pure-Python wheel with one module, `<name>.VALUE`."""
    version = "1.0"
    files = {
        f"{name}/__init__.py": f"VALUE = {value!r}\n",
        f"{name}-{version}.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ),
        f"{name}-{version}.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: gpulayer-tests\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=")
        record.append(f"{path},sha256={digest.decode()},{len(text.encode())}")
    record.append(f"{name}-{version}.dist-info/RECORD,,")
    files[f"{name}-{version}.dist-info/RECORD"] = "\n".join(record) + "\n"
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for path, text in files.items():
            z.writestr(path, text)
    return wheel


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GPULAYER_HOME", str(tmp_path / "home"))
    with gl.Client() as c:
        yield c


def test_spec_rejects_pip_options_and_paths_outside_the_bundle():
    with pytest.raises(ValueError, match="no pip options"):
        EnvironmentSpec(requirements=["--index-url https://example.invalid"])
    with pytest.raises(ValueError, match="relative path"):
        EnvironmentSpec(requirements_file="../secrets.txt")
    assert EnvironmentSpec().empty and not EnvironmentSpec(requirements=["x"]).empty


def test_cache_builds_once_and_reuses(tmp_path):
    import sys

    wheel = make_wheel(tmp_path)
    cache = EnvironmentCache(tmp_path / "envs")
    first = cache.prepare(sys.executable, [str(wheel)])
    assert not first.cached
    again = cache.prepare(sys.executable, [str(wheel)])
    assert again.cached and again.python == first.python
    assert again.seconds < first.seconds
    import subprocess

    out = subprocess.run(
        [first.python, "-c", "import gltestpkg, gpulayer; print(gltestpkg.VALUE)"],
        capture_output=True,
        text=True,
    )
    # The job's package is installed; the worker's own packages (gpulayer) stay visible.
    assert out.returncode == 0 and out.stdout.strip() == "42", out.stderr


def test_failed_build_is_reported_and_not_cached(tmp_path):
    import sys

    cache = EnvironmentCache(tmp_path / "envs")
    missing = str(tmp_path / "missing-1.0-py3-none-any.whl")
    with pytest.raises(GpuLayerError, match="environment build failed"):
        cache.prepare(sys.executable, [missing])
    with pytest.raises(GpuLayerError):  # tried again, not taken from the cache
        cache.prepare(sys.executable, [missing])


def test_job_with_requirements_end_to_end(client, tmp_path):
    wheel = make_wheel(tmp_path)
    code = "import gltestpkg; print('value', gltestpkg.VALUE)"
    first = client.submit(gl.command(["python", "-c", code], requirements=[str(wheel)]))
    assert first.wait(300).state == JobState.SUCCEEDED, first.logs_text()
    assert "value 42" in first.logs_text()
    second = client.submit(gl.command(["python", "-c", code], requirements=[str(wheel)]))
    assert second.wait(300).state == JobState.SUCCEEDED
    done = [e for e in second.events() if e.type == "task.succeeded"]
    assert done and done[0].data["prepare_cached"] is True
    assert "environment reused" in done[0].message
    first_done = [e for e in first.events() if e.type == "task.succeeded"][0]
    assert first_done.data["prepare_cached"] is False
    # A job without requirements still runs with the worker's interpreter.
    plain = client.submit(gl.command(["python", "-c", "import sys; print(sys.prefix)"]))
    assert plain.wait(60).state == JobState.SUCCEEDED
    assert "envs" not in plain.logs_text()


def test_requirements_file_in_the_source_bundle(client, tmp_path):
    wheel = make_wheel(tmp_path, name="glfilepkg", value="7")
    src = tmp_path / "app"
    src.mkdir()
    (src / "requirements.txt").write_text(f"{wheel}\n")
    (src / "main.py").write_text("import glfilepkg; print('file', glfilepkg.VALUE)\n")
    job = client.submit(
        gl.command(
            ["python", "main.py"],
            source=client.package(src),
            requirements_file="requirements.txt",
        )
    )
    assert job.wait(300).state == JobState.SUCCEEDED, job.logs_text()
    assert "file 7" in job.logs_text()


def test_job_fails_in_preparation_with_the_installer_message(client, tmp_path):
    bad = str(tmp_path / "nope-1.0-py3-none-any.whl")
    job = client.submit(gl.command(["python", "-c", "print('never')"], requirements=[bad]))
    st = job.wait(300)
    assert st.state == JobState.FAILED
    assert "environment build failed" in (st.error or "") + str([t.error for t in st.tasks])
    assert "never" not in job.logs_text()
