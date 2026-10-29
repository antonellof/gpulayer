"""`gpulayer up` + `gpulayer join` + gl.connect: real processes, TLS with a pinned cert."""

import os
import re
import shutil
import subprocess
import sys
import time

import pytest

import gpulayer as gl
from gpulayer.errors import Unauthorized

pytestmark = pytest.mark.integration


def _start(args, home, log):
    env = {**os.environ, "GPULAYER_HOME": str(home)}
    for k in ("GPULAYER_URL", "GPULAYER_TOKEN", "GPULAYER_FINGERPRINT"):
        env.pop(k, None)
    return subprocess.Popen(
        [sys.executable, "-m", "gpulayer.cli.main", *args],
        env=env,
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
    )


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")
def test_up_join_connect(tmp_path):
    up_log = tmp_path / "up.log"
    up = _start(["up", "--host", "127.0.0.1", "--port", "0", "--tls"], tmp_path / "server", up_log)
    procs = [up]
    try:
        for _ in range(60):
            if "Press Ctrl" in up_log.read_text():
                break
            time.sleep(0.5)
        text = up_log.read_text()
        url = re.search(r"GpuLayer is up at (\S+)", text).group(1)
        join_token = re.search(r"--token (glj_\S+)", text).group(1)
        client_token = re.search(r'token="(glc_[^"]+)"', text).group(1)
        fingerprint = re.search(r"--fingerprint (sha256:[0-9a-f]{64})", text).group(1)
        assert url.startswith("https://127.0.0.1:")

        procs.append(
            _start(
                [
                    "join",
                    url,
                    "--token",
                    join_token,
                    "--fingerprint",
                    fingerprint,
                    "--name",
                    "box2",
                ],
                tmp_path / "box2",
                tmp_path / "join.log",
            )
        )
        client = gl.connect(url, token=client_token, fingerprint=fingerprint)
        for _ in range(60):
            if len(client.workers()) == 2:
                break
            time.sleep(0.5)
        assert sorted(w.capabilities.origin.value for w in client.workers()) == ["existing"] * 2

        from remote_pkg.funcs import add

        assert add.options(client=client).map([(1, 2), (3, 4)]) == [3, 7]

        with pytest.raises(Unauthorized, match="fingerprint"):
            gl.Client(url=url, token=client_token, fingerprint="sha256:" + "0" * 64).workers()
        with pytest.raises(Unauthorized):
            gl.Client(url=url, token="glc_wrong", fingerprint=fingerprint).workers()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=30)
