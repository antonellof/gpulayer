"""@gl.remote: real execution through the embedded control plane."""

import pytest

import gpulayer as gl
from gpulayer.errors import ValidationFailed
from gpulayer.sdk.remote import _function_ref

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GPULAYER_HOME", str(tmp_path / "home"))
    with gl.Client() as c:
        yield c


def test_local_call_is_plain_python():
    from remote_pkg.funcs import add

    assert add(2, 3) == 5


def test_run_map_and_options(client):
    from remote_pkg.funcs import add

    f = add.options(client=client)
    assert f.ref == "remote_pkg.funcs:add"
    assert f.run(2, b=40, stream_logs=False) == 42
    assert f.map([(1, 1), (2, 2), (3, 3)]) == [2, 4, 6]
    with pytest.raises(gl.RemoteError, match="cuda"):
        f.options(accelerator="cuda").run(1)


def test_remote_failure_raises(client):
    from remote_pkg.funcs import boom

    with pytest.raises(gl.RemoteError, match="ValueError: nope"):
        boom.options(client=client).run(stream_logs=False)


@pytest.mark.distributed
def test_distributed_function(client):
    pytest.importorskip("torch")
    from remote_pkg.funcs import world_sum

    assert world_sum.options(client=client).run(stream_logs=False) == {"world": 2, "sum": 3.0}


def test_closures_rejected():
    def outer():
        def inner():
            return 1

        return inner

    with pytest.raises(ValidationFailed, match="top level"):
        _function_ref(outer())
    with pytest.raises(ValidationFailed):
        _function_ref(lambda: 1)


def test_distributed_contract_needs_one_entry():
    from pydantic import ValidationError

    from gpulayer.contracts import DistributedJob

    with pytest.raises(ValidationError):
        DistributedJob()
    with pytest.raises(ValidationError):
        DistributedJob(entrypoint="t.py", function="m:f")
    DistributedJob(function="m:f", function_args=[1])
