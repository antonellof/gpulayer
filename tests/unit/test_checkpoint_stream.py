"""The worker stores each checkpoint the job announces (ADR-0025). Simulated control plane."""

from __future__ import annotations

import os
import threading

from gpulayer.worker.agent import _CheckpointStream


class Channel:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.stored = threading.Event()

    def put_artifact(self, data, name, kind):
        self.sent.append(data)
        return f"ref-{len(self.sent)}"

    def checkpoint(self, task_id, ref):
        self.stored.set()


def test_a_checkpoint_with_the_same_time_and_size_is_still_sent(tmp_path):
    """Linux stamps files from a coarse clock: a checkpoint saved on SIGTERM a few ms after
    the epoch's had the same time and size, was skipped, and the next attempt repeated steps."""
    path = tmp_path / "last.ckpt"
    channel = Channel()
    stream = _CheckpointStream(channel, "task-1", path)
    path.write_bytes(b"step-10")
    stamp = path.stat().st_mtime_ns
    stream.notify()
    assert channel.stored.wait(10)
    path.write_bytes(b"step-12")  # same size
    os.utime(path, ns=(stamp, stamp))  # same time
    stream.close()
    assert channel.sent == [b"step-10", b"step-12"]


def test_an_unchanged_checkpoint_is_not_sent_again(tmp_path):
    path = tmp_path / "last.ckpt"
    channel = Channel()
    stream = _CheckpointStream(channel, "task-1", path)
    path.write_bytes(b"step-10")
    stream.notify()
    assert channel.stored.wait(10)
    stream.close()
    assert channel.sent == [b"step-10"]
