"""A finished task's outcome survives a control plane that is briefly unreachable.

Regression: on Kubernetes, a worker finished a task while the server pod restarted; the
refused connection killed the reporting thread, the outcome was lost, and the server kept the
task running (the worker shown busy) after it came back."""

import threading

from gpulayer.contracts import TaskOutcome, TaskState
from gpulayer.errors import Conflict
from gpulayer.worker.agent import WorkerAgent


class FlakyChannel:
    def __init__(self, failures):
        self.failures = list(failures)
        self.reported = []

    def finish(self, task_id, outcome):
        if self.failures:
            raise self.failures.pop(0)
        self.reported.append((task_id, outcome.state))


def agent(channel, retry_seconds=5.0):
    a = WorkerAgent.__new__(WorkerAgent)  # only what _report uses
    a.channel = channel
    a.stop = threading.Event()
    a.REPORT_RETRY_SECONDS = retry_seconds
    return a


def test_outcome_is_retried_until_the_server_is_back():
    ch = FlakyChannel([ConnectionRefusedError(111, "refused"), ConnectionResetError()])
    assert agent(ch)._report("task-1", TaskOutcome(state=TaskState.SUCCEEDED))
    assert ch.reported == [("task-1", TaskState.SUCCEEDED)]


def test_a_rejection_is_final():
    ch = FlakyChannel([Conflict("task belongs to another worker")])
    assert not agent(ch)._report("task-1", TaskOutcome(state=TaskState.SUCCEEDED))
    assert ch.reported == []


def test_gives_up_after_the_retry_window():
    ch = FlakyChannel([ConnectionRefusedError(111, "refused")] * 50)
    assert not agent(ch, retry_seconds=0.8)._report("task-1", TaskOutcome(state=TaskState.FAILED))
