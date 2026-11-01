"""docs/pytorch/api.md is generated from the docstrings and must match the code."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).parents[2]


def test_api_reference_is_up_to_date():
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_api_reference.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_every_public_member_has_a_docstring():
    text = (ROOT / "docs" / "pytorch" / "api.md").read_text()
    empty = [line for line in text.splitlines() if line.endswith("|  |")]
    assert not empty, "members without a docstring:\n" + "\n".join(empty)
