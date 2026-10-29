"""The version is declared in several places; they must agree."""

import re
from pathlib import Path

import gpulayer

ROOT = Path(__file__).parents[2]


def test_versions_agree():
    # tomllib needs Python 3.11; the [project] version line is enough here.
    project = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    assert project
    project = project.group(1)
    site = re.search(r'VERSION = "([^"]+)"', (ROOT / "site/src/lib/version.ts").read_text())
    changelog = re.search(r"^## (\d\S*)", (ROOT / "CHANGELOG.md").read_text(), re.M)
    assert gpulayer.__version__ == project
    assert site and site.group(1) == project
    assert changelog and changelog.group(1) == project
