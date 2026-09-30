from __future__ import annotations

import tomllib
from pathlib import Path

from tensorstead.version import VERSION


def test_application_version_matches_package_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    package_version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

    assert package_version == VERSION
