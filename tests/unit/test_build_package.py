"""Per-build wheel archiving (rollback recoverability).

``scripts/build_package.py`` archives each build's wheel + sdist + manifest into
``dist/builds/<build_number>/`` so a deploy that goes wrong can be rolled back by
pointing ``TENSORSTEAD_DEPLOY_ARTIFACT`` at a previous build directory. The build
number is monotonic and the archive directory is never overwritten, which is the
whole point: the prior build's wheel survives on disk.

These tests exercise the archiving logic directly (no ``uv build``) by importing
the script as a module.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_package.py"

pytestmark = pytest.mark.unit


def _build_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_package", SCRIPT)
    assert spec and spec.loader, "could not load build_package.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(build_number: int, wheel_name: str) -> dict:
    return {
        "build_number": build_number,
        "built_at": "2026-08-11T00:00:00+00:00",
        "package_version": "0.1.8",
        "wheel": wheel_name,
        "sha256": f"sha-for-build-{build_number}",
        "git_revision": f"rev-{build_number}",
        "source_tree_dirty": False,
    }


def _make_fake_artifact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_archive_build_copies_artifacts_into_a_per_build_directory(tmp_path: Path) -> None:
    """A build's wheel, sdist, and manifest land in dist/builds/<n>/."""
    bp = _build_module()
    archive_build = bp._archive_build

    dist = tmp_path / "dist"
    wheel = dist / "tensorstead-0.1.8-py3-none-any.whl"
    sdist = dist / "tensorstead-0.1.8.tar.gz"
    _make_fake_artifact(wheel, b"wheel-bytes-1")
    _make_fake_artifact(sdist, b"sdist-bytes-1")
    record = _record(1, wheel.name)

    build_dir = archive_build(dist, wheel, sdist, record)

    assert build_dir == dist / "builds" / "1"
    assert (build_dir / wheel.name).read_bytes() == b"wheel-bytes-1"
    assert (build_dir / sdist.name).read_bytes() == b"sdist-bytes-1"
    manifest = json.loads((build_dir / f"{wheel.name}.build.json").read_text())
    assert manifest["build_number"] == 1
    assert manifest["sha256"] == "sha-for-build-1"
    assert manifest["git_revision"] == "rev-1"


def test_two_builds_leave_independent_non_overwritten_artifacts(tmp_path: Path) -> None:
    """Build N+1 does not touch build N's archived wheel (rollback recoverability)."""
    bp = _build_module()
    archive_build = bp._archive_build

    dist = tmp_path / "dist"
    # uv build refreshes the canonical wheel in dist/ each build; simulate that
    # by writing different bytes for each "build" before archiving.
    wheel = dist / "tensorstead-0.1.8-py3-none-any.whl"
    sdist = dist / "tensorstead-0.1.8.tar.gz"

    _make_fake_artifact(wheel, b"wheel-bytes-1")
    _make_fake_artifact(sdist, b"sdist-bytes-1")
    first = archive_build(dist, wheel, sdist, _record(1, wheel.name))

    _make_fake_artifact(wheel, b"wheel-bytes-2")
    _make_fake_artifact(sdist, b"sdist-bytes-2")
    second = archive_build(dist, wheel, sdist, _record(2, wheel.name))

    assert first != second
    # Build 1's archived wheel is untouched by build 2.
    assert (first / wheel.name).read_bytes() == b"wheel-bytes-1"
    assert (second / wheel.name).read_bytes() == b"wheel-bytes-2"
    assert json.loads((first / f"{wheel.name}.build.json").read_text())["build_number"] == 1
    assert json.loads((second / f"{wheel.name}.build.json").read_text())["build_number"] == 2


def test_archive_build_refuses_to_overwrite_an_existing_build(tmp_path: Path) -> None:
    """A reused build number fails loudly rather than destroying prior history."""
    bp = _build_module()
    archive_build = bp._archive_build

    dist = tmp_path / "dist"
    wheel = dist / "tensorstead-0.1.8-py3-none-any.whl"
    sdist = dist / "tensorstead-0.1.8.tar.gz"
    _make_fake_artifact(wheel, b"wheel-bytes")
    _make_fake_artifact(sdist, b"sdist-bytes")

    archive_build(dist, wheel, sdist, _record(7, wheel.name))
    with pytest.raises(FileExistsError):
        archive_build(dist, wheel, sdist, _record(7, wheel.name))


def test_archive_build_works_without_an_sdist(tmp_path: Path) -> None:
    """A wheel-only build still archives the wheel and manifest."""
    bp = _build_module()
    archive_build = bp._archive_build

    dist = tmp_path / "dist"
    wheel = dist / "tensorstead-0.1.8-py3-none-any.whl"
    _make_fake_artifact(wheel, b"wheel-bytes")
    build_dir = archive_build(dist, wheel, None, _record(5, wheel.name))

    assert (build_dir / wheel.name).read_bytes() == b"wheel-bytes"
    assert not list(build_dir.glob("*.tar.gz"))
    assert (build_dir / f"{wheel.name}.build.json").exists()
