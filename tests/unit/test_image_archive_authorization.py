"""An image archive is authorized as a whole before the daemon loads any of it.

CodeGuard CG-04: ``import_image`` loaded the archive and returned only the first
image's id, so an archive carrying the expected image plus another image, or
extra tags, passed the identity check and left the rest in the daemon. A
compromised source agent answering a legitimate distribution can send one.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.container_engine.docker_py import DockerEngine
from tensorstead.agent.image_archive import ArchiveRefused, authorize, declared_images

pytestmark = pytest.mark.unit


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _image(tag: str) -> tuple[str, dict[str, bytes], dict[str, Any]]:
    """One image: its manifest digest, its blobs, and its index descriptor."""
    config = json.dumps({"architecture": "arm64", "os": "linux", "tag": tag}).encode()
    layer = f"layer-{tag}".encode()
    manifest = json.dumps(
        {"config": {"digest": _digest(config)}, "layers": [{"digest": _digest(layer)}]}
    ).encode()
    blobs = {_digest(config): config, _digest(layer): layer, _digest(manifest): manifest}
    descriptor: dict[str, Any] = {"digest": _digest(manifest)}
    return _digest(manifest), blobs, descriptor


def _archive(
    path: Path,
    images: list[tuple[str, dict[str, bytes], dict[str, Any]]],
    *,
    names: list[list[str]] | None = None,
    repo_tags: list[str] | None = None,
) -> None:
    descriptors = []
    blobs: dict[str, bytes] = {}
    for i, (_id, image_blobs, descriptor) in enumerate(images):
        d = dict(descriptor)
        for name in (names or [[]] * len(images))[i]:
            d["annotations"] = {"io.containerd.image.name": name}
        descriptors.append(d)
        blobs.update(image_blobs)
    with tarfile.open(path, "w") as tar:

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        add("index.json", json.dumps({"manifests": descriptors}).encode())
        if repo_tags is not None:
            add("manifest.json", json.dumps([{"RepoTags": repo_tags}]).encode())
        for digest, payload in blobs.items():
            add(f"blobs/sha256/{digest.partition(':')[2]}", payload)


class _RecordingDocker:
    """A docker-py stand-in that records loads and removals, like the reviewer's probe."""

    def __init__(self, loads_as: list[str], present: list[str] | None = None) -> None:
        self.loads_as = loads_as
        self.loaded = 0
        self.removed: list[str] = []
        self.present = list(present or [])
        outer = self

        class images:  # mirrors docker-py
            @staticmethod
            def load(_payload: Any) -> list[Any]:
                outer.loaded += 1
                return [type("Image", (), {"id": i})() for i in outer.loads_as]

            @staticmethod
            def list(**_filters: Any) -> list[Any]:
                return [type("Image", (), {"id": i})() for i in outer.present]

            @staticmethod
            def remove(image: str, **_options: Any) -> None:
                outer.removed.append(image)

        self.images = images


def _engine(docker: _RecordingDocker) -> DockerEngine:
    engine = DockerEngine(client=docker)
    engine._archive_client = docker  # the archive client is the same recording fake
    return engine


def test_an_extra_image_is_refused_before_anything_is_loaded(tmp_path: Path) -> None:
    expected = _image("expected")
    for order in ([expected, _image("attacker")], [_image("attacker"), expected]):
        archive = tmp_path / "a.tar"
        _archive(archive, order)
        docker = _RecordingDocker(loads_as=[expected[0]])
        with pytest.raises(ImageBuildError, match="declares 2 images"):
            _engine(docker).import_image(archive_path=str(archive), expected_id=expected[0])
        assert docker.loaded == 0


@pytest.mark.parametrize("where", ["annotation", "repo_tags"])
def test_an_unauthorized_name_is_refused_before_loading(tmp_path: Path, where: str) -> None:
    expected = _image("expected")
    archive = tmp_path / "a.tar"
    if where == "annotation":
        _archive(archive, [expected], names=[["docker.io/library/vllm:latest"]])
    else:
        _archive(archive, [expected], repo_tags=["repo/vllm:tag", "vllm:latest"])
    docker = _RecordingDocker(loads_as=[expected[0]])

    with pytest.raises(ImageBuildError, match="not authorized"):
        _engine(docker).import_image(
            archive_path=str(archive),
            expected_id=expected[0],
            allowed_names=frozenset({"repo/vllm:tag"}),
        )
    assert docker.loaded == 0


def test_the_wrong_image_is_refused_before_loading(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar"
    _archive(archive, [_image("other")])
    docker = _RecordingDocker(loads_as=["sha256:anything"])
    with pytest.raises(ImageBuildError, match="not the expected"):
        _engine(docker).import_image(archive_path=str(archive), expected_id=_image("expected")[0])
    assert docker.loaded == 0


def test_a_load_that_differs_from_the_declaration_removes_only_what_it_introduced(
    tmp_path: Path,
) -> None:
    expected = _image("expected")
    archive = tmp_path / "a.tar"
    _archive(archive, [expected])
    docker = _RecordingDocker(
        loads_as=[expected[0], "sha256:smuggled", "sha256:preexisting"],
        present=["sha256:preexisting"],
    )
    with pytest.raises(ImageBuildError, match="not the one image it declared"):
        _engine(docker).import_image(archive_path=str(archive), expected_id=expected[0])
    assert "sha256:smuggled" in docker.removed
    assert "sha256:preexisting" not in docker.removed


def test_the_authorized_image_and_name_load(tmp_path: Path) -> None:
    expected = _image("expected")
    archive = tmp_path / "a.tar"
    _archive(archive, [expected], repo_tags=["repo/vllm:tag"])
    docker = _RecordingDocker(loads_as=[expected[0]])
    image_id = _engine(docker).import_image(
        archive_path=str(archive),
        expected_id=expected[0],
        allowed_names=frozenset({"repo/vllm:tag"}),
    )
    assert image_id == expected[0]
    assert docker.loaded == 1 and docker.removed == []


def test_familiar_and_qualified_names_are_the_same_name(tmp_path: Path) -> None:
    expected = _image("expected")
    archive = tmp_path / "a.tar"
    _archive(archive, [expected], names=[["docker.io/library/alpine:3.22"]])
    image = authorize(
        declared_images(str(archive)), expected_id=None, allowed_names={"alpine:3.22"}
    )
    assert expected[0] in image.identities


def test_an_archive_without_an_index_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("repositories")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(ArchiveRefused, match=r"index\.json"):
        declared_images(str(archive))
