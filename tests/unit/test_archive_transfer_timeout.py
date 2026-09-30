"""Moving an image archive gets a timeout suited to it.

The first peer transfer that got all the way through TLS then died in local
import::

    archive for 'local/dspark-deepseek-v4-flash:0.1.1' could not be loaded:
    UnixHTTPConnectionPool(host='localhost', port=None): Read timed out.
    (read timeout=60)

after five minutes of work that had already succeeded. ``images.load`` sends a
multi-gigabyte archive over the Unix socket and then waits for one response while
Docker unpacks it, and it was inheriting the SDK's 60-second default — which is
right for ordinary engine calls and wrong for this one.

The engine now uses a **separate** client for archive-scale calls. Not a raised
global default: a lightweight call that hangs should still fail in a minute
rather than thirty. Bounded rather than disabled: an import that will never
finish must still end and be reportable.

``export_image`` was changed alongside ``import_image`` without waiting for it to
fail on hardware — it moves the same archive in the other direction, and the
source agent runs it before any peer can fetch.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from typing import Any

import pytest

from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.container_engine.docker_py import _ARCHIVE_TIMEOUT_SECONDS, DockerEngine

pytestmark = pytest.mark.unit


def _whole_image_archive() -> bytes:
    """The smallest ``docker save`` output that is a complete image."""
    config = b'{"architecture":"arm64","os":"linux"}'
    manifest = json.dumps(
        {"config": {"digest": _digest(config)}, "layers": [{"digest": _digest(b"layer")}]}
    ).encode()
    blobs = {
        _digest(config): config,
        _digest(b"layer"): b"layer",
        _digest(manifest): manifest,
    }
    index = json.dumps({"manifests": [{"digest": _digest(manifest)}]}).encode()

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:

        def _add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        _add("index.json", index)
        for digest, payload in blobs.items():
            _add(f"blobs/sha256/{digest.partition(':')[2]}", payload)
    return buffer.getvalue()


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _FakeDockerModule:
    """Stands in for the ``docker`` package to record how clients are built."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def from_env(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return _FakeClient()


class _FakeClient:
    class images:  # mirrors docker-py's attribute name
        @staticmethod
        def load(_payload: bytes) -> list[Any]:
            return [type("Image", (), {"id": "sha256:imported"})()]

        @staticmethod
        def get(_reference: str) -> Any:
            class _Image:
                id = "sha256:x"

                def save(self, **_kwargs: Any) -> list[bytes]:
                    # A *whole* archive, minimal but self-contained. These
                    # tests are about which client does the work, and
                    # ``export_image`` now also refuses an archive missing the
                    # blobs its own manifest names -- so an
                    # empty payload would fail them for an unrelated reason.
                    return [_whole_image_archive()]

            return _Image()


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> _FakeDockerModule:
    module = _FakeDockerModule()
    monkeypatch.setitem(sys.modules, "docker", module)
    return module


def test_the_archive_client_is_built_with_the_dedicated_timeout(
    fake_docker: _FakeDockerModule, tmp_path: Any
) -> None:
    """The defect: this client used to be built with the 60s default."""
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"tar")

    DockerEngine().import_image(archive_path=str(archive))

    assert fake_docker.calls == [{"timeout": _ARCHIVE_TIMEOUT_SECONDS}], fake_docker.calls


def test_the_dedicated_timeout_exceeds_the_sdk_default() -> None:
    """Pinned against docker-py's own constant, not a number we remember."""
    from docker.constants import DEFAULT_TIMEOUT_SECONDS  # type: ignore[import-untyped]

    assert _ARCHIVE_TIMEOUT_SECONDS > DEFAULT_TIMEOUT_SECONDS
    assert DEFAULT_TIMEOUT_SECONDS == 60, (
        "the default this works around changed; re-read the finding before adjusting"
    )


def test_the_timeout_is_bounded() -> None:
    """Not disabled. An import that will never finish must still terminate."""
    assert _ARCHIVE_TIMEOUT_SECONDS is not None
    assert 0 < _ARCHIVE_TIMEOUT_SECONDS <= 3600


class _RecordingImagesClient:
    """Records exactly what ``images.load`` was called with, not its bytes."""

    def __init__(self) -> None:
        self.received: list[Any] = []
        outer = self

        class images:  # mirrors docker-py's attribute name
            @staticmethod
            def load(payload: Any) -> list[Any]:
                outer.received.append(payload)
                return [type("Image", (), {"id": "sha256:imported"})()]

        self.images = images


def test_import_image_passes_a_file_object_not_a_read_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Hand the open file to ``images.load``, not the fully-read bytes.

    ``handle.read()`` loaded the entire archive -- tens of gigabytes for the
    images this product builds -- into one Python ``bytes`` object before
    ``images.load`` ever saw it. docker-py passes ``data`` straight through to
    requests, which streams a file-like object in chunks instead; the fix is
    to hand it the open file, not its contents.
    """
    recorder = _RecordingImagesClient()

    class _RecordingDockerModule:
        def from_env(self, **_kwargs: Any) -> Any:
            return recorder

    monkeypatch.setitem(sys.modules, "docker", _RecordingDockerModule())
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"tar-bytes")

    DockerEngine().import_image(archive_path=str(archive))

    assert len(recorder.received) == 1
    payload = recorder.received[0]
    assert not isinstance(payload, (bytes, bytearray)), (
        f"expected a file object, got the fully-read archive as {type(payload)}"
    )
    assert hasattr(payload, "read"), "expected a file-like object"


def test_ordinary_calls_keep_the_short_default(fake_docker: _FakeDockerModule) -> None:
    """The reason this is a second client rather than a raised global default.

    A lightweight engine call that hangs should still fail in a minute.
    """
    DockerEngine().image_digest("repo/image:tag")

    assert fake_docker.calls == [{}], (
        f"the ordinary client was built with {fake_docker.calls}; raising the "
        f"global default would make every hung call wait for the archive bound"
    )


def test_export_uses_the_archive_client_too(fake_docker: _FakeDockerModule, tmp_path: Any) -> None:
    """Same archive, other direction — fixed without waiting for it to fail."""
    DockerEngine().export_image(reference="local/x:1", archive_path=str(tmp_path / "out.tar"))

    assert fake_docker.calls == [{"timeout": _ARCHIVE_TIMEOUT_SECONDS}]


def test_a_timed_out_import_names_the_limit_it_reached(tmp_path: Any) -> None:
    """A read timeout that does not say which bound was hit is not actionable.

    An operator cannot otherwise distinguish a too-short limit from an engine
    that is genuinely stuck — which is exactly the reading that had to be made
    from the hardware failure.
    """

    class _TimingOut:
        class images:  # mirrors docker-py's attribute name
            @staticmethod
            def load(_payload: bytes) -> list[Any]:
                raise TimeoutError("Read timed out. (read timeout=1800)")

    archive = tmp_path / "image.tar"
    archive.write_bytes(b"tar")

    with pytest.raises(ImageBuildError) as caught:
        DockerEngine(client=_TimingOut()).import_image(archive_path=str(archive))

    message = str(caught.value)
    assert "Read timed out" in message
    assert f"managed archive timeout {_ARCHIVE_TIMEOUT_SECONDS}s" in message


def test_an_injected_client_is_still_used_for_archives() -> None:
    """Tests and callers that supply a client keep controlling both paths."""
    injected = _FakeClient()

    engine = DockerEngine(client=injected)

    assert engine._docker_for_archives() is injected
    assert engine._docker() is injected


# ---------------------------------------------------------------- ordering
#
# The first fix for this was inert in production and every test above still
# passed, because they all began with a fresh engine. ``_docker`` caches its
# lazily built 60s client on the same attribute the archive path consulted, so
# any archive call that *followed an ordinary one* silently got the short
# timeout -- and in ``build_and_distribute`` the source always builds before it
# exports. The tests modelled the fix rather than the sequence the product
# actually performs.
#
# These start from the ordinary call on purpose.


def test_export_after_an_ordinary_call_still_gets_the_archive_timeout(
    fake_docker: _FakeDockerModule, tmp_path: Any
) -> None:
    """The live defect: build_and_distribute builds, then exports, on one engine."""
    engine = DockerEngine()
    engine.image_digest("repo/image:tag")

    engine.export_image(reference="local/x:1", archive_path=str(tmp_path / "out.tar"))

    assert {"timeout": _ARCHIVE_TIMEOUT_SECONDS} in fake_docker.calls, (
        f"the export reused the ordinary 60s client; built {fake_docker.calls}"
    )


def test_import_after_an_ordinary_call_still_gets_the_archive_timeout(
    fake_docker: _FakeDockerModule, tmp_path: Any
) -> None:
    """The destination has the same exposure once it has made any engine call."""
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"tar")
    engine = DockerEngine()
    engine.image_digest("repo/image:tag")

    engine.import_image(archive_path=str(archive))

    assert {"timeout": _ARCHIVE_TIMEOUT_SECONDS} in fake_docker.calls, (
        f"the import reused the ordinary 60s client; built {fake_docker.calls}"
    )


def test_the_two_clients_stay_distinct_in_either_order(
    fake_docker: _FakeDockerModule, tmp_path: Any
) -> None:
    """Whichever comes first, each path keeps its own client and its own bound."""
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"tar")

    engine = DockerEngine()
    engine.import_image(archive_path=str(archive))  # archive path first
    engine.image_digest("repo/image:tag")  # then the ordinary one
    engine.export_image(reference="local/x:1", archive_path=str(tmp_path / "out.tar"))

    assert fake_docker.calls.count({"timeout": _ARCHIVE_TIMEOUT_SECONDS}) == 1, (
        f"the archive client was rebuilt or lost: {fake_docker.calls}"
    )
    assert fake_docker.calls.count({}) == 1, f"the ordinary client was rebuilt: {fake_docker.calls}"
    assert engine._docker() is not engine._docker_for_archives()
