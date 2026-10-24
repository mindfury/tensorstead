"""The ranged replication protocol: manifest, parallel byte ranges, in-place writes.

The archive protocol it replaces made four serial passes over a model over one
TLS stream and held a 200 Gb fabric to about 4 Gbps. What must not change is
everything the archive path already guaranteed: the digest is recomputed over
what landed, a peer's description of the tree is untrusted input checked before
anything is written, nothing reaches ``available`` without verification, and a
failure leaves no staging behind.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import threading
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent import replication as replication_module
from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import (
    PeerReplicationService,
    ReplicationError,
    directory_digest,
    parse_byte_range,
    safe_relative_path,
)

pytestmark = pytest.mark.unit

_MODEL = "huggingface:org/model"


def _small_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pieces of a few bytes, so small test files still span many ranges."""
    monkeypatch.setattr(replication_module, "_PIECE", 7)
    monkeypatch.setattr(replication_module, "_IO", 3)


def _acquisition(root: Path) -> ModelAcquisitionService:
    return ModelAcquisitionService(store_dir=root / "store", marker_dir=root / "markers")


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[PeerReplicationService, str]:
    """A source agent holding an available replica of a tree with every entry kind."""
    _small_pieces(monkeypatch)
    acquisition = _acquisition(tmp_path / "source")
    staging = acquisition.staging_path(_MODEL)
    (staging / "nested" / "deeper").mkdir(parents=True)
    (staging / "empty-dir").mkdir()
    (staging / "config.json").write_bytes(b'{"a": 1}')
    (staging / "empty-file").write_bytes(b"")
    (staging / "nested" / "deeper" / "weights.bin").write_bytes(os.urandom(1000))
    (staging / "exactly-one-piece").write_bytes(b"x" * 7)
    digest = directory_digest(staging)
    acquisition.promote(_MODEL, staging, content_digest=digest)
    return PeerReplicationService(acquisition=acquisition), digest


class _Session:
    """A ``PeerSession`` answering from a source service, as the routes would."""

    def __init__(self, peer: _InProcessPeer) -> None:
        self._peer = peer

    def fetch_range(self, path: str, offset: int, length: int, target: Path) -> None:
        self._peer.ranges.append((path, offset, length))
        if self._peer.fail_on is not None and (path, offset) == self._peer.fail_on:
            raise ReplicationError("replication_failed", "connection reset")
        source = self._peer.source.file_for(_MODEL, path)
        data = b"".join(self._peer.source.stream_file(source, offset, offset + length))
        if self._peer.corrupt:
            data = bytes(b ^ 0xFF for b in data)
        # Out of order, as several connections finish: later pieces of a
        # file usually land before earlier ones.
        if self._peer.jitter:
            threading.Event().wait(((offset * 7919) % 13) / 4000)
        fd = os.open(target, os.O_WRONLY)
        try:
            os.pwrite(fd, data, offset)
        finally:
            os.close(fd)

    def close(self) -> None:
        self._peer.closed += 1


class _InProcessPeer:
    def __init__(self, source: PeerReplicationService, **behaviour: Any) -> None:
        self.source = source
        self.manifest_override: list[Any] | None = behaviour.get("manifest")
        self.corrupt = behaviour.get("corrupt", False)
        self.fail_on: tuple[str, int] | None = behaviour.get("fail_on")
        self.jitter = behaviour.get("jitter", True)
        self.ranges: list[tuple[str, int, int]] = []
        self.closed = 0
        self.archive_fetches = 0

    def fetch_manifest(self, **_kw: Any) -> list[Any] | None:
        if self.manifest_override is not None:
            return self.manifest_override
        return [entry.as_json() for entry in self.source.manifest(_MODEL)]

    def open_session(self, **_kw: Any) -> _Session:
        return _Session(self)

    def fetch_content(self, **_kw: Any) -> Any:
        self.archive_fetches += 1
        path, _ = self.source.serve(_MODEL)
        yield from self.source.stream_archive(path)


def _no_staging_left_behind(root: Path) -> bool:
    return not list(root.rglob("*.staging.*"))


def test_a_tree_round_trips_over_many_ranges_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, digest = _source(tmp_path, monkeypatch)
    peer = _InProcessPeer(source)
    destination = _acquisition(tmp_path / "dest")
    service = PeerReplicationService(acquisition=destination, peer_client=peer)

    outcome = service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert (outcome.state, outcome.content_digest) == ("available", digest)
    assert directory_digest(destination.local_path(_MODEL)) == digest
    assert outcome.size_bytes == 8 + 1000 + 7
    assert len(peer.ranges) > 100, "the large file must have been fetched as many ranges"
    assert peer.archive_fetches == 0
    assert peer.closed == 1
    assert _no_staging_left_behind(tmp_path / "dest")


def test_bytes_that_differ_from_the_digest_are_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, digest = _source(tmp_path, monkeypatch)
    destination = _acquisition(tmp_path / "dest")
    service = PeerReplicationService(
        acquisition=destination, peer_client=_InProcessPeer(source, corrupt=True)
    )

    with pytest.raises(ReplicationError) as caught:
        service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert caught.value.code == "verification_failed"
    replica = destination.get_replica(_MODEL)
    assert replica is not None and replica.state == "failed"
    assert not destination.local_path(_MODEL).exists()
    assert _no_staging_left_behind(tmp_path / "dest")


def test_a_failed_range_fails_the_copy_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, digest = _source(tmp_path, monkeypatch)
    peer = _InProcessPeer(source, fail_on=("nested/deeper/weights.bin", 14))
    destination = _acquisition(tmp_path / "dest")
    service = PeerReplicationService(acquisition=destination, peer_client=peer)

    with pytest.raises(ReplicationError, match="connection reset"):
        service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert peer.closed == 1, "the session's connections must be closed on failure too"
    assert not destination.local_path(_MODEL).exists()
    assert _no_staging_left_behind(tmp_path / "dest")


def test_a_peer_without_a_manifest_is_copied_over_the_archive_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agents upgrade one at a time; an older source still serves only the tar."""
    source, digest = _source(tmp_path, monkeypatch)

    class _OldPeer(_InProcessPeer):
        def fetch_manifest(self, **_kw: Any) -> None:
            return None

    peer = _OldPeer(source)
    destination = _acquisition(tmp_path / "dest")
    service = PeerReplicationService(acquisition=destination, peer_client=peer)

    outcome = service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert outcome.content_digest == digest
    assert peer.archive_fetches == 1 and peer.ranges == []


def _dir(path: str) -> dict[str, Any]:
    return {"path": path, "type": "d", "size": 0}


def _file(path: str, size: int = 1) -> dict[str, Any]:
    return {"path": path, "type": "f", "size": size}


@pytest.mark.parametrize(
    "manifest",
    [
        pytest.param([_file("/etc/passwd")], id="absolute"),
        pytest.param([_dir("a"), _file("a/../../escape")], id="traversal"),
        pytest.param([_file("..")], id="dotdot"),
        pytest.param([_dir("a"), _file("a//b")], id="empty-component"),
        pytest.param([_file("./a")], id="dot-component"),
        pytest.param([_file("a\x00b")], id="nul"),
        pytest.param([_file("a"), _file("a")], id="duplicate"),
        pytest.param([_file("missing/parent")], id="undeclared-parent"),
        pytest.param([_file("a"), _file("a/b")], id="file-as-parent"),
        pytest.param([_file("a", -1)], id="negative-size"),
        pytest.param([{"path": "a", "type": "d", "size": 5}], id="sized-directory"),
        pytest.param([{"path": "a", "type": "l", "size": 0}], id="link-type"),
        pytest.param([{"path": "a", "type": "f", "size": True}], id="bool-size"),
        pytest.param([{"path": "a", "type": "f", "size": 1.5}], id="float-size"),
        pytest.param(["a"], id="not-an-object"),
    ],
)
def test_an_unsafe_manifest_is_refused_before_anything_is_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: list[Any]
) -> None:
    source, digest = _source(tmp_path, monkeypatch)
    peer = _InProcessPeer(source, manifest=manifest)
    destination = _acquisition(tmp_path / "dest")
    service = PeerReplicationService(acquisition=destination, peer_client=peer)

    with pytest.raises(ReplicationError):
        service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert peer.ranges == []
    assert not (tmp_path / "escape").exists()
    assert _no_staging_left_behind(tmp_path / "dest")


def test_a_manifest_past_the_ceiling_is_refused_before_anything_is_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSORSTEAD_MAX_ARTIFACT_BYTES", "100")
    source, digest = _source(tmp_path, monkeypatch)
    peer = _InProcessPeer(source)
    service = PeerReplicationService(acquisition=_acquisition(tmp_path / "d"), peer_client=peer)

    with pytest.raises(ReplicationError) as caught:
        service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert caught.value.code == "artifact_too_large"
    assert peer.ranges == []


def test_a_manifest_with_too_many_entries_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSORSTEAD_MAX_ARTIFACT_MEMBERS", "3")
    source, digest = _source(tmp_path, monkeypatch)
    peer = _InProcessPeer(source)
    service = PeerReplicationService(acquisition=_acquisition(tmp_path / "d"), peer_client=peer)

    with pytest.raises(ReplicationError, match="more than 3 entries"):
        service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")
    assert peer.ranges == []


def test_free_space_is_checked_against_the_real_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest finally says how big a copy is, so the preflight can use it.

    Enough free space to pass the 1 GiB floor, not enough for this model plus
    the reserve.
    """
    floor = replication_module._MIN_FREE_BYTES_TO_ATTEMPT
    usage = type("Usage", (), {"free": floor + 10, "total": 0, "used": 0})()
    monkeypatch.setattr(replication_module.shutil, "disk_usage", lambda _p: usage)
    source, digest = _source(tmp_path, monkeypatch)
    peer = _InProcessPeer(source)
    service = PeerReplicationService(acquisition=_acquisition(tmp_path / "d"), peer_client=peer)

    with pytest.raises(ReplicationError) as caught:
        service.replicate(model_id=_MODEL, content_digest=digest, source_endpoint="https://p")

    assert caught.value.code == "artifact_too_large"
    assert "free" in caught.value.message
    assert peer.ranges == []


def test_ordered_hashing_matches_sha256_whatever_order_pieces_land_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_pieces(monkeypatch)
    data = os.urandom(100)
    path = tmp_path / "f"
    path.write_bytes(data)
    state = replication_module._OrderedFileHash(path, len(data))
    pieces = sorted(range(-(-len(data) // 7)), key=lambda piece: (piece * 5) % 11)

    for piece in pieces:
        state.landed.add(piece)
        if state.ready():
            state.advance()

    assert state.complete
    assert state.result() == (len(data), hashlib.sha256(data).digest())


def test_the_archive_path_still_unpacks_with_a_large_copy_buffer(tmp_path: Path) -> None:
    """The legacy path's unpack used tarfile's 16 KiB default; it was its own slow pass."""
    seen: list[Any] = []

    class _Capturing(tarfile.TarFile):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.append(kwargs.get("copybufsize"))
            super().__init__(*args, **kwargs)

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        info = tarfile.TarInfo("weights.bin")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    archive = tmp_path / "a.tar"
    archive.write_bytes(payload.getvalue())
    staging = tmp_path / "staging"
    staging.mkdir()
    service = PeerReplicationService(acquisition=_acquisition(tmp_path))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(replication_module.tarfile, "TarFile", _Capturing)
        service._unpack(archive, staging)

    assert seen == [replication_module._IO]
    assert (staging / "weights.bin").read_bytes() == b"abc"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a", "a"),
        ("a/b.safetensors", "a/b.safetensors"),
        ("", None),
        ("/a", None),
        ("a/", None),
        ("a/./b", None),
        ("a/../b", None),
        (None, None),
        (7, None),
    ],
)
def test_safe_relative_path(value: Any, expected: str | None) -> None:
    assert safe_relative_path(value) == expected


def test_parse_byte_range() -> None:
    assert parse_byte_range(None, 10) is None
    assert parse_byte_range("bytes=0-9", 10) == (0, 10)
    assert parse_byte_range("bytes=3-3", 10) == (3, 4)
    for bad in ("bytes=0-10", "bytes=5-4", "bytes=-5", "bytes=5-", "items=0-1", "bytes=0-1,3-4"):
        with pytest.raises(ValueError):
            parse_byte_range(bad, 10)
