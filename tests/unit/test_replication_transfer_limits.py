"""Bounded artifact transfer.

Three failures shared one root cause: nothing capped how much memory or disk
one artifact transfer could consume, because nothing was watching. Model
serving built the *entire* tar in memory before yielding a single byte
(``stream_archive``), and the inbound pull had no notion of "too much" at
all -- a timeout bounds how long a transfer may run, not how many bytes it
may write.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent import replication as replication_module
from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import PeerReplicationService, ReplicationError

pytestmark = pytest.mark.unit


def _acquisition(tmp_path: Path) -> ModelAcquisitionService:
    return ModelAcquisitionService(store_dir=tmp_path / "store", marker_dir=tmp_path / "markers")


class _FakePeer:
    """A ``PeerClient`` whose ``fetch_content`` yields exactly the given chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.calls = 0

    def fetch_content(self, **_kwargs: Any) -> Any:
        self.calls += 1
        yield from self._chunks


def _no_staging_left_behind(root: Path) -> bool:
    return not any(p.name.count(".staging.") for p in root.rglob("*.staging.*"))


# ----------------------------------------------------------- stream_archive


def test_stream_archive_round_trips_directory_contents(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (source / "weights.bin").write_bytes(b"\x00\x01" * 10_000)

    service = PeerReplicationService(acquisition=_acquisition(tmp_path))
    produced = b"".join(service.stream_archive(source))

    with tarfile.open(fileobj=io.BytesIO(produced), mode="r:") as tar:
        names = sorted(m.name for m in tar.getmembers())
        assert names == ["config.json", "weights.bin"]
        config_member = tar.extractfile("config.json")
        weights_member = tar.extractfile("weights.bin")
        assert config_member is not None and config_member.read() == b'{"a": 1}'
        assert weights_member is not None and weights_member.read() == b"\x00\x01" * 10_000


def test_stream_archive_builds_its_buffer_on_the_model_s_own_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix moves the buffer off RAM.

    The previous version's harm was an in-memory ``BytesIO`` holding a second
    full copy of a multi-gigabyte model. The replacement must land on disk,
    and specifically on the *same* filesystem as the model directory it is
    archiving (``path.parent`` -- the managed store, which by construction
    already has room for one model's worth of data) rather than the platform
    default temp directory, which is not guaranteed to have room and, on some
    Linux configurations, is itself RAM-backed (tmpfs) -- reintroducing the
    exact hazard this fix removes.
    """
    seen_dirs: list[Any] = []
    real_temporary_file = tempfile.TemporaryFile

    def _capturing_temporary_file(*args: Any, **kwargs: Any) -> Any:
        seen_dirs.append(kwargs.get("dir"))
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setattr(replication_module.tempfile, "TemporaryFile", _capturing_temporary_file)

    source = tmp_path / "model"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"w" * 1000)

    service = PeerReplicationService(acquisition=_acquisition(tmp_path))
    b"".join(service.stream_archive(source))

    assert seen_dirs == [source.parent]


def test_stream_archive_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"w" * 1000)
    before = set(source.parent.iterdir())

    service = PeerReplicationService(acquisition=_acquisition(tmp_path))
    b"".join(service.stream_archive(source))

    after = set(source.parent.iterdir())
    assert after == before, f"stream_archive left new entries behind: {after - before}"


# --------------------------------------------------- _pull_into / replicate


def test_replicate_refuses_a_peer_that_sends_more_than_the_configured_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSORSTEAD_MAX_ARTIFACT_BYTES", "10")
    peer = _FakePeer([b"0123456789", b"eleven more bytes past the ceiling"])
    service = PeerReplicationService(acquisition=_acquisition(tmp_path), peer_client=peer)

    with pytest.raises(ReplicationError) as caught:
        service.replicate(model_id="m1", content_digest="sha256:x", source_endpoint="https://peer")

    assert "ceiling" in str(caught.value)
    assert _no_staging_left_behind(tmp_path / "store")


def test_replicate_refuses_to_start_when_free_disk_is_below_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight: refused before the peer is ever contacted, not partway in.

    Against a real near-full disk reading, not an absurdly high configured
    ceiling: comparing free space to the *ceiling* itself was the bug review
    caught -- it refused an ordinary
    small transfer on any host with less free space than the (200 GiB,
    generous, worst-case) default, which every real host with a modest disk
    has. The preflight's own threshold is a small fixed floor now, so this
    test simulates a disk that is genuinely below it.
    """
    tiny_free = replication_module._MIN_FREE_BYTES_TO_ATTEMPT - 1

    def _fake_disk_usage(_path: Any) -> Any:
        return type("Usage", (), {"free": tiny_free, "total": 0, "used": 0})()

    monkeypatch.setattr(replication_module.shutil, "disk_usage", _fake_disk_usage)
    peer = _FakePeer([b"data"])
    service = PeerReplicationService(acquisition=_acquisition(tmp_path), peer_client=peer)

    with pytest.raises(ReplicationError) as caught:
        service.replicate(model_id="m1", content_digest="sha256:x", source_endpoint="https://peer")

    assert "free" in str(caught.value).lower()
    assert peer.calls == 0, "must refuse before ever contacting the peer"


def test_replicate_is_unaffected_by_a_ceiling_larger_than_free_disk(tmp_path: Path) -> None:
    """The regression itself: a generous ceiling must not block an ordinary
    small transfer just because the host has less free space than it.
    """
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        info = tarfile.TarInfo("weights.bin")
        data = b"w" * 100
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    peer = _FakePeer([payload.getvalue()])
    service = PeerReplicationService(acquisition=_acquisition(tmp_path), peer_client=peer)

    outcome = service.replicate(
        model_id="m3",
        content_digest=_content_digest_of(payload.getvalue()),
        source_endpoint="https://peer",
    )

    assert outcome.state == "available"
    assert peer.calls == 1


def test_a_normal_transfer_under_the_ceiling_is_unaffected(tmp_path: Path) -> None:
    """The guard must not tax the ordinary case it was not built for."""
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        info = tarfile.TarInfo("weights.bin")
        data = b"w" * 1000
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    peer = _FakePeer([payload.getvalue()])
    service = PeerReplicationService(acquisition=_acquisition(tmp_path), peer_client=peer)

    outcome = service.replicate(
        model_id="m2",
        content_digest=_content_digest_of(payload.getvalue()),
        source_endpoint="https://peer",
    )

    assert outcome.state == "available"


def _content_digest_of(archive_bytes: bytes) -> str:
    """Compute the digest ``replicate`` will verify against: the *unpacked*
    directory digest, not the archive bytes, matching ``directory_digest``.
    """
    import hashlib

    digest = hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            digest.update(member.name.encode())
            digest.update(b"\0")
            extracted = tar.extractfile(member)
            assert extracted is not None
            digest.update(extracted.read())
    return f"sha256:{digest.hexdigest()}"
