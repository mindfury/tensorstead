"""Replication refuses an artifact before writing anything over budget.

CodeGuard CG-06: the transfer ceiling bounded wire bytes only. ``extractall``
then expanded whatever the archive declared -- a sparse member can declare
gigabytes in a few kilobytes (the reviewer's probe: 8 MiB logical from a 10 KiB
tar) -- with no bound on member count, logical size or time, and hashing then
read all of it. Each check below must fire *before* the offending entry is
extracted.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import PeerReplicationService, ReplicationError

pytestmark = pytest.mark.unit


class _Tar:
    """Yields crafted entries and records what extraction was asked to write."""

    def __init__(self, members: list[tarfile.TarInfo]) -> None:
        self._members = iter(members)
        self.extracted: list[str] = []

    def next(self) -> tarfile.TarInfo | None:
        return next(self._members, None)

    def extract(self, member: tarfile.TarInfo, _path: Any, **_kw: Any) -> None:
        self.extracted.append(member.name)


def _entry(name: str, size: int = 1, kind: bytes = tarfile.REGTYPE) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.type = kind
    return info


@pytest.fixture
def service(tmp_path: Path) -> PeerReplicationService:
    acquisition = ModelAcquisitionService(store_dir=tmp_path / "s", marker_dir=tmp_path / "m")
    return PeerReplicationService(acquisition=acquisition)


def test_a_sparse_member_is_refused_before_extraction(
    service: PeerReplicationService, tmp_path: Path
) -> None:
    sparse = _entry("weights", size=8 * 1024 * 1024, kind=tarfile.GNUTYPE_SPARSE)
    tar = _Tar([_entry("config.json"), sparse])
    with pytest.raises(ReplicationError, match="not a plain file"):
        service._extract_within_budget(tar, tmp_path)  # type: ignore[arg-type]
    assert tar.extracted == ["config.json"]


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_links_and_special_files_are_refused(
    service: PeerReplicationService, tmp_path: Path, kind: bytes
) -> None:
    tar = _Tar([_entry("x", kind=kind)])
    with pytest.raises(ReplicationError, match="not a plain file"):
        service._extract_within_budget(tar, tmp_path)  # type: ignore[arg-type]
    assert tar.extracted == []


def test_cumulative_logical_size_over_the_ceiling_is_refused(
    service: PeerReplicationService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSORSTEAD_MAX_ARTIFACT_BYTES", "100")
    tar = _Tar([_entry("a", size=60), _entry("b", size=60)])
    with pytest.raises(ReplicationError, match="transfer ceiling"):
        service._extract_within_budget(tar, tmp_path)  # type: ignore[arg-type]
    assert tar.extracted == ["a"]


def test_too_many_entries_are_refused(
    service: PeerReplicationService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSORSTEAD_MAX_ARTIFACT_MEMBERS", "3")
    tar = _Tar([_entry(f"e{i}", size=0) for i in range(10)])
    with pytest.raises(ReplicationError, match="more than 3 entries"):
        service._extract_within_budget(tar, tmp_path)  # type: ignore[arg-type]
    assert len(tar.extracted) == 3


def test_the_disk_reserve_is_kept(
    service: PeerReplicationService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    real = shutil.disk_usage

    def tight(path: Any) -> Any:
        usage = real(path)
        return type(usage)(usage.total, usage.used, 2 * 1024 * 1024 * 1024)  # 2 GiB free

    monkeypatch.setattr(shutil, "disk_usage", tight)
    tar = _Tar([_entry("big", size=1536 * 1024 * 1024)])  # would leave 0.5 GiB
    with pytest.raises(ReplicationError, match="bytes free"):
        service._extract_within_budget(tar, tmp_path)  # type: ignore[arg-type]
    assert tar.extracted == []


def test_a_real_archive_with_a_link_is_refused_and_nothing_escapes(
    service: PeerReplicationService, tmp_path: Path
) -> None:
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        data = b"{}"
        info = tarfile.TarInfo("config.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ReplicationError, match="not a plain file"):
        service._unpack(archive, staging)
    assert not (staging / "escape").exists()
