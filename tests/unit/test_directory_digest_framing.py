"""The model-tree digest is framed, versioned, and refuses what it cannot describe.

CodeGuard CG-05: v1 hashed ``path NUL bytes`` per file with no length framing,
so ``{a: b"x", b: b"y"}`` and ``{a: b"xb\\0y"}`` produced the same digest -- a
peer answering a replication could substitute a differently shaped tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import (
    DIGEST_PREFIX,
    PeerReplicationService,
    ReplicationError,
    UnsupportedTreeEntry,
    directory_digest,
)

pytestmark = pytest.mark.unit


def _tree(root: Path, files: dict[str, bytes], dirs: tuple[str, ...] = ()) -> Path:
    root.mkdir(parents=True)
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def test_the_reviewers_two_trees_now_differ(tmp_path: Path) -> None:
    two = _tree(tmp_path / "two", {"a": b"x", "b": b"y"})
    one = _tree(tmp_path / "one", {"a": b"xb\x00y"})
    assert directory_digest(two) != directory_digest(one)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"a": b""}, {}),  # an empty file is not nothing
        ({"ab": b"c"}, {"a": b"bc"}),  # names and contents can't trade bytes
        ({"a/b": b""}, {"a": b"", "b": b""}),
        ({"a": b"\x00\x00\x00\x00\x00\x00\x00\x01"}, {"a": b"", "b": b""}),
    ],
)
def test_boundary_like_trees_are_distinct(
    tmp_path: Path, left: dict[str, bytes], right: dict[str, bytes]
) -> None:
    assert directory_digest(_tree(tmp_path / "l", left)) != directory_digest(
        _tree(tmp_path / "r", right)
    )


def test_an_empty_directory_counts(tmp_path: Path) -> None:
    plain = _tree(tmp_path / "plain", {"a": b"x"})
    with_dir = _tree(tmp_path / "with", {"a": b"x"}, dirs=("empty",))
    assert directory_digest(plain) != directory_digest(with_dir)


def test_creation_order_does_not_matter(tmp_path: Path) -> None:
    first = _tree(tmp_path / "1", {"b": b"2", "a": b"1", "d/c": b"3"})
    second = _tree(tmp_path / "2", {"d/c": b"3", "a": b"1", "b": b"2"})
    assert directory_digest(first) == directory_digest(second)
    assert directory_digest(first).startswith(DIGEST_PREFIX)


@pytest.mark.parametrize("kind", ["file", "dir"])
def test_a_link_is_refused_not_followed(tmp_path: Path, kind: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x")
    root = _tree(tmp_path / "tree", {"a": b"x"})
    target = outside / "secret" if kind == "file" else outside
    os.symlink(target, root / "link")
    with pytest.raises(UnsupportedTreeEntry):
        directory_digest(root)


def test_a_legacy_unframed_digest_is_refused_at_verification(tmp_path: Path) -> None:
    acquisition = ModelAcquisitionService(store_dir=tmp_path / "s", marker_dir=tmp_path / "m")
    service = PeerReplicationService(acquisition=acquisition, peer_client=None)
    with pytest.raises(ReplicationError, match="legacy unframed digest"):
        service._verify("m1", f"{DIGEST_PREFIX}{'0' * 64}", "sha256:" + "0" * 64, None)
