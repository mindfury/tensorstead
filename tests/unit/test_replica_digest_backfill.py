"""A replica fetched before downloads were digested can be digested in place.

Such a model could not be copied node to node -- replication refuses without
a digest to verify against -- so placing it on a second node meant a second
download from upstream (Nemotron Bolt on the estate, 2026-10-21).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import UnsupportedTreeEntry, directory_digest

pytestmark = pytest.mark.unit

_ID = "huggingface:nvidia/Model"


def _service_with_replica(tmp_path: Path) -> ModelAcquisitionService:
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    staging = tmp_path / "models" / ".staging-test"
    (staging / "sub").mkdir(parents=True)
    (staging / "empty").mkdir()
    (staging / "config.json").write_text("{}")
    (staging / "sub" / "model.safetensors").write_bytes(b"x" * 1000)
    service.promote(_ID, staging, resolved_revision="abc", content_digest=None)
    return service


def test_a_digestless_replica_gains_the_digest_a_download_would_have_recorded(
    tmp_path: Path,
) -> None:
    service = _service_with_replica(tmp_path)
    assert service.get_replica(_ID).content_digest is None  # type: ignore[union-attr]

    digest, size = service.digest_replica(_ID)

    replica = service.get_replica(_ID)
    assert replica is not None
    assert digest == directory_digest(Path(replica.local_path))
    assert digest.startswith("tsdir2-sha256:")
    assert size == 1002
    assert (replica.content_digest, replica.size_bytes) == (digest, size)
    assert replica.state == "available"


def test_digesting_twice_gives_the_same_answer(tmp_path: Path) -> None:
    service = _service_with_replica(tmp_path)
    assert service.digest_replica(_ID) == service.digest_replica(_ID)


def test_a_replica_that_is_not_here_is_refused(tmp_path: Path) -> None:
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    with pytest.raises(LookupError, match="no available replica"):
        service.digest_replica(_ID)


def test_a_tree_with_a_link_is_refused_not_given_an_ambiguous_digest(tmp_path: Path) -> None:
    service = _service_with_replica(tmp_path)
    replica = service.get_replica(_ID)
    assert replica is not None
    os.symlink("config.json", Path(replica.local_path) / "alias.json")
    with pytest.raises(UnsupportedTreeEntry):
        service.digest_replica(_ID)
    assert service.get_replica(_ID).content_digest is None  # type: ignore[union-attr]
