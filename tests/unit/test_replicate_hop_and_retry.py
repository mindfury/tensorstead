"""A large copy is waited for, and re-running one that already finished is free.

On the estate, copying a 95 GB checkpoint from spark-471b to spark-6d42 was
marked failed after 30 s (the default per-call bound) while both agents carried
on, the failure told the operator to check ``image_list``, and a retry would
have wiped the finished copy and transferred it again.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import PeerReplicationService, directory_digest
from tensorstead.coordinator.node_http import NodeHTTPClient
from tensorstead.domain.models import Node

pytestmark = pytest.mark.unit


def _node(name: str = "spark-6d42.internal") -> Node:
    return Node(
        id="01M05TMJ7C4GX59EVPK95MNSPB",
        name=name,
        agent_endpoint="https://10.100.89.2:8443",
        agent_contract_version="1.15",
        agent_cert_fingerprint="sha256:abc",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )


def test_the_coordinator_waits_for_a_copy_as_long_as_for_a_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def call(self: Any, node: Node, method: str, path: str, **kw: Any) -> dict[str, Any]:
        seen[path] = kw.get("timeout")
        return {}

    monkeypatch.setattr(NodeHTTPClient, "_call", call)
    client = NodeHTTPClient(management_token="t")
    client.replicate_model(_node(), model={"model_id": "m"}, source_node=_node("spark-471b"))
    client.acquire_model(
        _node(), source_id="huggingface", source_model_id="m", revision="r", credential=None
    )
    assert seen["/agent/v1/models:replicate"] == seen["/agent/v1/models:acquire"] >= 3600


def test_a_timed_out_model_copy_points_at_the_model_list() -> None:
    error = NodeHTTPClient._timeout_error(
        _node(), "/agent/v1/models:replicate", 30.0, httpx.ReadTimeout("timed out")
    )
    assert "model_list" in error.message
    assert "image_list" not in error.message
    assert error.detail["work_may_still_be_running"] is True


class _Peer:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_content(self, **_kw: Any) -> Any:
        self.calls += 1
        raise AssertionError("a replica already here was fetched again")


def _held(tmp_path: Path) -> tuple[ModelAcquisitionService, str]:
    acquisition = ModelAcquisitionService(
        store_dir=tmp_path / "store", marker_dir=tmp_path / "markers"
    )
    staging = tmp_path / "store" / ".staging-x"
    staging.mkdir(parents=True)
    (staging / "config.json").write_text("{}")
    digest = directory_digest(staging)
    acquisition.promote("huggingface:m", staging, content_digest=digest)
    return acquisition, digest


def test_a_replica_already_here_with_the_same_digest_is_not_copied_again(
    tmp_path: Path,
) -> None:
    acquisition, digest = _held(tmp_path)
    peer = _Peer()
    service = PeerReplicationService(acquisition=acquisition, peer_client=peer)

    outcome = service.replicate(
        model_id="huggingface:m", content_digest=digest, source_endpoint="https://peer:8443"
    )

    assert peer.calls == 0
    assert (outcome.state, outcome.content_digest) == ("available", digest)


def test_a_different_digest_still_copies(tmp_path: Path) -> None:
    acquisition, _digest = _held(tmp_path)
    peer = _Peer()
    service = PeerReplicationService(acquisition=acquisition, peer_client=peer)
    with pytest.raises(Exception, match="fetched again"):
        service.replicate(
            model_id="huggingface:m",
            content_digest="tsdir2-sha256:" + "0" * 64,
            source_endpoint="https://peer:8443",
        )
    assert peer.calls == 1


def test_a_retry_records_the_revision_a_copied_replica_was_stored_without(tmp_path: Path) -> None:
    """An approval binds a revision; a copied replica without one could not start."""
    acquisition, digest = _held(tmp_path)
    assert acquisition.get_replica("huggingface:m").resolved_revision is None  # type: ignore[union-attr]
    service = PeerReplicationService(acquisition=acquisition, peer_client=_Peer())
    service.replicate(
        model_id="huggingface:m",
        content_digest=digest,
        source_endpoint="https://peer:8443",
        resolved_revision="b9ce18a6",
    )
    assert acquisition.get_replica("huggingface:m").resolved_revision == "b9ce18a6"  # type: ignore[union-attr]
