"""Unit tests for ``ModelService.delete_image`` and ``reconcile_images``.

`image delete` was a registry operation. The coordinator deleted its own row,
called no node, and printed "Image removed from node": 48 deletes, zero bytes
freed, 430 GB of images left on two nodes with no record naming them any more.
Its reference guard read only ``current_revision`` while its docstring promised
every retained one — the same defect ``_model_referrers`` had, fixed for models
by an earlier finding and left standing here.

These tests pin the four things that make the delete honest:

* the guard inspects every retained revision, so it fires before anything runs;
* the node is asked first, and the record is dropped only if the node agreed;
* a node that refuses or cannot be reached leaves **the record in place**,
  because the bytes are still there and the record is the only handle on them;
* a record whose object the node no longer holds can still be reaped, which is
  the one direction the drift ever went.

They run against a real migrated SQLite repository with a stub node client, so
the service layer runs its real transaction path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.errors import StillReferencedError
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import (
    Deployment,
    DeploymentRevision,
    ImageOrigin,
    ImageRecord,
    Model,
    ModelSource,
    Node,
)
from tensorstead.ports.node_client import AgentCallError
from tensorstead.service.models_ import ModelService

pytestmark = pytest.mark.unit

_MIGRATIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)

_DIGEST = "sha256:" + "ab" * 32
_OTHER_DIGEST = "sha256:" + "cd" * 32


class _StubNodeClient:
    """Records image calls and can be told how the node answers.

    ``present`` is what the node reports holding; ``remove_error`` makes the
    removal fail the way an in-use image or an unreachable node does.
    """

    def __init__(self) -> None:
        self.present: dict[str, list[dict[str, str]]] = {}
        self.remove_error: Exception | None = None
        self.remove_calls: list[tuple[str, str]] = []
        self.list_calls: list[str] = []

    def remove_image(self, node: Node, *, reference: str) -> dict[str, Any]:
        self.remove_calls.append((node.id, reference))
        if self.remove_error is not None:
            raise self.remove_error
        rows = self.present.get(node.id, [])
        match = [r for r in rows if r["reference"] == reference]
        if not match:
            return {"reference": reference, "removed": False, "digest": None}
        self.present[node.id] = [r for r in rows if r["reference"] != reference]
        return {"reference": reference, "removed": True, "digest": match[0]["digest"]}

    def list_images(self, node: Node) -> list[dict[str, Any]]:
        self.list_calls.append(node.id)
        if self.remove_error is not None:
            raise self.remove_error
        return list(self.present.get(node.id, []))


@pytest.fixture
def service() -> tuple[ModelService, SQLiteRepository, _StubNodeClient]:
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repo = SQLiteRepository(conn)
    client = _StubNodeClient()
    return ModelService(repo, client), repo, client


def _node(repo: SQLiteRepository, name: str = "spark-01") -> Node:
    node = Node(
        id=new_ulid(),
        name=name,
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.15",
        agent_cert_fingerprint="AA:BB:CC",
        platform_facts={},
        registered_at=datetime.now(),
    )
    repo.save_node(node)
    return node


def _image(
    repo: SQLiteRepository,
    node: Node,
    reference: str = "local/vllm:tag",
    digest: str = _DIGEST,
) -> ImageRecord:
    record = ImageRecord(
        node_id=node.id,
        reference=reference,
        digest=digest,
        pulled_at=datetime.now(),
        origin=ImageOrigin.PULLED,
    )
    repo.save_image(record)
    return record


def _deployment_referencing(
    repo: SQLiteRepository, node: Node, digest: str, *, at_revision: int, current: int
) -> Deployment:
    """A deployment whose revision ``at_revision`` uses ``digest`` on ``node``."""
    source = ModelSource("huggingface", True, True)
    repo.save_model_source(source)
    model = Model(
        id=new_ulid(),
        source_id=source.id,
        source_model_id="nvidia/Qwen3.8-Flash-Next-NVFP4",
        resolved_revision="rev1",
        revision_pinned=True,
    )
    repo.save_model(model)
    deployment = Deployment(
        id=new_ulid(), name="flash-next", desired_state="stopped", current_revision=current
    )
    repo.save_deployment(deployment)
    for n in range(1, current + 1):
        repo.insert_revision(
            DeploymentRevision(
                deployment_id=deployment.id,
                revision=n,
                model_id=model.id,
                model_source_id=model.source_id,
                source_model_id=model.source_model_id,
                resolved_revision="rev1",
                revision_pinned=True,
                runtime_type="vllm",
                runtime_version="0.27.1",
                image_reference="local/vllm:tag",
                image_digest=digest if n == at_revision else _OTHER_DIGEST,
                runtime_config={},
                participating_nodes=(node.id,),
                endpoint="10.0.0.11:8000",
                origin_platform_facts={},
            )
        )
    return deployment


# --------------------------------------------------------------- the guard


def test_image_referrers_checks_all_retained_revisions(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """An image used only by a retained (non-current) revision is still referenced.

    A deployment retains every revision for rollback, so a retained
    revision's ``image_digest`` is a live reference. Reading only
    ``current_revision`` is what let this guard pass.
    """
    svc, repo, _ = service
    node = _node(repo)
    _image(repo, node)
    _deployment_referencing(repo, node, _DIGEST, at_revision=1, current=3)

    assert svc._image_referrers(node.id, _DIGEST) == ["flash-next"]


def test_delete_refused_while_retained_revision_references_it(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """Refused with a typed error, before the node is called and before the row goes."""
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node)
    client.present[node.id] = [{"reference": "local/vllm:tag", "digest": _DIGEST}]
    _deployment_referencing(repo, node, _DIGEST, at_revision=1, current=3)

    with pytest.raises(StillReferencedError) as excinfo:
        svc.delete_image(node.id, _DIGEST)

    assert "flash-next" in str(excinfo.value)
    assert client.remove_calls == [], "the node must not be touched by a refused delete"
    assert len(repo.list_images()) == 1, "the record must survive a refusal"


# ------------------------------------------------- the delete reaches the node


def test_delete_removes_the_image_from_the_node_then_the_record(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """The defect itself: the delete must call the node, not only drop a row."""
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node)
    client.present[node.id] = [{"reference": "local/vllm:tag", "digest": _DIGEST}]

    result = svc.delete_image(node.id, _DIGEST)

    assert client.remove_calls == [(node.id, "local/vllm:tag")], (
        "48 deletes freed zero bytes because this call was never made"
    )
    assert result["outcome"] == "removed"
    assert result["removed_from_node"] is True
    assert repo.list_images() == []
    assert client.present[node.id] == []


def test_delete_removes_by_reference_not_digest(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """One object, three tags, three records — removing one record untags one tag.

    This is the ``ple1``/``ple2``/``ple3`` shape from the estate: three
    references on a single 31.1 GB object. Removing by digest would take all
    three at once, which no single record means.
    """
    svc, repo, client = service
    node = _node(repo)
    for tag in ("local/x:ple1", "local/x:ple2", "local/x:ple3"):
        _image(repo, node, reference=tag, digest=_DIGEST)
    tags = ("local/x:ple1", "local/x:ple2", "local/x:ple3")
    client.present[node.id] = [{"reference": t, "digest": _DIGEST} for t in tags]

    svc.delete_image(node.id, _DIGEST)

    assert len(client.remove_calls) == 1
    _, reference = client.remove_calls[0]
    assert reference.startswith("local/x:ple")
    assert len(client.present[node.id]) == 2, "the other two tags must survive"


def test_delete_reaps_the_record_when_the_node_does_not_hold_the_image(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """A record that outlived its object is still deletable, and says so.

    This is the direction the drift always went — 50 records against
    25 objects — and the outcome must not claim a removal that did not happen.
    """
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node)
    client.present[node.id] = []  # the object is already gone

    result = svc.delete_image(node.id, _DIGEST)

    assert result["outcome"] == "record_reaped"
    assert result["removed_from_node"] is False
    assert repo.list_images() == []


@pytest.mark.parametrize(
    ("code", "why"),
    [
        ("image_in_use", "a container still holds it, so the bytes are still there"),
        ("unreachable", "a delete that cannot reach the node has not deleted anything"),
    ],
)
def test_record_survives_when_the_node_refuses(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
    code: str,
    why: str,
) -> None:
    """The regression that matters: no record may be dropped on a failed removal.

    Dropping it is worse than failing. The record is the product's only handle
    on the object, so a record deleted after a failed removal makes the image
    unreachable rather than gone — which is how 430 GB came to need a manual
    sweep with a snapshot taken beforehand.
    """
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node)
    client.present[node.id] = [{"reference": "local/vllm:tag", "digest": _DIGEST}]
    client.remove_error = AgentCallError(code, "refused", node_id=node.id)

    with pytest.raises(AgentCallError):
        svc.delete_image(node.id, _DIGEST)

    assert len(repo.list_images()) == 1, why
    assert client.present[node.id], "the image is still on the node"


# ---------------------------------------------------------------- reconcile


def test_reconcile_reaps_records_whose_image_is_gone(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """Records outliving their objects are the drift, and reconcile removes them."""
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node, reference="local/gone:1", digest=_DIGEST)
    _image(repo, node, reference="local/here:1", digest=_OTHER_DIGEST)
    client.present[node.id] = [{"reference": "local/here:1", "digest": _OTHER_DIGEST}]

    result = svc.reconcile_images()

    assert [r["reference"] for r in result["reaped"]] == ["local/gone:1"]
    assert [i.reference for i in repo.list_images()] == ["local/here:1"]


def test_reconcile_reports_unrecorded_images_without_inventing_records(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """An image present with no record is reported, not adopted.

    Adopting it would have to guess an origin and a ``produced_by`` the product
    does not know, and that distinction is load-bearing.
    """
    svc, repo, client = service
    node = _node(repo)
    client.present[node.id] = [{"reference": "local/surprise:1", "digest": _DIGEST}]

    result = svc.reconcile_images()

    assert [r["reference"] for r in result["unrecorded"]] == ["local/surprise:1"]
    assert repo.list_images() == [], "reporting is not adopting"


def test_reconcile_keeps_a_record_whose_digest_is_present_under_another_reference(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """A retagging is not an orphan.

    The bytes are on the node under a different name. Reaping the record would
    delete the only record of something that is really there — the exact
    inference that produced the drift, run in reverse.
    """
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node, reference="local/old-name:1", digest=_DIGEST)
    client.present[node.id] = [{"reference": "local/new-name:1", "digest": _DIGEST}]

    result = svc.reconcile_images()

    assert result["reaped"] == []
    assert len(repo.list_images()) == 1


def test_reconcile_leaves_an_unreachable_nodes_records_alone(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """Absence of evidence is not evidence of absence.

    A node that cannot be reached has told us nothing about what it holds, and
    treating silence as "gone" is the inference this whole finding is about.
    """
    svc, repo, client = service
    node = _node(repo)
    _image(repo, node)
    client.remove_error = AgentCallError("unreachable", "down", node_id=node.id)

    result = svc.reconcile_images()

    assert result["reaped"] == []
    assert [r["node_id"] for r in result["unreachable"]] == [node.id]
    assert len(repo.list_images()) == 1


def test_delete_does_not_gate_on_the_registration_snapshot(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """A node recorded at 1.13 whose agent can remove images is not refused.

    ``Node.agent_contract_version`` is captured once at registration and never
    rewritten — contracts/api.py records the defect of an operator watching both
    nodes sit at "1.0" while every live probe said 1.6. Gating this delete on it
    would refuse a capability an upgraded node has had for days, and the estate's
    own nodes are registered at 1.13, so it would have refused every delete
    until someone re-registered them. The agent's 404 is the authority instead,
    and it leaves the record standing (see the refusal cases above).
    """
    svc, repo, client = service
    node = Node(
        id=new_ulid(),
        name="spark-alpha",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.13",
        agent_cert_fingerprint="AA:BB:CC",
        platform_facts={},
        registered_at=datetime.now(),
    )
    repo.save_node(node)
    _image(repo, node)
    client.present[node.id] = [{"reference": "local/vllm:tag", "digest": _DIGEST}]

    result = svc.delete_image(node.id, _DIGEST)

    assert result["outcome"] == "removed"
    assert client.remove_calls == [(node.id, "local/vllm:tag")]
