"""Unit tests for ``ModelService.delete``.

The defect was ``model_delete`` returning ``internal_error`` on a call that had
partially succeeded: ``_model_referrers`` inspected only ``current_revision``,
so a model referenced by a *retained* (non-current) revision passed the guard;
the replica deletes committed; then ``delete_model`` hit the enforced
foreign key ``deployment_revisions.model_id -> models.id`` and raised, which
the catch-all flattened to ``internal_error`` with empty detail. ``model_list``
then showed ``replicas: []`` under a model row that was still present.

These tests pin the two halves of the fix: the referrers check inspects every
retained revision (so the guard fires *before* any delete), and the delete is
atomic (so a failure can never leave the half-state). They run against a real
migrated SQLite repository with a stub node client, so the service layer runs
its real transaction path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.errors import StillReferencedError
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import (
    Deployment,
    DeploymentRevision,
    Model,
    ModelReplica,
    ModelSource,
    Node,
    ReplicaState,
)
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


class _StubNodeClient:
    """A node client that only records ``delete_model`` calls."""

    def __init__(self) -> None:
        self.delete_calls: list[str] = []

    def delete_model(self, node: Node, model_id: str) -> dict:
        self.delete_calls.append(model_id)
        return {"status": "deleted"}


@pytest.fixture
def service() -> tuple[ModelService, SQLiteRepository, _StubNodeClient]:
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repo = SQLiteRepository(conn)
    client = _StubNodeClient()
    return ModelService(repo, client), repo, client


def _node(repo: SQLiteRepository) -> Node:
    node = Node(
        id=new_ulid(),
        name="spark-01",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.0",
        agent_cert_fingerprint="AA:BB:CC",
        platform_facts={},
        registered_at=datetime.now(),
    )
    repo.save_node(node)
    return node


def _model(repo: SQLiteRepository, source: ModelSource, name: str) -> Model:
    model = Model(
        id=new_ulid(),
        source_id=source.id,
        source_model_id=name,
        resolved_revision="rev1",
        revision_pinned=True,
    )
    repo.save_model(model)
    return model


def _replica(repo: SQLiteRepository, model: Model, node: Node) -> ModelReplica:
    replica = ModelReplica(
        model_id=model.id,
        node_id=node.id,
        local_path=f"/var/lib/tensorstead/models/{model.source_model_id}",
        state=ReplicaState.AVAILABLE,
        verified_at=datetime.now(),
    )
    repo.save_replica(replica)
    return replica


def _revision(
    repo: SQLiteRepository, deployment: Deployment, model: Model, node: Node, n: int
) -> None:
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
            runtime_version="0.6.0",
            image_reference="repo/vllm:tag",
            image_digest="",
            runtime_config={},
            participating_nodes=(node.id,),
            endpoint="10.0.0.11:8000",
            origin_platform_facts={},
        )
    )


def test_model_referrers_checks_all_retained_revisions(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """A model referenced only by a retained (non-current) revision is still referenced.

    Previously ``_model_referrers`` read only ``current_revision`` and missed
    this case, so the guard passed and the foreign key fired after the
    replicas had been deleted.
    """
    svc, repo, _ = service
    source = ModelSource("huggingface", True, True)
    repo.save_model_source(source)
    node = _node(repo)
    old_model = _model(repo, source, "org/old")
    new_model = _model(repo, source, "org/new")
    _replica(repo, old_model, node)

    deployment = Deployment(
        id=new_ulid(), name="qwen36-27b", desired_state="stopped", current_revision=2
    )
    repo.save_deployment(deployment)
    _revision(repo, deployment, old_model, node, 1)  # retained, references old_model
    _revision(repo, deployment, new_model, node, 2)  # current

    referrers = svc._model_referrers(old_model.id)
    assert referrers == ["qwen36-27b"], (
        "a retained revision's model_id is a live reference; only checking"
        "current_revision is the D6 root cause"
    )
    # The new model is referenced by the current revision too.
    assert svc._model_referrers(new_model.id) == ["qwen36-27b"]


def test_model_delete_refused_while_retained_revision_references(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """Deleting a model referenced by a retained revision is refused with a typed error.

    Refused *before* any replica is removed, so the replicas survive intact.
    """
    svc, repo, client = service
    source = ModelSource("huggingface", True, True)
    repo.save_model_source(source)
    node = _node(repo)
    old_model = _model(repo, source, "org/old")
    new_model = _model(repo, source, "org/new")
    _replica(repo, old_model, node)

    deployment = Deployment(
        id=new_ulid(), name="qwen36-27b", desired_state="stopped", current_revision=2
    )
    repo.save_deployment(deployment)
    _revision(repo, deployment, old_model, node, 1)
    _revision(repo, deployment, new_model, node, 2)

    with pytest.raises(StillReferencedError) as exc_info:
        svc.delete(old_model.id)

    # The error names the referring deployment, so an operator can act on it.
    assert "qwen36-27b" in exc_info.value.as_failure()["detail"]["referrers"]
    # The agent was never asked to delete, and the replica is still recorded.
    assert client.delete_calls == []
    assert repo.list_replicas(old_model.id), "the replica must not be half-deleted"
    assert repo.get_model(old_model.id) is not None


def test_model_delete_succeeds_when_no_revision_references_it(
    service: tuple[ModelService, SQLiteRepository, _StubNodeClient],
) -> None:
    """A model nothing references deletes cleanly: replicas and row gone together."""
    svc, repo, client = service
    source = ModelSource("huggingface", True, True)
    repo.save_model_source(source)
    node = _node(repo)
    free_model = _model(repo, source, "org/free")
    _replica(repo, free_model, node)

    svc.delete(free_model.id)

    assert client.delete_calls == [free_model.id]
    assert repo.list_replicas(free_model.id) == []
    assert repo.get_model(free_model.id) is None
