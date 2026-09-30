"""Contract test the SQLite repository.

Verifies the repository implements the persistence contract faithfully: each
entity round-trips, the schema enforces the data-model uniqueness rules, and
the retained record survives a close-and-reopen cycle (the
coordinator's record is authoritative and durable).

Observed state is never written here by construction: the schema has no table
for it and the repository has no method that persists one.
That is by construction.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import (
    Deployment,
    DeploymentRevision,
    ImageOrigin,
    ImageRecord,
    Model,
    ModelReplica,
    ModelSource,
    Node,
    Operation,
    OperationKind,
    OperationState,
    ReplicaState,
)

pytestmark = pytest.mark.contract

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[SQLiteRepository, Path]:
    """Build a migrated in-memory repository against a temp DB file."""
    db_path = tmp_path / "coordinator.db"
    conn = connect(db_path)
    migrate(conn, MIGRATIONS_DIR)
    return SQLiteRepository(conn), db_path


def make_node() -> Node:
    return Node(
        id=new_ulid(),
        name="spark-01",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.0",
        agent_cert_fingerprint="AA:BB:CC",
        platform_facts={"cpu_arch": "aarch64", "os_family": "linux", "memory_is_unified": True},
        registered_at=datetime.now(),
    )


def make_model(source: ModelSource) -> Model:
    return Model(
        id=new_ulid(),
        source_id=source.id,
        source_model_id="org/model",
        resolved_revision="e1f2a3b",
        revision_pinned=True,
        size_bytes=1234,
        content_digest="sha256:abc",
    )


def test_node_round_trip(repo: tuple[SQLiteRepository, Path]) -> None:
    repository, _ = repo
    node = make_node()
    repository.save_node(node)
    got = repository.get_node(node.id)
    assert got is not None
    assert got.name == "spark-01"
    assert got.platform_facts == node.platform_facts
    assert got.registered_at == node.registered_at


def test_model_and_replica_round_trip(repo: tuple[SQLiteRepository, Path]) -> None:
    repository, _ = repo
    source = ModelSource("huggingface", supports_revision_pinning=True, requires_credential=True)
    repository.save_model_source(source)
    model = make_model(source)
    repository.save_model(model)
    node = make_node()
    repository.save_node(node)
    replica = ModelReplica(
        model_id=model.id,
        node_id=node.id,
        local_path="/var/lib/tensorstead/models/org/model",
        state=ReplicaState.AVAILABLE,
        verified_at=datetime.now(),
    )
    repository.save_replica(replica)

    assert repository.get_model(model.id) is not None
    assert repository.get_replica(model.id, node.id) is not None
    replicas = repository.list_replicas(model.id)
    assert len(replicas) == 1
    assert replicas[0].state == ReplicaState.AVAILABLE


def test_deployment_and_revisions_round_trip(repo: tuple[SQLiteRepository, Path]) -> None:
    repository, _ = repo
    node = make_node()
    repository.save_node(node)
    source = ModelSource("huggingface", True, True)
    repository.save_model_source(source)
    model = make_model(source)
    repository.save_model(model)

    deployment = Deployment(
        id=new_ulid(),
        name="llama-70b",
        desired_state="stopped",
        current_revision=1,
        running_revision=None,
    )
    repository.save_deployment(deployment)
    revision = DeploymentRevision(
        deployment_id=deployment.id,
        revision=1,
        model_id=model.id,
        model_source_id="huggingface",
        source_model_id="org/model",
        resolved_revision="e1f2a3b",
        revision_pinned=True,
        runtime_type="vllm",
        runtime_version="0.6.0",
        image_reference="repo/vllm:tag",
        image_digest="sha256:abc",
        runtime_config={"tensor_parallel_size": 2},
        participating_nodes=(node.id,),
        endpoint="10.0.0.11:8000",
        origin_platform_facts={"cpu_arch": "aarch64"},
    )
    repository.insert_revision(revision)

    assert repository.get_deployment(deployment.id) is not None
    got_rev = repository.get_revision(deployment.id, 1)
    assert got_rev is not None
    assert got_rev.participating_nodes == (node.id,)
    assert got_rev.runtime_config == {"tensor_parallel_size": 2}


def test_operation_round_trip(repo: tuple[SQLiteRepository, Path]) -> None:
    repository, _ = repo
    op = Operation(
        id=new_ulid(),
        kind=OperationKind.START,
        target_type="deployment",
        target_id=new_ulid(),
        deployment_revision=1,
        state=OperationState.SUCCEEDED,
        per_node_outcomes={"01J00000000000000000000000": {"state": "succeeded"}},
    )
    repository.save_operation(op)
    got = repository.get_operation(op.id)
    assert got is not None
    assert got.kind == OperationKind.START
    assert got.state == OperationState.SUCCEEDED
    assert got.per_node_outcomes == op.per_node_outcomes


def test_retained_record_survives_close_and_reopen(
    repo: tuple[SQLiteRepository, Path], tmp_path: Path
) -> None:
    """The retained record survives a close-and-reopen cycle."""
    repository, db_path = repo
    node = make_node()
    repository.save_node(node)

    # Reopen a fresh connection against the same file (simulates a restart).
    conn2 = connect(db_path)
    migrate(conn2, MIGRATIONS_DIR)
    repository2 = SQLiteRepository(conn2)
    got = repository2.get_node(node.id)
    assert got is not None
    assert got.name == node.name
    assert got.agent_cert_fingerprint == node.agent_cert_fingerprint


def test_model_uniqueness_constraint(repo: tuple[SQLiteRepository, Path]) -> None:
    """(source_id, source_model_id, resolved_revision) is unique."""
    repository, _ = repo
    source = ModelSource("huggingface", True, True)
    repository.save_model_source(source)
    model = make_model(source)
    repository.save_model(model)
    # A duplicate (same three key columns) must be rejected by the schema.
    with pytest.raises(sqlite3.IntegrityError):
        repository.save_model(
            Model(
                id=new_ulid(),
                source_id="huggingface",
                source_model_id="org/model",
                resolved_revision="e1f2a3b",
                revision_pinned=True,
            )
        )


def test_resolve_non_terminal_operations(repo: tuple[SQLiteRepository, Path]) -> None:
    """Pending/running operations are marked failed with outcome_unknown."""
    repository, _ = repo
    pending = Operation(
        id=new_ulid(), kind=OperationKind.START, target_type="deployment", target_id=new_ulid()
    )
    done = Operation(
        id=new_ulid(),
        kind=OperationKind.STOP,
        target_type="deployment",
        target_id=new_ulid(),
        state=OperationState.SUCCEEDED,
        finished_at=datetime.now(),
    )
    repository.save_operation(pending)
    repository.save_operation(done)

    repository.resolve_non_terminal_operations(finished_at=datetime.now())

    got_pending = repository.get_operation(pending.id)
    got_done = repository.get_operation(done.id)
    assert got_pending is not None
    assert got_pending.state == OperationState.FAILED
    assert got_pending.failure_reason is not None
    assert got_pending.failure_reason["code"] == "outcome_unknown"
    # A terminal operation is left untouched.
    assert got_done is not None
    assert got_done.state == OperationState.SUCCEEDED


def test_image_provenance_round_trips_through_the_store(
    repo: tuple[SQLiteRepository, Path],
) -> None:
    """A built image's origin and producer survive save -> list.

    The migration added ``origin`` and ``produced_by`` columns, but the write
    path dropped them and the read path did not fetch them, so every listed
    image read as ``origin='pulled'`` regardless of how it arrived. That is the
    recurring defect in this codebase -- a record the surface never shows, with
    nothing comparing them -- and the success criterion was false at the persistence layer
    despite the schema claiming otherwise. This pins the round-trip so the
    write/read path cannot silently drop provenance again.
    """
    repository, _ = repo
    node = make_node()
    repository.save_node(node)

    built = ImageRecord(
        node_id=node.id,
        reference="local/vllm:patched",
        digest="sha256:built",
        pulled_at=datetime.now(),
        origin=ImageOrigin.BUILT,
        produced_by="vllm-xgrammar",
    )
    repository.save_image(built)

    listed = repository.list_images()
    assert len(listed) == 1
    got = listed[0]
    assert got.origin is ImageOrigin.BUILT, f"origin round-tripped as {got.origin!r}, not BUILT"
    assert got.produced_by == "vllm-xgrammar"
    assert got.is_registry_digest is False

    # A pulled image records no producer and reads back as a registry digest.
    pulled = ImageRecord(
        node_id=node.id,
        reference="nvcr.io/nvidia/vllm:26.07-py3",
        digest="sha256:pulled",
        pulled_at=datetime.now(),
        origin=ImageOrigin.PULLED,
        produced_by=None,
    )
    repository.save_image(pulled)
    by_ref = {img.reference: img for img in repository.list_images()}
    assert by_ref["nvcr.io/nvidia/vllm:26.07-py3"].origin is ImageOrigin.PULLED
    assert by_ref["nvcr.io/nvidia/vllm:26.07-py3"].is_registry_digest is True


def test_delete_model_and_replicas_is_atomic_under_fk_violation(
    repo: tuple[SQLiteRepository, Path],
) -> None:
    """A model referenced by a retained revision deletes atomically.

    The half-state was ``model_list`` reporting ``replicas: []`` under a model
    row that was still present: ``delete_replica`` committed per replica, then
    ``delete_model`` hit the enforced foreign key
    ``deployment_revisions.model_id -> models.id`` and the already-deleted
    replicas stayed gone. ``delete_model_and_replicas`` wraps both deletes in
    one transaction, so a failure on the model delete rolls the replica deletes
    back -- the store can never be observed half-deleted.
    """
    repository, _ = repo
    node = make_node()
    repository.save_node(node)
    source = ModelSource("huggingface", True, True)
    repository.save_model_source(source)
    model = make_model(source)
    repository.save_model(model)
    repository.save_replica(
        ModelReplica(
            model_id=model.id,
            node_id=node.id,
            local_path="/var/lib/tensorstead/models/org/model",
            state=ReplicaState.AVAILABLE,
            verified_at=datetime.now(),
        )
    )
    # A retained deployment revision references the model, so the model row
    # delete will violate the enforced foreign key.
    deployment = Deployment(
        id=new_ulid(),
        name="llama-70b",
        desired_state="stopped",
        current_revision=1,
    )
    repository.save_deployment(deployment)
    repository.insert_revision(
        DeploymentRevision(
            deployment_id=deployment.id,
            revision=1,
            model_id=model.id,
            model_source_id="huggingface",
            source_model_id="org/model",
            resolved_revision="e1f2a3b",
            revision_pinned=True,
            runtime_type="vllm",
            runtime_version="0.6.0",
            image_reference="repo/vllm:tag",
            image_digest="sha256:abc",
            runtime_config={"tensor_parallel_size": 2},
            participating_nodes=(node.id,),
            endpoint="10.0.0.11:8000",
            origin_platform_facts={"cpu_arch": "aarch64"},
        )
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.delete_model_and_replicas(model.id)

    # Both rows survived the rollback -- no half-deleted state.
    assert repository.get_model(model.id) is not None, (
        "the model row must survive a rolled-back delete"
    )
    assert len(repository.list_replicas(model.id)) == 1, (
        "the replica rows must survive a rolled-back delete, not be left gone"
    )
