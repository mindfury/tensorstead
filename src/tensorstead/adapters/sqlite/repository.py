"""SQLite repository implementing the repository port.

Hand-written SQL is confined to this module. The repository maps between the
domain entities (``domain/models.py``) and rows in the schema from
``0001_initial.sql``. Multi-write operations use the explicit ``transaction``
context manager from ``connection.py``; single-row writes are atomic per row.

Observed state is never written here — the schema has no table for it, and
there is intentionally no method that writes one. This
is enforced structurally by the schema and by the guardrail test.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from threading import RLock
from typing import Any

from tensorstead.domain.approvals import CodeExecutionApproval
from tensorstead.domain.identity import assert_valid_ulid
from tensorstead.domain.models import (
    Credential,
    Deployment,
    DeploymentRevision,
    ImageBuildSpec,
    ImageOrigin,
    ImageRecord,
    InferenceCredential,
    Model,
    ModelReplica,
    ModelSource,
    Node,
    Operation,
    OperationKind,
    OperationState,
    ReplicaState,
    canonical_file_selector,
)


def _json(value: object) -> str:
    return json.dumps(value)


def _loads(value: str | None, default: Any = None) -> Any:
    """Decode a stored JSON column, returning ``default`` when null."""
    if value is None:
        return default
    return json.loads(value)


# A file selection is stored as its canonical patterns joined by newlines, so
# the UNIQUE constraint in migration 0006 compares selections rather than
# spellings. '' is the whole repository -- not NULL, because SQLite treats
# NULLs as distinct in a UNIQUE index and whole-repo models must still
# deduplicate.
def _encode_file_selector(patterns: tuple[str, ...]) -> str:
    return "\n".join(canonical_file_selector(patterns))


def _decode_file_selector(value: str | None) -> tuple[str, ...]:
    return canonical_file_selector(value.split("\n") if value else ())


class SQLiteRepository:
    """A ``Repository`` backed by SQLite.

    ``conn`` is owned by the caller (typically a single shared connection from
    ``connection.connect``). A lock serializes writes so the asyncio workers and
    the FastAPI app share the connection safely.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = RLock()

    # ------------------------------------------------------------------ nodes
    def save_node(self, node: Node) -> None:
        assert_valid_ulid(node.id, what="node id")
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO nodes (
                    id, name, agent_endpoint, agent_contract_version,
                    agent_cert_fingerprint, platform_facts, registered_at,
                    reserved, reserved_reason, agent_management_token_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.name,
                    node.agent_endpoint,
                    node.agent_contract_version,
                    node.agent_cert_fingerprint,
                    _json(node.platform_facts),
                    node.registered_at.isoformat(),
                    int(node.reserved),
                    node.reserved_reason,
                    node.agent_management_token_ref,
                ),
            )
            self._conn.commit()

    def get_node(self, node_id: str) -> Node | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._node_from_row(row) if row else None

    def get_node_by_name(self, name: str) -> Node | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM nodes WHERE name = ?", (name,)).fetchone()
        return self._node_from_row(row) if row else None

    def list_nodes(self) -> list[Node]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM nodes ORDER BY name").fetchall()
        return [self._node_from_row(r) for r in rows]

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            self._conn.commit()

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            name=row["name"],
            agent_endpoint=row["agent_endpoint"],
            agent_contract_version=row["agent_contract_version"],
            agent_cert_fingerprint=row["agent_cert_fingerprint"],
            platform_facts=_loads(row["platform_facts"], {}),
            registered_at=datetime.fromisoformat(row["registered_at"]),
            reserved=bool(row["reserved"]),
            reserved_reason=row["reserved_reason"],
            agent_management_token_ref=row["agent_management_token_ref"],
        )

    # ----------------------------------------------------------- model sources
    def save_model_source(self, source: ModelSource) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO model_sources
                    (id, supports_revision_pinning, requires_credential)
                VALUES (?, ?, ?)
                """,
                (source.id, int(source.supports_revision_pinning), int(source.requires_credential)),
            )
            self._conn.commit()

    def get_model_source(self, source_id: str) -> ModelSource | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM model_sources WHERE id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        return ModelSource(
            id=row["id"],
            supports_revision_pinning=bool(row["supports_revision_pinning"]),
            requires_credential=bool(row["requires_credential"]),
        )

    # -------------------------------------------------------------- models
    def save_model(self, model: Model) -> None:
        assert_valid_ulid(model.id, what="model id")
        with self._lock:
            # Plain INSERT (not OR REPLACE) so the data-model uniqueness rule
            # (source_id, source_model_id, resolved_revision) is enforced by the
            # schema rather than swallowed by an upsert.
            # Callers find an existing logical model first and reuse it.
            self._conn.execute(
                """
                INSERT INTO models (
                    id, source_id, source_model_id, resolved_revision,
                    revision_pinned, size_bytes, content_digest, file_selector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model.id,
                    model.source_id,
                    model.source_model_id,
                    model.resolved_revision,
                    int(model.revision_pinned),
                    model.size_bytes,
                    model.content_digest,
                    _encode_file_selector(model.file_selector),
                ),
            )
            self._conn.commit()

    def update_model_content(
        self,
        model_id: str,
        *,
        content_digest: str | None = None,
        size_bytes: int | None = None,
    ) -> None:
        """Record what the agent measured about an already-identified model.

        A narrow UPDATE rather than an upsert through ``save_model``: the
        identity columns (source, model, revision) stay immutable and their
        uniqueness rule stays enforced by the schema. Only the two descriptive
        columns move, and only from ``NULL`` to a measured value — the digest
        later nodes verify a peer transfer against.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE models
                   SET content_digest = COALESCE(?, content_digest),
                       size_bytes     = COALESCE(?, size_bytes)
                 WHERE id = ?
                """,
                (content_digest, size_bytes, model_id),
            )
            self._conn.commit()

    def get_model(self, model_id: str) -> Model | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._model_from_row(row) if row else None

    def list_models(self) -> list[Model]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM models ORDER BY source_id, source_model_id"
            ).fetchall()
        return [self._model_from_row(r) for r in rows]

    def find_model(
        self,
        source_id: str,
        source_model_id: str,
        resolved_revision: str | None,
        file_selector: tuple[str, ...] = (),
    ) -> Model | None:
        """Find the logical model for one source/repo/revision/selection.

        ``file_selector`` is part of the key because it is part of identity
        (migration 0006). Omitting it means the whole repository, which keeps
        every existing caller finding exactly what it found before.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM models
                WHERE source_id = ? AND source_model_id = ? AND resolved_revision IS ?
                  AND file_selector = ?
                """,
                (
                    source_id,
                    source_model_id,
                    resolved_revision,
                    _encode_file_selector(file_selector),
                ),
            ).fetchone()
        return self._model_from_row(row) if row else None

    @staticmethod
    def _model_from_row(row: sqlite3.Row) -> Model:
        return Model(
            id=row["id"],
            source_id=row["source_id"],
            source_model_id=row["source_model_id"],
            resolved_revision=row["resolved_revision"],
            revision_pinned=bool(row["revision_pinned"]),
            size_bytes=row["size_bytes"],
            content_digest=row["content_digest"],
            file_selector=_decode_file_selector(row["file_selector"]),
        )

    # -------------------------------------------------------------- replicas
    def save_replica(self, replica: ModelReplica) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO model_replicas
                    (model_id, node_id, local_path, state, verified_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    replica.model_id,
                    replica.node_id,
                    replica.local_path,
                    replica.state.value,
                    replica.verified_at.isoformat() if replica.verified_at else None,
                ),
            )
            self._conn.commit()

    def list_replicas(self, model_id: str) -> list[ModelReplica]:
        # Lifecycle operations dispatch one agent call per node concurrently.
        # sqlite3 connections are not a safe shared cursor boundary even when
        # ``check_same_thread`` is disabled, so reads share the repository lock
        # with writes.  Without this, concurrent replica reads can observe an
        # unrelated row shape and turn a healthy two-node start into a flaky
        # ``ReplicaState(None)`` failure.
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM model_replicas WHERE model_id = ?", (model_id,)
            ).fetchall()
        return [self._replica_from_row(r) for r in rows]

    def get_replica(self, model_id: str, node_id: str) -> ModelReplica | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM model_replicas WHERE model_id = ? AND node_id = ?",
                (model_id, node_id),
            ).fetchone()
        return self._replica_from_row(row) if row else None

    def delete_model(self, model_id: str) -> None:
        """Delete a logical model row. Replicas are deleted separately."""
        with self._lock:
            self._conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            self._conn.commit()

    def delete_replica(self, model_id: str, node_id: str) -> None:
        """Delete one replica row."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM model_replicas WHERE model_id = ? AND node_id = ?",
                (model_id, node_id),
            )
            self._conn.commit()

    def delete_model_and_replicas(self, model_id: str) -> None:
        """Delete a model row and all its replicas atomically.

        A single transaction: the replicas and the model row commit together
        or roll back together, so a failure on the model delete (for example a
        foreign-key violation from a retained deployment revision) cannot leave
        the replicas already gone while the model row remains -- the half-state
        that previously surfaced as ``internal_error`` on a call that had
        partially succeeded.
        """
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("DELETE FROM model_replicas WHERE model_id = ?", (model_id,))
                self._conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    @staticmethod
    def _replica_from_row(row: sqlite3.Row) -> ModelReplica:
        return ModelReplica(
            model_id=row["model_id"],
            node_id=row["node_id"],
            local_path=row["local_path"],
            state=ReplicaState(row["state"]),
            verified_at=(
                datetime.fromisoformat(row["verified_at"]) if row["verified_at"] else None
            ),
        )

    # ------------------------------------------------------------ credentials
    def save_credential(self, credential: Credential) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO credentials (source_id, name, secret_ref, is_default, set_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    credential.source_id,
                    credential.name,
                    credential.secret_ref,
                    int(credential.is_default),
                    credential.set_at.isoformat(),
                ),
            )
            self._conn.commit()

    def list_credentials(self) -> list[Credential]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM credentials ORDER BY source_id, name"
            ).fetchall()
        return [
            Credential(
                source_id=r["source_id"],
                name=r["name"],
                secret_ref=r["secret_ref"],
                is_default=bool(r["is_default"]),
                set_at=datetime.fromisoformat(r["set_at"]),
            )
            for r in rows
        ]

    def get_credential(self, source_id: str, name: str) -> Credential | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM credentials WHERE source_id = ? AND name = ?",
                (source_id, name),
            ).fetchone()
        if not row:
            return None
        return Credential(
            source_id=row["source_id"],
            name=row["name"],
            secret_ref=row["secret_ref"],
            is_default=bool(row["is_default"]),
            set_at=datetime.fromisoformat(row["set_at"]),
        )

    def delete_credential(self, source_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM credentials WHERE source_id = ? AND name = ?",
                (source_id, name),
            )
            self._conn.commit()

    # ----------------------------------------------------------- deployments
    def save_deployment(self, deployment: Deployment) -> None:
        assert_valid_ulid(deployment.id, what="deployment id")
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO deployments (
                    id, name, desired_state, current_revision, running_revision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    deployment.id,
                    deployment.name,
                    deployment.desired_state,
                    deployment.current_revision,
                    deployment.running_revision,
                    deployment.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
            ).fetchone()
        return self._deployment_from_row(row) if row else None

    def list_deployments(self) -> list[Deployment]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM deployments ORDER BY name").fetchall()
        return [self._deployment_from_row(r) for r in rows]

    def delete_deployment(self, deployment_id: str) -> None:
        """Delete a deployment and its revisions.

        Called only after the agent confirms the runtime instance and unit
        were removed. Model artifacts and images are retained —
        they are in separate tables and are not touched here.
        """
        with self._lock:
            self._conn.execute(
                "DELETE FROM deployment_revisions WHERE deployment_id = ?",
                (deployment_id,),
            )
            self._conn.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
            self._conn.commit()

    @staticmethod
    def _deployment_from_row(row: sqlite3.Row) -> Deployment:
        return Deployment(
            id=row["id"],
            name=row["name"],
            desired_state=row["desired_state"],
            current_revision=row["current_revision"],
            running_revision=row["running_revision"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------ deployment revisions
    def insert_revision(self, revision: DeploymentRevision) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO deployment_revisions (
                    deployment_id, revision, model_id, model_source_id,
                    source_model_id, resolved_revision, revision_pinned,
                    runtime_type, runtime_version, image_reference, image_digest,
                    runtime_config, participating_nodes, endpoint,
                    origin_platform_facts, restore_on_boot, created_at,
                    code_approval_fingerprint, code_approval_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.deployment_id,
                    revision.revision,
                    revision.model_id,
                    revision.model_source_id,
                    revision.source_model_id,
                    revision.resolved_revision,
                    int(revision.revision_pinned),
                    revision.runtime_type,
                    revision.runtime_version,
                    revision.image_reference,
                    revision.image_digest,
                    _json(revision.runtime_config),
                    _json(list(revision.participating_nodes)),
                    revision.endpoint,
                    _json(revision.origin_platform_facts),
                    int(revision.restore_on_boot),
                    revision.created_at.isoformat(),
                    revision.code_approval_fingerprint,
                    revision.code_approval_id,
                ),
            )
            self._conn.commit()

    def get_revision(self, deployment_id: str, revision: int) -> DeploymentRevision | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM deployment_revisions
                WHERE deployment_id = ? AND revision = ?
                """,
                (deployment_id, revision),
            ).fetchone()
        return self._revision_from_row(row) if row else None

    def list_revisions(self, deployment_id: str) -> list[DeploymentRevision]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM deployment_revisions
                WHERE deployment_id = ? ORDER BY revision
                """,
                (deployment_id,),
            ).fetchall()
        return [self._revision_from_row(r) for r in rows]

    @staticmethod
    def _revision_from_row(row: sqlite3.Row) -> DeploymentRevision:
        return DeploymentRevision(
            deployment_id=row["deployment_id"],
            revision=row["revision"],
            model_id=row["model_id"],
            model_source_id=row["model_source_id"],
            source_model_id=row["source_model_id"],
            resolved_revision=row["resolved_revision"],
            revision_pinned=bool(row["revision_pinned"]),
            runtime_type=row["runtime_type"],
            runtime_version=row["runtime_version"],
            image_reference=row["image_reference"],
            image_digest=row["image_digest"],
            runtime_config=_loads(row["runtime_config"], {}),
            participating_nodes=tuple(_loads(row["participating_nodes"], [])),
            endpoint=row["endpoint"],
            origin_platform_facts=_loads(row["origin_platform_facts"], {}),
            # Rows written before migration 0003 carry the column's DEFAULT 0,
            # so a deployment that predates this loses boot persistence rather
            # than keeping it silently.
            restore_on_boot=bool(row["restore_on_boot"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            # Rows written before migration 0009 carry the column's DEFAULT '',
            # which reads as "authorized by nothing" -- which is exactly what
            # they were, since no approval could exist before that migration.
            code_approval_fingerprint=row["code_approval_fingerprint"],
            code_approval_id=row["code_approval_id"],
        )

    # ------------------------------------------------------------- operations
    def save_operation(self, operation: Operation) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO operations (
                    id, kind, target_type, target_id, deployment_revision,
                    state, failure_reason, per_node_outcomes, progress,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation.id,
                    operation.kind.value,
                    operation.target_type,
                    operation.target_id,
                    operation.deployment_revision,
                    operation.state.value,
                    _json(operation.failure_reason) if operation.failure_reason else None,
                    _json(operation.per_node_outcomes) if operation.per_node_outcomes else None,
                    _json(operation.progress) if operation.progress else None,
                    operation.started_at.isoformat() if operation.started_at else None,
                    operation.finished_at.isoformat() if operation.finished_at else None,
                ),
            )
            self._conn.commit()

    def get_operation(self, operation_id: str) -> Operation | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM operations WHERE id = ?", (operation_id,)
            ).fetchone()
        return self._operation_from_row(row) if row else None

    def list_operations(self, deployment_id: str | None = None) -> list[Operation]:
        if deployment_id is None:
            with self._lock:
                rows = self._conn.execute("SELECT * FROM operations ORDER BY started_at").fetchall()
        else:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM operations WHERE target_id = ? ORDER BY started_at",
                    (deployment_id,),
                ).fetchall()
        return [self._operation_from_row(r) for r in rows]

    def resolve_non_terminal_operations(self, *, finished_at: datetime) -> None:
        """Mark pending/running operations failed with ``outcome_unknown``.

        Called on coordinator startup. The coordinator does not probe the node to
        decide what really happened and does not replay, retry, or roll back —
        it lost the evidence, so it records that rather than a guess.
        """
        reason = {
            "code": "outcome_unknown",
            "message": "operation interrupted by coordinator termination",
        }
        with self._lock:
            self._conn.execute(
                """
                UPDATE operations
                SET state = 'failed', failure_reason = ?, finished_at = ?
                WHERE state IN ('pending', 'running')
                """,
                (_json(reason), finished_at.isoformat()),
            )
            self._conn.commit()

    @staticmethod
    def _operation_from_row(row: sqlite3.Row) -> Operation:
        return Operation(
            id=row["id"],
            kind=OperationKind(row["kind"]),
            target_type=row["target_type"],
            target_id=row["target_id"],
            deployment_revision=row["deployment_revision"],
            state=OperationState(row["state"]),
            failure_reason=_loads(row["failure_reason"], None),
            per_node_outcomes=_loads(row["per_node_outcomes"], None),
            progress=_loads(row["progress"], None),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        )

    # ---------------------------------------------------------------- images
    def save_image(self, image: ImageRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO image_records
                    (node_id, reference, digest, pulled_at, origin, produced_by)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    image.node_id,
                    image.reference,
                    image.digest,
                    image.pulled_at.isoformat(),
                    image.origin.value,
                    image.produced_by,
                ),
            )
            self._conn.commit()

    # ------------------------------------------------ image build specs
    def save_build_spec(self, spec: ImageBuildSpec) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO image_build_specs
                    (name, base_image, steps, entrypoint, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    spec.name,
                    spec.base_image,
                    json.dumps(list(spec.steps)),
                    json.dumps(list(spec.entrypoint)),
                    spec.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def list_build_specs(self) -> list[ImageBuildSpec]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM image_build_specs ORDER BY name").fetchall()
        return [self._build_spec(row) for row in rows]

    def get_build_spec(self, name: str) -> ImageBuildSpec | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM image_build_specs WHERE name = ?", (name,)
            ).fetchone()
        return self._build_spec(row) if row is not None else None

    def delete_build_spec(self, name: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM image_build_specs WHERE name = ?", (name,))
            self._conn.commit()

    # -------------------------------------------- code-execution approvals
    def save_code_approval(self, approval: CodeExecutionApproval) -> None:
        """Insert one approval; a duplicate tuple is a conflict, never a replace.

        ``INSERT`` rather than ``INSERT OR REPLACE`` on purpose. The tuple is
        unique, so re-approving an existing one would otherwise silently
        overwrite who approved it and why -- and would leave a delete removing
        only one of two rows that authorize the same thing, which is a
        revocation that does not revoke.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO code_execution_approvals
                    (id, option, runtime_type, model_source_id, source_model_id,
                     model_revision, image_digest, fingerprint, reason, approved_by,
                     policy_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.id,
                    approval.option,
                    approval.runtime_type,
                    approval.model_source_id,
                    approval.source_model_id,
                    approval.model_revision,
                    approval.image_digest,
                    approval.fingerprint,
                    approval.reason,
                    approval.approved_by,
                    approval.policy_version,
                    approval.created_at.isoformat(),
                ),
            )

    def list_code_approvals(self) -> list[CodeExecutionApproval]:
        with self._conn:
            rows = self._conn.execute(
                "SELECT * FROM code_execution_approvals ORDER BY created_at DESC"
            ).fetchall()
        return [self._code_approval(row) for row in rows]

    def get_code_approval(self, approval_id: str) -> CodeExecutionApproval | None:
        with self._conn:
            row = self._conn.execute(
                "SELECT * FROM code_execution_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        return self._code_approval(row) if row is not None else None

    def find_code_approval(self, fingerprint: str) -> CodeExecutionApproval | None:
        """The approval matching an exact tuple, by its content address."""
        with self._conn:
            row = self._conn.execute(
                "SELECT * FROM code_execution_approvals WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
        if row is None:
            return None
        approval = self._code_approval(row)
        # The stored fingerprint is what the index found; the recomputed one is
        # what the tuple actually means. They can only differ if a row was
        # written by something other than this class -- and an approval whose
        # key does not match its own contents must not authorize anything.
        if approval.fingerprint != fingerprint:
            return None
        return approval

    def delete_code_approval(self, approval_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM code_execution_approvals WHERE id = ?", (approval_id,))

    @staticmethod
    def _code_approval(row: Any) -> CodeExecutionApproval:
        return CodeExecutionApproval(
            id=row["id"],
            option=row["option"],
            runtime_type=row["runtime_type"],
            model_source_id=row["model_source_id"],
            source_model_id=row["source_model_id"],
            model_revision=row["model_revision"],
            image_digest=row["image_digest"],
            reason=row["reason"],
            approved_by=row["approved_by"],
            policy_version=row["policy_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _build_spec(row: Any) -> ImageBuildSpec:
        # A spec recorded before the column existed reads as the image's own
        # entrypoint, which is what it always had.
        keys = row.keys() if hasattr(row, "keys") else []
        raw_entrypoint = row["entrypoint"] if "entrypoint" in keys else "[]"
        return ImageBuildSpec(
            entrypoint=tuple(json.loads(raw_entrypoint or "[]")),
            name=row["name"],
            base_image=row["base_image"],
            steps=tuple(json.loads(row["steps"])),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ---------------------------------------------- inference credentials
    def save_inference_credential(self, credential: InferenceCredential) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO inference_credentials (name, secret_ref, set_at)
                VALUES (?, ?, ?)
                """,
                (credential.name, credential.secret_ref, credential.set_at.isoformat()),
            )
            self._conn.commit()

    def list_inference_credentials(self) -> list[InferenceCredential]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM inference_credentials ORDER BY name"
            ).fetchall()
        return [
            InferenceCredential(
                name=row["name"],
                secret_ref=row["secret_ref"],
                set_at=datetime.fromisoformat(row["set_at"]),
            )
            for row in rows
        ]

    def get_inference_credential(self, name: str) -> InferenceCredential | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM inference_credentials WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return InferenceCredential(
            name=row["name"],
            secret_ref=row["secret_ref"],
            set_at=datetime.fromisoformat(row["set_at"]),
        )

    def delete_inference_credential(self, name: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM inference_credentials WHERE name = ?", (name,))
            self._conn.commit()

    def bind_inference_credential(self, deployment_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO deployment_inference_credentials
                    (deployment_id, name, bound_at)
                VALUES (?, ?, ?)
                """,
                (deployment_id, name, datetime.now().astimezone().isoformat()),
            )
            self._conn.commit()

    def unbind_inference_credential(self, deployment_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM deployment_inference_credentials WHERE deployment_id = ?",
                (deployment_id,),
            )
            self._conn.commit()

    def get_bound_inference_credential(self, deployment_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM deployment_inference_credentials WHERE deployment_id = ?",
                (deployment_id,),
            ).fetchone()
        return str(row["name"]) if row is not None else None

    def list_images(self) -> list[ImageRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM image_records ORDER BY node_id").fetchall()
        return [
            ImageRecord(
                node_id=r["node_id"],
                reference=r["reference"],
                digest=r["digest"],
                pulled_at=datetime.fromisoformat(r["pulled_at"]),
                origin=ImageOrigin(r["origin"]),
                produced_by=r["produced_by"],
            )
            for r in rows
        ]

    def delete_image(self, node_id: str, digest: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM image_records WHERE node_id = ? AND digest = ?",
                (node_id, digest),
            )
            self._conn.commit()
