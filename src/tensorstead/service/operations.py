"""Operation tracking service.

Every state-changing management action is a long-running operation recorded in
the store, with progress readable while it runs and a guaranteed terminal
outcome. This module owns creating, updating, and retrieving
those records.

The store is **not a log store**: it retains structured operation
records with concise failure info and bounded excerpts, never log streams.
Observed/read operations are never recorded — only state-changing
attempts.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from tensorstead.domain.errors import NotFoundError
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Operation, OperationKind, OperationState
from tensorstead.ports.repository import Repository


def _now() -> datetime:
    return datetime.now().astimezone()


class OperationService:
    """Create and advance long-running operation records."""

    def __init__(self, repository: Repository) -> None:
        self._repo = repository

    def retarget(self, operation_id: str, target_id: str) -> Operation:
        """Bind an operation to the entity it turned out to create.

        A creation cannot name its target up front — the id does not exist
        until the entity does — so the operation begins against a placeholder
        and is bound here once the row is written. Without this, a deployment's
        own creation is missing from its history, and the export's operation
        block starts at the first ``start``.
        """
        operation = replace(self.get(operation_id), target_id=target_id)
        self._repo.save_operation(operation)
        return operation

    def begin(
        self,
        *,
        kind: OperationKind,
        target_type: str,
        target_id: str,
        deployment_revision: int | None = None,
    ) -> Operation:
        """Create a new pending operation and return it."""
        operation = Operation(
            id=new_ulid(),
            kind=kind,
            target_type=target_type,
            target_id=target_id,
            deployment_revision=deployment_revision,
            state=OperationState.PENDING,
            started_at=_now(),
        )
        self._repo.save_operation(operation)
        return operation

    def mark_running(
        self, operation_id: str, *, progress: dict[str, Any] | None = None
    ) -> Operation:
        """Record that the work has started.

        Called from every route immediately after ``begin``, because that is
        the moment the work begins. Without it an operation went straight from
        ``pending`` to terminal and never reported itself as running, so the
        requirement that progress be readable *while* an operation
        runs could not be met, and startup resolution of non-terminal
        rows could only ever find ``pending`` ones.
        """
        return self.set_progress(operation_id, progress=progress or {})

    def set_progress(self, operation_id: str, *, progress: dict[str, Any]) -> Operation:
        operation = self.get(operation_id)
        operation = replace(operation, state=OperationState.RUNNING, progress=progress)
        self._repo.save_operation(operation)
        return operation

    def succeed(
        self,
        operation_id: str,
        *,
        per_node_outcomes: dict[str, Any] | None = None,
    ) -> Operation:
        operation = self.get(operation_id)
        operation = replace(
            operation,
            state=OperationState.SUCCEEDED,
            per_node_outcomes=per_node_outcomes,
            finished_at=_now(),
        )
        self._repo.save_operation(operation)
        return operation

    def fail(
        self,
        operation_id: str,
        *,
        code: str,
        message: str,
        node_id: str | None = None,
        detail: dict[str, Any] | None = None,
        per_node_outcomes: dict[str, Any] | None = None,
    ) -> Operation:
        operation = self.get(operation_id)
        reason: dict[str, Any] = {"code": code, "message": message}
        if node_id is not None:
            reason["node_id"] = node_id
        if detail:
            reason["detail"] = detail
        operation = replace(
            operation,
            state=OperationState.FAILED,
            failure_reason=reason,
            per_node_outcomes=per_node_outcomes,
            finished_at=_now(),
        )
        self._repo.save_operation(operation)
        return operation

    def get(self, operation_id: str) -> Operation:
        operation = self._repo.get_operation(operation_id)
        if operation is None:
            raise NotFoundError(f"no operation with id {operation_id!r}")
        return operation

    def list(self, deployment_id: str | None = None) -> list[Operation]:
        return self._repo.list_operations(deployment_id)
