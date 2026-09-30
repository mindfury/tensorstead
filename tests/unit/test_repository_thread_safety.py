"""One connection, many threads: reads must take the lock too.

The coordinator shares a single ``sqlite3`` connection across its asyncio
workers and the FastAPI app (``check_same_thread=False``), and the repository
serialises access with a lock. It serialised **writes**. Twenty read paths went
straight to ``self._conn.execute`` with no lock at all.

That is fine until two long operations run at once. Acquiring a model and
building an image simultaneously — both on background threads, both writing —
while a client polls their operations produced:

    sqlite3.InterfaceError: bad parameter or other API misuse

from ``get_operation``, and both operations failed. Concurrent use of one
connection is not safe merely because the writes agree among themselves; a read
interleaved into another thread's statement corrupts the connection's state.

This drives real threads against a real migrated database, because the defect
only exists when they overlap.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Node, Operation, OperationKind, OperationState

pytestmark = pytest.mark.unit

_MIGRATIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)


@pytest.fixture
def repo(tmp_path: Path) -> SQLiteRepository:
    conn = connect(tmp_path / "concurrent.db")
    migrate(conn, _MIGRATIONS)
    return SQLiteRepository(conn)


def _operation(op_id: str) -> Operation:
    return Operation(
        id=op_id,
        kind=OperationKind.MODEL_ACQUIRE,
        state=OperationState.RUNNING,
        target_type="model",
        target_id=new_ulid(),
        started_at=datetime.now().astimezone(),
    )


def test_concurrent_reads_and_writes_do_not_misuse_the_connection(
    repo: SQLiteRepository,
) -> None:
    """The exact shape that broke: writers running while a poller reads.

    Without the lock on reads this raises
    ``sqlite3.InterfaceError: bad parameter or other API misuse``.
    """
    ids = [new_ulid() for _ in range(40)]
    errors: list[BaseException] = []

    def write() -> None:
        try:
            for op_id in ids:
                repo.save_operation(_operation(op_id))
        except BaseException as exc:  # the assertion is "none escaped"
            errors.append(exc)

    def poll() -> None:
        try:
            for _ in range(200):
                repo.list_operations()
                for op_id in ids[:5]:
                    repo.get_operation(op_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write), threading.Thread(target=poll)]
    threads += [threading.Thread(target=poll) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"concurrent access raised: {errors[:3]}"
    assert len(repo.list_operations()) == len(ids)


def test_reads_of_different_tables_interleave_safely(repo: SQLiteRepository) -> None:
    """Not just operations: every read path shares the one connection."""
    node = Node(
        id=new_ulid(),
        name="spark-test",
        agent_endpoint="https://n:8443",
        agent_contract_version="1.14",
        agent_cert_fingerprint="aa",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )
    repo.save_node(node)
    errors: list[BaseException] = []

    def churn() -> None:
        try:
            for _ in range(150):
                repo.list_nodes()
                repo.get_node(node.id)
                repo.get_node_by_name(node.name)
                repo.list_models()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=churn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"concurrent reads raised: {errors[:3]}"
