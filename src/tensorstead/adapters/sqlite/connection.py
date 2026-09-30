"""SQLite connection with WAL mode and explicit transactions.

SQLite in WAL mode is the coordinator's store. Hand-written SQL
is confined to this adapter; nothing outside ``adapters/sqlite/`` touches SQL
or a connection object. The boundary is separation of concerns, not a
portability layer — it implies no PostgreSQL compatibility goal in v1.

Transactions are explicit: callers decide whether a multi-write operation
commits atomically under an explicit lock rather than relying on
autocommit surprises. WAL mode is enabled per-connection so readers never block
a writer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: str | Path, *, foreign_keys: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection in WAL mode with explicit transaction control.

    ``check_same_thread=False`` allows the coordinator's asyncio workers and the
    FastAPI app to share a single connection; the repository serializes access
    with a lock where needed. Foreign keys are enforced (the schema relies on
    them for referential integrity).
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON" if foreign_keys else "PRAGMA foreign_keys=OFF")
    return conn
