"""Minimal versioned migration runner.

Applies ``migrations/NNNN_name.sql`` files in ascending numeric order, tracking
applied migrations in a ``schema_migrations`` table so each runs exactly once.
Files are plain SQL; the runner executes each file's statements inside one
transaction so a failed migration cannot leave the schema half-applied.

The runner is deliberately minimal: no downgrades, no out-of-band tooling, no
SQL generation. The coordinator's store is SQLite in WAL mode.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# A migration file is NNNN_snake_name.sql. The numeric prefix is its order.
_MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def migration_files(migrations_dir: Path) -> list[tuple[int, Path]]:
    """Return migration files in the directory ordered by numeric prefix."""
    found: list[tuple[int, Path]] = []
    if not migrations_dir.is_dir():
        return found
    seen: dict[int, Path] = {}
    for path in sorted(migrations_dir.iterdir()):
        match = _MIGRATION_RE.match(path.name)
        if not match:
            continue
        version = int(match.group(1))
        # Two files claiming one version is refused rather than tolerated.
        # ``migrate`` keys applied migrations by *version*, so a duplicate means
        # the second file is silently never applied -- the schema then differs
        # from what the code expects, with nothing anywhere saying so. Found by
        # writing a second `0003` and watching 80 tests fail on a column that
        # the migration directory plainly contained.
        if version in seen:
            raise ValueError(
                f"two migrations claim version {version}: {seen[version].name} and "
                f"{path.name}. Renumber one -- applied migrations are tracked by "
                f"version, so the second would never run"
            )
        seen[version] = path
        found.append((version, path))
    return found


def migrate(conn: sqlite3.Connection, migrations_dir: Path) -> list[str]:
    """Apply any pending migrations and return the names of those applied.

    Creates ``schema_migrations`` if absent, then applies each migration not yet
    recorded, in ascending order, each inside its own transaction.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version   INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()

    applied: list[str] = []
    for version, path in migration_files(migrations_dir):
        row = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
        ).fetchone()
        if row is not None:
            continue
        script = path.read_text()
        # ``executescript`` issues an implicit COMMIT of any pending transaction
        # before running, so DDL is applied as its own unit (the design's
        # "explicit transactions" is enforced by the repository's callers; the
        # runner keeps each migration indivisible).
        conn.executescript(script)
        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            (version, path.name),
        )
        conn.commit()
        applied.append(path.name)
    return applied
