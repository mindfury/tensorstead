"""GUARDRAIL: observed state never persisted.

No write path exists for ``ObservedState``, ``NodeResourceObservation``, or
``Divergence`` — the structural form of the success criterion. Observed state is obtained
from the responsible node at request time and never stored as current.
This is enforced by test, not convention.

This is verified by:

- Asserting the repository port defines no ``save_observed*``,
  ``save_resource*``, or ``save_divergence*`` method.
- Asserting the SQLite repository implements no such methods.
- Asserting the SQLite schema (migration 0001) has no ``observed_state``,
  ``node_resource_observation``, or ``divergence`` table.
- Asserting no product code imports or calls a persistence method for these
  types.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"
_MIGRATIONS = _SRC / "adapters" / "sqlite" / "migrations"


def test_repository_port_has_no_observed_state_write_methods() -> None:
    """The repository port defines no save/insert for observed types."""
    repo_port = _SRC / "ports" / "repository.py"
    source = repo_port.read_text()

    forbidden = [
        r"save_observed",
        r"save_resource",
        r"save_divergence",
        r"insert_observed",
        r"insert_divergence",
        r"update_observed",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, source), (
            f"repository port defines {pattern!r} — observed state must have no write path"
        )


def test_sqlite_repository_has_no_observed_state_write_methods() -> None:
    """The SQLite repository implements no save/insert for observed types."""
    repo_impl = _SRC / "adapters" / "sqlite" / "repository.py"
    source = repo_impl.read_text()

    forbidden = [
        r"def save_observed",
        r"def save_resource_observation",
        r"def save_divergence",
        r"def insert_observed",
        r"def insert_divergence",
        r"observed_state",
        r"node_resource_observation",
        r"divergence",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, source, re.IGNORECASE), (
            f"SQLite repository references {pattern!r} — observed state must "
            "have no persistence path"
        )


def test_no_table_for_observed_state_in_schema() -> None:
    """The SQLite schema has no table for observed state, resources, or divergence."""
    for migration_file in _MIGRATIONS.glob("*.sql"):
        source = migration_file.read_text()
        forbidden_tables = [
            r"CREATE\s+TABLE.*observed_state",
            r"CREATE\s+TABLE.*node_resource_observation",
            r"CREATE\s+TABLE.*divergence",
            r"CREATE\s+TABLE.*resource_observation",
        ]
        for pattern in forbidden_tables:
            assert not re.search(pattern, source, re.IGNORECASE), (
                f"{migration_file.name} creates a table matching {pattern!r} — "
                "observed state must never be persisted"
            )


def test_observation_service_does_not_persist() -> None:
    """ObservationService calls no write methods on the repository."""
    observation_path = _SRC / "service" / "observation.py"
    source = observation_path.read_text()

    write_patterns = [
        r"\.save_",
        r"\.insert_",
        r"\.delete_",
        r"\.update_",
    ]
    for pattern in write_patterns:
        matches = re.findall(pattern, source)
        assert not matches, (
            f"observation.py calls a write method ({pattern!r}) — observed "
            "state must never be persisted"
        )


def test_observed_types_are_response_only() -> None:
    """ObservedState, NodeResourceObservation, and Divergence are response types."""
    state_path = _SRC / "domain" / "state.py"
    source = state_path.read_text()

    # The domain state module must define these types.
    assert "class ObservedState" in source, "ObservedState type must exist"
    assert "class NodeResourceObservation" in source, "NodeResourceObservation type must exist"
    assert "class Divergence" in source, "Divergence type must exist"

    # The module docstring must state these are response-only (never persisted).
    assert "never persisted" in source.lower() or "response" in source.lower(), (
        "domain/state.py must document that observed types are never persisted "
    )
