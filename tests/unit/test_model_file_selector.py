"""File selection is part of a model's identity (migration 0006).

A repository is not always one model. GGUF publishers ship twenty-odd
quantizations of the same weights in a single repo, so acquiring "the repo"
is neither what an operator wants nor a download that fits on the appliance.
Selecting is therefore necessary — and the moment it exists, the selection has
to be recorded, because a model row naming ``unsloth/...-GGUF`` while the node
holds one of its twenty-three files describes something that does not exist.
That is this estate's defining failure mode, and the acquire path is where it
would have been introduced.

These tests pin the properties that keep the record true: two selections of one
repository are two models, they occupy two directories, and a whole-repository
acquisition is byte-for-byte the thing it was before any of this existed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Model, canonical_file_selector, local_model_id

pytestmark = pytest.mark.unit

_MIGRATIONS = Path("src/tensorstead/adapters/sqlite/migrations")


# ------------------------------------------------------------------ canonical
def test_a_selection_is_order_and_duplicate_insensitive() -> None:
    """Two spellings of one selection must be one selection.

    Identity is compared as a string in SQL; if ``["b","a"]`` and ``["a","b"]``
    stored differently, the uniqueness rule would stop deduplicating and the
    same model would be acquired twice under two rows.
    """
    assert canonical_file_selector(["b.gguf", "a.gguf"]) == ("a.gguf", "b.gguf")
    assert canonical_file_selector(["a.gguf", "a.gguf"]) == ("a.gguf",)
    assert canonical_file_selector("solo.gguf") == ("solo.gguf",)
    assert canonical_file_selector(None) == ()
    assert canonical_file_selector(["  ", "x.gguf "]) == ("x.gguf",)


def test_the_model_canonicalizes_what_it_is_constructed_with() -> None:
    """A model built from a payload and one read from SQLite are one value."""
    model = Model(id=new_ulid(), source_id="hf", source_model_id="r", file_selector=("b", "a"))
    assert model.file_selector == ("a", "b")


# ------------------------------------------------------------- node-local ids
def test_a_whole_repository_keeps_the_id_it_always_had() -> None:
    """No model acquired before selection existed may change its store path."""
    assert local_model_id("huggingface", "nvidia/Qwen3.6-27B-NVFP4") == (
        "huggingface:nvidia/Qwen3.6-27B-NVFP4"
    )


def test_two_selections_of_one_repository_get_two_directories() -> None:
    """The failure this prevents: the second quant landing in the first's dir.

    The agent's reuse check compares resolved revisions, and for two quants of
    one repo they are identical — so a shared directory would return the first
    model's weights as a cache hit for the second.
    """
    q8 = local_model_id("huggingface", "unsloth/repo-GGUF", ("m-UD-Q8_K_XL.gguf",))
    q4 = local_model_id("huggingface", "unsloth/repo-GGUF", ("m-UD-Q4_K_M.gguf",))
    assert q8 != q4
    assert q8 != local_model_id("huggingface", "unsloth/repo-GGUF")


def test_one_selection_written_two_ways_is_one_directory() -> None:
    a = local_model_id("hf", "r", ("b.gguf", "a.gguf"))
    b = local_model_id("hf", "r", ("a.gguf", "b.gguf"))
    assert a == b


# -------------------------------------------------------------------- storage
def _repo(tmp_path: Path) -> SQLiteRepository:
    conn = connect(tmp_path / "x.db")
    migrate(conn, _MIGRATIONS)
    conn.execute(
        "INSERT INTO model_sources (id, supports_revision_pinning, requires_credential) "
        "VALUES ('hf', 1, 1)"
    )
    conn.commit()
    return SQLiteRepository(conn)


def test_two_selections_of_one_revision_coexist(tmp_path: Path) -> None:
    """The UNIQUE constraint the migration widened.

    Before 0006 this was ``UNIQUE (source_id, source_model_id,
    resolved_revision)`` and the second insert raised — one repository could
    only ever be one model.
    """
    repo = _repo(tmp_path)
    for selector in (("q8.gguf",), ("q4.gguf",)):
        repo.save_model(
            Model(
                id=new_ulid(),
                source_id="hf",
                source_model_id="unsloth/repo-GGUF",
                resolved_revision="sha1",
                file_selector=selector,
            )
        )
    assert len(repo.list_models()) == 2


def test_the_same_selection_twice_is_still_refused(tmp_path: Path) -> None:
    """Widening the key must not stop it being a key."""
    repo = _repo(tmp_path)
    for _ in range(2):
        model = Model(
            id=new_ulid(),
            source_id="hf",
            source_model_id="unsloth/repo-GGUF",
            resolved_revision="sha1",
            file_selector=("q8.gguf",),
        )
        if _ == 0:
            repo.save_model(model)
        else:
            with pytest.raises(sqlite3.IntegrityError):
                repo.save_model(model)


def test_whole_repository_models_still_deduplicate(tmp_path: Path) -> None:
    """'' rather than NULL, because SQLite treats NULLs as distinct in UNIQUE.

    A nullable column would silently stop deduplicating every model that
    predates this change — the exact opposite of what the constraint is for.
    """
    repo = _repo(tmp_path)
    repo.save_model(
        Model(id=new_ulid(), source_id="hf", source_model_id="r", resolved_revision="sha1")
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_model(
            Model(id=new_ulid(), source_id="hf", source_model_id="r", resolved_revision="sha1")
        )


def test_find_model_distinguishes_selections(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wanted = Model(
        id=new_ulid(),
        source_id="hf",
        source_model_id="r",
        resolved_revision="sha1",
        file_selector=("q8.gguf",),
    )
    repo.save_model(wanted)
    assert repo.find_model("hf", "r", "sha1", ("q8.gguf",)) is not None
    assert repo.find_model("hf", "r", "sha1", ("q4.gguf",)) is None
    # Omitting the selector means the whole repository, which this is not.
    assert repo.find_model("hf", "r", "sha1") is None


def test_the_selection_survives_a_round_trip(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    model = Model(
        id=new_ulid(),
        source_id="hf",
        source_model_id="r",
        resolved_revision="sha1",
        file_selector=("mmproj-BF16.gguf", "m-UD-Q8_K_XL.gguf"),
    )
    repo.save_model(model)
    read_back = repo.get_model(model.id)
    assert read_back is not None
    assert read_back.file_selector == ("m-UD-Q8_K_XL.gguf", "mmproj-BF16.gguf")


def test_existing_models_are_backfilled_as_whole_repositories(tmp_path: Path) -> None:
    """0006 rebuilds the table; every row that predates it was acquired whole.

    Run against a store migrated only as far as 0005, so this exercises the
    upgrade an existing coordinator actually performs rather than a fresh
    schema that never had the old constraint.
    """
    import shutil

    older = tmp_path / "migrations-0005"
    older.mkdir()
    for _, path in _iter_migrations():
        if not path.name.startswith("0006"):
            shutil.copy(path, older / path.name)

    conn = connect(tmp_path / "y.db")
    migrate(conn, older)
    conn.execute(
        "INSERT INTO model_sources (id, supports_revision_pinning, requires_credential) "
        "VALUES ('hf', 1, 1)"
    )
    legacy_id = new_ulid()
    conn.execute(
        "INSERT INTO models (id, source_id, source_model_id, resolved_revision, "
        "revision_pinned, size_bytes) VALUES (?, 'hf', 'legacy/model', 'sha1', 1, 42)",
        (legacy_id,),
    )
    conn.execute(
        "INSERT INTO nodes (id, name, agent_endpoint, agent_contract_version, "
        "agent_cert_fingerprint, platform_facts, registered_at) "
        "VALUES ('01M0000000000000000000000A', 'n1', 'https://h', '1.13', 'aa', '{}', "
        "datetime('now'))"
    )
    conn.execute(
        "INSERT INTO model_replicas (model_id, node_id, local_path, state) "
        "VALUES (?, '01M0000000000000000000000A', '/var/lib/tensorstead/models/hf:legacy/model', "
        "'available')",
        (legacy_id,),
    )
    conn.commit()

    assert "0006_model_file_selector.sql" in migrate(conn, _MIGRATIONS)

    row = conn.execute("SELECT * FROM models").fetchone()
    assert row["file_selector"] == ""
    assert row["size_bytes"] == 42, "the rebuild must carry every column across"
    assert conn.execute("SELECT COUNT(*) c FROM models").fetchone()["c"] == 1
    # The replica still points at its model: dropping and renaming the parent
    # table must not orphan the rows that reference it.
    kept = conn.execute(
        "SELECT COUNT(*) c FROM model_replicas WHERE model_id = ?", (legacy_id,)
    ).fetchone()["c"]
    assert kept == 1


def _iter_migrations() -> list[tuple[int, Path]]:
    from tensorstead.adapters.sqlite.migrations import migration_files

    return migration_files(_MIGRATIONS)
