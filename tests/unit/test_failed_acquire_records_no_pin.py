"""A failed acquire must not record a moving label as a pinned revision.

Found by running a real acquire that failed: the coordinator
was left holding

    unsloth/Qwen3.6-35B-A3B-MTP-GGUF | main | revision_pinned = 1

``main`` is a moving label. Recording it as *pinned* is the claim this product
most exists to refuse — and it is the estate's
defining failure mode in miniature: a row that says something reproducible
about something that is not.

The path is narrow, which is why it survived. ``_save_replica`` runs only after
an acquire has already raised, so the row appears only when something else has
already gone wrong, and the revision it is handed is the one the *operator
asked for* rather than one any source resolved. ``revision_pinned`` was
inferred as ``revision is not None``, which reads "main" as concrete.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import ModelSource, Node
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


class _FailingNodeClient:
    """A node whose acquire always fails, which is the path under test."""

    def acquire_model(
        self,
        node: Node,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None,
        progress: object = None,
        file_selector: tuple[str, ...] = (),
    ) -> dict:
        raise AgentCallError(code="http_error", message="Internal Server Error", node_id=node.id)


@pytest.fixture
def service() -> tuple[ModelService, SQLiteRepository, Node]:
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repo = SQLiteRepository(conn)
    repo.save_model_source(
        ModelSource(id="huggingface", supports_revision_pinning=True, requires_credential=False)
    )
    node = Node(
        id=new_ulid(),
        name="spark-test",
        agent_endpoint="https://spark-test:8443",
        agent_contract_version="1.14",
        agent_cert_fingerprint="aa",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )
    repo.save_node(node)
    return ModelService(repo, _FailingNodeClient()), repo, node


def test_a_failed_acquire_of_a_moving_label_records_it_unpinned(
    service: tuple[ModelService, SQLiteRepository, Node],
) -> None:
    """The exact row the live failure produced, now refused."""
    models, repo, node = service
    with pytest.raises(AgentCallError):
        models.acquire(
            source_id="huggingface",
            source_model_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            revision=None,  # the operator named no revision -- "main" is inferred
            nodes=[node.id],
        )

    rows = repo.list_models()
    assert len(rows) == 1, "the failure recorded a model row"
    recorded = rows[0]
    assert recorded.resolved_revision == "main"
    assert recorded.revision_pinned is False, (
        "'main' is a moving label; recording it as pinned claims a "
        "reproducibility that does not exist"
    )


def test_the_success_path_still_records_a_pin(
    service: tuple[ModelService, SQLiteRepository, Node],
) -> None:
    """The default must not be weakened for the path that legitimately pins.

    A revision that came back from the agent is concrete, and saying so is what
    makes a deployment reproducible.
    """
    models, _repo, _node = service
    created = models._find_or_create_model(
        "huggingface", "nvidia/Qwen3.6-27B-NVFP4", "0893e1606ff3", []
    )
    assert created.revision_pinned is True


def test_a_failed_acquire_keeps_its_selection(
    service: tuple[ModelService, SQLiteRepository, Node],
) -> None:
    """Identity is still identity when the acquire fails.

    Otherwise a retry with the same selection would not find the failed row and
    a second one would appear beside it.
    """
    models, repo, node = service
    with pytest.raises(AgentCallError):
        models.acquire(
            source_id="huggingface",
            source_model_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            revision=None,
            nodes=[node.id],
            file_selector=("Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",),
        )
    rows = repo.list_models()
    assert len(rows) == 1
    assert rows[0].file_selector == ("Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",)
