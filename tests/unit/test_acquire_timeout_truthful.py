"""A timed-out acquire hop is waited out, and its placeholder does not outlive success.

On 2026-10-23 the 53 GB MAI-Code-1.1-Flash download took longer than the
one-hour hop bound. The coordinator recorded the acquire as failed -- while its
own message said the work was probably still running -- and the agent finished
fifteen minutes later. The re-run recorded the real model under its resolved
revision, and the failed placeholder under the unresolved one stayed beside it:
"failed, unpinned, 0 bytes" next to the available copy, indefinitely.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import ModelSource, Node, ReplicaState
from tensorstead.ports.node_client import AgentCallError
from tensorstead.service import models_
from tensorstead.service.models_ import ModelService

pytestmark = pytest.mark.unit

_MIGRATIONS = Path(__file__).resolve().parents[2] / "src/tensorstead/adapters/sqlite/migrations"
_MODEL = "azureml/MAI-Code-1.1-Flash-On-Device"
_RESOLVED = "2@" + "ab" * 64


class _Client:
    """Fails the first ``fail`` acquires with ``code``, then succeeds."""

    def __init__(self, fail: int, code: str = "agent_timeout") -> None:
        self.fail, self.code, self.calls = fail, code, 0

    def acquire_model(self, node: Node, **_kw: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.fail:
            raise AgentCallError(code=self.code, message="hop bound exceeded", node_id=node.id)
        return {"resolved_revision": _RESOLVED, "revision_pinned": True, "size_bytes": 10}


def _service(client: _Client) -> tuple[ModelService, SQLiteRepository, Node]:
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repo = SQLiteRepository(conn)
    repo.save_model_source(
        ModelSource(id="azureml", supports_revision_pinning=True, requires_credential=False)
    )
    node = Node(
        id=new_ulid(),
        name="spark-471b.internal",
        agent_endpoint="https://spark-471b:8443",
        agent_contract_version="1.15",
        agent_cert_fingerprint="aa",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )
    repo.save_node(node)
    return ModelService(repo, client), repo, node


def test_a_hop_timeout_is_waited_out_not_recorded_as_failure() -> None:
    client = _Client(fail=2)
    models, repo, node = _service(client)
    model = models.acquire(
        source_id="azureml", source_model_id=_MODEL, revision="2", nodes=[node.id]
    )
    assert client.calls == 3
    assert model.resolved_revision == _RESOLVED
    assert [(m.resolved_revision, m.revision_pinned) for m in repo.list_models()] == [
        (_RESOLVED, True)
    ]


def test_a_transfer_that_never_ends_is_reported_after_the_bounded_waits() -> None:
    client = _Client(fail=99)
    models, _repo, node = _service(client)
    with pytest.raises(AgentCallError) as caught:
        models.acquire(source_id="azureml", source_model_id=_MODEL, revision="2", nodes=[node.id])
    assert caught.value.code == "agent_timeout"
    assert client.calls == models_._ACQUIRE_TIMEOUT_REWAITS + 1


def test_a_real_failure_is_not_retried() -> None:
    client = _Client(fail=1, code="http_error")
    models, _repo, node = _service(client)
    with pytest.raises(AgentCallError):
        models.acquire(source_id="azureml", source_model_id=_MODEL, revision="2", nodes=[node.id])
    assert client.calls == 1


def test_success_removes_the_placeholder_an_earlier_failure_recorded() -> None:
    client = _Client(fail=1, code="http_error")
    models, repo, node = _service(client)
    with pytest.raises(AgentCallError):
        models.acquire(source_id="azureml", source_model_id=_MODEL, revision="2", nodes=[node.id])
    placeholder = repo.list_models()
    assert [(m.resolved_revision, m.revision_pinned) for m in placeholder] == [("2", False)]
    assert repo.list_replicas(placeholder[0].id)[0].state == ReplicaState.FAILED

    models.acquire(source_id="azureml", source_model_id=_MODEL, revision="2", nodes=[node.id])

    assert [m.resolved_revision for m in repo.list_models()] == [_RESOLVED]
