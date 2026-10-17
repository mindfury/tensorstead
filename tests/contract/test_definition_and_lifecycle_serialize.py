"""A definition change racing a lifecycle operation is never lost.

CodeGuard CG-08: ``modify`` did not take the lifecycle's per-deployment lock,
and lifecycle saves replaced the whole deployment row from a snapshot taken
before the modify. A modify accepted mid-start inserted revision 2; when the
start finished it wrote its stale snapshot and the pointer went back to 1, with
revision 2 orphaned. Concurrent modifies could also collide on the revision
number and surface a raw uniqueness error.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.domain.errors import ConcurrentModificationError
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def deployment() -> tuple[Any, str]:
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]
    op = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node],
        },
        headers=_AUTH,
    ).json()["operation_id"]
    poll_operation(client, op)
    model_id = client.get("/v1/models", headers=_AUTH).json()[0]["id"]
    client.post(
        "/v1/deployments",
        json={
            "name": "d",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "latest",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {},
            "participating_nodes": [node],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    return app, dep_id


def test_a_modify_during_start_is_not_rolled_back(
    deployment: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, dep_id = deployment
    node_client = app.state.node_client
    lifecycle = app.state.lifecycle_service
    deployments = app.state.deployments_service
    modify_result: dict[str, Any] = {}

    def modify() -> None:
        try:
            modify_result["value"] = deployments.modify(dep_id, runtime_config={"max_num_seqs": 2})
        except Exception as exc:  # recorded, asserted below
            modify_result["error"] = exc

    original = node_client.create_deployment

    def create_mid_flight(node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        # The reviewer's interleaving: a modify arrives while start is inside
        # its node call. It either completes now (no shared lock) or waits for
        # the start to finish (shared lock); the bounded join covers both.
        worker = threading.Thread(target=modify)
        worker.start()
        worker.join(timeout=0.5)
        create_mid_flight.worker = worker  # type: ignore[attr-defined]
        result: dict[str, Any] = original(node, payload)
        return result

    monkeypatch.setattr(node_client, "create_deployment", create_mid_flight)

    lifecycle.start(dep_id)
    create_mid_flight.worker.join(timeout=5)  # type: ignore[attr-defined]

    assert "error" not in modify_result, modify_result
    assert modify_result["value"]["revision"] == 2
    deployment_row = app.state.repository.get_deployment(dep_id)
    revisions = [r.revision for r in app.state.repository.list_revisions(dep_id)]
    assert revisions == [1, 2]
    assert deployment_row.current_revision == 2, "the accepted revision was rolled back"
    assert deployment_row.desired_state == "running"


def test_concurrent_modifies_serialize_without_raw_collisions(
    deployment: tuple[Any, str],
) -> None:
    app, dep_id = deployment
    deployments = app.state.deployments_service
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def modify(i: int) -> None:
        barrier.wait()
        try:
            deployments.modify(dep_id, runtime_config={"max_num_seqs": i + 1})
        except ConcurrentModificationError:
            pass  # a typed conflict is an acceptable answer
        except BaseException as exc:  # anything else is the defect
            errors.append(exc)

    threads = [threading.Thread(target=modify, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    revisions = [r.revision for r in app.state.repository.list_revisions(dep_id)]
    assert revisions == list(range(1, len(revisions) + 1)), "a revision was orphaned or skipped"
    assert app.state.repository.get_deployment(dep_id).current_revision == revisions[-1]


def test_the_repository_refuses_a_stale_advance_atomically(deployment: tuple[Any, str]) -> None:
    app, dep_id = deployment
    repo = app.state.repository
    rev1 = repo.get_revision(dep_id, 1)
    from dataclasses import replace

    with pytest.raises(ConcurrentModificationError):
        repo.advance_revision(replace(rev1, revision=2), expected_current=7)
    assert [r.revision for r in repo.list_revisions(dep_id)] == [1], "an orphan was left"
    assert repo.get_deployment(dep_id).current_revision == 1
