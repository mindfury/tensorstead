"""``modify`` applies the same node-validity rules as ``create`` (021, 019).

Two rules were enforced when a deployment was created and not when it was
modified, so a PATCH could store a definition that create would have refused:

- **021** — an unregistered node ID. The revision became the deployment's
  current revision and failed only later at start, after the record and any
  possible reality had already diverged.
- **019** — the same node named twice. ``len(participating_nodes) > 1``
  classifies it as a distributed deployment, so one agent is asked to occupy two
  positions in a runtime group: two rank-0 launches on one host, and a future
  map keyed by node ID that silently keeps only the second outcome.

The tests go through HTTP for the same reason as the expected-revision ones: the
gap was between a surface and a rule, and a service-level test asserts only that
the rule exists.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_ABSENT = "01J000000000000000000ABSENT"


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _node(client: TestClient, name: str, host: str) -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": f"https://{host}:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _setup(client: TestClient) -> tuple[str, str, str]:
    node_id = _node(client, "spark-01", "10.0.0.11")
    second = _node(client, "spark-02", "10.0.0.12")

    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))

    resp = client.post(
        "/v1/deployments",
        json={
            "name": "qwen36-27b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(
        next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == "qwen36-27b")
    )
    return dep_id, node_id, second


def _revision(client: TestClient, deployment_id: str) -> int:
    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    assert resp.status_code == 200
    return int(resp.json()["declared"]["revision"]["revision"])


def test_modify_refuses_an_unregistered_node(client: TestClient) -> None:
    """021: create refuses this; modify must too."""
    dep_id, _, _ = _setup(client)

    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"participating_nodes": [_ABSENT]},
        headers=_AUTH,
    )

    assert resp.status_code == 404, f"stored a revision naming an unknown node: {resp.text}"


def test_a_refused_modify_stores_no_revision(client: TestClient) -> None:
    """The refusal must land before the insert, not after."""
    dep_id, _, _ = _setup(client)

    client.patch(
        f"/v1/deployments/{dep_id}", json={"participating_nodes": [_ABSENT]}, headers=_AUTH
    )

    assert _revision(client, dep_id) == 1


def test_create_refuses_the_same_node_twice(client: TestClient) -> None:
    """019: one agent cannot occupy two positions in a runtime group."""
    _, node_id, _ = _setup(client)
    models = client.get("/v1/models", headers=_AUTH).json()

    resp = client.post(
        "/v1/deployments",
        json={
            "name": "duplicated",
            "model_id": models[0]["id"],
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": [node_id, node_id],
            "endpoint": "10.0.0.11:8001",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422, f"accepted one node as a two-node group: {resp.text}"
    assert node_id in resp.text, "the refusal must name the duplicate"


def test_modify_refuses_the_same_node_twice(client: TestClient) -> None:
    """The same rule on the other surface (019)."""
    dep_id, node_id, _ = _setup(client)

    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"participating_nodes": [node_id, node_id]},
        headers=_AUTH,
    )

    assert resp.status_code == 422, f"accepted one node as a two-node group: {resp.text}"
    assert _revision(client, dep_id) == 1


def test_distinct_nodes_keep_their_declared_order(client: TestClient) -> None:
    """Uniqueness must not become deduplication: order carries rank (019).

    Silently dropping a duplicate would hide the operator's error *and* change
    rank assignment, which is why this refuses rather than normalises.
    """
    dep_id, node_id, second = _setup(client)

    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={
            "participating_nodes": [second, node_id],
            "endpoint": "10.0.0.12:8000",
            # A two-node group needs parallelism that can span it (017).
            "runtime_config": {"tensor_parallel_size": 2},
        },
        headers=_AUTH,
    )

    assert resp.status_code == 200, resp.text
    detail = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    assert detail["declared"]["revision"]["participating_nodes"] == [second, node_id]
