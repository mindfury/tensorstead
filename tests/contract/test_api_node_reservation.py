"""A node an operator marks reserved refuses new work.

Declared, not observed: nothing about the node changes and nothing already
running there is touched. The only effect is that ``create``, ``modify``, and
``start`` refuse to name it -- ``stop`` never does, since a reservation says
"put no new work here", not "nothing already there may be stopped".

The motivating case: a standalone service (ComfyUI) can occupy a
node's real memory in a way Tensorstead cannot see until the moment a start is
actually attempted -- by which point a model may already have been acquired
onto that node for nothing. This is the operator's way of saying "not this
one" before any of that time is spent, at every entry point that accepts a
node name, not only the one call site someone happened to think of.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


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


def _reserve(client: TestClient, node_id: str, note: str = "outside Tensorstead") -> None:
    resp = client.post(f"/v1/nodes/{node_id}/reserve", json={"note": note}, headers=_AUTH)
    assert resp.status_code == 200, resp.text


def _unreserve(client: TestClient, node_id: str) -> None:
    resp = client.post(f"/v1/nodes/{node_id}/unreserve", headers=_AUTH)
    assert resp.status_code == 200, resp.text


def _model(client: TestClient, node_id: str) -> str:
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
    return str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))


def _create(client: TestClient, model_id: str, node_id: str, *, name: str = "qwen36-27b") -> Any:
    """POST /v1/deployments; returns the raw response so callers can check status."""
    return client.post(
        "/v1/deployments",
        json={
            "name": name,
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


def _created_deployment_id(client: TestClient, name: str) -> str:
    """The create response carries only an operation id (202); look the deployment up by name."""
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    return str(next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == name))


def test_create_refuses_a_reserved_node(client: TestClient) -> None:
    node_id = _node(client, "spark-01", "10.0.0.11")
    model_id = _model(client, node_id)
    _reserve(client, node_id, "ComfyUI running standalone here")

    resp = _create(client, model_id, node_id)

    assert resp.status_code == 422, f"deployed onto a reserved node: {resp.text}"
    assert resp.json()["code"] == "node_reserved"
    assert "ComfyUI running standalone here" in resp.text, (
        "the refusal must carry the operator's own reason"
    )


def test_modify_refuses_moving_a_deployment_to_a_reserved_node(client: TestClient) -> None:
    first = _node(client, "spark-01", "10.0.0.11")
    second = _node(client, "spark-02", "10.0.0.12")
    model_id = _model(client, first)
    assert _create(client, model_id, first).status_code == 202
    dep_id = _created_deployment_id(client, "qwen36-27b")
    _reserve(client, second)

    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"participating_nodes": [second], "endpoint": "10.0.0.12:8000"},
        headers=_AUTH,
    )

    assert resp.status_code == 422, f"moved a deployment onto a reserved node: {resp.text}"
    assert resp.json()["code"] == "node_reserved"


def test_start_refuses_a_node_reserved_after_the_deployment_was_created(
    client: TestClient,
) -> None:
    """The case create's own check cannot catch on its own.

    Created while the node was still open, reserved afterward -- exactly the
    sequence a standalone service showing up on an already-deployed node
    produces. If only ``create`` refused this, the deployment would sit ready
    and ``start`` would trample the reserved node anyway.
    """
    node_id = _node(client, "spark-01", "10.0.0.11")
    model_id = _model(client, node_id)
    assert _create(client, model_id, node_id).status_code == 202
    dep_id = _created_deployment_id(client, "qwen36-27b")
    _reserve(client, node_id, "hardware set aside")

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    assert resp.status_code == 422, f"started on a reserved node: {resp.text}"
    assert resp.json()["code"] == "node_reserved"
    assert "hardware set aside" in resp.text


def test_stop_is_never_refused_by_a_reservation(client: TestClient) -> None:
    """A reservation says 'no new work', never 'you may not stop what's there'."""
    node_id = _node(client, "spark-01", "10.0.0.11")
    model_id = _model(client, node_id)
    assert _create(client, model_id, node_id).status_code == 202
    dep_id = _created_deployment_id(client, "qwen36-27b")
    start = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert start.status_code == 202, start.text

    _reserve(client, node_id, "decommissioning")
    resp = client.post(f"/v1/deployments/{dep_id}:stop", headers=_AUTH)

    assert resp.status_code == 202, (
        f"a reservation blocked stopping what was already there: {resp.text}"
    )


def test_unreserving_allows_start_again(client: TestClient) -> None:
    node_id = _node(client, "spark-01", "10.0.0.11")
    model_id = _model(client, node_id)
    assert _create(client, model_id, node_id).status_code == 202
    dep_id = _created_deployment_id(client, "qwen36-27b")
    _reserve(client, node_id)

    refused = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert refused.status_code == 422

    _unreserve(client, node_id)
    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    assert resp.status_code == 202, (
        f"un-reserving did not restore the ability to start: {resp.text}"
    )


def test_reservation_is_visible_in_the_node_list(client: TestClient) -> None:
    """Declared facts belong in the cheap inventory call, not only node show."""
    node_id = _node(client, "spark-01", "10.0.0.11")
    other = _node(client, "spark-02", "10.0.0.12")
    _reserve(client, node_id, "outside Tensorstead")

    listed = {n["id"]: n for n in client.get("/v1/nodes", headers=_AUTH).json()}

    assert listed[node_id]["reserved"] is True
    assert listed[node_id]["reserved_reason"] == "outside Tensorstead"
    assert listed[other]["reserved"] is False


def test_reservation_is_visible_on_node_show(client: TestClient) -> None:
    node_id = _node(client, "spark-01", "10.0.0.11")
    _reserve(client, node_id, "outside Tensorstead")

    body = client.get(f"/v1/nodes/{node_id}", headers=_AUTH).json()

    assert body["reserved"] is True
    assert body["reserved_reason"] == "outside Tensorstead"


def test_a_new_node_is_unreserved_by_default(client: TestClient) -> None:
    """Every node that predates this feature must not silently become blocked."""
    node_id = _node(client, "spark-01", "10.0.0.11")

    body = client.get(f"/v1/nodes/{node_id}", headers=_AUTH).json()

    assert body["reserved"] is False
    assert body["reserved_reason"] == ""
