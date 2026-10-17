"""A bound inference credential reaches every start path, or the start refuses.

CodeGuard CG-01: ``start()`` dispatched ``_create_on`` without resolving the
deployment's binding, so an initial start sent no credential at all -- which the
agent reads as "use this node's default key, or none". Only ``restart()``
resolved it. A broken binding was likewise never resolved on start, so it could
not fail the start closed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_KEY = "sk-bound-synthetic-not-real"


def _register(client: TestClient, name: str, endpoint: str) -> str:
    resp = client.post("/v1/nodes", json={"name": name, "agent_endpoint": endpoint}, headers=_AUTH)
    assert resp.status_code == 201
    return str(resp.json()["id"])


@pytest.fixture
def group() -> tuple[TestClient, Any, str]:
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_client = app.state.node_client
    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    node_client.agents[node_b] = FakeNodeAgent()
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_a, node_b],
        },
        headers=_AUTH,
    )
    poll_operation(client, resp.json()["operation_id"])
    model_id = client.get("/v1/models", headers=_AUTH).json()[0]["id"]
    created = client.post(
        "/v1/deployments",
        json={
            "name": "bound",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "latest",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": [node_a, node_b],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert created.status_code == 202, created.text
    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    return client, node_client, dep_id


def _bind(client: TestClient, dep_id: str) -> None:
    assert (
        client.put("/v1/inference-credentials/prod", json={"value": _KEY}, headers=_AUTH)
    ).status_code == 200
    assert (
        client.put(
            f"/v1/deployments/{dep_id}/inference-credential", json={"name": "prod"}, headers=_AUTH
        )
    ).status_code == 200


@pytest.mark.parametrize("verb", ["start", "restart"])
def test_every_rank_receives_the_bound_key(group: Any, verb: str) -> None:
    client, node_client, dep_id = group
    _bind(client, dep_id)

    resp = client.post(f"/v1/deployments/{dep_id}:{verb}", headers=_AUTH)

    assert resp.status_code in (200, 202), resp.text
    sent = [payload.get("inference_credential") for _node, payload in node_client.create_payloads]
    assert sent and sent == [_KEY] * len(sent), sent
    assert len(sent) == 2


def test_a_broken_binding_refuses_before_any_node_is_touched(
    group: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, node_client, dep_id = group
    _bind(client, dep_id)

    # The stored secret becomes unreadable behind the binding.
    def unreadable(_ref: str) -> str:
        raise OSError("secret file unreadable")

    monkeypatch.setattr(client.app.state.inference_credentials._provider, "resolve", unreadable)
    node_client.node_calls.clear()

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    assert resp.status_code >= 400
    assert resp.json()["code"] == "credential_resolution_failed"
    assert node_client.node_calls == [], "refused only after touching nodes"
    assert node_client.create_payloads == []
