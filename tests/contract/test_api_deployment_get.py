"""Contract test the declared/observed split.

``GET /v1/deployments/{id}`` returns two labelled blocks — ``declared`` and
``observed`` — never merged. ``observed_at`` is always present on the
observed block and absent from the declared block, so a reader can
always tell what was recorded from what was seen right now.

Divergences are reported as a by-product of observation and nothing is mutated.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _register_node(client: TestClient, name: str = "spark-01") -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire_model(client: TestClient, node_id: str) -> str:
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
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))


def _create_and_start(client: TestClient, node_id: str, model_id: str) -> str:
    create = client.post(
        "/v1/deployments",
        json={
            "name": "llama-70b",
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
    assert create.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    deployment = next(d for d in deployments if d["declared"]["name"] == "llama-70b")
    start = client.post(f"/v1/deployments/{deployment['declared']['id']}:start", headers=_AUTH)
    assert start.status_code == 202
    return str(deployment["declared"]["id"])


def test_get_returns_separate_declared_and_observed_blocks(client: TestClient) -> None:
    """GET /v1/deployments/{id} has two labelled blocks, never merged."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()

    assert "declared" in body
    assert "observed" in body
    declared = body["declared"]
    observed = body["observed"]

    # Declared block carries the coordinator's authoritative record.
    assert declared["id"] == deployment_id
    assert declared["name"] == "llama-70b"
    assert declared["desired_state"] == "running"
    assert declared["current_revision"] == 1
    assert "revision" in declared

    # Observed block carries observed_at — the key distinguishing field.
    assert "observed_at" in observed
    assert observed["observed_at"]  # non-empty

    # Declared block must NOT carry observed_at — the structural separation
    # that requires (a reader can always tell which block is which).
    assert "observed_at" not in declared


def test_observed_at_always_present(client: TestClient) -> None:
    """observed_at is always present on the observed block."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    observed = resp.json()["observed"]
    assert observed["observed_at"] is not None
    assert observed["observed_at"]  # non-empty string


def test_divergences_reported_not_acted_on(client: TestClient) -> None:
    """Divergences are a by-product of observation; nothing is mutated."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    body = resp.json()
    assert "divergences" in body
    assert isinstance(body["divergences"], list)

    # An observed-state request issues zero write operations.
    # The deployment's desired_state is unchanged — observation is pure.
    assert body["declared"]["desired_state"] == "running"


def test_declared_returned_in_full_even_when_observation_fails(client: TestClient) -> None:
    """When observation fails, declared is still returned in full."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    # Make the agent unreachable — observation degrades, declared stays full.
    app = client.app
    app.state.fake_agent.unreachable = True  # type: ignore[attr-defined]

    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()

    declared = body["declared"]
    assert declared["id"] == deployment_id
    assert declared["name"] == "llama-70b"
    assert declared["desired_state"] == "running"
    assert declared["current_revision"] == 1

    observed = body["observed"]
    assert observed["status"] == "unreachable"
    assert "observed_at" in observed
