"""GUARDRAIL: observation is pure.

An observed-state request issues zero write operations against a node and
zero writes to the store. Observation is read-only; detecting a divergence
mutates nothing. Host state changes only when a client
explicitly invokes reconcile.

This is verified by:

- Asserting the observation agent route has no write path (no POST/PUT/DELETE
  on ``/agent/v1/deployments/{id}/observed``).
- Asserting the coordinator's observation service calls no write methods on
  the repository (no ``save_*``, ``insert_*``, ``delete_*``).
- Asserting an observed-state request against a running deployment does not
  change the deployment's desired_state or running_revision.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"


def test_observation_route_is_read_only() -> None:
    """The agent observation route is GET-only — no write path."""
    observation_path = _SRC / "agent" / "routes" / "observation.py"
    source = observation_path.read_text()

    # The route must be a GET handler.
    assert "@router.get" in source, "observation route must be GET (read-only)"
    # No POST, PUT, DELETE, or PATCH on the observation route.
    assert "@router.post" not in source, "POST write path — write-path violation"
    assert "@router.put" not in source, "PUT write path — write-path violation"
    assert "@router.delete" not in source, "DELETE write path — write-path violation"
    assert "@router.patch" not in source, "PATCH write path — write-path violation"


def test_observation_service_calls_no_write_methods() -> None:
    """ObservationService calls no save_*/insert_*/delete_* on the repository."""
    observation_path = _SRC / "service" / "observation.py"
    source = observation_path.read_text()

    # No write methods on the repository should be called.
    write_patterns = [
        r"\.save_",
        r"\.insert_",
        r"\.delete_",
        r"\.update_",
    ]
    for pattern in write_patterns:
        matches = re.findall(pattern, source)
        assert not matches, (
            f"observation.py calls a write method ({pattern!r}) — "
            "observation must issue zero write operations"
        )


def test_observed_request_does_not_mutate_state() -> None:
    """An observed-state request does not change declared state."""
    app, _ = build_test_coordinator()
    client = TestClient(app)

    # Register a node, acquire a model, create and start a deployment.
    node_resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert node_resp.status_code == 201
    node_id = node_resp.json()["id"]

    model_resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert model_resp.status_code == 202
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, model_resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = next(m["id"] for m in models if m["source_model_id"] == "org/model")

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
    dep_id = deployment["declared"]["id"]

    start = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert start.status_code == 202

    # Capture declared state before observation.
    before = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    declared_before = before["declared"]

    # Observe the deployment — this must not mutate anything.
    observed = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    assert observed["declared"] == declared_before, (
        "observed-state request changed declared state — write-path violation"
    )

    # Observe again — still no change.
    observed_again = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    assert observed_again["declared"] == declared_before, (
        "repeated observation changed declared state — write-path violation"
    )
