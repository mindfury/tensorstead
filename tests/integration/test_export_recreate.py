"""Integration test recreation from an export.

The independent test: export a known deployment, destroy it, regenerate it
from the export on a comparable node, and compare model revision, runtime
version, image digest, and runtime configuration against the original. The
recreation is a **new deployment with a new id** at revision 1 — not a
continuation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _setup_deployment(client: TestClient, name: str = "llama-70b") -> tuple[str, str]:
    """Register → acquire → create; return (deployment_id, model_id)."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    node_id = str(resp.json()["id"])

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
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))

    resp = client.post(
        "/v1/deployments",
        json={
            "name": name,
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2, "max_model_len": 8192},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == name))
    return dep_id, model_id


def test_us4_export_destroy_recreate(client: TestClient) -> None:
    """Export, destroy, recreate; the four acceptance-criterion values match."""
    dep_id, _ = _setup_deployment(client)

    # Start it so the image digest is resolved — the acceptance criterion compares digests.
    start = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert start.status_code == 202

    # Export revision 1 (current).
    export = client.get(f"/v1/deployments/{dep_id}/export?revision=1", headers=_AUTH)
    assert export.status_code == 200
    original = export.json()

    # Destroy it.
    removed = client.delete(f"/v1/deployments/{dep_id}", headers=_AUTH)
    assert removed.status_code == 202
    assert client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).status_code == 404

    # Recreate from the export.
    recreated = client.post(
        "/v1/deployments:create-from-export", json={"export": original}, headers=_AUTH
    )
    assert recreated.status_code == 202

    # The recreation is a new deployment with a new id.
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    new_declared = next(
        d["declared"]
        for d in deployments
        if d["declared"]["name"] == "llama-70b" and d["declared"]["id"] != dep_id
    )
    assert new_declared["id"] != dep_id
    assert new_declared["current_revision"] == 1

    # Compare the four values against the original export.
    new_rev = new_declared["revision"]
    assert new_rev["model"]["resolved_revision"] == original["model"]["revision"]
    assert new_rev["runtime_version"] == original["runtime"]["version"]
    assert new_rev["image_reference"] == original["image"]["reference"]
    assert new_rev["runtime_config"] == original["runtime_config"]


def test_us4_recreate_missing_node_is_error(client: TestClient) -> None:
    """A missing node name is an error, never a silent substitution."""
    dep_id, _ = _setup_deployment(client)
    export = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()

    # Point the export at a node that does not exist in the inventory.
    export["placement"]["nodes"] = ["nonexistent-node"]
    resp = client.post("/v1/deployments:create-from-export", json={"export": export}, headers=_AUTH)
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert "nonexistent-node" in resp.json()["message"]
