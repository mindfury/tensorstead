"""Contract test deployment modify config semantics.

``PATCH /v1/deployments/{id}`` with ``runtime_config`` **merges** by default:
keys named override, keys not named are kept, so a one-setting modify can no
longer drop its neighbours. ``replace_config: true`` replaces the whole map —
the destructive reading, reachable only when named. The response carries the
configuration actually recorded.
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


def _setup(client: TestClient) -> tuple[str, str]:
    node_id = _register(client)
    model_id = _acquire(client, node_id)
    resp = client.post(
        "/v1/deployments",
        json={
            "name": "qwen36-27b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            # Three settings; the modify will touch one and must not lose the others.
            "runtime_config": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.4,
                "tool_call_parser": "hermes",
            },
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    deployment_id = next(
        d["declared"]["id"] for d in deployments if d["declared"]["name"] == "qwen36-27b"
    )
    return str(deployment_id), model_id


def _register(client: TestClient) -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire(client: TestClient, node_id: str) -> str:
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


def _revision_config(client: TestClient, deployment_id: str) -> dict:
    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    assert resp.status_code == 200
    return dict(resp.json()["declared"]["revision"]["runtime_config"])


def test_modify_config_merges_and_keeps_untouched_keys(client: TestClient) -> None:
    """A one-key modify patches that key and keeps every other."""
    dep_id, _ = _setup(client)
    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_config": {"tool_call_parser": "qwen3_xml"}},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    recorded = _revision_config(client, dep_id)
    assert recorded["tool_call_parser"] == "qwen3_xml"
    assert recorded["gpu_memory_utilization"] == 0.4  # the neighbour stayed
    # The response carries what was recorded.
    assert resp.json()["runtime_config"] == recorded


def test_modify_config_replace_drops_untouched_keys(client: TestClient) -> None:
    """``replace_config`` drops keys not named — the explicit destructive reading."""
    dep_id, _ = _setup(client)
    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_config": {"tool_call_parser": "qwen3_xml"}, "replace_config": True},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    recorded = _revision_config(client, dep_id)
    assert recorded["tool_call_parser"] == "qwen3_xml"
    assert "gpu_memory_utilization" not in recorded  # dropped, not kept
    assert resp.json()["runtime_config"] == recorded


def test_modify_without_config_leaves_config_unchanged(client: TestClient) -> None:
    """A modify naming no config does not touch the config (PATCH semantics)."""
    dep_id, _ = _setup(client)
    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_version": "0.7.0"},
        headers=_AUTH,
    )
    assert resp.status_code == 200
    recorded = _revision_config(client, dep_id)
    assert recorded["gpu_memory_utilization"] == 0.4
    assert recorded["tool_call_parser"] == "hermes"
