"""Renaming a deployment.

A name is the handle every surface resolves a deployment by, and it was
write-once, so names drifted from what deployments serve. A rename changes the
name and nothing else: no revision, no restart, no change to what is served.
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


def _setup(client: TestClient) -> tuple[str, str]:
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    node = str(resp.json()["id"])
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/m",
            "revision": "e1f2a3b",
            "nodes": [node],
        },
        headers=_AUTH,
    )
    poll_operation(client, resp.json()["operation_id"])
    model = str(next(m["id"] for m in client.get("/v1/models", headers=_AUTH).json()))
    return node, model


def _create(client: TestClient, node: str, model: str, name: str, endpoint: str) -> Any:
    return client.post(
        "/v1/deployments",
        json={
            "name": name,
            "model_id": model,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node],
            "endpoint": endpoint,
        },
        headers=_AUTH,
    )


def _declared(client: TestClient) -> dict[str, dict[str, Any]]:
    return {
        d["declared"]["name"]: d["declared"]
        for d in client.get("/v1/deployments", headers=_AUTH).json()
    }


def _id(client: TestClient, name: str) -> str:
    return str(_declared(client)[name]["id"])


def test_a_running_deployment_can_be_renamed_and_nothing_else_changes(client: TestClient) -> None:
    node, model = _setup(client)
    _create(client, node, model, "qwen38-27b-nvfp4-sglang-mtp-tp1-20260928", "10.0.0.11:8000")
    dep = _id(client, "qwen38-27b-nvfp4-sglang-mtp-tp1-20260928")
    assert client.post(f"/v1/deployments/{dep}:start", headers=_AUTH).status_code == 202
    before = _declared(client)["qwen38-27b-nvfp4-sglang-mtp-tp1-20260928"]

    resp = client.post(
        f"/v1/deployments/{dep}:rename",
        json={"name": "qwen3.8-27b-nvfp4-sglang-dflash2"},
        headers=_AUTH,
    )

    assert resp.status_code == 200, resp.text
    after = _declared(client)
    assert "qwen38-27b-nvfp4-sglang-mtp-tp1-20260928" not in after
    renamed = after["qwen3.8-27b-nvfp4-sglang-dflash2"]
    assert renamed["id"] == dep
    for field in ("desired_state", "current_revision", "running_revision"):
        assert renamed[field] == before[field], field
    assert len(client.get(f"/v1/deployments/{dep}/revisions", headers=_AUTH).json()) == 1
    # The new name is the handle from now on.
    assert client.post(f"/v1/deployments/{dep}:stop", headers=_AUTH).status_code == 202


def test_a_name_another_deployment_holds_is_refused(client: TestClient) -> None:
    node, model = _setup(client)
    _create(client, node, model, "first", "10.0.0.11:8000")
    _create(client, node, model, "second", "10.0.0.11:8001")

    resp = client.post(
        f"/v1/deployments/{_id(client, 'second')}:rename", json={"name": "first"}, headers=_AUTH
    )

    assert resp.status_code == 409
    assert resp.json()["code"] == "already_exists"
    assert set(_declared(client)) == {"first", "second"}


@pytest.mark.parametrize("bad", ["", " ", "-leading-dash", "has space", "a/b", "x" * 129])
def test_an_invalid_name_is_refused(client: TestClient, bad: str) -> None:
    node, model = _setup(client)
    _create(client, node, model, "dep", "10.0.0.11:8000")

    resp = client.post(
        f"/v1/deployments/{_id(client, 'dep')}:rename", json={"name": bad}, headers=_AUTH
    )

    assert resp.status_code == 422
    assert "dep" in _declared(client)


def test_renaming_to_the_same_name_is_a_no_op(client: TestClient) -> None:
    node, model = _setup(client)
    _create(client, node, model, "dep", "10.0.0.11:8000")

    resp = client.post(
        f"/v1/deployments/{_id(client, 'dep')}:rename", json={"name": "dep"}, headers=_AUTH
    )

    assert resp.status_code == 200


def test_creating_a_duplicate_name_is_refused_cleanly(client: TestClient) -> None:
    """It used to reach the store and fail as a bare integrity error."""
    node, model = _setup(client)
    assert _create(client, node, model, "dep", "10.0.0.11:8000").status_code == 202

    resp = _create(client, node, model, "dep", "10.0.0.11:8001")

    assert resp.status_code == 409
    assert resp.json()["code"] == "already_exists"
    assert len(_declared(client)) == 1
