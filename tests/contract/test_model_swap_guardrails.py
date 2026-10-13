"""Model-swap guardrails.

An agent repointed a running deployment at a different model and restarted it;
the new revision kept the old model's host config and crashed on warmup. A
different model is a different deployment, so changing the model of a running
deployment is refused, and two running deployments may not serve under the
same declared model name.
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


def _node(client: TestClient) -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _model(client: TestClient, node_id: str, source_model_id: str) -> str:
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": source_model_id,
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == source_model_id))


def _deployment(
    client: TestClient,
    node_id: str,
    model_id: str,
    *,
    name: str,
    endpoint: str,
    config: dict[str, Any] | None = None,
    runtime_type: str = "vllm",
) -> str:
    base: dict[str, Any] = {"tensor_parallel_size": 1} if runtime_type == "vllm" else {}
    resp = client.post(
        "/v1/deployments",
        json={
            "name": name,
            "model_id": model_id,
            "runtime_type": runtime_type,
            "runtime_version": "0.6.0",
            "image_reference": f"repo/{runtime_type}:tag",
            "runtime_config": {**base, **(config or {})},
            "participating_nodes": [node_id],
            "endpoint": endpoint,
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202, resp.text
    listed = client.get("/v1/deployments", headers=_AUTH).json()
    return str(next(d for d in listed if d["declared"]["name"] == name)["declared"]["id"])


def _post(client: TestClient, dep_id: str, verb: str) -> Any:
    return client.post(f"/v1/deployments/{dep_id}:{verb}", headers=_AUTH)


def _desired(client: TestClient, dep_id: str) -> str:
    listed = client.get("/v1/deployments", headers=_AUTH).json()
    return str(
        next(d for d in listed if d["declared"]["id"] == dep_id)["declared"]["desired_state"]
    )


# --- model change ------------------------------------------------------------


def test_changing_the_model_of_a_running_deployment_is_refused(client: TestClient) -> None:
    node = _node(client)
    old, new = _model(client, node, "org/old"), _model(client, node, "org/new")
    dep = _deployment(client, node, old, name="serving", endpoint="10.0.0.11:8000")
    assert _post(client, dep, "start").status_code == 202

    resp = client.patch(f"/v1/deployments/{dep}", json={"model_id": new}, headers=_AUTH)

    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "model_change_refused"
    assert body["detail"]["runbook"] == "model-swap"
    assert "huggingface:org/old" in body["message"]
    # Nothing was recorded: the current revision is still the first.
    revisions = client.get(f"/v1/deployments/{dep}/revisions", headers=_AUTH).json()
    assert len(revisions) == 1


def test_a_stopped_deployment_may_change_model_with_a_warning(client: TestClient) -> None:
    node = _node(client)
    old, new = _model(client, node, "org/old"), _model(client, node, "org/new")
    dep = _deployment(client, node, old, name="idle", endpoint="10.0.0.11:8000")

    resp = client.patch(f"/v1/deployments/{dep}", json={"model_id": new}, headers=_AUTH)

    assert resp.status_code in (200, 202), resp.text
    warnings = resp.json().get("warnings", [])
    assert any("carry over unchanged" in w for w in warnings)


def test_other_changes_to_a_running_deployment_are_still_allowed(client: TestClient) -> None:
    node = _node(client)
    model = _model(client, node, "org/old")
    dep = _deployment(client, node, model, name="serving", endpoint="10.0.0.11:8000")
    assert _post(client, dep, "start").status_code == 202

    resp = client.patch(
        f"/v1/deployments/{dep}",
        json={"runtime_config": {"max_model_len": 4096}, "model_id": model},
        headers=_AUTH,
    )

    assert resp.status_code in (200, 202), resp.text


# --- served-name collisions --------------------------------------------------


def test_a_second_deployment_under_the_same_served_name_cannot_start(
    client: TestClient,
) -> None:
    node = _node(client)
    a, b = _model(client, node, "org/a"), _model(client, node, "org/b")
    first = _deployment(
        client,
        node,
        a,
        name="first",
        endpoint="10.0.0.11:8000",
        config={"served_model_name": "chat"},
    )
    second = _deployment(
        client,
        node,
        b,
        name="second",
        endpoint="10.0.0.11:8001",
        config={"served_model_name": "chat"},
    )
    assert _post(client, first, "start").status_code == 202

    resp = _post(client, second, "start")

    assert resp.status_code == 409
    assert resp.json()["code"] == "served_name_conflict"
    assert resp.json()["detail"]["conflicting_deployment"] == "first"
    assert _desired(client, second) == "stopped"


def test_the_name_declared_through_extra_args_counts(client: TestClient) -> None:
    """Runtimes without a modelled field declare it in ``extra_args`` (llama.cpp's ``alias``)."""
    node = _node(client)
    a, b = _model(client, node, "org/a"), _model(client, node, "org/b")
    first = _deployment(
        client,
        node,
        a,
        name="first",
        endpoint="10.0.0.11:8000",
        config={"served_model_name": "chat"},
    )
    second = _deployment(
        client,
        node,
        b,
        name="second",
        endpoint="10.0.0.11:8001",
        config={"extra_args": {"alias": "chat"}},
        runtime_type="llamacpp",
    )
    assert _post(client, first, "start").status_code == 202

    assert _post(client, second, "start").status_code == 409


def test_a_refused_restart_leaves_the_deployment_running(client: TestClient) -> None:
    node = _node(client)
    a, b = _model(client, node, "org/a"), _model(client, node, "org/b")
    first = _deployment(
        client,
        node,
        a,
        name="first",
        endpoint="10.0.0.11:8000",
        config={"served_model_name": "chat"},
    )
    second = _deployment(
        client,
        node,
        b,
        name="second",
        endpoint="10.0.0.11:8001",
        config={"served_model_name": "other"},
    )
    assert _post(client, first, "start").status_code == 202
    assert _post(client, second, "start").status_code == 202
    # Renamed while running: recorded, not applied until a restart.
    modified = client.patch(
        f"/v1/deployments/{second}",
        json={"runtime_config": {"served_model_name": "chat"}},
        headers=_AUTH,
    )
    assert any("already served" in w for w in modified.json().get("warnings", []))

    resp = _post(client, second, "restart")

    assert resp.status_code == 409
    assert resp.json()["code"] == "served_name_conflict"
    assert _desired(client, second) == "running"


def test_distinct_or_undeclared_names_start_normally(client: TestClient) -> None:
    node = _node(client)
    a, b, c = (_model(client, node, f"org/{x}") for x in "abc")
    first = _deployment(
        client,
        node,
        a,
        name="first",
        endpoint="10.0.0.11:8000",
        config={"served_model_name": "chat"},
    )
    second = _deployment(
        client,
        node,
        b,
        name="second",
        endpoint="10.0.0.11:8001",
        config={"served_model_name": "code"},
    )
    third = _deployment(client, node, c, name="third", endpoint="10.0.0.11:8002")
    for dep in (first, second, third):
        assert _post(client, dep, "start").status_code == 202
