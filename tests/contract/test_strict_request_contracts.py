"""Request bodies are strict, and nothing a client sends is echoed back in an error.

CodeGuard CG-11: several routes took a raw ``dict``, silently ignoring unknown
fields and turning wrong types into 500s (a numeric ``from_file`` raised
``TypeError``). CG-12: FastAPI's default 422 echoed each rejected value, so a
malformed credential request returned the secret it carried.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_SENTINEL = "REVIEW_SYNTHETIC_SECRET_NOT_REAL_7f3a"


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("put", "/v1/inference-credentials/x", {"value": "v", "unexpected": 1}),
        ("put", "/v1/inference-credentials/x", {"from_file": 7}),
        ("put", "/v1/inference-credentials/x", {"value": None}),
        ("put", "/v1/inference-credentials/x", {}),
        ("put", "/v1/inference-credentials/x", {"value": "v", "from_env": "HOME"}),
        ("put", "/v1/deployments/d/inference-credential", {"name": "x", "extra": True}),
        ("post", "/v1/code-approvals", {"option": "trust_remote_code"}),
        ("put", "/v1/buildspecs/s", {"base_image": "b", "steps": ["x"] * 201}),
        ("put", "/v1/buildspecs/s", {"base_image": "b", "steps": "not-a-list"}),
        ("post", "/v1/images:build", {"spec": "s", "reference": "r"}),
        (
            "post",
            "/v1/images:build",
            {"spec": "s", "reference": "r", "node_id": "a", "nodes": ["b"]},
        ),
        ("post", "/v1/images:import", {"node_id": "n", "reference": "r"}),
        ("post", "/v1/deployments:create-from-export", {"export": {"kind": "x"}}),
        ("post", "/v1/deployments:create-from-export", {"apiVersion": "v1"}),
    ],
)
def test_malformed_bodies_are_a_structured_422(
    client: TestClient, method: str, path: str, body: dict[str, Any]
) -> None:
    resp = getattr(client, method)(path, json=body, headers=_AUTH)
    assert resp.status_code == 422, (path, resp.status_code, resp.text)
    assert resp.json()["code"] == "invalid_request"


def test_invalid_input_writes_nothing(client: TestClient) -> None:
    client.put("/v1/inference-credentials/x", json={"from_file": 7}, headers=_AUTH)
    client.put("/v1/buildspecs/s", json={"base_image": "b", "steps": "nope"}, headers=_AUTH)
    assert client.get("/v1/inference-credentials", headers=_AUTH).json() == []
    assert client.get("/v1/buildspecs", headers=_AUTH).json() == []


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        # The reviewer's probe: the secret inside a wrong-typed field.
        ("put", "/v1/credentials/huggingface/x", {"secret": [_SENTINEL]}),
        ("put", "/v1/credentials/huggingface/x", {"secret": _SENTINEL, "from_env": "HOME"}),
        ("put", "/v1/inference-credentials/x", {"value": [_SENTINEL]}),
        ("put", "/v1/inference-credentials/x", {"value": _SENTINEL, "from_file": "f"}),
        ("put", "/v1/inference-credentials/x", {"value": _SENTINEL, _SENTINEL: 1}),
        ("put", "/v1/inference-credentials/x", {"from_file": _SENTINEL}),  # imports disabled
        ("put", "/v1/inference-credentials/x", {"from_env": _SENTINEL}),  # unset variable
    ],
)
def test_no_secret_is_echoed_anywhere(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    method: str,
    path: str,
    body: dict[str, Any],
) -> None:
    caplog.set_level(logging.DEBUG)
    resp = getattr(client, method)(path, json=body, headers=_AUTH)
    assert 400 <= resp.status_code < 500, (resp.status_code, resp.text)
    assert _SENTINEL not in resp.text
    assert _SENTINEL not in caplog.text
    operations = client.get("/v1/operations", headers=_AUTH).text
    assert _SENTINEL not in operations


def test_create_from_export_honours_the_endpoint_override(client: TestClient) -> None:
    """``--endpoint`` was sent by the CLI and never read by the route."""
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
    export = {
        "apiVersion": "tensorstead/v1",
        "kind": "DeploymentExport",
        "deployment": {"name": "moved"},
        "model": {
            "source": "huggingface",
            "id": "org/model",
            "revision": "e1f2a3b",
            "revision_pinned": True,
        },
        "runtime": {"type": "vllm", "version": "latest"},
        "image": {"reference": "repo/vllm:tag"},
        "runtime_config": {},
        "placement": {"nodes": ["spark-01"], "endpoint": "10.0.0.99:8000"},
    }
    resp = client.post(
        "/v1/deployments:create-from-export",
        json={"export": export, "endpoint": "10.0.0.11:8123"},
        headers=_AUTH,
    )
    assert resp.status_code == 202, resp.text
    declared = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]
    assert declared["revision"]["endpoint"] == "10.0.0.11:8123"
