"""An image a retained revision ran cannot be deleted from under it.

CodeGuard CG-10: service-created revisions store ``image_digest=''`` and the
real digest lives only in the image inventory. The deletion guard compared the
revision's empty field, so an image a current or retained revision ran was never
protected. These go through the public API only and seed no digest anywhere.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_IMAGE = "repo/vllm:tag"


def _digest(reference: str) -> str:
    return "sha256:" + hashlib.sha256(reference.encode()).hexdigest()


@pytest.fixture
def running() -> tuple[TestClient, str, str]:
    app, _ = build_test_coordinator()
    client = TestClient(app)
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
    model_id = client.get("/v1/models", headers=_AUTH).json()[0]["id"]
    client.post(
        "/v1/deployments",
        json={
            "name": "d",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "latest",
            "image_reference": _IMAGE,
            "runtime_config": {},
            "participating_nodes": [node],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    dep = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    assert client.post(f"/v1/deployments/{dep}:start", headers=_AUTH).status_code == 202
    return client, node, dep


def _delete(client: TestClient, node: str, digest: str) -> Any:
    return client.delete(f"/v1/images/{node}/{digest}", headers=_AUTH)


def test_the_image_a_stopped_deployment_ran_is_protected(running: Any) -> None:
    client, node, dep = running
    assert client.post(f"/v1/deployments/{dep}:stop", headers=_AUTH).status_code == 202

    resp = _delete(client, node, _digest(_IMAGE))

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "still_referenced"


def test_a_retained_revision_keeps_its_image_after_the_deployment_moves_on(
    running: Any,
) -> None:
    client, node, dep = running
    assert (
        client.patch(
            f"/v1/deployments/{dep}", json={"image_reference": "repo/vllm:new"}, headers=_AUTH
        ).status_code
        == 200
    )
    assert client.post(f"/v1/deployments/{dep}:restart", headers=_AUTH).status_code in (200, 202)

    # Revision 1 ran the old image; it is retained for rollback.
    resp = _delete(client, node, _digest(_IMAGE))

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "still_referenced"


def test_an_unreferenced_image_can_still_be_deleted(running: Any) -> None:
    client, node, dep = running
    assert client.post(f"/v1/deployments/{dep}:stop", headers=_AUTH).status_code == 202
    assert client.delete(f"/v1/deployments/{dep}", headers=_AUTH).status_code in (200, 202, 204)

    resp = _delete(client, node, _digest(_IMAGE))

    assert resp.status_code == 200, resp.text


def test_a_revision_without_digest_evidence_is_protected_by_reference(running: Any) -> None:
    """Conservative for records made before materializations were recorded."""
    client, node, _dep = running
    repo = client.app.state.repository
    with repo._lock:  # simulate a pre-CG-10 database: no binding rows
        repo._conn.execute("DELETE FROM revision_images")
        repo._conn.commit()

    resp = _delete(client, node, _digest(_IMAGE))

    assert resp.status_code == 409, resp.text
