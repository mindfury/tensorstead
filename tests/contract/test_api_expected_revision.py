"""``expected_revision`` must reach the service, not just the request model.

The requirement's optimistic half was implemented everywhere except the one line that
connects it. The request model accepted ``expected_revision``, the CLI sent
``--expect-revision``, the MCP tool sent it, and ``DeploymentService.modify``
checked it — but the PATCH route did not forward it. A caller that explicitly
said "apply this only if the deployment is still at revision N" was told it had
succeeded after losing exactly the race the parameter exists to detect.

Every existing test passed. ``test_optimistic_revision.py`` proved each surface
*exposes* the field and called ``_check_expected_revision`` directly; nothing
went through the route, which is where the value was being dropped. That is the
shape of this defect class: each half was correct and nothing checked the join.

These tests therefore go through HTTP deliberately. A service-level test cannot
fail for this bug.
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


def _setup(client: TestClient) -> str:
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
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))

    resp = client.post(
        "/v1/deployments",
        json={
            "name": "qwen36-27b",
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
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    return str(
        next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == "qwen36-27b")
    )


def _current_revision(client: TestClient, deployment_id: str) -> int:
    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    assert resp.status_code == 200
    return int(resp.json()["declared"]["revision"]["revision"])


def test_a_stale_expected_revision_is_refused_through_the_route(client: TestClient) -> None:
    """The bug itself: PATCH with a superseded expectation must conflict."""
    dep_id = _setup(client)

    # Someone else modifies first; the deployment advances to revision 2.
    assert (
        client.patch(
            f"/v1/deployments/{dep_id}", json={"runtime_version": "0.7.0"}, headers=_AUTH
        ).status_code
        == 200
    )
    assert _current_revision(client, dep_id) == 2

    # Our caller still believes it is at 1 and says so.
    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_version": "0.8.0", "expected_revision": 1},
        headers=_AUTH,
    )

    assert resp.status_code == 409, (
        f"expected_revision was accepted and ignored: {resp.status_code} {resp.text}"
    )


def test_the_refusal_names_what_was_expected_and_what_is_true(client: TestClient) -> None:
    """A conflict the caller cannot act on is barely better than none."""
    dep_id = _setup(client)
    client.patch(f"/v1/deployments/{dep_id}", json={"runtime_version": "0.7.0"}, headers=_AUTH)

    body = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_version": "0.8.0", "expected_revision": 1},
        headers=_AUTH,
    ).json()

    assert body.get("code") == "concurrent_modification", body
    assert body.get("detail") == {"expected": 1, "actual": 2}, body


def test_a_refused_modify_records_no_revision(client: TestClient) -> None:
    """The losing caller must change nothing — a reported conflict, not a partial write."""
    dep_id = _setup(client)
    client.patch(f"/v1/deployments/{dep_id}", json={"runtime_version": "0.7.0"}, headers=_AUTH)

    client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_version": "0.8.0", "expected_revision": 1},
        headers=_AUTH,
    )

    assert _current_revision(client, dep_id) == 2
    resp = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH)
    assert resp.json()["declared"]["revision"]["runtime_version"] == "0.7.0"


def test_a_current_expected_revision_still_applies(client: TestClient) -> None:
    """The check must refuse staleness, not the feature: a correct expectation passes."""
    dep_id = _setup(client)

    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"runtime_version": "0.7.0", "expected_revision": 1},
        headers=_AUTH,
    )

    assert resp.status_code == 200
    assert _current_revision(client, dep_id) == 2


def test_omitting_expected_revision_is_unaffected(client: TestClient) -> None:
    """Absent means "no opinion" — wiring the check must not make it mandatory."""
    dep_id = _setup(client)

    resp = client.patch(
        f"/v1/deployments/{dep_id}", json={"runtime_version": "0.7.0"}, headers=_AUTH
    )

    assert resp.status_code == 200
