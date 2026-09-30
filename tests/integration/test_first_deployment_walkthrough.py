"""The first working deployment as an executable walkthrough.

The success criterion says an operator "completes a first working deployment — node
registered, model acquired, deployment serving inference — without issuing any
manual shell command against the host". Its entire automated verification was
that `docs/first-deployment.md` exists on disk. A document's presence is not
evidence anybody can follow it.

This drives the documented path against the fake boundaries and asserts the
property the success criterion is actually about: **every value the path requires can be
obtained from the product**, and no step needs a shell on a host.

It cannot prove the prose is clear — that stays a human judgement, recorded in
the hardware run. It can prove the path exists, is reachable through the
product alone, and that no step demands knowledge the product will not give.
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


def test_the_documented_first_deployment_path_needs_nothing_but_the_product(
    client: TestClient,
) -> None:
    """Walk register -> acquire -> create -> start using only product output."""
    # 1. Register a node. The operator supplies a name and an address, which
    #    are theirs to know; nothing else is required of them.
    node = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert node.status_code == 201, node.text
    node_id = node.json()["id"]

    # 2. Discover the runtimes rather than knowing them.
    runtimes = client.get("/v1/runtimes", headers=_AUTH)
    assert runtimes.status_code == 200
    available = [r["type"] for r in runtimes.json()]
    assert available, "the product must name its runtimes; guessing is not a path"
    runtime = available[0]

    # 3. Acquire a model. The identifier is the operator's choice from an
    #    upstream source, which is knowledge the product cannot supply.
    acquire = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "nodes": [node_id],
            "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        },
        headers=_AUTH,
    )
    assert acquire.status_code == 202, acquire.text
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, acquire.json()["operation_id"])

    # 4. Find the acquired model through the product, not by remembering an id.
    models = client.get("/v1/models", headers=_AUTH).json()
    assert models, "an acquired model must be discoverable by listing"
    model_id = models[0]["id"]

    # 5. Create the deployment. Every value here came from a previous product
    #    response, except the ones only the operator can decide: a name, an
    #    endpoint, and the image they intend to run.
    created = client.post(
        "/v1/deployments",
        json={
            "name": "first-deployment",
            "model_id": model_id,
            "runtime_type": runtime,
            "runtime_version": "latest",
            "image_reference": "registry.example/vllm:1",
            "participating_nodes": [node_id],
            "endpoint": "spark-01:8000",
            "runtime_config": {},
        },
        headers=_AUTH,
    )
    assert created.status_code == 202, created.text

    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    deployment_id = deployments[0]["declared"]["id"]

    # 6. Created, not running: the host is untouched until start.
    assert deployments[0]["declared"]["desired_state"] == "stopped"

    # 7. Start it.
    started = client.post(f"/v1/deployments/{deployment_id}:start", headers=_AUTH)
    assert started.status_code == 202, started.text

    # 8. The declared state reflects the request, and observed state is
    #    separately reported so the operator can tell them apart.
    status = client.get(f"/v1/deployments/{deployment_id}/status", headers=_AUTH)
    assert status.status_code == 200
    assert "observed" in status.json()


def test_no_step_of_the_documented_path_requires_a_shell() -> None:
    """The success criterion's actual claim: zero manual shell commands against the host.

    Reads the router directly rather than ``app.routes``: since Starlette 1.4
    / FastAPI 0.141 ``include_router`` no longer flattens routes onto the app,
    so walking the app sees nothing and passes vacuously. The first version of
    this test did exactly that -- it inspected zero routes and reported
    success, which is the failure mode this whole suite exists to catch.
    """
    from tensorstead.coordinator.routes import router

    forbidden = {"command", "script", "shell", "exec", "ssh", "cmd"}
    inspected = 0
    offenders: list[str] = []

    for route in router.routes:
        path = getattr(route, "path", "")
        inspected += 1
        body = getattr(route, "body_field", None)
        model = getattr(body, "type_", None)
        for field in getattr(model, "model_fields", {}) or {}:
            if field.lower() in forbidden:
                offenders.append(f"{path}:{field}")

    assert inspected > 20, (
        f"only {inspected} routes inspected; the router shape changed and this "
        "check has stopped looking at anything"
    )
    assert offenders == [], (
        f"management routes accept host-command-shaped fields: {offenders}. "
        "the first-deployment path must need no shell."
    )


def test_the_walkthrough_covers_what_the_document_claims(client: TestClient) -> None:
    """The document and the walkthrough must describe the same path.

    A walkthrough that drifts from the documentation verifies something nobody
    is being told to do. This checks the four steps it exercises are all
    present in the document, so the two cannot silently diverge.
    """
    from pathlib import Path

    doc = Path("docs/first-deployment.md").read_text(encoding="utf-8").lower()

    for step in ("node register", "model acquire", "deployment create", "deployment start"):
        assert step in doc, (
            f"docs/first-deployment.md no longer documents {step!r}, which this "
            "walkthrough exercises as part of the first-deployment path"
        )
