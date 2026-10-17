"""A group's image comes from the registry once; the other nodes copy it.

Before staging, every rank's agent resolved its image alone, so a two-node start
on fresh nodes pulled the same image twice -- 9.7 GB each for GLM-5.3's runtime.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.domain.errors import ImageStagingError
from tensorstead.ports.node_client import AgentCallError
from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}
_PINNED = "repo/vllm@sha256:" + "ab" * 32


def _register(client: TestClient, name: str, endpoint: str) -> str:
    resp = client.post("/v1/nodes", json={"name": name, "agent_endpoint": endpoint}, headers=_AUTH)
    assert resp.status_code == 201
    return str(resp.json()["id"])


@pytest.fixture
def estate() -> tuple[TestClient, Any, str, str, FakeNodeAgent, FakeNodeAgent]:
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_client = app.state.node_client
    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    agent_a = node_client.agent
    agent_b = FakeNodeAgent()
    node_client.agents[node_b] = agent_b
    return client, app.state.image_stager, node_a, node_b, agent_a, agent_b


def test_pulled_once_on_the_first_node_and_copied_to_the_rest(estate: Any) -> None:
    _client, stager, node_a, node_b, agent_a, agent_b = estate

    outcomes = stager.stage(_PINNED, [node_a, node_b])

    assert agent_a.image_pulls == [_PINNED]
    assert agent_b.image_pulls == [], "the second node must not pull from the registry"
    assert outcomes[node_a]["status"] == "pulled"
    assert outcomes[node_b]["status"] == "distributed"
    assert outcomes[node_b]["image_id"] == outcomes[node_a]["image_id"]


def test_a_node_that_already_holds_it_is_the_source(estate: Any) -> None:
    _client, stager, node_a, node_b, agent_a, agent_b = estate
    agent_b.images[_PINNED] = "sha256:" + "ab" * 32

    outcomes = stager.stage(_PINNED, [node_a, node_b])

    assert agent_a.image_pulls == [] and agent_b.image_pulls == []
    assert outcomes[node_b]["status"] == "present"
    assert outcomes[node_a]["status"] == "distributed"


def test_a_copy_held_only_by_its_digest_counts_as_present(estate: Any) -> None:
    """A distributed digest-pinned image is held by bare id; that is still the image."""
    _client, stager, node_a, node_b, agent_a, agent_b = estate
    stager.stage(_PINNED, [node_a, node_b])

    outcomes = stager.stage(_PINNED, [node_a, node_b])

    assert {o["status"] for o in outcomes.values()} == {"present"}
    assert agent_a.image_pulls == [_PINNED] and agent_b.image_pulls == []


def test_a_failed_copy_is_refused_not_pulled_again(
    estate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, stager, node_a, node_b, _agent_a, agent_b = estate

    def refuse(node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        raise AgentCallError(
            code="image_distribution_failed", message="peer said no", node_id=node.id
        )

    monkeypatch.setattr(client.app.state.node_client, "distribute_image", refuse)

    with pytest.raises(ImageStagingError, match=r"spark-02.*not pulled from the registry again"):
        stager.stage(_PINNED, [node_a, node_b])
    assert agent_b.image_pulls == []


def test_an_unreachable_node_is_left_for_the_start_to_report(estate: Any) -> None:
    _client, stager, node_a, node_b, agent_a, agent_b = estate
    agent_b.unreachable = True

    outcomes = stager.stage(_PINNED, [node_a, node_b])

    assert outcomes[node_b]["status"] == "unreachable"
    assert outcomes[node_a]["status"] == "pulled"
    assert agent_a.image_pulls == [_PINNED]


def test_the_pull_route_stages_and_records_per_node_outcomes(estate: Any) -> None:
    client, _stager, node_a, node_b, agent_a, agent_b = estate

    resp = client.post(
        "/v1/images:pull", json={"reference": _PINNED, "nodes": [node_a, node_b]}, headers=_AUTH
    )
    assert resp.status_code == 202
    operation = poll_operation(client, resp.json()["operation_id"])

    assert operation["state"] == "succeeded"
    assert operation["kind"] == "image_acquire"
    assert agent_a.image_pulls == [_PINNED] and agent_b.image_pulls == []
    assert operation["per_node_outcomes"][node_b]["status"] == "distributed"


def test_a_two_node_start_stages_its_image_first(estate: Any) -> None:
    client, _stager, node_a, node_b, agent_a, agent_b = estate
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_a, node_b],
        },
        headers=_AUTH,
    )
    poll_operation(client, resp.json()["operation_id"])
    model_id = next(
        m["id"]
        for m in client.get("/v1/models", headers=_AUTH).json()
        if m["source_model_id"] == "org/model"
    )
    created = client.post(
        "/v1/deployments",
        json={
            "name": "group",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "latest",
            "image_reference": _PINNED,
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": [node_a, node_b],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert created.status_code in (200, 201, 202), created.text
    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]

    started = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert started.status_code in (200, 202), started.text

    assert agent_a.image_pulls == [_PINNED]
    assert agent_b.image_pulls == []
