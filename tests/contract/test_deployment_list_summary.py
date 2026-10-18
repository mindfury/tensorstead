"""Finding a deployment does not require reading every deployment's full config.

An agent asked to compare the GLM deployment with its vendor recipe called
``deployment_list``: 42 full declared/observed documents, ~70 KB from the API
and ~190 KB once MCP carried it as both text and structured content. Its
harness spilled that to a file and showed one stopped deployment. The list now
has a summary view, server-side filters applied before any node is asked, and
MCP's ``deployment_list`` uses both.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.mcp.client import CoordinatorClient
from tensorstead.mcp.server import build_server
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}

_SUMMARY_KEYS = {
    "id",
    "name",
    "desired_state",
    "current_revision",
    "running_revision",
    "source_model_id",
    "served_model_name",
    "runtime_type",
    "endpoint",
    "participating_nodes",
    "observed_status",
}


@pytest.fixture
def estate() -> tuple[TestClient, Any]:
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]
    for i, (name, repo, served) in enumerate(
        (
            ("glm5.3-flash-fp4-vllm-tp2-mtp5", "nvidia/GLM-5.3-Flash-FP4-Local", "glm-flash"),
            ("deepseek-v4-flash-nvfp4-vllm", "nvidia/DeepSeek-V4-Flash-NVFP4", "deepseek"),
            ("qwen3.8-27b-nvfp4-sglang", "Qwen/Qwen3.8-27B", None),
        )
    ):
        acquired = client.post(
            "/v1/models:acquire",
            json={
                "source_id": "huggingface",
                "source_model_id": repo,
                "revision": f"e1f2a3b{i}",
                "nodes": [node],
            },
            headers=_AUTH,
        )
        poll_operation(client, acquired.json()["operation_id"])
        model_id = next(
            m["id"]
            for m in client.get("/v1/models", headers=_AUTH).json()
            if m["source_model_id"] == repo
        )
        config: dict[str, Any] = {"tensor_parallel_size": 1}
        if served:
            config["served_model_name"] = served
        created = client.post(
            "/v1/deployments",
            json={
                "name": name,
                "model_id": model_id,
                "runtime_type": "vllm",
                "runtime_version": "0.6.0",
                "image_reference": "repo/vllm:tag",
                "runtime_config": config,
                "participating_nodes": [node],
                "endpoint": f"10.0.0.11:{8000 + i}",
            },
            headers=_AUTH,
        )
        assert created.status_code == 202, created.text
        poll_operation(client, created.json()["operation_id"])
    return client, app


def _counting_observations(app: Any) -> list[str]:
    observed: list[str] = []
    service = app.state.observation_service
    original = service.observe

    def observe(deployment_id: str, *args: Any, **kwargs: Any) -> Any:
        observed.append(deployment_id)
        return original(deployment_id, *args, **kwargs)

    service.observe = observe
    return observed


def _list(client: TestClient, **params: str) -> list[dict[str, Any]]:
    response = client.get("/v1/deployments", params=params, headers=_AUTH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    return body


def test_the_default_listing_is_unchanged(estate: tuple[TestClient, Any]) -> None:
    client, _ = estate
    body = _list(client)
    assert len(body) == 3
    assert {"declared", "observed", "divergences"} <= body[0].keys()


def test_a_summary_names_each_deployment_and_its_model_without_observing(
    estate: tuple[TestClient, Any],
) -> None:
    client, app = estate
    observed = _counting_observations(app)
    body = _list(client, view="summary")
    assert len(body) == 3
    assert all(set(item) == _SUMMARY_KEYS for item in body)
    glm = next(item for item in body if item["name"].startswith("glm"))
    assert glm["source_model_id"] == "nvidia/GLM-5.3-Flash-FP4-Local"
    assert glm["served_model_name"] == "glm-flash"
    assert glm["observed_status"] is None
    assert observed == [], "a summary listing asked the nodes about every deployment"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"model": "glm-5.3"}, {"glm5.3-flash-fp4-vllm-tp2-mtp5"}),  # source id, any case
        ({"model": "DEEPSEEK"}, {"deepseek-v4-flash-nvfp4-vllm"}),  # served name too
        ({"name": "qwen"}, {"qwen3.8-27b-nvfp4-sglang"}),
        ({"endpoint": ":8001"}, {"deepseek-v4-flash-nvfp4-vllm"}),
        ({"desired_state": "running"}, set()),
        ({"model": "flash", "name": "glm"}, {"glm5.3-flash-fp4-vllm-tp2-mtp5"}),  # AND
        ({"model": "no-such-model"}, set()),
    ],
)
def test_filters_narrow_the_listing(
    estate: tuple[TestClient, Any], params: dict[str, str], expected: set[str]
) -> None:
    client, _ = estate
    for view in ("summary", "full"):
        body = _list(client, view=view, **params)
        names = {item["name"] if view == "summary" else item["declared"]["name"] for item in body}
        assert names == expected, (view, params)


def test_only_the_filtered_deployments_are_observed(estate: tuple[TestClient, Any]) -> None:
    client, app = estate
    observed = _counting_observations(app)
    full = _list(client, model="glm")
    assert len(full) == 1 and len(observed) == 1

    observed.clear()
    summary = _list(client, view="summary", model="glm", observe="true")
    assert len(observed) == 1
    assert summary[0]["observed_status"] in {"running", "not_running", "unknown", "unreachable"}


def test_an_unknown_desired_state_is_refused(estate: tuple[TestClient, Any]) -> None:
    client, _ = estate
    response = client.get("/v1/deployments", params={"desired_state": "paused"}, headers=_AUTH)
    assert response.status_code == 422


def _mcp(client: TestClient, tool: str, arguments: dict[str, Any]) -> tuple[Any, int]:
    server = build_server(client=CoordinatorClient(client=client, token="test"))
    result = asyncio.run(server.call_tool(tool, arguments))
    assert not result.is_error
    # What an agent's context actually receives: every content block plus the
    # structured copy.
    size = sum(len(block.text) for block in result.content if hasattr(block, "text"))
    if result.structured_content is not None:
        size += len(json.dumps(result.structured_content))
    structured = result.structured_content or {}
    return structured.get("result", structured), size


def test_mcp_finds_a_deployment_by_model_in_a_small_result(
    estate: tuple[TestClient, Any],
) -> None:
    client, _ = estate
    found, size = _mcp(client, "deployment_list", {"model": "GLM-5.3"})
    assert [item["name"] for item in found] == ["glm5.3-flash-fp4-vllm-tp2-mtp5"]
    assert set(found[0]) == _SUMMARY_KEYS
    assert size < 2_000, f"one summary cost {size} bytes of agent context"


def test_mcp_lists_everything_compactly_by_default(estate: tuple[TestClient, Any]) -> None:
    client, _ = estate
    everything, size = _mcp(client, "deployment_list", {})
    assert len(everything) == 3
    # Measured ~770 bytes each across both MCP copies: 42 deployments come to
    # ~32 KB unfiltered, against ~190 KB for the full documents. Filtering is
    # the intended path; this bounds the unfiltered worst case.
    assert size / len(everything) < 1_000, f"{size / len(everything):.0f} bytes per deployment"


def test_the_tool_description_no_longer_denies_rename() -> None:
    tools = asyncio.run(build_server().list_tools())
    description = next(t.description for t in tools if t.name == "deployment_list") or ""
    assert "no rename" not in description
    assert "deployment_get" in description
