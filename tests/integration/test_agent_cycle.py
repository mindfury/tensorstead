"""Integration test agent-to-human knowledge transfer.

An agent completes a full deploy-and-inspect cycle **through MCP only**, and
then a human recovers everything essential **through the CLI**, having never
seen the session.

The claim under test is the requirement's real content: nothing an agent learns exists
only in its context. The agent is not a source of truth. What
makes that true here is not an MCP feature — it is the persistence model. The
agent's tool calls produce ``Deployment``, ``DeploymentRevision``, and
``Operation`` rows, so the export a human reads later is built from the same
records any other client would have produced.

The four values the acceptance criterion names — model revision, runtime version, image digest,
runtime configuration — are checked individually, because "the export looks
right" is exactly the kind of judgement that passes while a field is quietly
missing.

Both surfaces are driven for real: MCP tools go through ``call_tool`` dispatch,
and the CLI goes through its Typer app.
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

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def stack() -> tuple[Any, TestClient]:
    """One coordinator, reachable through both an MCP server and the CLI."""
    app, _ = build_test_coordinator()
    http = TestClient(app)
    server = build_server(client=CoordinatorClient(client=http, token="test"))
    return server, http


def _call(server: Any, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoke one MCP tool through the server's real dispatch path.

    Reads ``structured_content``, which the SDK derives from each tool's
    declared return type. List-returning tools arrive wrapped as
    ``{"result": [...]}``; that envelope is unwrapped here so a one-element list
    stays a list rather than being mistaken for a single record.
    """
    result = asyncio.run(server.call_tool(tool, arguments or {}))
    assert not result.is_error, f"MCP tool {tool} failed: {result.content}"

    structured = result.structured_content
    if structured is None:  # no declared return type; fall back to the text block
        return json.loads(result.content[0].text)
    if set(structured) == {"result"}:
        return structured["result"]
    return structured


def _agent_deploys(server: Any, http: TestClient) -> dict[str, Any]:
    """The whole agent-side cycle: register, acquire, define, start, inspect."""
    node = _call(
        server,
        "node_register",
        {"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
    )
    node_id = node["id"]

    acquired = _call(
        server,
        "model_acquire",
        {
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
    )
    # Acquisition runs on a background thread; poll to terminal before reading
    # the model list.
    poll_operation(http, acquired["operation_id"])
    models = _call(server, "model_list")
    model_id = next(m["id"] for m in models if m["source_model_id"] == "org/model")

    _call(
        server,
        "deployment_create",
        {
            "name": "llama-70b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2, "max_model_len": 8192},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
    )
    deployments = _call(server, "deployment_list")
    declared = next(d["declared"] for d in deployments if d["declared"]["name"] == "llama-70b")

    _call(server, "deployment_start", {"deployment_id": declared["id"]})
    return dict(declared)


def test_an_agent_completes_a_full_cycle_through_mcp_only(stack: tuple[Any, TestClient]) -> None:
    """Deploy and inspect using supported management operations only."""
    server, _http = stack
    declared = _agent_deploys(server, _http)

    # Inspect: declared and observed arrive as separate blocks.
    body = _call(server, "deployment_get", {"deployment_id": declared["id"]})
    assert "declared" in body and "observed" in body
    assert body["declared"]["desired_state"] == "running"
    assert body["observed"]["per_node"]

    status = _call(server, "deployment_status", {"deployment_id": declared["id"]})
    assert status["observed"]["status"] == "running"


def test_a_human_recovers_the_four_values_through_the_cli(
    stack: tuple[Any, TestClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The export carries what the agent never told anyone."""
    server, http = stack
    declared = _agent_deploys(server, http)

    from typer.testing import CliRunner

    from tensorstead.cli import commands as cli_commands
    from tensorstead.cli.main import app as cli_app

    monkeypatch.setattr(cli_commands, "_client", lambda: http)
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "test")

    out = tmp_path / "recovered.yaml"
    result = CliRunner().invoke(cli_app, ["deployment", "export", "llama-70b", "-o", str(out)])
    assert result.exit_code == 0, result.output

    import yaml

    export = yaml.safe_load(out.read_text())

    # The four values, each checked on its own.
    assert export["model"]["revision"] == "e1f2a3b", "model revision"
    assert export["runtime"]["version"] == "0.6.0", "runtime version"
    assert export["runtime_config"] == {
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
    }, "runtime configuration"

    # Image digest: the *field* is carried, which is what the export format
    # guarantees. Its value stays null here because no image is really pulled —
    # the fake agent runs no container engine. Populating it is what the
    # hardware quickstart exercises, so asserting a value in this tier
    # would only be asserting that the fake lies convincingly.
    assert "digest" in export["image"], "image digest field is carried by the export"
    assert export["image"]["reference"] == "repo/vllm:tag"

    # And it describes the deployment the agent actually made.
    assert export["deployment"]["name"] == declared["name"]


def test_nothing_essential_exists_only_in_the_agent_session(
    stack: tuple[Any, TestClient],
) -> None:
    """The records outlive the session that created them.

    A *fresh* client, holding none of the agent's context, reaches the same
    facts. That is what "the agent is not a source of truth" means concretely.
    """
    server, http = stack
    declared = _agent_deploys(server, http)

    fresh = build_server(client=CoordinatorClient(client=http, token="test"))

    revisions = _call(fresh, "deployment_revisions", {"deployment_id": declared["id"]})
    assert revisions, "the agent's work is a retained revision"
    assert revisions[0]["runtime_version"] == "0.6.0"

    operations = _call(fresh, "operation_list", {"deployment_id": declared["id"]})
    kinds = {op["kind"] for op in operations}
    assert {"deployment_create", "start"} <= kinds, (
        f"the agent's lifecycle actions are recorded operations, got {sorted(kinds)}"
    )


def test_the_cli_and_mcp_report_the_same_deployment(stack: tuple[Any, TestClient]) -> None:
    """Both surfaces call the same service, so they cannot disagree."""
    server, http = stack
    declared = _agent_deploys(server, http)

    via_mcp = _call(server, "deployment_get", {"deployment_id": declared["id"]})
    via_api = http.get(f"/v1/deployments/{declared['id']}", headers=_AUTH).json()

    assert via_mcp["declared"] == via_api["declared"]


def test_an_agent_receives_unreachable_as_a_value_not_an_error(
    stack: tuple[Any, TestClient],
) -> None:
    """``unreachable`` is an ordinary result.

    An agent that saw a failed call here would retry rather than report, which
    is precisely the stale-state behaviour this rule exists to prevent.
    """
    server, http = stack
    declared = _agent_deploys(server, http)
    http.app.state.fake_agent.unreachable = True  # type: ignore[attr-defined]

    status = _call(server, "deployment_status", {"deployment_id": declared["id"]})

    assert status["observed"]["status"] == "unreachable"
    per_node = status["observed"]["per_node"]
    assert all(n["status"] == "unreachable" for n in per_node.values())


def test_declared_survives_an_unreachable_node_for_an_agent_too(
    stack: tuple[Any, TestClient],
) -> None:
    """Declared is returned in full on every surface."""
    server, http = stack
    declared = _agent_deploys(server, http)
    http.app.state.fake_agent.unreachable = True  # type: ignore[attr-defined]

    body = _call(server, "deployment_get", {"deployment_id": declared["id"]})

    assert body["declared"]["revision"]["runtime_version"] == "0.6.0"
    assert body["declared"]["revision"]["image_reference"] == "repo/vllm:tag"
