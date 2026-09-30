"""A destination failure survives to the terminal operation record.

The fix for the defect was verified in two places that never met: the agent returns a
structured ``image_distribution_failed`` body, and ``build_and_distribute``
retains a code when one is raised at it. Nothing asserted the **join** — that a
real agent response, translated by the real ``NodeHTTPClient``, reaches
``per_node_outcomes`` on the terminal ``image_build`` operation with both its
code and its actionable message intact.

That is the same gap this batch keeps finding: each half tested, the seam
between them assumed. The fix was declined on exactly those grounds, and that
was right.

The chain exercised here is real at every step except the socket:

1. the **real agent app** produces the 502 body, from a real
   ``ImageDistributionError`` raised by the distribution service;
2. the **real** ``NodeHTTPClient._error_from_response`` translates that response
   into an ``AgentCallError`` — this is the step that could flatten a structured
   failure into ``http_error`` carrying a JSON blob, and once did;
3. the **real** ``ImageBuildService.build_and_distribute`` records the per-node
   outcome;
4. the **real** coordinator route records the terminal operation.

Only the TCP connection is replaced, by handing the agent's own response object
to the coordinator's own translator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.agent.image_distribution import ImageDistributionError
from tensorstead.coordinator.node_http import NodeHTTPClient
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.node_agent import FakeNodeAgent
from tests.fakes.service_manager import FakeServiceManager
from tests.helpers import FakeNodeClient, build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_PINNED = "nvcr.io/nvidia/vllm@sha256:0123456789abcdef"
_TLS_FAILURE = (
    "could not fetch 'local/dspark-deepseek-v4-flash:0.1.1' from "
    "https://spark-alpha.internal:8443: [SSL: CERTIFICATE_VERIFY_FAILED] "
    "certificate verify failed: unable to get local issuer certificate"
)


def _real_agent_refusal(tmp_path: Path, message: str) -> httpx.Response:
    """The actual HTTP response a destination agent returns for a failed fetch.

    Produced by the real agent app and its real exception handler rather than
    hand-written, so if that handler's shape changes this test changes with it
    instead of asserting a shape nothing produces any more.
    """
    app = build_agent_app(
        management_token="mgmt",
        replication_token="repl",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )

    class _Refusing:
        def pull_from_peer(self, **kwargs: Any) -> str:
            raise ImageDistributionError(message, reference=str(kwargs.get("reference", "")))

    app.state.image_distribution = _Refusing()
    resp = TestClient(app).post(
        "/agent/v1/images:distribute",
        json={
            "reference": "local/dspark-deepseek-v4-flash:0.1.1",
            "source_endpoint": "https://spark-alpha.internal:8443",
            "expected_image_id": "sha256:b1c50db6",
        },
        headers={"Authorization": "Bearer mgmt"},
    )
    assert resp.status_code == 502, resp.text
    return httpx.Response(resp.status_code, json=resp.json())


def _coordinator_whose_destination_refuses(tmp_path: Path, message: str) -> TestClient:
    """A coordinator whose distribute hop fails the way a real agent fails it."""
    refusal = _real_agent_refusal(tmp_path, message)

    class _RefusingClient(FakeNodeClient):
        def distribute_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
            # The coordinator's own translation of the agent's own response.
            raise NodeHTTPClient(management_token="t")._error_from_response(node, refusal)

    app, _ = build_test_coordinator()
    agent = FakeNodeAgent()
    app.state.image_builds._client = _RefusingClient(agent)
    return TestClient(app)


def _two_nodes(client: TestClient) -> list[str]:
    ids = []
    for name, host in (("spark-alpha", "10.0.0.11"), ("spark-beta", "10.0.0.12")):
        resp = client.post(
            "/v1/nodes",
            json={"name": name, "agent_endpoint": f"https://{host}:8443"},
            headers=_AUTH,
        )
        assert resp.status_code == 201, resp.text
        ids.append(str(resp.json()["id"]))
    return ids


def _build(client: TestClient, nodes: list[str]) -> dict:
    client.put("/v1/buildspecs/dspark", json={"base_image": _PINNED, "steps": []}, headers=_AUTH)
    resp = client.post(
        "/v1/images:build",
        json={
            "spec": "dspark",
            "nodes": nodes,
            "reference": "local/dspark-deepseek-v4-flash:0.1.1",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202, resp.text
    return poll_operation(client, resp.json()["operation_id"], auth=_AUTH)


def test_the_operation_retains_the_destination_code(tmp_path: Path) -> None:
    """``image_distribution_failed`` must survive to ``per_node_outcomes``."""
    client = _coordinator_whose_destination_refuses(tmp_path, _TLS_FAILURE)
    nodes = _two_nodes(client)

    operation = _build(client, nodes)

    assert operation["state"] == "failed", operation
    outcome = operation["per_node_outcomes"][nodes[1]]
    assert outcome["status"] == "failed"
    assert outcome["code"] == "image_distribution_failed", (
        f"the destination's code was lost between the agent and the operation: {outcome}"
    )


def test_the_operation_retains_the_actionable_message(tmp_path: Path) -> None:
    """The reason is what makes a retry decidable — a code alone is not enough.

    This exact message is what finally identified the real blocker: the
    destination did not trust the source's certificate. Had it
    been flattened, that would still be unknown.
    """
    client = _coordinator_whose_destination_refuses(tmp_path, _TLS_FAILURE)
    nodes = _two_nodes(client)

    operation = _build(client, nodes)

    detail = operation["per_node_outcomes"][nodes[1]]["detail"]
    assert "CERTIFICATE_VERIFY_FAILED" in detail, (
        f"the destination's reason did not survive to the operation: {detail!r}"
    )
    assert "Internal Server Error" not in detail


def test_the_source_node_outcome_is_unaffected(tmp_path: Path) -> None:
    """The node that built it still reports what it produced.

    Partial success is an overall failure, and the work already done is neither
    unwound nor erased from the record.
    """
    client = _coordinator_whose_destination_refuses(tmp_path, _TLS_FAILURE)
    nodes = _two_nodes(client)

    operation = _build(client, nodes)

    source = operation["per_node_outcomes"][nodes[0]]
    assert source["status"] == "built"
    assert source["image_id"], "the source's produced identifier was lost"


def test_the_failure_reason_names_the_failed_node(tmp_path: Path) -> None:
    """An operator must be able to see *which* node without reading every outcome."""
    client = _coordinator_whose_destination_refuses(tmp_path, _TLS_FAILURE)
    nodes = _two_nodes(client)

    operation = _build(client, nodes)

    reason = operation["failure_reason"]
    assert reason["code"] == "image_build_failed"
    assert "spark-beta" in reason["message"], reason
    assert reason["detail"]["failed_nodes"] == ["spark-beta"]
