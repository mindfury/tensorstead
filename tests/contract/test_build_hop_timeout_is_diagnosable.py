"""A build that outlives the caller's bound must still say what happened (084).

The official Flash-Next runtime compiles vLLM from source. Submitted through the
managed path it died at **1800.129 seconds** — the coordinator→agent build hop's
bound, to a tenth of a second — and the operation recorded:

    code    image_build_failed
    message The read operation timed out
    detail  {}
    per_node_outcomes  null

Which names no node, no phase, and no bound, and is indistinguishable from a
build that never started. The image was then produced out of band and imported,
so the estate's own TP=2 candidate pins an artifact whose provenance runs
through a tar file instead of the recorded spec the coordinator holds. A number
caused exactly the record-and-reality gap this product exists to close.

The cause was that `httpx.TimeoutException` is not an `AgentCallError`, so
nothing between the socket and the operation record classified it — the same
defect one layer up, on the caller's side of the same hop.

Two things are asserted here, and the second matters more than the first:

- the bound is named in the failure, so an operator can tell "too slow" from
  "never started";
- the failure says **whether work is still running on the node**, because a read
  timeout leaves the agent compiling, and a blind retry races a build that is
  still going.

The bound itself is exercised at 0.001s rather than four hours: what is under
test is the shape of the record, not the duration.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tensorstead.coordinator.node_http import NodeHTTPClient
from tensorstead.domain.models import Node
from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import FakeNodeClient, build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_PINNED = "nvcr.io/nvidia/vllm@sha256:0123456789abcdef"
_REFERENCE = "local/vllm-flash-next-official:d4d703c"


def _node() -> Node:
    from datetime import datetime

    return Node(
        id="01M05TMJ2ESS9KKRN1Q0MFJZC4",
        name="spark-alpha.internal",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.14",
        agent_cert_fingerprint="sha256:abc",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )


def _timeout_error(exc: httpx.TimeoutException) -> Any:
    """The client's own classification of a transport timeout."""
    return NodeHTTPClient._timeout_error(_node(), "/agent/v1/images:build", 1800.0, exc)


def test_a_read_timeout_names_the_bound_it_hit() -> None:
    """ "The read operation timed out" told an operator nothing about which bound."""
    error = _timeout_error(httpx.ReadTimeout("The read operation timed out"))

    assert error.code == "agent_timeout"
    assert "1800" in error.message
    assert "spark-alpha.internal" in error.message
    assert error.detail["timeout_seconds"] == 1800.0
    assert error.detail["path"] == "/agent/v1/images:build"


def test_a_read_timeout_warns_that_the_node_is_probably_still_building() -> None:
    """The operationally important half: a retry would race a live compile."""
    error = _timeout_error(httpx.ReadTimeout("timed out"))

    assert error.detail["work_may_still_be_running"] is True
    assert "image_list" in error.message, "the operator was not told what to check"


def test_a_connect_timeout_says_the_work_never_started() -> None:
    """Distinguishable, because one is safe to retry and the other is not."""
    error = _timeout_error(httpx.ConnectTimeout("connect timed out"))

    assert error.detail["work_may_still_be_running"] is False
    assert "retry is safe" in error.message


def test_the_timeout_reaches_the_operation_record_diagnosably() -> None:
    """End to end: the whole point is that the *operation* carries this.

    The defect was never in the exception; it was that nothing between the
    socket and the operation record classified it, so `_record_failure` wrote
    `str(exc)` and an empty detail.
    """

    class _TimingOutClient(FakeNodeClient):
        def build_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
            raise NodeHTTPClient._timeout_error(
                node, "/agent/v1/images:build", 1800.0, httpx.ReadTimeout("timed out")
            )

    app, _ = build_test_coordinator()
    app.state.image_builds._client = _TimingOutClient(FakeNodeAgent())
    client = TestClient(app)

    node_id = client.post(
        "/v1/nodes",
        json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]
    client.put(
        "/v1/buildspecs/flash-next",
        json={"base_image": _PINNED, "steps": ["make -j2"]},
        headers=_AUTH,
    )
    accepted = client.post(
        "/v1/images:build",
        json={"spec": "flash-next", "node_id": node_id, "reference": _REFERENCE},
        headers=_AUTH,
    )
    assert accepted.status_code == 202, accepted.text

    operation = poll_operation(client, accepted.json()["operation_id"], auth=_AUTH)

    assert operation["state"] == "failed"
    reason = operation["failure_reason"]
    # The three facts the 2026-09-05 record was missing.
    assert "1800" in reason["message"], reason["message"]
    assert "spark-alpha" in reason["message"], reason["message"]
    assert reason["detail"].get("work_may_still_be_running") is True, reason["detail"]
    assert reason["detail"], "detail was empty, which is the defect this closes"
    # And it must still name what was being built.
    assert reason["detail"].get("reference") == _REFERENCE
    assert reason["detail"].get("spec") == "flash-next"


def test_the_build_bound_is_above_a_from_source_compile() -> None:
    """1800s was a judgement; the compile that broke it took 1800.129s.

    Asserted as a floor rather than an exact value so the number can be revised
    by measurement without editing a test, while a revision back down to
    something a source build cannot fit fails here.
    """
    from tensorstead.coordinator.node_http import _BUILD_TIMEOUT_SECONDS

    assert _BUILD_TIMEOUT_SECONDS > 1800.0, "the bound that was measured as too small"
    # 7200s is headroom over the measured lower bound, not a fitted value: the
    # only fact in evidence is that the compile ran past 1800s without
    # finishing. Nobody knows what it actually needs, which is why the tests
    # above -- about the record a timeout leaves -- matter more than this one.
    assert _BUILD_TIMEOUT_SECONDS >= 7200.0, "generous headroom over the one measured number"
