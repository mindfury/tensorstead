"""The advisory endpoint pre-flight check.

The agent has implemented `GET /agent/v1/endpoint-check` for a while and the
coordinator never called it, so the requirement was dead. These pin the
property that makes it safe to call: it warns and can never refuse.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.service.deployments import DeploymentService

pytestmark = pytest.mark.unit


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.id = name


class _Repo:
    def __init__(self) -> None:
        self.nodes = {"n1": _Node("spark-01"), "n2": _Node("spark-02")}

    def get_node(self, node_id: str) -> Any:
        return self.nodes.get(node_id)


class _Client:
    def __init__(self, bound: bool | Exception) -> None:
        self.bound = bound
        self.calls: list[int] = []

    def check_endpoint(self, node: Any, port: int) -> dict[str, Any]:
        self.calls.append(port)
        if isinstance(self.bound, Exception):
            raise self.bound
        return {"port": port, "bound": self.bound}


def _service(client: Any) -> DeploymentService:
    return DeploymentService(_Repo(), {}, client)  # type: ignore[arg-type]


def test_a_bound_port_produces_a_warning_that_says_it_is_advisory() -> None:
    """It must never be represented as a guarantee."""
    warnings = _service(_Client(bound=True)).endpoint_preflight(["n1"], "spark-01:8000")

    assert len(warnings) == 1
    assert "8000" in warnings[0]
    assert "advisory only" in warnings[0]
    assert "created regardless" in warnings[0], (
        "the message must say the deployment was not blocked"
    )


def test_a_free_port_produces_no_warning() -> None:
    assert _service(_Client(bound=False)).endpoint_preflight(["n1"], "spark-01:8000") == []


def test_every_participating_node_is_checked() -> None:
    client = _Client(bound=True)
    warnings = _service(client).endpoint_preflight(["n1", "n2"], "x:8000")

    assert len(client.calls) == 2
    assert len(warnings) == 2


@pytest.mark.parametrize(
    "failure",
    [ConnectionError("unreachable"), TimeoutError("slow"), RuntimeError("agent too old")],
)
def test_a_check_that_cannot_be_performed_is_not_a_finding(failure: Exception) -> None:
    """An unreachable node during create is ordinary, not a warning."""
    assert _service(_Client(bound=failure)).endpoint_preflight(["n1"], "x:8000") == []


@pytest.mark.parametrize("endpoint", ["no-port", "host:", "host:notanumber", ""])
def test_an_unparseable_endpoint_is_skipped_rather_than_raising(endpoint: str) -> None:
    """A malformed endpoint must not turn an advisory check into a failure."""
    client = _Client(bound=True)
    assert _service(client).endpoint_preflight(["n1"], endpoint) == []
    assert client.calls == []


def test_without_a_node_client_the_check_is_simply_not_performed() -> None:
    """Its absence must never change whether a deployment can be created."""
    service = DeploymentService(_Repo(), {}, None)  # type: ignore[arg-type]
    assert service.endpoint_preflight(["n1"], "x:8000") == []


def test_an_unknown_node_is_skipped() -> None:
    client = _Client(bound=True)
    assert _service(client).endpoint_preflight(["ghost"], "x:8000") == []
    assert client.calls == []
