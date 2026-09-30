"""Hardware test: two-node serving.

A deployment spanning two real appliances, coordinated by the product and
executed by a runtime that distributes. This is the test that proves it on
metal rather than against fakes.

**Safety** (tests/hardware/README.md): every resource this creates carries a
run-unique `tensorstead-test-` name, a pre-existing name is a failure rather than
something to adopt, and teardown removes only what this run made. Nothing here
enumerates or touches resources it did not create.

Skipped unless `--hardware` **and** `TENSORSTEAD_HARDWARE_ENABLED=1` **and** two
nodes are configured. A missing second node skips rather than degrading to a
single-node run — a distribution test that quietly stops testing distribution
while still reporting green is worse than no test at all.

This test was authored without access to the hardware it targets. Expect the
first real run to need adjustment; that first run is the point of it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not os.environ.get("TENSORSTEAD_HARDWARE_ENABLED"),
        reason="hardware tests require TENSORSTEAD_HARDWARE_ENABLED=1",
    ),
]

_RUN_ID = f"{int(time.time())}-{os.getpid()}"
_NAME = f"tensorstead-test-multinode-{_RUN_ID}"

_TIMEOUT_SECONDS = 600
_POLL_SECONDS = 5


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


@pytest.fixture(scope="module")
def client() -> Any:
    """An HTTP client bound to the coordinator under test."""
    import httpx

    api = _env("TENSORSTEAD_API")
    token = _env("TENSORSTEAD_MGMT_TOKEN")
    if not api:
        pytest.skip("TENSORSTEAD_API is not set")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(base_url=api, headers=headers, timeout=60.0) as http:
        yield http


@pytest.fixture(scope="module")
def two_nodes(client: Any) -> tuple[str, str]:
    """Resolve two registered nodes by name, or skip."""
    name_a, name_b = _env("TENSORSTEAD_TEST_NODE"), _env("TENSORSTEAD_TEST_NODE_B")
    if not name_a or not name_b:
        pytest.skip("two nodes required: set TENSORSTEAD_TEST_NODE and TENSORSTEAD_TEST_NODE_B")

    nodes = client.get("/v1/nodes").json()
    by_name = {n["name"]: n["id"] for n in nodes}
    missing = {name_a, name_b} - set(by_name)
    if missing:
        pytest.skip(f"nodes not registered with the coordinator: {sorted(missing)}")
    return by_name[name_a], by_name[name_b]


@pytest.fixture
def deployment(client: Any, two_nodes: tuple[str, str]) -> Iterator[str]:
    """Create the test deployment; always remove it, including on failure."""
    node_a, node_b = two_nodes

    # Rule 2: a pre-existing name is a collision, never something to adopt.
    existing = {d["declared"]["name"] for d in client.get("/v1/deployments").json()}
    assert _NAME not in existing, (
        f"a deployment named {_NAME!r} already exists; refusing to touch a resource "
        f"this run did not create (tests/hardware/README.md)"
    )

    model_id = _acquire_model(client, [node_a, node_b])
    created = client.post(
        "/v1/deployments",
        json={
            "name": _NAME,
            "model_id": model_id,
            "runtime_type": os.environ.get("TENSORSTEAD_TEST_RUNTIME", "vllm"),
            "runtime_version": os.environ.get("TENSORSTEAD_TEST_RUNTIME_VERSION", "latest"),
            "image_reference": os.environ["TENSORSTEAD_TEST_IMAGE"],
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": [node_a, node_b],
            "endpoint": os.environ["TENSORSTEAD_TEST_ENDPOINT"],
        },
    )
    assert created.status_code == 202, created.text
    dep_id = _find_deployment_id(client, _NAME)

    try:
        yield dep_id
    finally:
        # Rule 4: teardown removes only this run's resource, failure or not.
        client.delete(f"/v1/deployments/{dep_id}")


def _acquire_model(client: Any, nodes: list[str]) -> str:
    """Acquire the configured test model onto both nodes (acquire once, replicate)."""
    source_model_id = os.environ["TENSORSTEAD_TEST_MODEL"]
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": os.environ.get("TENSORSTEAD_TEST_SOURCE", "huggingface"),
            "source_model_id": source_model_id,
            "revision": _env("TENSORSTEAD_TEST_REVISION"),
            "nodes": nodes,
        },
    )
    assert resp.status_code == 202, resp.text
    _await_operation(client, resp.json()["operation_id"])

    models = client.get("/v1/models").json()
    return str(next(m["id"] for m in models if m["source_model_id"] == source_model_id))


def _find_deployment_id(client: Any, name: str) -> str:
    for deployment in client.get("/v1/deployments").json():
        if deployment["declared"]["name"] == name:
            return str(deployment["declared"]["id"])
    raise AssertionError(f"deployment {name!r} was created but not found")


def _await_operation(client: Any, operation_id: str) -> dict[str, Any]:
    """Poll one operation to a terminal state, or fail with its reason."""
    deadline = time.time() + _TIMEOUT_SECONDS
    while time.time() < deadline:
        operation = client.get(f"/v1/operations/{operation_id}").json()
        if operation["state"] in ("succeeded", "failed"):
            assert operation["state"] == "succeeded", (
                f"operation {operation['kind']} failed: {operation.get('failure_reason')}"
            )
            return dict(operation)
        time.sleep(_POLL_SECONDS)
    raise AssertionError(f"operation {operation_id} did not finish in {_TIMEOUT_SECONDS}s")


def test_two_node_deployment_reaches_a_serving_state(
    client: Any, deployment: str, two_nodes: tuple[str, str]
) -> None:
    """Start across both appliances and confirm both are serving."""
    node_a, node_b = two_nodes

    started = client.post(f"/v1/deployments/{deployment}:start")
    assert started.status_code == 202, started.text
    _await_operation(client, started.json()["operation_id"])

    observed = client.get(f"/v1/deployments/{deployment}/status").json()["observed"]

    # Every participating node reports for itself.
    assert set(observed["per_node"]) == {node_a, node_b}
    for node_id, node_observed in observed["per_node"].items():
        assert node_observed["status"] == "running", (
            f"node {node_id} is {node_observed['status']}: {node_observed.get('detail')}"
        )
    assert observed["status"] == "running"


def test_all_participating_nodes_are_reported(client: Any, deployment: str) -> None:
    """Declared state names both nodes, and observation covers both."""
    body = client.get(f"/v1/deployments/{deployment}").json()

    declared_nodes = body["declared"]["revision"]["participating_nodes"]
    assert len(declared_nodes) == 2
    assert set(body["observed"]["per_node"]) == set(declared_nodes)


def test_the_endpoint_is_reachable_on_the_serving_node(client: Any, deployment: str) -> None:
    """The runtime formed its own distributed group and is answering.

    The product asserts reachability of the endpoint it recorded; it does not
    send an inference request, because it is never on that path.
    """
    started = client.post(f"/v1/deployments/{deployment}:start")
    if started.status_code == 202:
        _await_operation(client, started.json()["operation_id"])

    observed = client.get(f"/v1/deployments/{deployment}/status").json()["observed"]
    reachable = [n.get("endpoint_reachable") for n in observed["per_node"].values()]
    assert any(reachable), f"no participating node reports a reachable endpoint: {observed}"


def test_stop_applies_to_every_participating_node(client: Any, deployment: str) -> None:
    """A lifecycle operation covers the whole deployment, not one node."""
    client.post(f"/v1/deployments/{deployment}:start")
    stopped = client.post(f"/v1/deployments/{deployment}:stop")
    assert stopped.status_code == 202, stopped.text
    _await_operation(client, stopped.json()["operation_id"])

    observed = client.get(f"/v1/deployments/{deployment}/status").json()["observed"]
    for node_id, node_observed in observed["per_node"].items():
        assert node_observed["status"] != "running", f"node {node_id} is still running"


@pytest.mark.skipif(
    os.environ.get("TENSORSTEAD_TEST_RUNTIME", "vllm") == "vllm",
    reason="needs a non-distributing runtime configured as TENSORSTEAD_TEST_RUNTIME",
)
def test_a_non_distributing_runtime_is_refused_on_two_nodes(
    client: Any, two_nodes: tuple[str, str]
) -> None:
    """llama.cpp across two nodes is refused on real hardware."""
    node_a, node_b = two_nodes
    resp = client.post(
        "/v1/deployments",
        json={
            "name": f"{_NAME}-refused",
            "model_id": _acquire_model(client, [node_a]),
            "runtime_type": "llamacpp",
            "runtime_version": os.environ.get("TENSORSTEAD_TEST_RUNTIME_VERSION", "latest"),
            "image_reference": os.environ["TENSORSTEAD_TEST_IMAGE"],
            "runtime_config": {},
            "participating_nodes": [node_a, node_b],
            "endpoint": os.environ["TENSORSTEAD_TEST_ENDPOINT"],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "runtime_not_distributed"
    # Nothing was created, so there is nothing to clean up.
    names = {d["declared"]["name"] for d in client.get("/v1/deployments").json()}
    assert f"{_NAME}-refused" not in names
