"""Hardware test: colocated vs. split deployment.

The same product path must complete in **both** arrangements:

1. **Self-contained** — coordinator, agent, and inference runtime all on one
   inference host. No separate management machine exists.
2. **Split** — the coordinator on a different host from the agent.

And it must behave **identically** in both. That symmetry is the requirement.
Neither arrangement may become a user-facing assumption: a
product that quietly works better colocated has an undocumented dependency on
colocation, and one that assumes separation cannot run on a single appliance at
all — which is the common case for someone with one box.

``test_no_colocation_assumption`` already asserts this structurally over the
source: no loopback default for an agent target, no shared-filesystem shortcut,
every agent call over the contract. This test is the behavioural counterpart —
the structural check cannot prove that the two arrangements *behave* the same,
only that no code path obviously distinguishes them.

**Safety** (tests/hardware/README.md): run-unique `tensorstead-test-` names, a
pre-existing name is a failure rather than something to adopt, and teardown
removes only what this run created.

Skipped unless ``--hardware``, ``TENSORSTEAD_HARDWARE_ENABLED=1``, and the relevant
coordinator URLs are configured. The split half additionally requires a
coordinator that is genuinely on another host; it skips rather than pretending,
because running both halves against the same endpoint would compare a thing to
itself and always pass.

Authored without access to the hardware it targets. Expect the first real run to
need adjustment.
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

_TIMEOUT_SECONDS = 600
_POLL_SECONDS = 5

# The two arrangements under test.
#   TENSORSTEAD_API_COLOCATED — a coordinator running *on* the inference host
#   TENSORSTEAD_API_SPLIT     — a coordinator on a different host from the agent
_ARRANGEMENTS = ("colocated", "split")


def _api_for(arrangement: str) -> str | None:
    return os.environ.get(f"TENSORSTEAD_API_{arrangement.upper()}")


def _client_for(arrangement: str) -> Any:
    import httpx

    api = _api_for(arrangement)
    if not api:
        pytest.skip(f"TENSORSTEAD_API_{arrangement.upper()} is not set")
    token = os.environ.get("TENSORSTEAD_MGMT_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=api, headers=headers, timeout=60.0)


def _await_operation(client: Any, operation_id: str) -> dict[str, Any]:
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


def _complete_us1_path(client: Any, arrangement: str) -> dict[str, Any]:
    """Register → acquire → create → start → observe. Returns the observed facts.

    Deliberately the *whole* product path rather than a smoke check: the question is
    whether the product behaves the same end to end, and a shortened path could
    miss a difference that only appears once a container is actually running.
    """
    name = f"tensorstead-test-{arrangement}-{_RUN_ID}"
    node_name = os.environ["TENSORSTEAD_TEST_NODE"]

    nodes = client.get("/v1/nodes").json()
    by_name = {n["name"]: n["id"] for n in nodes}
    if node_name not in by_name:
        pytest.skip(f"node {node_name!r} is not registered with the {arrangement} coordinator")
    node_id = by_name[node_name]

    existing = {d["declared"]["name"] for d in client.get("/v1/deployments").json()}
    assert name not in existing, (
        f"a deployment named {name!r} already exists; refusing to touch a resource this "
        f"run did not create (tests/hardware/README.md)"
    )

    source_model_id = os.environ["TENSORSTEAD_TEST_MODEL"]
    acquired = client.post(
        "/v1/models:acquire",
        json={
            "source_id": os.environ.get("TENSORSTEAD_TEST_SOURCE", "huggingface"),
            "source_model_id": source_model_id,
            "revision": os.environ.get("TENSORSTEAD_TEST_REVISION"),
            "nodes": [node_id],
        },
    )
    assert acquired.status_code == 202, acquired.text
    _await_operation(client, acquired.json()["operation_id"])

    models = client.get("/v1/models").json()
    model_id = next(m["id"] for m in models if m["source_model_id"] == source_model_id)

    created = client.post(
        "/v1/deployments",
        json={
            "name": name,
            "model_id": model_id,
            "runtime_type": os.environ.get("TENSORSTEAD_TEST_RUNTIME", "vllm"),
            "runtime_version": os.environ.get("TENSORSTEAD_TEST_RUNTIME_VERSION", "latest"),
            "image_reference": os.environ["TENSORSTEAD_TEST_IMAGE"],
            "runtime_config": {},
            "participating_nodes": [node_id],
            "endpoint": os.environ["TENSORSTEAD_TEST_ENDPOINT"],
        },
    )
    assert created.status_code == 202, created.text

    deployments = client.get("/v1/deployments").json()
    dep_id = next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == name)

    started = client.post(f"/v1/deployments/{dep_id}:start")
    assert started.status_code == 202, started.text
    _await_operation(client, started.json()["operation_id"])

    body = client.get(f"/v1/deployments/{dep_id}").json()
    export = client.get(f"/v1/deployments/{dep_id}/export").json()
    return {
        "deployment_id": dep_id,
        "name": name,
        "declared": body["declared"],
        "observed": body["observed"],
        "export": export,
    }


@pytest.fixture(params=_ARRANGEMENTS)
def arrangement_result(request: pytest.FixtureRequest) -> Iterator[dict[str, Any]]:
    """Complete the product path in one arrangement; always clean up afterwards."""
    arrangement = request.param
    client = _client_for(arrangement)
    result: dict[str, Any] | None = None
    try:
        result = _complete_us1_path(client, arrangement)
        result["arrangement"] = arrangement
        yield result
    finally:
        if result is not None:
            client.delete(f"/v1/deployments/{result['deployment_id']}")
        client.close()


def test_us1_completes_in_this_arrangement(arrangement_result: dict[str, Any]) -> None:
    """The whole path works, colocated or split."""
    observed = arrangement_result["observed"]
    assert observed["status"] == "running", (
        f"{arrangement_result['arrangement']}: deployment is {observed['status']}"
    )
    assert any(n.get("endpoint_reachable") for n in observed["per_node"].values()), (
        f"{arrangement_result['arrangement']}: no node reports a reachable endpoint"
    )


def test_no_management_step_needed_a_separate_machine(
    arrangement_result: dict[str, Any],
) -> None:
    """The colocated run proves no separate management host is required.

    Recorded explicitly because the failure mode is silent: a product that
    happens to work colocated but assumes separation in one code path fails only
    for the single-appliance operator, who is the common case.
    """
    if arrangement_result["arrangement"] != "colocated":
        pytest.skip("this claim is about the colocated arrangement")

    assert arrangement_result["declared"]["desired_state"] == "running"


def test_both_arrangements_produce_the_same_facts() -> None:
    """Colocated and split are indistinguishable in what they produce.

    Runs the path in both and compares the reproducibility-critical values. If
    the two coordinators are not genuinely on different hosts this skips, since
    comparing an endpoint with itself would pass while proving nothing.
    """
    colocated_api, split_api = _api_for("colocated"), _api_for("split")
    if not colocated_api or not split_api:
        pytest.skip("both TENSORSTEAD_API_COLOCATED and TENSORSTEAD_API_SPLIT must be set")
    if colocated_api == split_api:
        pytest.skip("the two arrangements point at the same coordinator; nothing to compare")

    results = {}
    clients = {}
    try:
        for arrangement in _ARRANGEMENTS:
            clients[arrangement] = _client_for(arrangement)
            results[arrangement] = _complete_us1_path(clients[arrangement], arrangement)

        first, second = (results[a]["export"] for a in _ARRANGEMENTS)

        # The four reproducibility values must match. The deployment *name*
        # differs by construction (run-unique per arrangement), which is why
        # this compares the substance rather than the whole document.
        assert first["model"]["revision"] == second["model"]["revision"]
        assert first["runtime"]["version"] == second["runtime"]["version"]
        assert first["image"]["digest"] == second["image"]["digest"]
        assert first["runtime_config"] == second["runtime_config"]

        # And observed state agrees.
        for arrangement in _ARRANGEMENTS:
            assert results[arrangement]["observed"]["status"] == "running"
    finally:
        for arrangement, result in results.items():
            clients[arrangement].delete(f"/v1/deployments/{result['deployment_id']}")
        for client in clients.values():
            client.close()
