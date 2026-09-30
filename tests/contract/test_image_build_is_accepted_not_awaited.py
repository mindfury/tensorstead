"""An image build returns before it finishes.

``image_build`` ran the whole build and distribution inside the request. A build
routinely takes many minutes on a Spark, so the MCP client's read window expired
and reported::

    coordinator_unreachable: The read operation timed out

which is indistinguishable from a dead coordinator. The caller could not tell
which of three things had happened:

1. the request was never accepted, and retrying is safe;
2. it was accepted and is still running, so retrying starts a *second*
   concurrent build or races the first one's distribution phase;
3. it was accepted and terminally failed, and the real error is unread.

``operation_list`` could not disambiguate it either, because a synchronous build
registered no operation at all. There was nothing to poll.

This is the defect fixed for ``model_acquire``, in the same file,
found again because that fix was applied to the operation that had failed rather
than to the class of long-running operations. The CLI had even papered over its
own half by raising its client timeout to an hour -- honest for the CLI, useless
for every other client, and a comment there already described the product
"telling an operator a lie about work that succeeded".

**What these tests can and cannot prove.** The obvious test — hold the build and
assert the POST returns quickly — cannot be written here, and finding out why is
worth recording. ``TestClient`` drives the app through a blocking portal that
waits for work scheduled on the loop's default executor, so a held build blocks
the *client* even though the route has already returned. Measured: with the work
held for six seconds, ``model_acquire`` blocks for six seconds too, and it has
been correctly asynchronous since that fix. A wall-clock assertion through
this harness would therefore fail against a correct route and pass against
nothing.

So these assert the properties that *are* observable in-process: the response
shape, that the work runs on a different thread than the request (which is what
"not awaited inline" means mechanically), and that the operation is registered
and pollable — the thing whose absence made the outcome ambiguous.

The wall-clock property needs a live coordinator over real HTTP. It is not
asserted here rather than asserted misleadingly.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_PINNED = "nvcr.io/nvidia/vllm@sha256:0123456789abcdef"


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _node(client: TestClient) -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _spec(client: TestClient) -> None:
    client.put("/v1/buildspecs/s", json={"base_image": _PINNED, "steps": []}, headers=_AUTH)


def test_the_build_runs_off_the_request_thread(client: TestClient) -> None:
    """The defect itself, in the form this harness can actually observe.

    "Accepted, not awaited" means mechanically that the build executes somewhere
    other than the request. A synchronous route runs it on the request's own
    thread; this must not. See the module docstring for why the wall-clock form
    of this assertion is untestable through ``TestClient``.
    """
    node_id = _node(client)
    _spec(client)

    ran_on: dict[str, int] = {}
    service = client.app.state.image_builds  # type: ignore[attr-defined]
    original = service._client.build_image

    def record(node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        ran_on["build"] = threading.get_ident()
        return original(node, payload)  # type: ignore[no-any-return]

    service._client.build_image = record  # type: ignore[method-assign]
    try:
        resp = client.post(
            "/v1/images:build",
            json={"spec": "s", "node_id": node_id, "reference": "local/vllm:threaded"},
            headers=_AUTH,
        )
        assert resp.status_code == 202, resp.text
        assert resp.json()["operation_id"], "accepted with no id is nothing to poll"
        poll_operation(client, resp.json()["operation_id"], auth=_AUTH)
    finally:
        service._client.build_image = original  # type: ignore[method-assign]

    assert ran_on.get("build"), "the build never ran"
    assert ran_on["build"] != threading.get_ident(), (
        "the build ran on the caller's thread; the route is still doing the work inline"
    )


def test_the_accepted_operation_names_what_it_is_building(client: TestClient) -> None:
    """Polling a fresh operation must show which build it is, not an opaque id.

    This is what makes the ambiguity answerable: an operator who lost a response
    can list operations and see the reference they submitted.
    """
    node_id = _node(client)
    _spec(client)

    resp = client.post(
        "/v1/images:build",
        json={"spec": "s", "node_id": node_id, "reference": "local/vllm:named"},
        headers=_AUTH,
    )
    operation = poll_operation(client, resp.json()["operation_id"], auth=_AUTH)

    assert operation["kind"] == "image_build"
    assert operation["progress"]["reference"] == "local/vllm:named"
    assert operation["progress"]["spec"] == "s"


def test_the_build_appears_in_the_operation_list(client: TestClient) -> None:
    """``operation_list`` returned nothing for a build, so nothing could be polled."""
    node_id = _node(client)
    _spec(client)

    resp = client.post(
        "/v1/images:build",
        json={"spec": "s", "node_id": node_id, "reference": "local/vllm:listed"},
        headers=_AUTH,
    )
    operation_id = resp.json()["operation_id"]
    poll_operation(client, operation_id, auth=_AUTH)

    listed = client.get("/v1/operations", headers=_AUTH).json()
    assert any(op["id"] == operation_id for op in listed), (
        f"the build operation is absent from operation_list: {listed}"
    )
    assert any(op["kind"] == "image_build" for op in listed)


def test_an_unrecorded_spec_is_refused_before_acceptance(client: TestClient) -> None:
    """Outcome 1 stays a clean refusal rather than becoming a poll.

    A name that was never recorded is knowable without doing any work, so
    accepting it and failing asynchronously would turn a 404 into a round trip
    and put a failure in the history for work that never started.
    """
    node_id = _node(client)

    resp = client.post(
        "/v1/images:build",
        json={"spec": "absent", "node_id": node_id, "reference": "x"},
        headers=_AUTH,
    )

    assert resp.status_code == 404
    assert "absent" in resp.text
    assert not client.get("/v1/operations", headers=_AUTH).json(), (
        "a refused request recorded an operation"
    )
