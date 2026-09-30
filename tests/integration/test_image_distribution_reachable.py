"""Building once and copying is reachable from outside the test suite.

`ImageBuildService.build_and_distribute` was written, unit-tested,
and then called by nothing. No route, no CLI, no MCP tool reached it — the
orchestration that makes a multi-node deployment possible existed only as
something the tests could see.

That is why a tensor-parallel deployment on a locally built image was not
possible. Both nodes must run the *same* image identifier; building separately
on each produces a different one for the same spec, and the resulting digest
comparison is silently meaningless.

The design notes drew the lesson after the first real build: the tests
"stopped at 'the build returns an identifier' and never asked whether anything
could then use it". This follows the artifact to its purpose.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


def _build(client: TestClient, payload: dict) -> dict:
    """POST a build and poll its operation to terminal.

    A build now returns 202 with an operation id and runs on a background
    thread, so a test that posts and immediately reads races the work. Some of
    these tests passed only because the fake engine is instant and the GIL
    happened to schedule the executor first -- a latent flake, and the same
    race ``poll_operation`` was written for when model acquire went async.
    """
    resp = client.post("/v1/images:build", json=payload, headers=_AUTH)
    assert resp.status_code == 202, resp.text
    return poll_operation(client, resp.json()["operation_id"], auth=_AUTH)


def _two_nodes(client: TestClient) -> list[str]:
    ids = []
    for name, host in (("spark-alpha", "10.0.0.11"), ("spark-beta", "10.0.0.12")):
        response = client.post(
            "/v1/nodes",
            json={"name": name, "agent_endpoint": f"https://{host}:8443"},
            headers=_AUTH,
        )
        assert response.status_code == 201, response.text
        ids.append(str(response.json()["id"]))
    return ids


def _record_spec(client: TestClient) -> None:
    response = client.put(
        "/v1/buildspecs/vllm-patched",
        json={"base_image": "vllm/vllm-openai:v0.6.0", "steps": ["pip install xgrammar==0.2.1"]},
        headers=_AUTH,
    )
    assert response.status_code in (200, 201), response.text


def test_a_multi_node_build_produces_one_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the whole operation exists for.

    One image id across both nodes. Two independent builds would give two, and
    every later digest comparison would be comparing unlike things while
    reporting agreement.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_ids = _two_nodes(client)
    _record_spec(client)

    operation = _build(
        client, {"spec": "vllm-patched", "nodes": node_ids, "reference": "local/vllm:patched"}
    )

    assert operation["state"] == "succeeded", operation
    per_node = operation["per_node_outcomes"]
    assert set(per_node) == set(node_ids)
    identifiers = {entry["image_id"] for entry in per_node.values()}
    assert len(identifiers) == 1, f"the same spec produced {len(identifiers)} identifiers"
    assert per_node[node_ids[0]]["status"] == "built"
    assert per_node[node_ids[1]]["status"] == "distributed"


def test_a_single_node_build_still_takes_the_simple_path() -> None:
    """The existing shape is unchanged; `nodes` is additive."""
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_ids = _two_nodes(client)
    _record_spec(client)

    operation = _build(
        client, {"spec": "vllm-patched", "node_id": node_ids[0], "reference": "local/vllm:one"}
    )

    assert operation["state"] == "succeeded", operation
    # A single-node build distributes to nobody, so it reports no per-node map.
    assert not operation["per_node_outcomes"]


def test_a_node_that_cannot_receive_the_image_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial success is failure, and it says which node.

    The nodes that did receive it are left alone rather than unwound: the
    product never reverses work it has done on a host without being asked.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_ids = _two_nodes(client)
    _record_spec(client)

    service = app.state.image_builds
    original = service._client.distribute_image

    def refuse(node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("peer refused the archive")

    service._client.distribute_image = refuse  # type: ignore[method-assign]
    try:
        operation = _build(
            client, {"spec": "vllm-patched", "nodes": node_ids, "reference": "local/vllm:x"}
        )
    finally:
        service._client.distribute_image = original  # type: ignore[method-assign]

    # A partial build is a failed operation, not a succeeded one that happens to
    # mention a failure in its detail.
    assert operation["state"] == "failed", operation
    reason = operation["failure_reason"]
    assert reason["code"] == "image_build_failed"
    assert reason["detail"]["failed_nodes"], "a failure that names no node is not actionable"
    assert operation["per_node_outcomes"][node_ids[0]]["status"] == "built", (
        "the node that succeeded was unwound; the product does not reverse work unasked"
    )


def test_the_built_image_provenance_survives_to_image_list() -> None:
    """The artifact's recorded origin is discoverable, not just built.

    The design's lesson after the first real build was that the tests "stopped at
    'the build returns an identifier' and never asked whether anything could then
    use it". This follows the artifact one step further: after a produce-once-
    then-copy build, `image list` must show the image on both nodes, and each
    entry must carry the origin and the spec that produced it. The origin was
    recorded but was not surfaced on `image list`, so the acceptance criterion -- "100%
    of managed images report a digest and an origin" -- was true in the store
    and false on the surface. Asserting the provenance here is what keeps that
    from happening again.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_ids = _two_nodes(client)
    _record_spec(client)

    operation = _build(
        client, {"spec": "vllm-patched", "nodes": node_ids, "reference": "local/vllm:patched"}
    )
    assert operation["state"] == "succeeded", operation
    image_id = operation["per_node_outcomes"][node_ids[0]]["image_id"]

    images = client.get("/v1/images", headers=_AUTH).json()
    by_node = {img["node_id"]: img for img in images}
    assert set(by_node) == set(node_ids), "the image should be present on both nodes"

    for node_id in node_ids:
        entry = by_node[node_id]
        assert entry["reference"] == "local/vllm:patched"
        assert entry["digest"] == image_id, (
            f"node {node_id} lists digest {entry.get('digest')!r}, "
            f"but the build produced {image_id!r}"
        )
        assert entry["origin"] == "built", (
            f"node {node_id} origin is {entry.get('origin')!r}, not 'built'"
        )
        assert entry["produced_by"] == "vllm-patched"
        assert entry["is_registry_digest"] is False, (
            "a locally built image presented as a registry digest"
        )


def test_a_build_spec_can_make_the_image_self_starting() -> None:
    """The mechanism is recorded, the content stays pinned.

    An image that must prepare itself before serving does that in its own
    ENTRYPOINT. Recording it in the build spec keeps the *mechanism*
    provenance-tracked while whatever it reads at start — a file shipped inside
    the acquired checkpoint — stays pinned to the model revision rather than
    frozen into the image.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)

    response = client.put(
        "/v1/buildspecs/dspark-vllm",
        json={
            "base_image": "ghcr.io/anemll/dspark-vllm-gx10:0.1.1",
            "steps": ["pip install --no-cache-dir some-dep"],
            "entrypoint": ["/usr/local/bin/dspark-entrypoint"],
        },
        headers=_AUTH,
    )

    assert response.status_code in (200, 201), response.text
    assert response.json()["entrypoint"] == ["/usr/local/bin/dspark-entrypoint"]


def test_a_build_spec_describes_itself_the_same_way_everywhere() -> None:
    """Recording one and listing it must not disagree about its fields.

    `record_spec` returned `entrypoint` unconditionally while `list_specs`
    never returned it, so the same object described itself differently
    depending on which call you made. Noticed reading `buildspec list` against
    the live store immediately after deploying the column — small, and the same
    shape as every larger instance of a record that stopped matching another.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)

    plain = client.put(
        "/v1/buildspecs/plain",
        json={"base_image": "repo/base@sha256:" + "0" * 64, "steps": ["true"]},
        headers=_AUTH,
    ).json()
    starting = client.put(
        "/v1/buildspecs/self-starting",
        json={
            "base_image": "repo/base@sha256:" + "0" * 64,
            "steps": ["true"],
            "entrypoint": ["/entrypoint.sh"],
        },
        headers=_AUTH,
    ).json()

    listed = {s["name"]: s for s in client.get("/v1/buildspecs", headers=_AUTH).json()}

    assert plain.keys() == listed["plain"].keys(), (
        "recording a spec and listing it describe the same object differently"
    )
    assert starting.keys() == listed["self-starting"].keys()
    assert "entrypoint" not in plain, "a spec using the image's own entrypoint says nothing"
    assert "entrypoint" not in listed["plain"]
    assert starting["entrypoint"] == ["/entrypoint.sh"]
    assert listed["self-starting"]["entrypoint"] == ["/entrypoint.sh"]
