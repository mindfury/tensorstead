"""The managed-image surface through the real coordinator app.

The unit tests drive the service directly. This exercises the API the way a
client does, so a route wired to the wrong service method, or a payload key
that does not match, fails here rather than on an appliance.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

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


_PINNED = "nvcr.io/nvidia/vllm@sha256:0123456789abcdef"
_XGRAMMAR_FIX = "python3 -m pip install --no-deps xgrammar==0.2.1 apache-tvm-ffi==0.1.9"


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


def test_recording_a_spec_stores_it_without_building(client: TestClient) -> None:
    """The real case: repairing NVIDIA's xgrammar packaging."""
    resp = client.put(
        "/v1/buildspecs/vllm-xgrammar",
        json={"base_image": _PINNED, "steps": [_XGRAMMAR_FIX]},
        headers=_AUTH,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["base_is_pinned"] is True
    assert "warning" not in body

    listed = client.get("/v1/buildspecs", headers=_AUTH).json()
    assert [s["name"] for s in listed] == ["vllm-xgrammar"]
    assert listed[0]["steps"] == [_XGRAMMAR_FIX]


def test_an_unpinned_base_is_recorded_with_a_warning(client: TestClient) -> None:
    """Reported, never refused."""
    resp = client.put(
        "/v1/buildspecs/loose",
        json={"base_image": "nvcr.io/nvidia/vllm:26.07-py3", "steps": [_XGRAMMAR_FIX]},
        headers=_AUTH,
    )

    assert resp.status_code == 200
    assert resp.json()["base_is_pinned"] is False
    assert "not reproducible" in resp.json()["warning"]
    assert client.get("/v1/buildspecs", headers=_AUTH).json()[0]["name"] == "loose"


def test_building_reports_an_image_id_and_provenance(client: TestClient) -> None:
    """Never presented as a registry digest.

    The build response used to carry this. It now returns an operation id
    so the assertion follows the provenance to its durable
    home: the image record, which is where an operator reads it anyway and
    where it survives the operation being pruned.
    """
    node_id = _node(client)
    client.put("/v1/buildspecs/s", json={"base_image": _PINNED, "steps": []}, headers=_AUTH)

    operation = _build(client, {"spec": "s", "node_id": node_id, "reference": "local/vllm:patched"})
    assert operation["state"] == "succeeded", operation

    images = client.get("/v1/images", headers=_AUTH).json()
    image = next(i for i in images if i["reference"] == "local/vllm:patched")
    assert image["origin"] == "built"
    assert image["produced_by"] == "s"
    # The distinction in the form the inventory expresses it. The build *response* used
    # to omit `digest` entirely and name the value `image_id`; the image record
    # carries the identifier under `digest` with the flag that says what it is
    # not. The flag is the load-bearing part -- it is what lets an operator tell
    # a built image from a pulled one.
    assert image["is_registry_digest"] is False
    assert image["digest"], "a built image still reports its identifier"


def test_a_built_image_appears_in_the_image_inventory(client: TestClient) -> None:
    """It must be discoverable, which is what `image list` is for.

    The success criterion requires every managed image to report a digest *and an
    origin*. The origin is recorded but was not surfaced by `image list`, so the
    success criterion was true in the store and false on the surface -- an
    operator could not tell a pulled image from one built from a named spec, the
    distinction the design makes load-bearing. Asserting `reference` alone let the
    gap survive; the assertion now follows the provenance to the surface.
    """
    node_id = _node(client)
    client.put("/v1/buildspecs/s", json={"base_image": _PINNED, "steps": []}, headers=_AUTH)
    _build(client, {"spec": "s", "node_id": node_id, "reference": "local/vllm:patched"})

    images = client.get("/v1/images", headers=_AUTH).json()
    matched = [i for i in images if i.get("reference") == "local/vllm:patched"]
    assert matched, "the built image is not in the inventory"
    image = matched[0]
    assert image["origin"] == "built", (
        f"origin recorded as {image.get('origin')!r}, not surfaced as 'built'"
    )
    assert image["produced_by"] == "s"
    assert image["is_registry_digest"] is False, (
        "a locally built image presented as a registry digest"
    )


def test_every_image_in_the_inventory_reports_an_origin(client: TestClient) -> None:
    """The success criterion: 100% of managed images report a digest and an origin.

    A future record added without surfacing its origin would silently drop the
    field; this fails rather than letting absence read as a default.
    """
    node_id = _node(client)
    client.put("/v1/buildspecs/s", json={"base_image": _PINNED, "steps": []}, headers=_AUTH)
    _build(client, {"spec": "s", "node_id": node_id, "reference": "local/vllm:a"})

    images = client.get("/v1/images", headers=_AUTH).json()
    assert images, "at least one image is expected"
    for image in images:
        assert image.get("origin") in ("pulled", "built", "imported"), (
            f"image {image.get('reference')!r} reports no origin: {image}"
        )
        assert "digest" in image, f"image {image.get('reference')!r} reports no digest: {image}"


def test_building_an_unrecorded_spec_is_a_clean_refusal(client: TestClient) -> None:
    node_id = _node(client)

    resp = client.post(
        "/v1/images:build",
        json={"spec": "absent", "node_id": node_id, "reference": "x"},
        headers=_AUTH,
    )

    assert resp.status_code == 404
    assert "absent" in resp.text


def test_deleting_a_spec_that_produced_nothing_succeeds(client: TestClient) -> None:
    client.put("/v1/buildspecs/s", json={"base_image": _PINNED, "steps": []}, headers=_AUTH)

    assert client.delete("/v1/buildspecs/s", headers=_AUTH).status_code == 200
    assert client.get("/v1/buildspecs", headers=_AUTH).json() == []


def test_every_managed_image_route_requires_authentication(client: TestClient) -> None:
    """The surface is gated like every other management route."""
    for method, path in [
        ("get", "/v1/buildspecs"),
        ("put", "/v1/buildspecs/x"),
        ("delete", "/v1/buildspecs/x"),
        ("post", "/v1/images:build"),
        ("post", "/v1/images:import"),
    ]:
        call = getattr(client, method)
        response = call(path) if method in ("get", "delete") else call(path, json={})
        assert response.status_code in (401, 403), f"{method} {path} was not gated"
