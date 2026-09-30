"""The agent can remove an image, and say which of three things happened.

Before this the agent had ``pull``, ``build``, ``import``, ``probe``, ``serve`` and
``distribute``, and no way to remove anything. The engine had ``remove_image`` and
a docstring stating that an operator-facing delete "does not reach the daemon at
all". So ``image delete`` deleted a coordinator row, printed "Image removed from
node", and freed nothing — 48 times, across 430 GB.

The coordinator's decision depends on telling three outcomes apart, so each one
is pinned here:

* removed — the node freed it, and the record may go;
* not present — nothing to free, and the record may still go (it had outlived
  its object, which is how the estate reached 50 records against 25 objects);
* in use — the bytes are still there, so the record must **not** go, and the
  refusal has to carry a code the coordinator can act on rather than a bare 500.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_REFERENCE = "local/vllm:26.07-xgrammar-0.2.1"
_DIGEST = "sha256:" + "d8" * 32


def _client(tmp_path: Path, engine: FakeContainerEngine) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def test_removing_a_held_image_reports_removed(tmp_path: Path) -> None:
    """The call the product never made."""
    engine = FakeContainerEngine()
    engine.local_images[_REFERENCE] = _DIGEST
    client = _client(tmp_path, engine)

    resp = client.post("/agent/v1/images:remove", json={"reference": _REFERENCE}, headers=_AUTH)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["removed"] is True
    assert body["digest"] == _DIGEST
    assert engine.removed_images == [_REFERENCE], "the daemon must actually be asked"
    assert _REFERENCE not in engine.local_images


def test_removing_an_absent_image_is_a_success_that_removed_nothing(tmp_path: Path) -> None:
    """Idempotent, and honest about it.

    The coordinator reaps its record on this answer, so it must be a success —
    but ``removed: false`` is what stops it reporting a removal that did not
    happen, which is the wording defect this finding is named for.
    """
    engine = FakeContainerEngine()
    client = _client(tmp_path, engine)

    resp = client.post("/agent/v1/images:remove", json={"reference": _REFERENCE}, headers=_AUTH)

    assert resp.status_code == 200, resp.text
    assert resp.json()["removed"] is False
    assert engine.removed_images == []


def test_an_image_a_container_holds_is_refused_with_a_code(tmp_path: Path) -> None:
    """Refused, not forced, and the reason survives the hop.

    ``force=True`` is right for unwinding a rejected import and wrong here: the
    operator asked to delete an image something is still running from, and the
    honest answer is that they cannot yet. A bare 500 would strand the
    coordinator, which must keep its record on exactly this outcome.
    """
    engine = FakeContainerEngine()
    engine.local_images[_REFERENCE] = _DIGEST
    engine.in_use_images.add(_REFERENCE)
    client = _client(tmp_path, engine)

    resp = client.post("/agent/v1/images:remove", json={"reference": _REFERENCE}, headers=_AUTH)

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "image_in_use"
    assert engine.removed_images == [], "an in-use image must survive the refusal"
    assert _REFERENCE in engine.local_images


def test_removal_requires_the_management_token(tmp_path: Path) -> None:
    """Removing an image is a management call, not an artifact-replication one."""
    engine = FakeContainerEngine()
    engine.local_images[_REFERENCE] = _DIGEST
    client = _client(tmp_path, engine)

    resp = client.post("/agent/v1/images:remove", json={"reference": _REFERENCE})

    assert resp.status_code in (401, 403), resp.text
    assert engine.removed_images == []


def test_listing_reports_what_the_daemon_holds(tmp_path: Path) -> None:
    """The read that makes records checkable instead of merely trusted."""
    engine = FakeContainerEngine()
    engine.local_images[_REFERENCE] = _DIGEST
    engine.local_images["local/other:1"] = "sha256:" + "11" * 32
    client = _client(tmp_path, engine)

    resp = client.get("/agent/v1/images", headers=_AUTH)

    assert resp.status_code == 200, resp.text
    rows = {row["reference"]: row["digest"] for row in resp.json()}
    assert rows == {_REFERENCE: _DIGEST, "local/other:1": "sha256:" + "11" * 32}
