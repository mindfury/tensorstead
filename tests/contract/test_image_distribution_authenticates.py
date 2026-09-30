"""The image-replication hop authenticates itself.

``image_build(..., nodes=[a, b])` builds on one node and copies to the rest.
The destination agent fetches the archive from the source agent's
``GET /agent/v1/images/{reference}/content``, which is gated by the
**replication** token — deliberately, so an agent holding only that role can
fetch artifacts and issue no management instructions.

The coordinator sent the destination ``reference``, ``source_endpoint``, and
``expected_image_id``. It never sent a credential. The destination passed that
absent value through, and ``image_distribution`` sends an ``Authorization``
header only when the token is non-empty — so the peer request went out bare and
the source answered 401. Every multi-node image build would have failed its
distribution step, and a production agent cannot be configured open by accident:
``build_agent_app_from_env`` requires ``TENSORSTEAD_REPLICATION_TOKEN``.

Nothing caught it. ``FakeNodeClient.distribute_image`` returns the expected
identifier without contacting a source, and the unit tests drove a purpose-built
client. Neither has a source side, so neither has a source side that can refuse.

These tests use **distinct management and replication tokens** and a real source
agent, which is the arrangement under which the bug is visible at all: with one
shared token, or no token, the request would have succeeded either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.agent.image_distribution import ImageDistributionService
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_MANAGEMENT = "management-token"
_REPLICATION = "replication-token-quite-different"
_REFERENCE = "local/dspark-deepseek-v4-flash:0.1.1"
_IMAGE_ID = "sha256:distributed"


def _agent(tmp_path: Path, *, name: str) -> TestClient:
    engine = FakeContainerEngine()
    engine.local_images[_REFERENCE] = _IMAGE_ID
    root = tmp_path / name
    app = build_agent_app(
        management_token=_MANAGEMENT,
        replication_token=_REPLICATION,
        container_engine=engine,
        service_manager=FakeServiceManager(),
        store_dir=root / "store",
        marker_dir=root / "markers",
    )
    app.state.image_store_path = str(root / "images")
    return TestClient(app)


def test_the_source_refuses_a_fetch_with_no_credential(tmp_path: Path) -> None:
    """The gate the distribution hop was walking into blind."""
    source = _agent(tmp_path, name="source")

    resp = source.get(f"/agent/v1/images/{_REFERENCE}/content")

    assert resp.status_code == 401


def test_the_source_refuses_the_management_token(tmp_path: Path) -> None:
    """Role separation: management authority does not fetch artifacts."""
    source = _agent(tmp_path, name="source")

    resp = source.get(
        f"/agent/v1/images/{_REFERENCE}/content",
        headers={"Authorization": f"Bearer {_MANAGEMENT}"},
    )

    assert resp.status_code == 401


def test_the_destination_sends_its_own_replication_credential(tmp_path: Path) -> None:
    """The fix: the token comes from the agent's own state, not the request.

    Captured at the outbound peer call, because that is where the empty header
    was produced. Asserting on the request body would only prove the coordinator
    stopped sending something it never sent.
    """
    sent: dict[str, Any] = {}

    class _CapturingDistribution(ImageDistributionService):
        def pull_from_peer(self, **kwargs: Any) -> str:
            sent.update(kwargs)
            return str(kwargs["expected_image_id"])

    destination = _agent(tmp_path, name="destination")
    destination.app.state.image_distribution = _CapturingDistribution(
        FakeContainerEngine(), str(tmp_path / "dest-images")
    )

    resp = destination.post(
        "/agent/v1/images:distribute",
        json={
            "reference": _REFERENCE,
            "source_endpoint": "https://10.0.0.11:8443",
            "expected_image_id": _IMAGE_ID,
        },
        headers={"Authorization": f"Bearer {_MANAGEMENT}"},
    )

    assert resp.status_code == 200, resp.text
    assert sent["token"] == _REPLICATION, (
        f"the peer fetch would go out with {sent['token']!r}; the source requires "
        f"the replication token and answers 401 without it"
    )


def test_the_coordinator_cannot_supply_the_credential(tmp_path: Path) -> None:
    """A credential must not travel through the coordinator to reach this hop.

    Both agents already hold it. Accepting one here would put a secret into
    coordinator request bodies, logs, and operation records to authenticate a
    hop between two parties that are each already authorised.
    """
    destination = _agent(tmp_path, name="destination")
    # Installed so that reintroducing the field fails on the assertion below
    # rather than on whatever the default service touches on its way out.
    destination.app.state.image_distribution = ImageDistributionService(
        FakeContainerEngine(), str(tmp_path / "dest-images")
    )

    resp = destination.post(
        "/agent/v1/images:distribute",
        json={
            "reference": _REFERENCE,
            "source_endpoint": "https://10.0.0.11:8443",
            "expected_image_id": _IMAGE_ID,
            "replication_token": "smuggled",
        },
        headers={"Authorization": f"Bearer {_MANAGEMENT}"},
    )

    assert resp.status_code == 422, (
        "the agent hop is extra='forbid'; a credential field must be refused, "
        "not accepted and ignored"
    )
