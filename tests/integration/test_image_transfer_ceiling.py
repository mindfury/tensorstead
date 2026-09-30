"""Inbound image transfer is bounded.

``pull_from_peer`` wrote every received chunk to disk with no ceiling at all --
a compromised or unauthenticated peer could stream until the managed image
store filled, and the request timeout bounds duration, not bytes. This is a
real-TLS test for the same reason ``test_peer_transfer_trusts_the_managed_ca.py``
is: the thing under test is what actually crosses the wire, and a mocked
transport would only prove the mock was mockable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent import image_distribution as image_distribution_module
from tensorstead.agent.image_distribution import ImageDistributionError, ImageDistributionService
from tests.fakes.tls_ca import TLSPeerServer, issue_private_ca

pytestmark = pytest.mark.integration

_REFERENCE = "local/oversized-image:0.1.1"
_IMAGE_ID = "sha256:0000000000000000000000000000000000000000000000000000000000000"


class _LoadingEngine:
    def import_image(self, *, archive_path: str) -> str:
        return _IMAGE_ID

    def verify_image_materializable(self, *, image_id: str) -> None:
        """Whole on arrival. This test's subject is the transfer, not the load."""


def test_pull_from_peer_refuses_a_body_larger_than_the_configured_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSORSTEAD_MAX_ARTIFACT_BYTES", "10")
    ca_pem, cert_pem, key_pem = issue_private_ca(tmp_path)
    body = b"this body is well past a ten byte ceiling"
    server = TLSPeerServer(body, cert_pem, key_pem)
    store = tmp_path / "images"
    service = ImageDistributionService(_LoadingEngine(), str(store), str(ca_pem))

    with server as endpoint, pytest.raises(ImageDistributionError) as caught:
        service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )

    assert "ceiling" in str(caught.value)
    assert not list(store.glob(".staging-*")), "a staged archive survived an aborted transfer"


def test_pull_from_peer_refuses_to_start_when_free_disk_is_below_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight: refused before any connection, not partway through one.

    Against a real near-full disk reading, not an absurdly high configured
    ceiling: comparing free space to the ceiling itself was the bug review
    caught, refusing an ordinary small
    transfer on any host with less free space than the (200 GiB, generous,
    worst-case) default. The preflight's own threshold is a small fixed
    floor now, so this simulates a disk genuinely below it.
    """
    tiny_free = image_distribution_module._MIN_FREE_BYTES_TO_ATTEMPT - 1

    def _fake_disk_usage(_path: Any) -> Any:
        return type("Usage", (), {"free": tiny_free, "total": 0, "used": 0})()

    monkeypatch.setattr(image_distribution_module.shutil, "disk_usage", _fake_disk_usage)
    store = tmp_path / "images"
    service = ImageDistributionService(_LoadingEngine(), str(store), None)

    with pytest.raises(ImageDistributionError) as caught:
        service.pull_from_peer(
            reference=_REFERENCE,
            # Nothing needs to listen here: a genuine preflight never dials out.
            source_endpoint="https://127.0.0.1:1",
            expected_image_id=_IMAGE_ID,
        )

    assert "free" in str(caught.value).lower()


def test_a_transfer_under_the_ceiling_is_unaffected(tmp_path: Path) -> None:
    """The guard must not tax the ordinary case it was not built for."""
    ca_pem, cert_pem, key_pem = issue_private_ca(tmp_path)
    server = TLSPeerServer(b"small image body", cert_pem, key_pem)
    service = ImageDistributionService(_LoadingEngine(), str(tmp_path / "images"), str(ca_pem))

    with server as endpoint:
        arrived = service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )

    assert arrived == _IMAGE_ID
