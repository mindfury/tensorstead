"""A peer's reason for refusing an image copy reaches the destination's error.

Issue #61: the destination called ``raise_for_status()`` on the streamed
response without reading it, so ``ImageDistributionError`` -- and therefore the
operation record -- carried only ``Client error '404 Not Found'``. The holder
had sent a ``code`` and ``message`` saying why; neither arrived.

Real TLS for the reason ``test_image_transfer_ceiling.py`` gives: the subject
is what crosses the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.image_distribution import ImageDistributionError, ImageDistributionService
from tests.fakes.tls_ca import TLSPeerServer, issue_private_ca

pytestmark = pytest.mark.integration

_REFERENCE = "nvcr.io/nvidia/tensorrt-llm/release@sha256:" + "1" * 64
_IMAGE_ID = "sha256:" + "1" * 64


class _UnusedEngine:
    def import_image(self, **_kwargs: Any) -> str:
        raise AssertionError("a refused fetch must not load anything")


def _refused_by(tmp_path: Path, body: bytes, status: int) -> ImageDistributionError:
    ca_pem, cert_pem, key_pem = issue_private_ca(tmp_path)
    store = tmp_path / "images"
    service = ImageDistributionService(_UnusedEngine(), str(store), str(ca_pem))
    with (
        TLSPeerServer(body, cert_pem, key_pem, status=status) as endpoint,
        pytest.raises(ImageDistributionError) as caught,
    ):
        service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )
    assert not list(store.glob(".staging-*")), "a refused fetch left a staged archive"
    return caught.value


def _agent_error(code: str, message: str) -> bytes:
    return json.dumps({"detail": {"code": code, "message": message, "detail": {}}}).encode()


def test_the_peers_code_and_message_reach_the_error(tmp_path: Path) -> None:
    body = _agent_error("image_export_failed", "content digest sha256:abc: not found")

    error = _refused_by(tmp_path, body, 500)

    assert "HTTP 500 image_export_failed" in error.message
    assert "content digest sha256:abc: not found" in error.message


def test_a_refusal_without_a_json_body_still_names_the_status(tmp_path: Path) -> None:
    error = _refused_by(tmp_path, b"<html>bad gateway</html>", 502)

    assert "HTTP 502" in error.message
    assert "<html>" not in error.message, "an unparsed body was echoed into the record"


def test_a_long_peer_message_is_clipped(tmp_path: Path) -> None:
    error = _refused_by(tmp_path, _agent_error("image_export_failed", "x" * 50_000), 500)

    assert len(error.message) < 2_000, "an untrusted peer could flood the operation record"
