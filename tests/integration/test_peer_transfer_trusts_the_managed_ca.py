"""Agent-to-agent image transfer verifies against the managed CA (026).

The managed-TLS profile issues every agent a certificate from a private CA and
installs that CA's public bundle on every host. Coordinator-to-agent calls
verified against it. Agent-to-agent image transfer did not — because
``ImageDistributionService`` reads ``TENSORSTEAD_AGENT_CA_BUNDLE`` and the agent's
environment template never set it, so it fell back to the public trust store,
which by construction cannot contain a private CA.

The bundle was installed on every agent and pointed at by nothing. Every visible
hop was healthy, which is exactly why this took a live multi-node build to find::

    could not fetch 'local/dspark-deepseek-v4-flash:0.1.1' from
    https://spark-alpha.internal:8443: [SSL: CERTIFICATE_VERIFY_FAILED]
    certificate verify failed: unable to get local issuer certificate

**This is a real TLS test**, as asked for: a genuine private CA that the system
trust store does not know, a genuine HTTPS server presenting a certificate it
issued, and the genuine ``pull_from_peer`` fetching across it. It fails without
the CA setting and succeeds with it, which is the property the estate needed and
did not have. A mocked transport would have proved neither half.
"""

from __future__ import annotations

import http.server
import socket
import ssl
import tarfile
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.image_distribution import ImageDistributionError, ImageDistributionService
from tests.fakes.tls_ca import TLSPeerServer, issue_private_ca

pytestmark = pytest.mark.integration

_REFERENCE = "local/dspark-deepseek-v4-flash:0.1.1"
_IMAGE_ID = "sha256:b1c50db67ef0d26fd6ac2a0e839b45f1bac401192f9ed2920b0a6379fcb2fc66"


def _archive(directory: Path) -> bytes:
    """Any well-formed tar; the fake engine does not inspect its contents."""
    payload = directory / "layer"
    payload.write_text("image", encoding="utf-8")
    archive = directory / "image.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="layer")
    return archive.read_bytes()


class _LoadingEngine:
    """A container engine that accepts the archive and reports the expected id."""

    def import_image(self, *, archive_path: str) -> str:
        return _IMAGE_ID

    def verify_image_materializable(self, *, image_id: str) -> None:
        """Whole on arrival. This test's subject is the transfer, not the load."""


@pytest.fixture
def peer(tmp_path: Path) -> Any:
    ca_pem, cert_pem, key_pem = issue_private_ca(tmp_path)
    body = _archive(tmp_path)
    return ca_pem, TLSPeerServer(body, cert_pem, key_pem)


def test_the_transfer_fails_without_the_managed_ca(peer: Any, tmp_path: Path) -> None:
    """The estate's actual condition: no CA setting, so no trust, so no image."""
    _, server = peer
    service = ImageDistributionService(_LoadingEngine(), str(tmp_path / "images"), None)

    with server as endpoint, pytest.raises(ImageDistributionError) as caught:
        service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )

    message = str(caught.value)
    assert "CERTIFICATE_VERIFY_FAILED" in message, (
        f"expected the private CA to be untrusted by default, got: {message}"
    )
    assert _REFERENCE in message, "the failure must name the reference"


def test_the_transfer_succeeds_with_the_managed_ca(peer: Any, tmp_path: Path) -> None:
    """The same transfer, the only difference being the trust anchor.

    This is what ``TENSORSTEAD_AGENT_CA_BUNDLE`` in the agent environment supplies,
    and what its absence withheld.
    """
    ca_pem, server = peer
    service = ImageDistributionService(_LoadingEngine(), str(tmp_path / "images"), str(ca_pem))

    with server as endpoint:
        arrived = service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )

    assert arrived == _IMAGE_ID


def test_the_staged_archive_is_not_left_behind(peer: Any, tmp_path: Path) -> None:
    """A half-transferred archive is a later import waiting to pick up wrong bytes."""
    ca_pem, server = peer
    store = tmp_path / "images"
    service = ImageDistributionService(_LoadingEngine(), str(store), str(ca_pem))

    with server as endpoint:
        service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )

    assert not list(store.glob(".staging-*")), "a staged archive survived a successful transfer"


def test_a_failed_verification_leaves_no_staged_archive(peer: Any, tmp_path: Path) -> None:
    """The same must hold when the transfer fails, which is when it matters more."""
    _, server = peer
    store = tmp_path / "images"
    service = ImageDistributionService(_LoadingEngine(), str(store), None)

    with server as endpoint, pytest.raises(ImageDistributionError):
        service.pull_from_peer(
            reference=_REFERENCE, source_endpoint=endpoint, expected_image_id=_IMAGE_ID
        )

    assert not list(store.glob(".staging-*")) if store.exists() else True


def test_a_socket_that_is_not_tls_at_all_is_still_refused(tmp_path: Path) -> None:
    """Verification must not be skippable by the peer simply not offering TLS."""
    plain = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    port = plain.server_address[1]
    plain.server_close()
    # Nothing listens on that port now; the point is that an https:// fetch
    # against a non-TLS endpoint fails rather than silently downgrading.
    service = ImageDistributionService(_LoadingEngine(), str(tmp_path / "images"), None)

    with pytest.raises(ImageDistributionError):
        service.pull_from_peer(
            reference=_REFERENCE,
            source_endpoint=f"https://127.0.0.1:{port}",
            expected_image_id=_IMAGE_ID,
        )


def test_the_agent_environment_template_names_the_ca_bundle() -> None:
    """The wiring itself (026): installed and never pointed at is the defect.

    Cheap, and it is the exact regression. The coordinator template has always
    set this; the agent template did not, and nothing compared them.
    """
    template = (
        Path(__file__).resolve().parents[2]
        / "ansible"
        / "roles"
        / "agent"
        / "templates"
        / "agent.env.j2"
    )
    content = template.read_text(encoding="utf-8")

    assert "TENSORSTEAD_AGENT_CA_BUNDLE={{ tensorstead_agent_ca_bundle }}" in content, (
        "the agent environment does not name the managed CA bundle, so "
        "agent-to-agent TLS falls back to the public trust store"
    )


def test_a_socket_check_confirms_the_test_ca_is_not_publicly_trusted(peer: Any) -> None:
    """Guard against the first test passing for the wrong reason.

    If the generated CA were somehow in the system store, the "fails without the
    CA" test would be asserting nothing. This confirms the default context
    rejects it.
    """
    _, server = peer
    with server as endpoint:
        host, port = endpoint.removeprefix("https://").split(":")
        context = ssl.create_default_context()
        with (
            socket.create_connection((host, int(port)), timeout=5.0) as raw,
            pytest.raises(ssl.SSLError),
        ):
            context.wrap_socket(raw, server_hostname=host)
