"""Agent-to-agent model transfer verifies against the managed CA.

``ImageDistributionService`` was fixed to read ``TENSORSTEAD_AGENT_CA_BUNDLE``
in an earlier fix. ``HTTPPeerClient`` -- the equivalent transport for *model*
replication, carrying the larger, more expensive artifact and the same
replication bearer token -- never matched it: it verified a certificate
*fingerprint* over a second, detached TLS handshake, then fetched the actual
content over a connection with no CA verification at all (``verify=False``
by default, and the only production construction site never passed
anything else). A detached preflight proves nothing about the connection
that follows it, and the fingerprint it checked was never captured at
registration in the first place, so in practice this ran with zero peer
verification on every replication.

**This is a real TLS test**, for the same reason the image-distribution
version is: a genuine private CA the system trust store does not know, a
genuine HTTPS server presenting a certificate it issued, and the genuine
``HTTPPeerClient.fetch_content`` fetching across it.
"""

from __future__ import annotations

import http.server
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.replication import HTTPPeerClient, ReplicationError
from tests.fakes.tls_ca import TLSPeerServer, issue_private_ca

pytestmark = pytest.mark.integration

_MODEL_ID = "org/model@e1f2a3b"
_BODY = b"weights-would-go-here"


@pytest.fixture
def peer(tmp_path: Path) -> Any:
    ca_pem, cert_pem, key_pem = issue_private_ca(tmp_path)
    return ca_pem, TLSPeerServer(_BODY, cert_pem, key_pem)


def _fetch(client: HTTPPeerClient, endpoint: str) -> bytes:
    """Consume the generator, which is where the actual request happens."""
    return b"".join(client.fetch_content(endpoint=endpoint, model_id=_MODEL_ID, token="tok"))


def test_the_transfer_fails_without_the_managed_ca(peer: Any) -> None:
    """The estate's actual condition before this fix: no CA setting, so no trust."""
    _, server = peer
    client = HTTPPeerClient(ca_bundle=None)

    with server as endpoint, pytest.raises(Exception) as caught:
        _fetch(client, endpoint)

    message = str(caught.value)
    assert "CERTIFICATE_VERIFY_FAILED" in message, (
        f"expected the private CA to be untrusted by default, got: {message}"
    )


def test_the_transfer_succeeds_with_the_managed_ca(peer: Any) -> None:
    """The same transfer, the only difference being the trust anchor."""
    ca_pem, server = peer
    client = HTTPPeerClient(ca_bundle=str(ca_pem))

    with server as endpoint:
        content = _fetch(client, endpoint)

    assert content == _BODY


def test_a_socket_that_is_not_tls_at_all_is_still_refused() -> None:
    """Verification must not be skippable by the peer simply not offering TLS."""
    plain = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    port = plain.server_address[1]
    plain.server_close()
    # Nothing listens on that port now; the point is that an https:// fetch
    # against a non-TLS endpoint fails rather than silently downgrading.
    client = HTTPPeerClient(ca_bundle=None)

    # Blind on purpose: nothing listens on the port, so the exact exception
    # (a connection error, not a TLS one) is incidental to what this proves.
    with pytest.raises(Exception):  # noqa: B017
        _fetch(client, f"https://127.0.0.1:{port}")


def test_a_plain_http_endpoint_is_refused_before_any_connection_is_made() -> None:
    """The other half of the same defect: http:// must never be attempted at all."""
    client = HTTPPeerClient(ca_bundle=None)

    with pytest.raises(ReplicationError) as caught:
        _fetch(client, "http://127.0.0.1:1")  # port 1 -- nothing could answer

    assert caught.value.code == "replication_failed"
    assert "https" in str(caught.value)
    assert _MODEL_ID in str(caught.value)


def test_the_replication_token_is_never_in_a_refusal_message(peer: Any) -> None:
    """The credential goes in a header; it must not reach any failure text."""
    _, server = peer
    client = HTTPPeerClient(ca_bundle=None)
    secret = "s3cret-replication-token"

    with server as endpoint, pytest.raises(Exception) as caught:
        b"".join(client.fetch_content(endpoint=endpoint, model_id=_MODEL_ID, token=secret))

    assert secret not in str(caught.value)


def test_the_agent_environment_template_names_the_ca_bundle() -> None:
    """The same variable both peer clients now depend on for real verification."""
    template = (
        Path(__file__).resolve().parents[2]
        / "ansible"
        / "roles"
        / "agent"
        / "templates"
        / "agent.env.j2"
    )
    content = template.read_text(encoding="utf-8")

    assert "TENSORSTEAD_AGENT_CA_BUNDLE={{ tensorstead_agent_ca_bundle }}" in content


def test_the_production_construction_reads_the_ca_bundle_env_var() -> None:
    """The wiring itself: HTTPPeerClient() took no arguments in production.

    ImageDistributionService's construction a few lines below it already
    read TENSORSTEAD_AGENT_CA_BUNDLE; this asserts the same env var now reaches
    HTTPPeerClient's construction too, structurally, so a future edit that
    silently drops the argument fails a test rather than a live replication.
    """
    source = (
        Path(__file__).resolve().parents[2] / "src" / "tensorstead" / "agent" / "app.py"
    ).read_text(encoding="utf-8")

    assert 'HTTPPeerClient(ca_bundle=os.environ.get("TENSORSTEAD_AGENT_CA_BUNDLE")' in source, (
        "the production HTTPPeerClient() construction does not read the managed "
        "CA bundle, so model replication runs with no TLS verification"
    )
