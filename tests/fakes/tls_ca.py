"""A private CA and an HTTPS server signed by it, for real-TLS tests.

Extracted from ``test_peer_transfer_trusts_the_managed_ca.py``
so a second peer client (``HTTPPeerClient``) can
be tested against the same genuine TLS trust boundary rather than either
duplicating ~150 lines of X.509 extension handling or going untested.
"""

from __future__ import annotations

import datetime
import http.server
import ssl
import threading
from pathlib import Path
from typing import Any


def issue_private_ca(directory: Path) -> tuple[Path, Path, Path]:
    """A CA the public trust store has never heard of, and a cert it signed.

    Returns ``(ca_pem, server_cert_pem, server_key_pem)``.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "tensorstead-test-ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        # Modern OpenSSL refuses a chain whose CA carries no subject key
        # identifier ("Missing Authority Key Identifier"), so a CA without these
        # would fail for a reason unrelated to trust and make the test below
        # pass for the wrong one.
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        # ...and one that permits signing certificates. Without it OpenSSL
        # rejects the chain with "CA cert does not include key usage extension",
        # which again is not the trust failure this test is about.
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = directory / "ca.pem"
    ca_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_pem = directory / "server.pem"
    cert_pem.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_pem = directory / "server.key"
    key_pem.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return ca_pem, cert_pem, key_pem


class TLSPeerServer:
    """An HTTPS server standing in for a peer agent's content endpoint.

    Serves ``body`` unconditionally on any path -- the caller under test
    supplies the URL, and this only needs to be a real TLS endpoint behind
    it, not a faithful route implementation.
    """

    def __init__(self, body: bytes, cert: Path, key: Path) -> None:
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: Any) -> None:
                return

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert), keyfile=str(key))
        self._server.socket = context.wrap_socket(self._server.socket, server_side=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        # "localhost" rather than the bound address: the certificate's subject
        # names it, and hostname verification is part of what is under test.
        return f"https://localhost:{self._server.server_address[1]}"

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
