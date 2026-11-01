"""A ranged replication between two real agents over real TLS.

The unit tests drive the protocol in process. This drives it the way the estate
does: the genuine agent app under uvicorn with a certificate from a private CA,
and the genuine ``HTTPPeerClient`` -- whose range path is ``http.client``, not
``httpx`` -- fetching across it, verifying that CA, reusing its connections,
and falling back to the archive protocol against a peer that predates the
manifest.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import Depends, FastAPI

from tensorstead.agent import replication as replication_module
from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.app import build_agent_app
from tensorstead.agent.replication import (
    HTTPPeerClient,
    PeerReplicationService,
    ReplicationError,
    directory_digest,
)
from tests.fakes.tls_ca import issue_private_ca

pytestmark = pytest.mark.integration

_MODEL = "huggingface:org/model"


@contextmanager
def _serving(app: FastAPI, cert: Path, key: Path) -> Iterator[str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, ssl_certfile=str(cert), ssl_keyfile=str(key), log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            thread.join(0.01)
            assert thread.is_alive(), "the test server did not start"
        yield f"https://localhost:{port}"
    finally:
        server.handle_exit(signal.SIGINT, None)
        thread.join(10)
        sock.close()


@pytest.fixture
def source(tmp_path: Path) -> tuple[ModelAcquisitionService, str, Path, Path, Path]:
    ca, cert, key = issue_private_ca(tmp_path)
    acquisition = ModelAcquisitionService(
        store_dir=tmp_path / "src" / "s", marker_dir=tmp_path / "src" / "m"
    )
    staging = acquisition.staging_path(_MODEL)
    (staging / "shards").mkdir(parents=True)
    for index in range(3):
        (staging / "shards" / f"model-{index}.safetensors").write_bytes(os.urandom(300_000))
    (staging / "config.json").write_bytes(b"{}")
    digest = directory_digest(staging)
    acquisition.promote(_MODEL, staging, content_digest=digest)
    return acquisition, digest, ca, cert, key


def _agent(acquisition: ModelAcquisitionService) -> FastAPI:
    return build_agent_app(
        management_token="mgmt",
        replication_token="repl",
        acquisition=acquisition,
        model_source=object(),
        container_engine=object(),
        service_manager=object(),
    )


def _destination(tmp_path: Path, ca: Path) -> tuple[PeerReplicationService, Any]:
    acquisition = ModelAcquisitionService(
        store_dir=tmp_path / "dst" / "s", marker_dir=tmp_path / "dst" / "m"
    )
    return (
        PeerReplicationService(
            acquisition=acquisition, peer_client=HTTPPeerClient(ca_bundle=str(ca))
        ),
        acquisition,
    )


def test_a_model_is_copied_as_ranges_over_reused_tls_connections(
    tmp_path: Path, source: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition, digest, ca, cert, key = source
    monkeypatch.setattr(replication_module, "_PIECE", 64 * 1024)
    monkeypatch.setenv("TENSORSTEAD_REPLICATION_CONNECTIONS", "4")
    connections: list[Any] = []
    real_take = replication_module._HTTPPeerSession._take

    def _counting_take(self: Any) -> Any:
        taken = real_take(self)
        connections.append(taken[0])
        return taken

    monkeypatch.setattr(replication_module._HTTPPeerSession, "_take", _counting_take)
    app = _agent(acquisition)
    app.state.replication.stream_archive = _refuse_archive
    service, destination = _destination(tmp_path, ca)

    with _serving(app, cert, key) as endpoint:
        outcome = service.replicate(
            model_id=_MODEL,
            content_digest=digest,
            source_endpoint=endpoint,
            replication_token="repl",
        )

    assert (outcome.state, outcome.content_digest) == ("available", digest)
    assert directory_digest(destination.local_path(_MODEL)) == digest
    assert len(connections) >= 15, "every shard spans several ranges"
    assert len({id(conn) for conn in connections}) <= 4, "connections must be reused"


def _refuse_archive(_path: Any) -> Any:
    raise AssertionError("a current peer must not be copied over the archive protocol")


def test_a_peer_without_the_manifest_route_is_copied_from_its_archive(
    tmp_path: Path, source: tuple
) -> None:
    """An agent that predates the ranged protocol answers the manifest with a bare 404."""
    from tensorstead.agent.app import require_replication
    from tensorstead.agent.routes.replication import router

    acquisition, digest, ca, cert, key = source
    current = _agent(acquisition)
    old = FastAPI()
    old.state = current.state
    legacy = next(r for r in router.routes if getattr(r, "path", "").endswith("}/content"))
    old.add_api_route(
        legacy.path,  # type: ignore[attr-defined]
        legacy.endpoint,  # type: ignore[attr-defined]
        dependencies=[Depends(require_replication)],
    )
    service, _ = _destination(tmp_path, ca)

    with _serving(old, cert, key) as endpoint:
        outcome = service.replicate(
            model_id=_MODEL,
            content_digest=digest,
            source_endpoint=endpoint,
            replication_token="repl",
        )

    assert outcome.content_digest == digest


def test_a_current_peer_refusing_the_model_is_not_mistaken_for_an_old_one(
    tmp_path: Path, source: tuple
) -> None:
    acquisition, digest, ca, cert, key = source
    service, _ = _destination(tmp_path, ca)

    with (
        _serving(_agent(acquisition), cert, key) as endpoint,
        pytest.raises(ReplicationError) as caught,
    ):
        service.replicate(
            model_id="huggingface:not/held",
            content_digest=digest,
            source_endpoint=endpoint,
            replication_token="repl",
        )

    assert "HTTP 404" in caught.value.message


def test_the_wrong_token_is_refused(tmp_path: Path, source: tuple) -> None:
    acquisition, digest, ca, cert, key = source
    service, destination = _destination(tmp_path, ca)

    with _serving(_agent(acquisition), cert, key) as endpoint, pytest.raises(ReplicationError):
        service.replicate(
            model_id=_MODEL,
            content_digest=digest,
            source_endpoint=endpoint,
            replication_token="mgmt",
        )

    assert not destination.local_path(_MODEL).exists()


def test_the_range_connections_verify_the_managed_ca(tmp_path: Path, source: tuple) -> None:
    """The range path has its own TLS client; it must not trust what httpx would refuse."""
    acquisition, _, _, cert, key = source
    target = tmp_path / "out"
    target.write_bytes(b"\0" * 10)
    client = HTTPPeerClient(ca_bundle=None)

    with _serving(_agent(acquisition), cert, key) as endpoint:
        session = client.open_session(endpoint=endpoint, model_id=_MODEL, token="repl")
        try:
            with pytest.raises(Exception) as caught:
                session.fetch_range("config.json", 0, 2, target)
        finally:
            session.close()

    assert "CERTIFICATE_VERIFY_FAILED" in str(caught.value)
    assert target.read_bytes() == b"\0" * 10


# -- the content listener ----------------------------------------------------


def _listening_agent(acquisition: ModelAcquisitionService, cert: Path, key: Path) -> FastAPI:
    from tensorstead.agent.content_listener import ContentListenerConfig

    return build_agent_app(
        management_token="mgmt",
        replication_token="repl",
        acquisition=acquisition,
        model_source=object(),
        container_engine=object(),
        service_manager=object(),
        content_listener=ContentListenerConfig(
            host="127.0.0.1", port=0, certfile=str(cert), keyfile=str(key)
        ),
    )


def _record_ports(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    ports: list[int] = []
    real_connect = replication_module._HTTPPeerSession._connect

    def _recording(self: Any, port: int) -> Any:
        ports.append(port)
        return real_connect(self, port)

    monkeypatch.setattr(replication_module._HTTPPeerSession, "_connect", _recording)
    return ports


def _port_of(endpoint: str) -> int:
    return int(endpoint.rsplit(":", 1)[1])


def test_ranges_come_from_the_content_listener_when_the_source_runs_one(
    tmp_path: Path, source: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition, digest, ca, cert, key = source
    monkeypatch.setattr(replication_module, "_PIECE", 64 * 1024)
    monkeypatch.setenv("TENSORSTEAD_REPLICATION_CONNECTIONS", "4")
    ports = _record_ports(monkeypatch)
    app = _listening_agent(acquisition, cert, key)
    app.state.replication.stream_archive = _refuse_archive
    service, destination = _destination(tmp_path, ca)

    with _serving(app, cert, key) as endpoint:
        listener_port = app.state.content_listener.port
        outcome = service.replicate(
            model_id=_MODEL,
            content_digest=digest,
            source_endpoint=endpoint,
            replication_token="repl",
        )

    assert (outcome.state, outcome.content_digest) == ("available", digest)
    assert directory_digest(destination.local_path(_MODEL)) == digest
    assert ports and set(ports) == {listener_port}
    assert listener_port != _port_of(endpoint)
    assert 0 < len(ports) <= 4, "the listener's connections are kept alive and reused"
    # The listener lives exactly as long as the app is served.
    assert app.state.content_listener is None
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", listener_port), timeout=2).close()


def test_an_unreachable_content_port_falls_back_to_the_main_port(
    tmp_path: Path, source: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A firewall that admits only the main port costs speed, not the copy."""
    from types import SimpleNamespace

    acquisition, digest, ca, cert, key = source
    ports = _record_ports(monkeypatch)
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()
    app = _agent(acquisition)
    app.state.content_listener = SimpleNamespace(port=closed_port)
    service, _ = _destination(tmp_path, ca)

    with _serving(app, cert, key) as endpoint:
        outcome = service.replicate(
            model_id=_MODEL,
            content_digest=digest,
            source_endpoint=endpoint,
            replication_token="repl",
        )

    assert outcome.content_digest == digest
    assert ports[0] == closed_port
    assert set(ports) == {closed_port, _port_of(endpoint)}
    # Connections opened concurrently may each try the content port once before
    # the first fallback lands; it is never retried per range.
    assert ports.count(closed_port) <= replication_module._DEFAULT_CONNECTIONS


@pytest.fixture
def listener(source: tuple) -> Iterator[tuple[int, Path]]:
    acquisition, _, ca, cert, key = source
    app = _listening_agent(acquisition, cert, key)
    with _serving(app, cert, key):
        yield app.state.content_listener.port, ca


def _ask(
    port: int, ca: Path, path: str, *, method: str = "GET", headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    import http.client
    import ssl

    conn = http.client.HTTPSConnection(
        "localhost", port, context=ssl.create_default_context(cafile=str(ca)), timeout=10
    )
    try:
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


_FILE = f"/agent/v1/models/{_MODEL}/content/file?path=config.json"
_TOKEN = {"Authorization": "Bearer repl"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer mgmt"},
        {"Authorization": "Bearer nope"},
        {"Authorization": "Basic repl"},
    ],
)
def test_the_listener_requires_the_replication_token(
    listener: tuple[int, Path], headers: dict[str, str]
) -> None:
    port, ca = listener
    status, _, body = _ask(port, ca, _FILE, headers=headers)
    assert status == 401
    assert b'"unauthorized"' in body


def test_the_listener_serves_a_range_with_the_main_ports_headers(
    listener: tuple[int, Path],
) -> None:
    port, ca = listener
    status, headers, body = _ask(port, ca, _FILE, headers={**_TOKEN, "Range": "bytes=0-0"})
    assert (status, body) == (206, b"{")
    assert headers["Content-Range"] == "bytes 0-0/2"
    assert headers["X-Contract-Version"]


@pytest.mark.parametrize(
    ("path", "status", "code"),
    [
        (f"/agent/v1/models/{_MODEL}/content/file?path=../escape", 404, b"not_found"),
        (f"/agent/v1/models/{_MODEL}/content/file?path=shards", 404, b"not_found"),
        ("/agent/v1/models/huggingface:not/held/content/file?path=a", 404, b"not_found"),
        (f"/agent/v1/models/{_MODEL}/content/file", 422, b"invalid_request"),
        (f"/agent/v1/models/{_MODEL}/content/manifest", 404, b"not_found"),
        ("/agent/v1/info", 404, b"not_found"),
    ],
)
def test_the_listener_refuses_what_the_main_port_refuses(
    listener: tuple[int, Path], path: str, status: int, code: bytes
) -> None:
    port, ca = listener
    got, _, body = _ask(port, ca, path, headers=_TOKEN)
    assert got == status
    assert code in body


def test_an_unsatisfiable_range_names_the_size(listener: tuple[int, Path]) -> None:
    port, ca = listener
    status, headers, _ = _ask(port, ca, _FILE, headers={**_TOKEN, "Range": "bytes=5-9"})
    assert status == 416
    assert headers["Content-Range"] == "bytes */2"


def test_the_listener_serves_nothing_but_get(listener: tuple[int, Path]) -> None:
    port, ca = listener
    status, _, _ = _ask(port, ca, _FILE, method="POST", headers=_TOKEN)
    assert status == 501
