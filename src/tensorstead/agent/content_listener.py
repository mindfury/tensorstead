"""The content listener: replication byte ranges served on many cores.

Through uvicorn, every byte the agent sends is encrypted on its one event-loop
thread. For management traffic that never matters. For a model copy it is the
whole ceiling: measured between two DGX Sparks, that thread sat at 92-100% of
one core for an entire 56.7 GB copy while the fabric carried 107 Gbps of plain
TCP in the same direction and both NVMe drives sustained 8 GB/s or more. The
copy ran at 13 Gbps.

This listener serves the one route that carries model bytes --
``GET /agent/v1/models/{id}/content/file`` -- on a second port, and gives each
connection a worker thread of its own. That thread holds a blocking TLS
socket: ``SSLSocket.sendall`` encrypts with the GIL released and
``FileIO.readinto`` reads with it released, so eight connections encrypt on
eight cores. Everything else -- the manifest, the archive, every management
route -- stays on the main port.

The route is the same route, not a lesser one: the same replication token,
compared the same way; the same ``file_for`` containment and ``available``
refusal; the same single-range ``Range`` handling. The certificate is the
agent's own, so a destination verifies it against the same CA.

On the idle-polling rule: the accept loop is a listener, the same thing
uvicorn runs for the main port. It waits in ``sock_accept`` and does nothing
until a peer connects; it makes no outbound call and starts no thread. Each
connection's thread comes from the worker pool Starlette already uses for
every synchronous route, and is held only while that peer is connected.
``tests/integration/test_no_idle_polling.py`` pins that shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import io
import json
import os
import socket
import ssl
import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import anyio
import anyio.to_thread

from tensorstead.agent.replication import REFUSAL_STATUS, ReplicationError, parse_byte_range
from tensorstead.contracts.version import CONTRACT_VERSION
from tensorstead.domain.tokens import tokens_match

_PREFIX = "/agent/v1/models/"
_SUFFIX = "/content/file"

# One read and one ``sendall`` per chunk. ``sendall`` encrypts the whole chunk
# in one call with the GIL released, so a larger chunk is less time holding it.
_SEND = 8 * 1024 * 1024

# A peer that connects and never completes a handshake, or goes quiet between
# requests on a keep-alive connection, is closed after these.
_HANDSHAKE_SECONDS = 30.0
_IDLE_SECONDS = 120.0


@dataclass(frozen=True)
class ContentListenerConfig:
    """Where the listener binds and the certificate it presents.

    ``port`` 0 binds an ephemeral port, which is for tests; the advertised port
    is whatever the socket was given.
    """

    host: str
    port: int
    certfile: str
    keyfile: str
    max_connections: int = 64


class ContentListener:
    """Serve replication byte ranges on a second port, one thread per connection."""

    def __init__(self, config: ContentListenerConfig, *, service: Any, token: str) -> None:
        if not token:
            raise ValueError("the content listener serves only under a replication token")
        self._config = config
        self._service = service
        self._token = token
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(config.certfile, config.keyfile)
        self._sock: socket.socket | None = None
        self._accepting: asyncio.Task[None] | None = None
        self._serving: set[asyncio.Task[None]] = set()
        self._live: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._limiter: anyio.CapacityLimiter | None = None
        self.port: int | None = None

    async def start(self) -> None:
        """Bind, listen, and begin accepting on the running event loop."""
        sock = socket.create_server(
            (self._config.host, self._config.port), backlog=self._config.max_connections
        )
        sock.setblocking(False)
        self._sock = sock
        self.port = sock.getsockname()[1]
        self._limiter = anyio.CapacityLimiter(self._config.max_connections)
        self._accepting = asyncio.get_running_loop().create_task(self._accept())

    async def close(self) -> None:
        """Stop accepting, end every live connection, and wait for their threads."""
        sock, self._sock = self._sock, None
        if self._accepting is not None:
            self._accepting.cancel()
            await asyncio.gather(self._accepting, return_exceptions=True)
        if sock is not None:
            sock.close()
        with self._lock:
            live = list(self._live)
        for conn in live:
            _shut(conn)
        await asyncio.gather(*self._serving, return_exceptions=True)
        self.port = None

    async def _accept(self) -> None:
        """Wait for peers. Nothing here runs until one connects."""
        loop = asyncio.get_running_loop()
        while self._sock is not None:
            try:
                conn, _ = await loop.sock_accept(self._sock)
            except OSError:
                # Out of descriptors, or worse. Retrying would spin, so stop
                # listening instead: the manifest stops advertising the port,
                # a destination already holding it is refused and falls back
                # to the main port, and the copy goes on more slowly.
                if self._sock is not None:
                    self._sock.close()
                    self._sock = None
                self.port = None
                return
            with self._lock:
                full = len(self._live) >= self._config.max_connections
                if not full:
                    self._live.add(conn)
            if full:
                conn.close()
                continue
            task = loop.create_task(self._serve(conn))
            self._serving.add(task)
            task.add_done_callback(self._serving.discard)

    async def _serve(self, conn: socket.socket) -> None:
        try:
            await anyio.to_thread.run_sync(self._serve_blocking, conn, limiter=self._limiter)
        finally:
            with self._lock:
                self._live.discard(conn)
            conn.close()

    def _serve_blocking(self, conn: socket.socket) -> None:
        """One peer's connection, start to finish, on this worker thread."""
        # ``conn`` stays the registered socket, for ``close()`` to shut down;
        # the TLS socket wraps a duplicate of it. A shutdown acts on the
        # connection, not the descriptor, so it reaches this thread either way.
        # Timeouts are set on the duplicate only: they change flags the two
        # descriptors share, and the duplicate is the one doing the I/O.
        raw = socket.socket(fileno=os.dup(conn.fileno()))
        raw.settimeout(_HANDSHAKE_SECONDS)
        try:
            tls = self._context.wrap_socket(raw, server_side=True)
        except OSError:
            raw.close()
            return
        try:
            _RangeHandler(tls, tls.getpeername(), self)
        except OSError:
            pass  # the peer went away, or close() ended the connection
        finally:
            tls.close()

    # -- what the handler asks of its listener --------------------------------

    def authorised(self, values: list[str]) -> bool:
        # Exactly one Authorization header, Bearer, matching: as AgentAuth.
        if len(values) != 1:
            return False
        scheme, _, credential = values[0].strip().partition(" ")
        return scheme.lower() == "bearer" and tokens_match(credential.strip(), self._token)

    def file_for(self, model_id: str, path: str) -> Any:
        return self._service.file_for(model_id, path)


def _shut(conn: socket.socket) -> None:
    with contextlib.suppress(OSError):
        conn.shutdown(socket.SHUT_RDWR)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """HTTP/1.1 keep-alive, one route, the main port's behaviour on it."""

    protocol_version = "HTTP/1.1"
    timeout = _IDLE_SECONDS
    server_version = "tensorstead-agent"
    sys_version = ""

    def __init__(self, request: Any, client_address: Any, listener: ContentListener) -> None:
        self.listener = listener
        super().__init__(request, client_address, listener)  # type: ignore[arg-type]

    def log_message(self, format: str, *args: Any) -> None:
        """No access log: the main port's is off too, and this one is per range."""

    def do_GET(self) -> None:
        split = urlsplit(self.path)
        if not (split.path.startswith(_PREFIX) and split.path.endswith(_SUFFIX)):
            self._refuse(404, "not_found", "not found")
            return
        model_id = unquote(split.path[len(_PREFIX) : -len(_SUFFIX)])
        if not self.listener.authorised(self.headers.get_all("Authorization") or []):
            self._refuse(401, "unauthorized", "replication token required")
            return
        paths = parse_qs(split.query, keep_blank_values=True).get("path", [])
        if not model_id or len(paths) != 1:
            self._refuse(422, "invalid_request", "one model and one path are required")
            return
        try:
            target = self.listener.file_for(model_id, paths[0])
        except ReplicationError as exc:
            self._refuse(REFUSAL_STATUS.get(exc.code, 500), exc.code, exc.message, exc.detail)
            return
        try:
            fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError:
            self._refuse(404, "not_found", "not a file of this replica", {"model_id": model_id})
            return
        with io.FileIO(fd, "rb") as source:
            size = os.fstat(fd).st_size
            try:
                requested = parse_byte_range(self.headers.get("Range"), size)
            except ValueError as exc:
                self._refuse(
                    416,
                    "range_not_satisfiable",
                    str(exc),
                    headers={"Content-Range": f"bytes */{size}"},
                )
                return
            start, end = requested or (0, size)
            self.send_response(206 if requested is not None else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(end - start))
            self.send_header("Accept-Ranges", "bytes")
            if requested is not None:
                self.send_header("Content-Range", f"bytes {start}-{end - 1}/{size}")
            self.send_header("X-Contract-Version", CONTRACT_VERSION)
            self.end_headers()
            self._send_range(source, start, end)

    def _send_range(self, source: io.FileIO, start: int, end: int) -> None:
        buffer = memoryview(bytearray(min(_SEND, max(end - start, 1))))
        source.seek(start)
        position = start
        while position < end:
            read = source.readinto(buffer[: min(len(buffer), end - position)])
            if not read:
                # The file shrank underneath the request. The length is already
                # promised, so the only honest answer is to end the connection
                # short; the destination refuses a short range.
                self.close_connection = True
                return
            self.connection.sendall(buffer[:read])
            position += read

    def _refuse(
        self,
        status: int,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            {"detail": {"code": code, "message": message, "detail": detail or {}}}
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Contract-Version", CONTRACT_VERSION)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
