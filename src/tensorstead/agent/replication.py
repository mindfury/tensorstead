"""Peer replication on the agent.

**The destination agent owns the operation.** The coordinator's instruction is
in effect "ensure this model revision is locally available; a source replica
exists at A" — it names a peer and then stays out of the way. No model byte
transits the coordinator process, which is what keeps a management
system out of the inference data plane even while it moves weights around.

Four steps, in this order and no other:

1. pull the artifact directly from the source node's agent;
2. write into **staging**;
3. verify the artifact against the expected ``content_digest``;
4. **atomically promote** the verified replica to ``available``.

Step 3 is the one that distinguishes replication from upstream acquisition. An
upstream source is authoritative for what it just downloaded; a peer is not —
it is relaying someone else's claim over a network — so the digest is
*recomputed here* and compared. A mismatch marks the replica ``failed`` and it
never becomes available. Staging plus atomic promotion makes that
structural rather than a check that could be skipped.

Still HTTPS over whatever IP path the two nodes already share -- no
topology-aware distribution, no peer swarming, no RDMA-specific protocol, no
multi-source striping. But not one stream: the destination reads the source's
**manifest** (every path and size) and pulls the files as byte ranges over
several connections at once, writing each range straight into its place in
staging. The archive protocol it replaces made four serial passes over the
model -- the source wrote a whole tar before sending a byte, then the
destination wrote the tar, unpacked it, and re-read everything to hash it --
over one TLS stream, and every one of those passes ran at roughly 4-5 Gbps.
That protocol is still served and still spoken, to and from a peer that
predates the manifest.

The parallelism is request-scoped: each pool is a ``with`` block inside
``replicate()``, so its threads are joined before the request returns and
nothing outlives it (the idle-polling guardrail's rule, and the same shape
``service/lifecycle.py`` already uses).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import ssl
import stat
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from tensorstead.agent.acquisition import AcquiredReplica, ModelAcquisitionService

# Read size for streaming and digesting; large enough to be efficient, small
# enough that a multi-gigabyte model never lands in memory at once.
_CHUNK = 1024 * 1024

# The ranged protocol's sizes. A piece is the unit one connection fetches in
# one request: big enough that a request's overhead vanishes, small enough that
# a model of a few large shards still spreads across every connection and a
# file finishes -- and starts hashing -- while others are still arriving. The
# I/O size is per read and per write; 1 MiB reads were measurably slower
# through both uvicorn's TLS and the client's.
_PIECE = 64 * 1024 * 1024
_IO = 8 * 1024 * 1024

# How many connections pull at once, and how many threads hash finished files.
# One uvicorn process encrypts on one event-loop thread, so the source caps out
# at a few connections' worth; eight measured at that cap with headroom.
_DEFAULT_CONNECTIONS = 8
_DEFAULT_HASHERS = 4

# A manifest is a list of paths and sizes; anything near this is not one.
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def _env_count(name: str, default: int, ceiling: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return max(1, min(ceiling, int(raw)))


# A configured ceiling on one artifact transfer:
# neither content endpoint in this protocol currently advertises an expected
# size up front, so this is the only bound available -- it stops a runaway or
# malicious peer from writing until the managed filesystem is full, since a
# request timeout bounds duration, not bytes.
_DEFAULT_MAX_TRANSFER_BYTES = 200 * 1024 * 1024 * 1024  # 200 GiB

# The preflight's own threshold -- deliberately *not* the ceiling above.
# Comparing free space to the full configured ceiling refused every transfer,
# however small, on any host with less free space than that generous worst-
# case number: review of this behaviour caught it
# directly, refusing an ordinary small test transfer on a host with ~14.6 GiB
# free against a 200 GiB default ceiling. Neither endpoint advertises a real
# expected size (a previous review already notes this), so this is a
# sanity floor -- "is the disk not already essentially full" -- not a claim
# the transfer will fit. A transfer too large for the disk it landed on still
# fails correctly, just not preemptively: the write raises ENOSPC and the
# existing cleanup path runs the same as any other mid-transfer failure.
_MIN_FREE_BYTES_TO_ATTEMPT = 1 * 1024 * 1024 * 1024  # 1 GiB


def _max_transfer_bytes() -> int:
    raw = os.environ.get("TENSORSTEAD_MAX_ARTIFACT_BYTES")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_TRANSFER_BYTES
    return int(raw)


class ReplicationError(Exception):
    """A peer replication attempt failed, with a structured reason."""

    def __init__(self, code: str, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


class PeerClient(Protocol):
    """The transport that fetches artifact bytes from a source agent.

    ``fetch_manifest`` and ``open_session`` are the ranged protocol. A
    transport without them speaks only the archive protocol, and
    ``PeerReplicationService`` uses that instead.
    """

    def fetch_content(
        self,
        *,
        endpoint: str,
        model_id: str,
        token: str | None = None,
        fingerprint: str | None = None,
    ) -> Iterable[bytes]:
        """Stream the source agent's ``GET /agent/v1/models/{id}/content``."""


class PeerSession(Protocol):
    """Connections to one source, shared by the threads of one transfer."""

    def fetch_range(self, path: str, offset: int, length: int, target: Path) -> None:
        """Write exactly ``length`` bytes of the peer's ``path`` into ``target`` at ``offset``."""

    def close(self) -> None:
        """Close every connection the session opened."""


@dataclass(frozen=True)
class ManifestEntry:
    """One entry of a model tree: a directory (``d``) or a regular file (``f``)."""

    path: str
    kind: str
    size: int

    def as_json(self) -> dict[str, Any]:
        return {"path": self.path, "type": self.kind, "size": self.size}


def tree_manifest(root: Path) -> list[ManifestEntry]:
    """Every directory and regular file under ``root``, sorted by path.

    The same entries ``directory_digest`` covers, refused on the same grounds,
    so a tree that can be digested can be described and nothing else can.
    """
    entries: list[ManifestEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append(ManifestEntry(rel, "d", 0))
            elif stat.S_ISREG(mode):
                entries.append(ManifestEntry(rel, "f", path.lstat().st_size))
            else:
                raise UnsupportedTreeEntry(
                    f"{rel!r} is neither a regular file nor a directory; "
                    f"model trees hold only those"
                )
    entries.sort(key=lambda entry: entry.path.encode("utf-8", "surrogateescape"))
    return entries


def safe_relative_path(value: Any) -> str | None:
    """``value`` if it is a normalized relative POSIX path, else ``None``.

    No absolute path, no ``.`` or ``..`` component, no empty component, no NUL.
    Applied on both sides: by the source to the path a peer asks for, and by
    the destination to every path a peer's manifest names.
    """
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        return None
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if PurePosixPath(value).is_absolute():
        return None
    return value


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The ``[start, end)`` a ``Range: bytes=a-b`` header asks for.

    ``None`` for no header -- the whole file. One closed range is all the
    destination ever sends, so anything else is refused with ``ValueError``.
    """
    if header is None:
        return None
    unit, _, spec = header.partition("=")
    first, dash, last = spec.partition("-")
    if unit.strip() != "bytes" or not dash or not first.isdigit() or not last.isdigit():
        raise ValueError(f"unsupported range {header!r}")
    start, end = int(first), int(last) + 1
    if start >= end or end > size:
        raise ValueError(f"range {header!r} is outside a {size}-byte file")
    return start, end


class _HTTPPeerSession:
    """A small pool of keep-alive HTTPS connections to one source agent.

    ``http.client`` rather than ``httpx`` on this path, measured: httpx parses
    every 64 KiB read in Python under the GIL, which held a parallel transfer to
    about 1 GB/s however many connections it ran. ``readinto`` here goes
    straight to ``SSLSocket.recv_into``, which decrypts with the GIL released.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        token: str | None,
        context: ssl.SSLContext,
        timeout: float,
    ) -> None:
        split = urlsplit(endpoint)
        if split.scheme != "https" or not split.hostname:
            raise ReplicationError(
                "replication_failed",
                f"refusing to fetch {model_id!r} from non-https peer endpoint {endpoint!r}",
                detail={"endpoint": endpoint},
            )
        self._host = split.hostname
        self._port = split.port or 443
        self._prefix = f"{split.path.rstrip('/')}/agent/v1/models/{model_id}/content/file"
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._endpoint = endpoint
        self._model_id = model_id
        self._context = context
        self._timeout = timeout
        self._idle: list[tuple[http.client.HTTPSConnection, memoryview]] = []
        self._opened: list[http.client.HTTPSConnection] = []
        self._lock = threading.Lock()

    def _take(self) -> tuple[http.client.HTTPSConnection, memoryview]:
        with self._lock:
            if self._idle:
                return self._idle.pop()
            conn = http.client.HTTPSConnection(
                self._host, self._port, context=self._context, timeout=self._timeout
            )
            self._opened.append(conn)
        return conn, memoryview(bytearray(_IO))

    def fetch_range(self, path: str, offset: int, length: int, target: Path) -> None:
        conn, buffer = self._take()
        reusable = False
        try:
            headers = {**self._headers, "Range": f"bytes={offset}-{offset + length - 1}"}
            conn.request("GET", f"{self._prefix}?path={quote(path, safe='')}", headers=headers)
            response = conn.getresponse()
            if response.status != 206:
                response.read(64 * 1024)
                raise ReplicationError(
                    "replication_failed",
                    f"peer refused to serve {path!r} of {self._model_id!r}: HTTP {response.status}",
                    detail={"endpoint": self._endpoint, "status": response.status, "path": path},
                )
            expected_range = f"bytes {offset}-{offset + length - 1}/"
            if response.getheader("Content-Length") != str(length) or not (
                response.getheader("Content-Range") or ""
            ).startswith(expected_range):
                raise ReplicationError(
                    "replication_failed",
                    f"peer answered a different range of {path!r} than was asked for",
                    detail={"endpoint": self._endpoint, "path": path},
                )
            fd = os.open(target, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                position, remaining = offset, length
                while remaining:
                    received = response.readinto(buffer[: min(remaining, _IO)])
                    if not received:
                        raise ReplicationError(
                            "replication_failed",
                            f"peer closed the connection partway through {path!r}",
                            detail={"endpoint": self._endpoint, "path": path},
                        )
                    _pwrite_all(fd, buffer[:received], position)
                    position += received
                    remaining -= received
            finally:
                os.close(fd)
            reusable = True
        finally:
            if reusable:
                with self._lock:
                    self._idle.append((conn, buffer))
            else:
                conn.close()

    def close(self) -> None:
        with self._lock:
            opened, self._opened, self._idle = self._opened, [], []
        for conn in opened:
            conn.close()


def _pwrite_all(fd: int, data: memoryview, offset: int) -> None:
    while data:
        written = os.pwrite(fd, data, offset)
        data = data[written:]
        offset += written


class HTTPPeerClient:
    """Fetch artifact bytes from a peer agent over TLS.

    Verified against the managed private CA named by ``ca_bundle``, in the
    same already-correct shape ``ImageDistributionService`` uses -- until this
    class was fixed it had never matched that shape, defaulting to no
    verification at all in production. The request also carries the
    **replication** token, not the management token.

    This used to verify a pinned certificate *fingerprint* instead, over a
    second TLS handshake opened just to read the peer's certificate and then
    discarded -- detached from the connection that actually carried the
    token and the bytes, so a DNS-rebinding or connection-race attacker
    could let the preflight reach the real peer and intercept the transfer
    that followed it. It was also, in practice, theatre: nothing has ever
    captured a fingerprint at registration, so the check never ran outside a
    test. The coordinator's own equivalent (``_pin_fingerprint`` in
    ``coordinator/node_http.py``) was already removed for exactly this
    reason -- "a control read as present while being absent" -- and this is
    that same cleanup applied to the other client that had it
    (the ``fingerprint`` parameter stays in the call shape,
    unused, for the same reason ``image_distribution.py`` keeps it).
    """

    def __init__(self, *, timeout: float = 300.0, ca_bundle: str | None = None) -> None:
        self._timeout = timeout
        self._ca_bundle = ca_bundle

    def _context(self) -> ssl.SSLContext:
        return ssl.create_default_context(cafile=self._ca_bundle)

    def fetch_manifest(
        self, *, endpoint: str, model_id: str, token: str | None = None
    ) -> list[Any] | None:
        """The source's manifest entries, or ``None`` for a peer that has no manifest.

        A peer that predates the ranged protocol answers its framework's bare
        404 for the unknown route; a current peer refusing *this model* answers
        with a structured ``detail``, and that is raised, not taken as "old".
        """
        import httpx

        if not endpoint.startswith("https://"):
            raise ReplicationError(
                "replication_failed",
                f"refusing to fetch {model_id!r} from non-https peer endpoint {endpoint!r}",
                detail={"endpoint": endpoint},
            )
        url = f"{endpoint.rstrip('/')}/agent/v1/models/{model_id}/content/manifest"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with (
            httpx.Client(verify=self._context(), timeout=self._timeout) as client,
            client.stream("GET", url, headers=headers) as response,
        ):
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > _MAX_MANIFEST_BYTES:
                    raise ReplicationError(
                        "replication_failed",
                        f"peer's manifest for {model_id!r} is larger than "
                        f"{_MAX_MANIFEST_BYTES} bytes; refused",
                        detail={"endpoint": endpoint},
                    )
            status = response.status_code
        try:
            payload = json.loads(body) if body else None
        except ValueError:
            payload = None
        if status == 404 and not (
            isinstance(payload, dict) and isinstance(payload.get("detail"), dict)
        ):
            return None
        if status >= 400:
            raise ReplicationError(
                "replication_failed",
                f"peer refused to describe {model_id!r}: HTTP {status}",
                detail={"endpoint": endpoint, "status": status},
            )
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ReplicationError(
                "replication_failed",
                f"peer's manifest for {model_id!r} is not a list of entries",
                detail={"endpoint": endpoint},
            )
        return entries

    def open_session(
        self, *, endpoint: str, model_id: str, token: str | None = None
    ) -> PeerSession:
        return _HTTPPeerSession(
            endpoint=endpoint,
            model_id=model_id,
            token=token,
            context=self._context(),
            timeout=self._timeout,
        )

    def fetch_content(
        self,
        *,
        endpoint: str,
        model_id: str,
        token: str | None = None,
        fingerprint: str | None = None,  # noqa: ARG002 - see class docstring
    ) -> Iterable[bytes]:
        import ssl

        import httpx

        if not endpoint.startswith("https://"):
            raise ReplicationError(
                "replication_failed",
                f"refusing to fetch {model_id!r} from non-https peer endpoint {endpoint!r}",
                detail={"endpoint": endpoint},
            )
        verify: bool | ssl.SSLContext = (
            ssl.create_default_context(cafile=self._ca_bundle) if self._ca_bundle else True
        )

        url = f"{endpoint.rstrip('/')}/agent/v1/models/{model_id}/content"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with (
            httpx.Client(verify=verify, timeout=self._timeout) as client,
            client.stream("GET", url, headers=headers) as response,
        ):
            if response.status_code >= 400:
                response.read()
                raise ReplicationError(
                    "replication_failed",
                    f"peer refused to serve {model_id!r}: HTTP {response.status_code}",
                    detail={"endpoint": endpoint, "status": response.status_code},
                )
            yield from response.iter_bytes(_CHUNK)


@dataclass
class ReplicationOutcome:
    """What one replication attempt produced."""

    model_id: str
    state: str  # available | failed
    content_digest: str | None = None
    size_bytes: int | None = None
    detail: str | None = None


# Versioned, because a digest is a recorded fact: the v1 format below is never
# silently reinterpreted. v1 hashed ``path NUL bytes`` for each file in order
# with no length framing, so file bytes could impersonate the next entry's name
# and contents -- ``{a: "x", b: "y"}`` and ``{a: "xb\\0y"}`` hashed identically
# (CodeGuard CG-05). A v1 digest is refused for verification, not trusted.
DIGEST_PREFIX = "tsdir2-sha256:"
_DOMAIN = b"tensorstead/directory-digest/v2\n"


class UnsupportedTreeEntry(ValueError):
    """A model tree holds something other than regular files and directories."""


def directory_digest(root: Path) -> str:
    """A deterministic, unambiguous digest over a directory's *contents* (v2).

    A canonical manifest, sorted by path: for every directory and regular file,
    a type byte, the length-prefixed UTF-8 path, the content length, and the
    content's SHA-256; then an entry count. Lengths frame every field, so no
    content can be read as a boundary. Directories are entries, so an empty one
    counts. Links, devices and the like are refused rather than followed or
    skipped: a model tree has no business holding them, and silently ignoring
    one would let two different trees share a digest.

    Contents only -- not a tar stream, which folds in mtimes and ordering --
    because what is verified is that the weights are the same weights.
    """
    entries: list[tuple[bytes, bytes, int, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix().encode("utf-8", "surrogateescape")
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append((rel, b"d", 0, bytes(32)))
            elif stat.S_ISREG(mode):
                size, content_digest = _file_sha256(path)
                entries.append((rel, b"f", size, content_digest))
            else:
                raise UnsupportedTreeEntry(
                    f"{rel.decode('utf-8', 'replace')!r} is neither a regular file nor a "
                    f"directory; model trees hold only those"
                )
    return _combine_digest(entries)


def _file_sha256(path: Path) -> tuple[int, bytes]:
    """A regular file's length and SHA-256, read from disk.

    Large reads into one reused buffer: ``hashlib`` releases the GIL on them, so
    several files hash on several cores at once.
    """
    content = hashlib.sha256()
    size = 0
    buffer = memoryview(bytearray(_IO))
    with path.open("rb", buffering=0) as handle:
        while read := handle.readinto(buffer):
            content.update(buffer[:read])
            size += read
    return size, content.digest()


class _OrderedFileHash:
    """A file's SHA-256, advanced over its pieces in order as they land.

    Pieces arrive out of order over several connections; SHA-256 does not
    accept them that way. Each ``advance`` reads back from disk every piece
    that is now contiguous with what was already hashed, so hashing keeps pace
    with the transfer instead of starting when the last piece arrives -- which
    left the last shard's whole hash as a tail after the network went idle.
    ``advance`` is never run twice at once for one file (the caller ensures
    it); ``landed`` and ``running`` are written only by the caller's thread.
    """

    def __init__(self, path: Path, size: int) -> None:
        self.path = path
        self.size = size
        self.landed: set[int] = set()
        self.running = False
        self._next = 0
        self._hashed = 0
        self._digest = hashlib.sha256()

    @property
    def complete(self) -> bool:
        return self._hashed == self.size

    def ready(self) -> bool:
        return self._next * _PIECE < self.size and self._next in self.landed

    def advance(self) -> None:
        buffer = memoryview(bytearray(_IO))
        with self.path.open("rb", buffering=0) as handle:
            handle.seek(self._hashed)
            while self.ready():
                end = min(self.size, (self._next + 1) * _PIECE)
                while self._hashed < end:
                    read = handle.readinto(buffer[: min(_IO, end - self._hashed)])
                    if not read:
                        raise ReplicationError(
                            "replication_failed", f"{self.path.name!r} is shorter than declared"
                        )
                    self._digest.update(buffer[:read])
                    self._hashed += read
                self._next += 1

    def result(self) -> tuple[int, bytes]:
        return self._hashed, self._digest.digest()


def _combine_digest(entries: list[tuple[bytes, bytes, int, bytes]]) -> str:
    """The v2 digest over per-entry ``(path, type, size, sha256)`` facts."""
    entries = sorted(entries, key=lambda entry: entry[0])
    digest = hashlib.sha256(_DOMAIN)
    for rel, kind, size, content_digest in entries:
        digest.update(kind)
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(size.to_bytes(8, "big"))
        digest.update(content_digest)
    digest.update(b"end")
    digest.update(len(entries).to_bytes(8, "big"))
    return f"{DIGEST_PREFIX}{digest.hexdigest()}"


def _directory_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class PeerReplicationService:
    """Pull a model replica from a peer agent, verify it, then promote it."""

    def __init__(
        self,
        *,
        acquisition: ModelAcquisitionService,
        peer_client: PeerClient | None = None,
    ) -> None:
        self._acquisition = acquisition
        self._peer = peer_client

    # ------------------------------------------------------------ destination
    def replicate(
        self,
        *,
        model_id: str,
        content_digest: str | None,
        source_endpoint: str,
        source_node_id: str | None = None,
        source_fingerprint: str | None = None,
        replication_token: str | None = None,
        resolved_revision: str | None = None,
    ) -> ReplicationOutcome:
        """Pull → stage → verify → promote.

        ``resolved_revision`` is recorded on the promoted replica. It used to be
        dropped, so a copied replica carried no revision and anything bound to
        one -- a code-execution approval -- refused to start on it (FP4-Local
        TP=2 on the estate: "no acquired model on this node matches ... with an
        immutable revision").

        Raises ``ReplicationError`` on an unreachable peer, a peer that will not
        serve, or a digest mismatch. In every failure case the replica is marked
        ``failed`` rather than left in a state that could read as available.
        """
        if self._peer is None:
            raise ReplicationError(
                "replication_unavailable",
                "no peer transport is configured on this agent",
            )

        # Already here, verified against the same digest: nothing to copy. This
        # is what makes re-running an acquire after a timed-out copy cheap --
        # the copy usually finished on the agents after the coordinator stopped
        # waiting, and a retry used to wipe it and start the transfer again.
        existing = self._acquisition.get_replica(model_id)
        if (
            content_digest is not None
            and existing is not None
            and existing.state == "available"
            and existing.content_digest == content_digest
        ):
            if existing.resolved_revision is None and resolved_revision is not None:
                # A copy made before revisions were carried: record it now, so
                # a retried acquire repairs the replica without moving a byte.
                self._acquisition.record_revision(model_id, resolved_revision)
            return ReplicationOutcome(
                model_id=existing.model_id,
                state=existing.state,
                content_digest=existing.content_digest,
                size_bytes=existing.size_bytes,
            )

        staging_dir = self._acquisition.staging_path(model_id)
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        archive = staging_dir.parent / f"{staging_dir.name}.tar"

        try:
            self._preflight(staging_dir, model_id)
            manifest = self._manifest(model_id, source_endpoint, replication_token)
            if manifest is None:
                self._pull_into(
                    archive, model_id, source_endpoint, replication_token, source_fingerprint
                )
                self._unpack(archive, staging_dir)
                computed = directory_digest(staging_dir)
                size_bytes = _directory_size(staging_dir)
            else:
                computed, size_bytes = self._pull_files(
                    manifest, staging_dir, model_id, source_endpoint, replication_token
                )
            self._verify(model_id, computed, content_digest, source_node_id)
        except ReplicationError:
            shutil.rmtree(staging_dir, ignore_errors=True)
            archive.unlink(missing_ok=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            archive.unlink(missing_ok=True)
            self._acquisition.mark_failed(model_id, detail=str(exc))
            raise ReplicationError("replication_failed", str(exc)) from exc
        finally:
            archive.unlink(missing_ok=True)

        replica: AcquiredReplica = self._acquisition.promote(
            model_id,
            staging_dir,
            resolved_revision=resolved_revision,
            size_bytes=size_bytes,
            content_digest=computed,
        )
        return ReplicationOutcome(
            model_id=replica.model_id,
            state=replica.state,
            content_digest=replica.content_digest,
            size_bytes=replica.size_bytes,
        )

    def _preflight(self, staging_dir: Path, model_id: str) -> None:
        """Refuse before contacting the peer when the disk is already essentially full."""
        free = shutil.disk_usage(staging_dir.parent).free
        if free < _MIN_FREE_BYTES_TO_ATTEMPT:
            raise ReplicationError(
                "artifact_too_large",
                f"refusing to start pulling {model_id!r}: only {free} bytes free, "
                f"below the {_MIN_FREE_BYTES_TO_ATTEMPT} byte minimum to attempt a transfer",
                detail={
                    "model_id": model_id,
                    "free_bytes": free,
                    "ceiling_bytes": _max_transfer_bytes(),
                },
            )

    def _manifest(
        self, model_id: str, endpoint: str, token: str | None
    ) -> list[ManifestEntry] | None:
        """The source's validated manifest, or ``None`` to use the archive protocol."""
        fetch = getattr(self._peer, "fetch_manifest", None)
        if fetch is None or getattr(self._peer, "open_session", None) is None:
            return None
        raw = fetch(endpoint=endpoint, model_id=model_id, token=token)
        return None if raw is None else self._validated_manifest(raw, model_id)

    def _validated_manifest(self, raw: list[Any], model_id: str) -> list[ManifestEntry]:
        """A peer's manifest, checked as the untrusted input it is.

        The same bounds the archive path enforces member by member, enforced
        here before anything is written: entry count, total size against the
        transfer ceiling, and -- because a manifest finally says how big the
        transfer is -- free space against the real size rather than a floor.
        Paths are normalized and relative, unique, and every parent is itself a
        declared directory, so nothing can be written outside staging or
        through a file.
        """
        max_members = int(os.environ.get("TENSORSTEAD_MAX_ARTIFACT_MEMBERS", "100000"))
        if len(raw) > max_members:
            raise ReplicationError(
                "replication_failed", f"the artifact has more than {max_members} entries; refused"
            )
        entries: list[ManifestEntry] = []
        kinds: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ReplicationError("replication_failed", "a manifest entry is not an object")
            path = safe_relative_path(item.get("path"))
            kind, size = item.get("type"), item.get("size")
            if (
                path is None
                or kind not in ("d", "f")
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or (kind == "d" and size != 0)
            ):
                raise ReplicationError(
                    "replication_failed",
                    f"manifest entry {str(item.get('path'))[:200]!r} is not a plain file or "
                    f"directory with a safe relative path; model trees hold only those",
                )
            if path in kinds:
                raise ReplicationError(
                    "replication_failed", f"manifest names {path!r} more than once"
                )
            kinds[path] = kind
            entries.append(ManifestEntry(path, kind, size))
        for entry in entries:
            parent = entry.path.rpartition("/")[0]
            if parent and kinds.get(parent) != "d":
                raise ReplicationError(
                    "replication_failed",
                    f"manifest entry {entry.path!r} has no declared parent directory",
                )
        total = sum(entry.size for entry in entries)
        ceiling = _max_transfer_bytes()
        if total > ceiling:
            raise ReplicationError(
                "artifact_too_large",
                f"peer's {model_id!r} is {total} bytes, past the configured ceiling of "
                f"{ceiling} bytes; refused",
                detail={"model_id": model_id, "size_bytes": total, "ceiling_bytes": ceiling},
            )
        return entries

    def _pull_files(
        self,
        entries: list[ManifestEntry],
        staging_dir: Path,
        model_id: str,
        endpoint: str,
        token: str | None,
    ) -> tuple[str, int]:
        """Pull every file as byte ranges in parallel; return ``(digest, size)``.

        Ranges are written in place with ``pwrite`` into files sized up front,
        so there is no archive and no unpack. Each file is hashed -- read back
        from disk, so the digest describes what landed rather than what was
        sent -- piece by piece as its leading pieces arrive, alongside the
        transfer. The digest is the same v2 ``directory_digest`` over the same
        entries; only the order in which its per-file facts are gathered differs.
        """
        total = sum(entry.size for entry in entries)
        free = shutil.disk_usage(staging_dir).free
        if free - total < _MIN_FREE_BYTES_TO_ATTEMPT:
            raise ReplicationError(
                "artifact_too_large",
                f"{model_id!r} is {total} bytes and only {free} bytes are free; "
                f"refusing to leave less than {_MIN_FREE_BYTES_TO_ATTEMPT} bytes free",
                detail={"model_id": model_id, "size_bytes": total, "free_bytes": free},
            )
        files = [entry for entry in entries if entry.kind == "f"]
        for entry in sorted(entries, key=lambda e: e.path.count("/")):
            target = staging_dir / entry.path
            if entry.kind == "d":
                target.mkdir()
                continue
            fd = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644
            )
            try:
                # Sized, not allocated: posix_fallocate is emulated by writing
                # zeros on a filesystem without native support, which would
                # write the model twice.
                os.ftruncate(fd, entry.size)
            finally:
                os.close(fd)

        hashes = {
            entry.path: _OrderedFileHash(staging_dir / entry.path, entry.size) for entry in files
        }
        connections = _env_count("TENSORSTEAD_REPLICATION_CONNECTIONS", _DEFAULT_CONNECTIONS, 64)
        hashers = _env_count("TENSORSTEAD_REPLICATION_HASHERS", _DEFAULT_HASHERS, 64)
        deadline = time.monotonic() + float(
            os.environ.get("TENSORSTEAD_MAX_UNPACK_SECONDS", str(4 * 60 * 60))
        )
        assert self._peer is not None  # noqa: S101 - narrowed in replicate()
        session: PeerSession = self._peer.open_session(  # type: ignore[attr-defined]
            endpoint=endpoint, model_id=model_id, token=token
        )
        try:
            with (
                ThreadPoolExecutor(connections, thread_name_prefix="replicate-pull") as pulls,
                ThreadPoolExecutor(hashers, thread_name_prefix="replicate-hash") as hashing,
            ):
                # future -> (is_hash, path, piece index)
                pending: dict[Future[Any], tuple[bool, str, int]] = {}

                def _advance(path: str) -> None:
                    state = hashes[path]
                    if not state.running and state.ready():
                        state.running = True
                        pending[hashing.submit(state.advance)] = (True, path, -1)

                for entry in files:
                    for offset in range(0, entry.size, _PIECE):
                        future = pulls.submit(
                            session.fetch_range,
                            entry.path,
                            offset,
                            min(_PIECE, entry.size - offset),
                            staging_dir / entry.path,
                        )
                        pending[future] = (False, entry.path, offset // _PIECE)
                try:
                    while pending:
                        timeout = deadline - time.monotonic()
                        if timeout <= 0:
                            raise ReplicationError(
                                "replication_failed", "pulling the artifact exceeded its deadline"
                            )
                        done, _ = wait(list(pending), timeout=timeout, return_when=FIRST_COMPLETED)
                        for future in done:
                            is_hash, path, piece = pending.pop(future)
                            future.result()
                            if is_hash:
                                hashes[path].running = False
                            else:
                                hashes[path].landed.add(piece)
                            _advance(path)
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise
        finally:
            session.close()

        # What is in staging must be exactly what the manifest declared -- the
        # digest is over the tree, so it is the tree that is checked.
        declared = {entry.path: entry.kind for entry in entries}
        found = {entry.path: entry.kind for entry in tree_manifest(staging_dir)}
        if found != declared:
            raise ReplicationError(
                "replication_failed", "staging does not hold exactly the declared entries"
            )
        if not all(state.complete for state in hashes.values()):
            raise ReplicationError("replication_failed", "not every file was hashed in full")
        hashed = {path: state.result() for path, state in hashes.items()}
        facts = [
            (
                entry.path.encode("utf-8", "surrogateescape"),
                entry.kind.encode(),
                *(hashed[entry.path] if entry.kind == "f" else (0, bytes(32))),
            )
            for entry in entries
        ]
        return _combine_digest(facts), sum(size for size, _ in hashed.values())

    def _pull_into(
        self,
        archive: Path,
        model_id: str,
        source_endpoint: str,
        token: str | None,
        fingerprint: str | None,
    ) -> None:
        """Stream the peer's content endpoint to a local archive.

        Bounded against both a full disk and a runaway peer:
        a preflight refuses to start a transfer this host could
        not possibly finish, and a running total refuses to keep writing past
        the configured ceiling. Neither endpoint in this protocol advertises
        an expected size today, so the ceiling -- not a per-transfer figure
        -- is what both checks are against; ``replicate()``'s caller already
        cleans up staging on any exception raised here.
        """
        # Type narrowing for mypy, not a validation step: the caller only
        # reaches here with a peer configured.
        assert self._peer is not None  # noqa: S101
        limit = _max_transfer_bytes()
        written = 0
        with archive.open("wb") as handle:
            for chunk in self._peer.fetch_content(
                endpoint=source_endpoint,
                model_id=model_id,
                token=token,
                fingerprint=fingerprint,
            ):
                written += len(chunk)
                if written > limit:
                    raise ReplicationError(
                        "artifact_too_large",
                        f"peer for {model_id!r} sent more than the configured "
                        f"ceiling of {limit} bytes; aborting",
                        detail={"model_id": model_id, "ceiling_bytes": limit},
                    )
                handle.write(chunk)

    def _unpack(self, archive: Path, staging_dir: Path) -> None:
        """Extract the pulled archive into staging.

        ``filter="data"`` refuses absolute paths, traversal, and special files.
        These bytes came from another host, so the extraction is treated as
        untrusted input even though that host is a peer we registered.
        """
        try:
            # tarfile's default 16 KiB copy buffer made unpacking its own
            # 4 Gbps pass.
            with tarfile.TarFile(archive, "r", copybufsize=_IO) as tar:
                self._extract_within_budget(tar, staging_dir)
        except (tarfile.TarError, OSError) as exc:
            raise ReplicationError(
                "replication_failed",
                f"could not unpack the artifact pulled from the peer: {exc}",
            ) from exc

    def _extract_within_budget(self, tar: tarfile.TarFile, staging_dir: Path) -> None:
        """Extract one member at a time, refusing before writing anything over budget.

        The transfer ceiling bounds the bytes received, not what they expand
        to: a sparse member can declare gigabytes in a few kilobytes, and
        nothing bounded the member count or the time spent (CodeGuard CG-06).
        Each member is checked before it is written -- type, cumulative
        logical size against the same ceiling, the free-space reserve, the
        member count, and an overall deadline -- in one streaming pass that
        never holds the member list. Hashing afterwards reads at most what was
        allowed in, so it is bounded too.
        """
        ceiling = _max_transfer_bytes()
        max_members = int(os.environ.get("TENSORSTEAD_MAX_ARTIFACT_MEMBERS", "100000"))
        deadline = time.monotonic() + float(
            os.environ.get("TENSORSTEAD_MAX_UNPACK_SECONDS", str(4 * 60 * 60))
        )
        total = 0
        count = 0
        while (member := tar.next()) is not None:
            count += 1
            if count > max_members:
                raise ReplicationError(
                    "replication_failed",
                    f"the artifact has more than {max_members} entries; refused",
                )
            if time.monotonic() > deadline:
                raise ReplicationError(
                    "replication_failed", "unpacking the artifact exceeded its deadline"
                )
            # Both signals: tarfile fills ``sparse`` while parsing PAX sparse
            # headers, while a GNU sparse member keeps its own type -- which
            # ``isfile()`` counts as regular.
            sparse = member.issparse() or member.type == tarfile.GNUTYPE_SPARSE
            if sparse or not (member.isfile() or member.isdir()):
                raise ReplicationError(
                    "replication_failed",
                    f"artifact entry {member.name!r} is not a plain file or directory; "
                    f"model trees hold only those",
                )
            total += member.size
            if total > ceiling:
                raise ReplicationError(
                    "replication_failed",
                    f"the artifact expands past the {ceiling}-byte transfer ceiling; refused",
                )
            free = shutil.disk_usage(staging_dir).free
            if free - member.size < _MIN_FREE_BYTES_TO_ATTEMPT:
                raise ReplicationError(
                    "replication_failed",
                    f"extracting {member.name!r} would leave less than "
                    f"{_MIN_FREE_BYTES_TO_ATTEMPT} bytes free; refused",
                )
            tar.extract(member, staging_dir, filter="data")

    def _verify(
        self,
        model_id: str,
        computed: str,
        expected: str | None,
        source_node_id: str | None,
    ) -> None:
        """Compare the staged artifact against the expected digest.

        An absent ``content_digest`` is not treated as "verified" — it is
        treated as nothing to verify against, and the replication is refused.
        Promoting an unverifiable copy is exactly the outcome staging exists to
        prevent.
        """
        if expected is None:
            self._acquisition.mark_failed(model_id, detail="no content_digest to verify against")
            raise ReplicationError(
                "verification_failed",
                f"refusing to promote {model_id!r}: no content_digest was supplied "
                f"to verify against",
                detail={"model_id": model_id},
            )
        if not expected.startswith(DIGEST_PREFIX):
            # A v1 digest is ambiguous (see DIGEST_PREFIX), so a copy that
            # matched it would prove nothing. Refused, never recomputed in the
            # old format to make it match.
            self._acquisition.mark_failed(model_id, detail="legacy unframed content digest")
            raise ReplicationError(
                "verification_failed",
                f"refusing to verify {model_id!r} against the legacy unframed digest "
                f"{expected!r}; re-acquire it on the holder to record a "
                f"{DIGEST_PREFIX.rstrip(':')} digest",
                detail={"model_id": model_id, "expected": expected},
            )
        if computed != expected:
            self._acquisition.mark_failed(
                model_id, detail=f"digest mismatch: expected {expected}, computed {computed}"
            )
            raise ReplicationError(
                "verification_failed",
                f"artifact pulled for {model_id!r} does not match the expected content digest",
                detail={
                    "model_id": model_id,
                    "expected": expected,
                    "computed": computed,
                    "source_node_id": source_node_id,
                },
            )

    # ----------------------------------------------------------------- source
    def serve(self, model_id: str) -> tuple[Path, str]:
        """The source side: hand back a locally ``available`` replica's path.

        Refuses anything that is not ``available``, which is what stops a
        staging copy from propagating to a third node and being promoted there
        on the strength of having arrived.
        """
        replica = self._acquisition.get_replica(model_id)
        if replica is None:
            raise ReplicationError(
                "not_found",
                f"no replica of {model_id!r} on this host",
                detail={"model_id": model_id},
            )
        if replica.state != "available":
            raise ReplicationError(
                "replica_not_available",
                f"replica of {model_id!r} is {replica.state}, not available; refusing to serve it",
                detail={"model_id": model_id, "state": replica.state},
            )
        path = Path(replica.local_path)
        if not path.exists():
            raise ReplicationError(
                "not_found",
                f"replica of {model_id!r} is recorded available but absent on disk",
                detail={"model_id": model_id},
            )
        return path, replica.content_digest or directory_digest(path)

    def manifest(self, model_id: str) -> list[ManifestEntry]:
        """The source side of the ranged protocol: what an ``available`` replica holds."""
        path, _ = self.serve(model_id)
        return tree_manifest(path)

    def file_for(self, model_id: str, relative: str) -> Path:
        """One regular file of an ``available`` replica, contained within it.

        The requested path is a peer's input: it must be a normalized relative
        path, every component must be a real directory (no link anywhere on
        the way), and it must end at a regular file.
        """
        root, _ = self.serve(model_id)
        safe = safe_relative_path(relative)
        if safe is None:
            raise ReplicationError(
                "not_found", "not a file of this replica", detail={"model_id": model_id}
            )
        target = root
        parts = safe.split("/")
        for index, part in enumerate(parts):
            target = target / part
            try:
                mode = target.lstat().st_mode
            except FileNotFoundError:
                mode = 0
            last = index == len(parts) - 1
            if not (stat.S_ISREG(mode) if last else stat.S_ISDIR(mode)):
                raise ReplicationError(
                    "not_found", "not a file of this replica", detail={"model_id": model_id}
                )
        return target

    def stream_file(self, path: Path, start: int, end: int) -> Iterator[bytes]:
        """Yield ``path``'s bytes ``[start, end)`` straight off disk.

        No staging copy and nothing built ahead of the first byte: each chunk
        is one ``pread``. A file that shrank underneath the request ends the
        stream short, which the destination refuses.
        """
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            position = start
            while position < end:
                chunk = os.pread(fd, min(_IO, end - position), position)
                if not chunk:
                    return
                position += len(chunk)
                yield chunk
        finally:
            os.close(fd)

    def stream_archive(self, path: Path) -> Iterable[bytes]:
        """Yield the model directory as an uncompressed tar stream.

        Bounded against RAM, not against a second copy of the whole tar
        The previous version built the entire
        tar in an in-memory ``BytesIO`` before the first yield -- for a
        multi-gigabyte model, a second full copy in RAM before a single byte
        reached the wire -- that second copy in RAM is the specific harm
        that kills the agent process under memory pressure. This builds the
        same tar into a temporary file on the *same filesystem as the model
        itself* (``path.parent`` -- the managed store, which by definition
        already has room for one model's worth of data) and streams it back
        off disk in bounded chunks, the same tail this method already used
        for the read side.

        This is not byte-for-byte incremental -- the whole tar is still
        written before the first yield. Achieving that would need either a
        second thread pausing ``tarfile``'s blocking write calls (product
        code may not start one: the idle-polling guardrail forbids it
        for exactly the reason a first version of this fix ran into --
        nothing then bounds how long that thread can outlive the request
        that started it) or a hand-rolled tar writer bypassing the standard
        library's header/format handling. The named harm is
        memory, not time-to-first-byte, and this removes it completely.
        """
        with tempfile.TemporaryFile(dir=path.parent) as buffer:
            # The digest includes directories, even empty ones. Hardlinked
            # regular files also count independently; serialize their bytes
            # rather than emitting links, which the receiver refuses.
            with tarfile.open(fileobj=buffer, mode="w", dereference=True) as tar:
                for item in sorted(path.rglob("*")):
                    mode = item.lstat().st_mode
                    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                        raise UnsupportedTreeEntry(
                            f"{item.relative_to(path).as_posix()!r} is neither a regular file "
                            f"nor a directory; model trees hold only those"
                        )
                    tar.add(item, arcname=str(item.relative_to(path)), recursive=False)
            buffer.seek(0)
            while chunk := buffer.read(_CHUNK):
                yield chunk
