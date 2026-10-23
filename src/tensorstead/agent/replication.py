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

Deliberately simple in v1: HTTP streaming over whatever IP path
the two nodes already share. No topology-aware distribution, no peer swarming,
no RDMA-specific protocol, no multi-source striping.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tensorstead.agent.acquisition import AcquiredReplica, ModelAcquisitionService

# Read size for streaming and digesting; large enough to be efficient, small
# enough that a multi-gigabyte model never lands in memory at once.
_CHUNK = 1024 * 1024

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
    """The transport that fetches artifact bytes from a source agent."""

    def fetch_content(
        self,
        *,
        endpoint: str,
        model_id: str,
        token: str | None = None,
        fingerprint: str | None = None,
    ) -> Iterable[bytes]:
        """Stream the source agent's ``GET /agent/v1/models/{id}/content``."""


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
                content = hashlib.sha256()
                size = 0
                with path.open("rb") as handle:
                    while chunk := handle.read(_CHUNK):
                        content.update(chunk)
                        size += len(chunk)
                entries.append((rel, b"f", size, content.digest()))
            else:
                raise UnsupportedTreeEntry(
                    f"{rel.decode('utf-8', 'replace')!r} is neither a regular file nor a "
                    f"directory; model trees hold only those"
                )
    entries.sort(key=lambda entry: entry[0])
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
    ) -> ReplicationOutcome:
        """Pull → stage → verify → promote.

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
            self._pull_into(
                archive, model_id, source_endpoint, replication_token, source_fingerprint
            )
            self._unpack(archive, staging_dir)
            computed = directory_digest(staging_dir)
            self._verify(model_id, computed, content_digest, source_node_id)
            size_bytes = _directory_size(staging_dir)
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
            size_bytes=size_bytes,
            content_digest=computed,
        )
        return ReplicationOutcome(
            model_id=replica.model_id,
            state=replica.state,
            content_digest=replica.content_digest,
            size_bytes=replica.size_bytes,
        )

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
        free = shutil.disk_usage(archive.parent).free
        if free < _MIN_FREE_BYTES_TO_ATTEMPT:
            raise ReplicationError(
                "artifact_too_large",
                f"refusing to start pulling {model_id!r}: only {free} bytes free, "
                f"below the {_MIN_FREE_BYTES_TO_ATTEMPT} byte minimum to attempt a transfer",
                detail={"model_id": model_id, "free_bytes": free, "ceiling_bytes": limit},
            )
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
            with tarfile.open(archive, "r:") as tar:
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
