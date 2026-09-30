"""Upstream acquisition on the agent.

The agent's model store lives on the node host. ``acquire`` stages a model into
a staging location, verifies it against the expected digest, then **atomically
promotes** it to ``available`` — so an interrupted or unverifiable transfer can
never be presented as an available replica. Staging plus atomic
promotion makes that structural rather than a check that could be skipped.

The agent has no database: its model store is the host filesystem. The staging
→ verified → available lifecycle is recorded on the filesystem via a marker, so
the fake agent and the real agent agree on the behaviour the conformance suite
asserts.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# ``local_model_id`` lives in the domain because the coordinator derives the
# same path when it records a replica; re-exported here so the agent's own
# callers keep importing it from the module that owns the store.
from tensorstead.domain.models import local_model_id

__all__ = ["local_model_id"]
from tensorstead.ports.model_source import AcquireProgress


class ModelAcquireError(Exception):
    """An upstream model acquisition failed after staging began."""


def _redact(text: str, credential: str | None) -> str:
    """Keep a per-request credential out of a detail that is written to disk.

    The failure detail is persisted in the replica marker and returned to the
    operator, and an upstream refusal is exactly the message most likely to
    quote the token it refused ("401 for token hf_..."). The
    credential is never written to the agent's disk or logs, so it is stripped
    here rather than at either call site.
    """
    if not credential:
        return text
    return text.replace(credential, "[REDACTED]")


@dataclass
class AcquiredReplica:
    """A model revision present on the node in a known state."""

    model_id: str
    resolved_revision: str | None
    local_path: str
    state: str  # staging | available | failed
    verified_at: datetime | None = None
    size_bytes: int | None = None
    content_digest: str | None = None


def _now() -> datetime:
    return datetime.now().astimezone()


def _safe_token(model_id: str) -> str:
    """Encode a model id (may contain ``:``, ``/``) into a safe filename token.

    Base64-url encoding is reversible, so ``list_replicas`` can recover the
    original model id from the marker filename.
    """
    import base64

    return base64.urlsafe_b64encode(model_id.encode()).decode().rstrip("=")


class ModelAcquisitionService:
    """Stage → verify → promote model acquisitions on the agent host.

    ``store_dir`` is the model store root (e.g. ``/var/lib/tensorstead/models``).
    ``marker_dir`` records each replica's state so the agent can answer
    ``GET /agent/v1/models`` without a database — the host filesystem is the
    source of truth.
    """

    def __init__(self, *, store_dir: Path, marker_dir: Path) -> None:
        self._store_dir = store_dir
        self._marker_dir = marker_dir
        # Per-model lock for acquire serialization. Two concurrent acquires
        # of the same model on one node must not both download into staging and
        # race the promote: the first installs an available replica, the second
        # reuses it. The lock is held across resolve → acquire → verify →
        # promote so the re-check after resolve sees the first's committed
        # marker rather than racing past an empty store.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, model_id: str) -> threading.Lock:
        """Return the per-model acquire lock, creating it on first use."""
        with self._locks_guard:
            if model_id not in self._locks:
                self._locks[model_id] = threading.Lock()
            return self._locks[model_id]

    def _ensure_dirs(self) -> None:
        """Create the store and marker dirs on first use, not at construction.

        Building the agent app must not require write access to ``/var/lib``;
        the dirs are created only when acquisition actually runs (the agent may
        be inspected or run on a host where these are pre-provisioned).
        """
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._marker_dir.mkdir(parents=True, exist_ok=True)

    def _replica_marker(self, model_id: str) -> Path:
        # ``source_id:source_model_id`` can contain ``:`` and ``/`` (repo ids
        # are ``org/model``), neither of which is a safe single filename.
        # Encode to a filesystem-safe token.
        return self._marker_dir / f"{_safe_token(model_id)}.json"

    def acquire(
        self,
        source: Any,
        *,
        model_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None,
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> AcquiredReplica:
        """Resolve a revision, stage the download, verify, then promote.

        On any exception before ``promote()`` is called, the replica is
        explicitly marked ``failed``. A ``promote()`` failure itself is not
        additionally marked here: ``promote()`` restores whatever state
        preceded the call (a prior ``available`` replica back in place, or
        nothing at all for a first acquire) rather than leaving a marker this
        method would then have to overwrite.
        Either way, no exception from this method ever leaves the marker
        claiming a replica is ``available`` when it is not. Returns the
        promoted ``AcquiredReplica`` on success.

        Serialized per model id: a concurrent second acquire of the same
        model reuses the first's available replica instead of racing it to the
        same staging/promote target. The re-check after ``resolve`` is what
        catches ``revision=None`` -- both calls resolve ``main`` to the same
        sha, so the second sees the first's available replica and returns it
        without re-downloading.
        """
        self._ensure_dirs()
        with self._lock_for(model_id):
            return self._acquire_locked(
                source,
                model_id=model_id,
                source_model_id=source_model_id,
                revision=revision,
                credential=credential,
                progress=progress,
                file_selector=file_selector,
            )

    def _acquire_locked(
        self,
        source: Any,
        *,
        model_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None,
        progress: AcquireProgress | None,
        file_selector: tuple[str, ...] = (),
    ) -> AcquiredReplica:
        """The acquire body, run under the per-model lock."""

        def _report(fraction: float, message: str) -> None:
            if progress is not None:
                progress(fraction, message)

        _report(0.0, "resolving revision")
        if hasattr(source, "resolve"):
            resolved = source.resolve(source_model_id, revision)
        else:
            resolved = revision or "main"

        # Re-check under the lock: a prior acquire for this model may have
        # already installed an available replica at this exact revision (a
        # concurrent call, or a retry after a timeout that the caller could not
        # distinguish from "never received"). Reusing it is the idempotent
        # outcome -- the alternative is a second download into a fresh staging
        # dir racing the first's promote and a duplicate on-node copy.
        # This is the layer that catches ``revision=None``: both calls resolve
        # ``main`` to the same sha, so the second sees the first's replica.
        existing = self.get_replica(model_id)
        if (
            existing is not None
            and existing.state == "available"
            and existing.resolved_revision == resolved
        ):
            _report(1.0, "available (reused)")
            return existing

        # Stage: download into a per-call unique staging dir.
        staging_dir = self.staging_path(model_id)
        staging_dir.mkdir(parents=True, exist_ok=True)

        _report(0.2, f"downloading {source_model_id}@{resolved}")
        try:
            result = source.acquire(
                source_model_id,
                resolved,
                str(staging_dir),
                credential,
                progress=progress,
                file_selector=file_selector,
            )
        except Exception as exc:
            # A failed transfer is neither reusable nor evidence of a replica.
            # Remove only this call's UUID-scoped staging tree; sibling acquires
            # and a prior promoted replica remain untouched.
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            detail = _redact(str(exc), credential)
            self._mark_failed(model_id, detail, existing=existing)
            raise ModelAcquireError(detail) from exc
        size_bytes = getattr(result, "size_bytes", None)
        content_digest = getattr(result, "content_digest", None)

        _report(0.9, "verifying")
        # Structural verification of the staged tree before the rename that
        # would make it count as ``available``. An interrupted download
        # can leave the tree missing ``config.json`` or a shard; without this
        # check the digest the source reports would be recorded and
        # ``verified_at`` stamped regardless, so a broken tree would be
        # promoted. The check is structural only -- the source is
        # authoritative for the content digest on the upstream path, and peer
        # replication recomputes one (agent/replication.py). Source-format
        # knowledge stays behind the source adapter, so the call
        # is guarded the same way ``resolve`` is: a source without ``verify``
        # keeps the previous behaviour. A failed verification marks the
        # replica ``failed`` and re-raises -- never promoted.
        if hasattr(source, "verify"):
            try:
                source.verify(staging_dir)
            except Exception as exc:
                # Same disposal as a failed transfer above: an unpromotable tree
                # is bytes nobody will read, and the detail it raises reaches the
                # same marker, so it gets the same redaction. The exception type
                # is deliberately *not* wrapped -- a verify failure is this
                # agent's own refusal to promote, not an upstream error, and
                # callers pin the original type.
                import shutil

                shutil.rmtree(staging_dir, ignore_errors=True)
                self._mark_failed(model_id, _redact(str(exc), credential), existing=existing)
                raise

        # The upstream source is authoritative for what it downloaded, so the
        # digest it reports is recorded rather than recomputed here. Peer
        # replication is the case that must verify (it is copying someone
        # else's claim), and it does — see agent/replication.py.
        replica = self.promote(
            model_id,
            staging_dir,
            resolved_revision=resolved,
            size_bytes=size_bytes,
            content_digest=content_digest,
        )
        _report(1.0, "available")
        return replica

    # ------------------------------------------- shared stage→promote surface
    def staging_path(self, model_id: str) -> Path:
        """Where a transfer is written before it is allowed to count.

        Public because peer replication stages through the *same* path
        and promotes through the *same* method. One promotion path means "a
        staging copy never becomes available" is enforced in one place rather
        than reimplemented per transfer mechanism.

        The staging name carries a random suffix so two concurrent acquires of
        the same model never share a staging directory: the deterministic
        ``{token}.staging`` name let a second acquire clobber the first's staging
        tree mid-download. Promotion stays deterministic (``_model_dir``), so
        the lock in ``acquire``, not the staging name, serializes the
        rename. Peer replication derives its archive name from
        ``staging_dir.name``, so the unique suffix flows through unchanged.
        """
        self._ensure_dirs()
        return self._model_dir(model_id).with_name(f"{_safe_token(model_id)}.staging.{uuid4().hex}")

    def promote(
        self,
        model_id: str,
        staging_dir: Path,
        *,
        resolved_revision: str | None = None,
        size_bytes: int | None = None,
        content_digest: str | None = None,
    ) -> AcquiredReplica:
        """Promote a verified staging directory to ``available``.

        The previous good replica, if any, is never destroyed before the
        replacement is installed and marked:
        it used to be removed unconditionally up front, so a rename that then
        failed -- a transient filesystem error, a killed process -- left no
        usable copy on disk while the marker still claimed the old one was
        ``available``. Now the old replica is moved aside to a sibling
        rollback name first and only deleted after the new one is installed
        *and* the marker rewritten to describe it; any failure in between
        restores the rollback copy to the final path and leaves the prior
        marker untouched, so the worst a failed replacement can do is leave
        the model exactly as it was before the attempt.
        """
        import shutil

        self._ensure_dirs()
        final_dir = self._model_dir(model_id)
        final_dir.parent.mkdir(parents=True, exist_ok=True)

        rollback_dir: Path | None = None
        if final_dir.exists():
            rollback_dir = final_dir.with_name(f"{final_dir.name}.rollback.{uuid4().hex}")
            os.rename(final_dir, rollback_dir)

        try:
            try:
                os.rename(staging_dir, final_dir)
            except OSError:
                # Atomic rename of a directory may fail cross-device; fall
                # back to a copy, which is still a single logical promote.
                # `final_dir` was already vacated above (or never existed),
                # so this clears only whatever the failed rename may have
                # partially left, never the prior good replica.
                shutil.rmtree(final_dir, ignore_errors=True)
                shutil.copytree(staging_dir, final_dir)
                shutil.rmtree(staging_dir, ignore_errors=True)

            replica = AcquiredReplica(
                model_id=model_id,
                resolved_revision=resolved_revision,
                local_path=str(final_dir),
                state="available",
                verified_at=_now(),
                size_bytes=size_bytes,
                content_digest=content_digest,
            )
            self._mark_available(replica)
        except BaseException:
            shutil.rmtree(final_dir, ignore_errors=True)
            if rollback_dir is not None:
                os.rename(rollback_dir, final_dir)
            raise

        if rollback_dir is not None:
            shutil.rmtree(rollback_dir, ignore_errors=True)
        return replica

    def mark_failed(self, model_id: str, detail: str) -> None:
        """Record a replica as ``failed`` so it is never offered as available."""
        self._ensure_dirs()
        self._mark(model_id, "failed", detail=detail)

    def local_path(self, model_id: str) -> Path:
        """The promoted location of a replica on this host."""
        return self._model_dir(model_id)

    def _resolve_within_store(self, value: str) -> Path | None:
        """Canonicalize ``value``, returned only if it resolves inside the store.

        Shared containment logic: extracted so
        the primary model path (``require_model_path``) and secondary model
        references (``managed_model_path``) apply exactly
        the same rule rather than two independently-maintained copies of it.

        ``resolve()`` before comparison is load-bearing: it collapses ``..``
        and follows symlinks, so a path that only *looks* contained -- or one
        inside the store pointing back out via a symlink -- is caught rather
        than trusted for where it appears to live.
        """
        try:
            candidate = Path(value).resolve()
            root = self._store_dir.resolve()
        except (OSError, RuntimeError):
            return None
        if candidate == root or root not in candidate.parents:
            return None
        if not candidate.is_dir():
            return None
        return candidate

    def managed_model_path(self, value: str) -> str | None:
        """Return ``value`` as a mountable path if this store owns it, else None.

        A runtime adapter may name configuration values that refer to further
        model weights -- a speculative-decoding drafter is the case that forced
        this. Those values come from operator configuration,
        so they cannot be mounted on the adapter's word alone: a string that
        became a bind-mount would let a deployment record name arbitrary host
        access, which is exactly what ``host_config`` refuses and what this rule
        exists to prevent.

        The rule is ownership, not shape. A path resolving inside this store's
        root, that exists, is something this agent downloaded, recorded and can
        account for -- mounting it is returning the operator's own acquisition
        to them. Anything else -- a HuggingFace repo id, a relative path, an
        absolute path elsewhere on the host, or a symlink pointing out of the
        store -- resolves to nothing and is left alone. The runtime then
        behaves exactly as it did before this existed, fetching what it was
        given.
        """
        if not value or ":" in value:
            # ':' is either a source-qualified id or a Docker mount separator.
            # Neither is a path this store hands out.
            return None
        candidate = self._resolve_within_store(value)
        return str(candidate) if candidate is not None else None

    def require_model_path(self, value: str) -> Path:
        """Refuse a primary model path that does not resolve inside the store.

        The primary model path used to reach
        Docker as a read-only host bind mount with no containment check at
        all: ``runtime_model_path`` only migrates a legacy colon-containing
        name, it never verified the result lives anywhere this agent
        manages. A management/MCP caller naming an outside directory with a
        minimal valid model shape (a bare ``config.json``) reached the host
        filesystem directly, read-only but still arbitrary. Secondary model
        references (a speculative-decoding drafter) already went through
        this same containment logic via ``managed_model_path``; this applies
        it to the path that matters most.

        Raises rather than returning ``None`` the way ``managed_model_path``
        does: a secondary reference that fails containment is legitimately
        left alone (it might be an upstream repo id the runtime resolves
        itself), but there is no such fallback for the primary path -- it is
        either a model this agent owns, or the start must refuse.
        """
        candidate = self._resolve_within_store(value)
        if candidate is None:
            raise ValueError(
                f"model path {value!r} does not resolve to a directory beneath "
                f"the managed store {self._store_dir.resolve()}"
            )
        return candidate

    def runtime_model_path(self, recorded_path: str) -> str:
        """Return a Docker-safe model path, migrating a legacy path if needed.

        Early releases stored a source-qualified model ID directly in the
        directory name (for example ``huggingface:nvidia/model``). Docker uses
        ``:`` as a bind-mount separator, so that valid host path cannot be
        mounted.  New stores use a URL-safe token.  When an older coordinator
        names a legacy path, rename it in-place on the same filesystem and
        update its local marker; this preserves the downloaded model bytes.
        """
        if ":" not in recorded_path:
            return recorded_path
        legacy_dir = Path(recorded_path)
        try:
            model_id = str(legacy_dir.relative_to(self._store_dir))
        except ValueError:
            return recorded_path
        safe_dir = self._model_dir(model_id)
        if legacy_dir.exists() and not safe_dir.exists():
            safe_dir.parent.mkdir(parents=True, exist_ok=True)
            os.rename(legacy_dir, safe_dir)
            replica = self.get_replica(model_id)
            if replica is not None:
                self._mark_available(
                    AcquiredReplica(
                        model_id=replica.model_id,
                        resolved_revision=replica.resolved_revision,
                        local_path=str(safe_dir),
                        state=replica.state,
                        verified_at=replica.verified_at,
                        size_bytes=replica.size_bytes,
                        content_digest=replica.content_digest,
                    )
                )
        return str(safe_dir) if safe_dir.exists() else recorded_path

    def _model_dir(self, model_id: str) -> Path:
        """One Docker-safe directory name for a source-qualified model ID."""
        return self._store_dir / _safe_token(model_id)

    def _mark_failed(self, model_id: str, detail: str, *, existing: AcquiredReplica | None) -> None:
        """Record a failed transfer without erasing an intact replica's record.

        ``_mark`` rewrites the whole marker, so marking a failure unconditionally
        replaced the description of a replica that had not failed. The ordinary
        case is a re-acquire at a *new* revision that 403s: the previously
        promoted tree is still on disk under ``_model_dir`` and still good, but
        the marker was rewritten to ``failed`` with no ``local_path``, so
        ``GET /agent/v1/models`` misreported the node and ``agent/replication``
        refused to serve bytes that were fine (it requires ``available``). That
        is the estate's defining failure mode: a record that stopped matching
        reality *because* something adjacent to it went wrong.

        The marker describes what is present on the node. A failed attempt that
        installed nothing changes nothing about what is present, so it leaves
        the marker alone; the failure reaches the operator by the raised error
        and the 502 it becomes, which is where a transient event belongs. Only
        when there is no intact replica to describe does ``failed`` become the
        truthful record of the model's state.
        """
        if existing is not None and existing.state == "available":
            return
        self._mark(model_id, "failed", detail=detail)

    def _mark(self, model_id: str, state: str, *, detail: str | None = None) -> None:
        record = {
            "model_id": model_id,
            "state": state,
            "updated_at": _now().isoformat(),
            "detail": detail,
        }
        self._write_marker_atomically(model_id, record)

    def _mark_available(self, replica: AcquiredReplica) -> None:
        record = {
            "model_id": replica.model_id,
            "resolved_revision": replica.resolved_revision,
            "state": "available",
            "local_path": replica.local_path,
            "verified_at": replica.verified_at.isoformat() if replica.verified_at else None,
            "size_bytes": replica.size_bytes,
            "content_digest": replica.content_digest,
            "updated_at": _now().isoformat(),
        }
        self._write_marker_atomically(replica.model_id, record)

    def _write_marker_atomically(self, model_id: str, record: dict[str, Any]) -> None:
        """Write a marker so a reader never observes a partially-written file.

        ``Path.write_text`` is open, write, close -- not atomic. A crash or
        kill between the write and the close leaves a truncated marker, which
        ``list_replicas`` treats as absent (it catches ``ValueError`` from
        ``json.loads``) -- a safe direction for a ``failed`` marker, but for
        the ``available`` marker ``promote()`` now depends on this write
        genuinely being all-or-nothing: writing to a sibling temp file and
        ``os.replace``-ing over the real path is atomic on the same
        filesystem.
        """
        import json

        marker = self._replica_marker(model_id)
        fd, tmp_name = tempfile.mkstemp(dir=marker.parent, prefix=f".{marker.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle)
            os.replace(tmp_name, marker)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def list_replicas(self) -> list[AcquiredReplica]:
        """Return every model replica on this host with its current state."""
        import json

        if not self._marker_dir.is_dir():
            return []
        replicas: list[AcquiredReplica] = []
        for marker in self._marker_dir.glob("*.json"):
            try:
                record = json.loads(marker.read_text())
            except (OSError, ValueError):
                continue
            state = record.get("state", "unknown")
            replicas.append(
                AcquiredReplica(
                    model_id=record["model_id"],
                    resolved_revision=record.get("resolved_revision"),
                    local_path=record.get("local_path", ""),
                    state=state,
                    verified_at=(
                        datetime.fromisoformat(record["verified_at"])
                        if record.get("verified_at")
                        else None
                    ),
                    size_bytes=record.get("size_bytes"),
                    content_digest=record.get("content_digest"),
                )
            )
        return replicas

    def get_replica(self, model_id: str) -> AcquiredReplica | None:
        for replica in self.list_replicas():
            if replica.model_id == model_id:
                return replica
        return None

    def remove_replica(self, model_id: str) -> None:
        """Remove a model replica and its marker from this host.

        Unconditional at this layer — the coordinator's *referenced* check
        governs whether deletion is permitted. Removes the model
        store directory and the marker file.
        """
        import shutil

        marker = self._replica_marker(model_id)
        if marker.exists():
            marker.unlink()
        final_dir = self._model_dir(model_id)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
