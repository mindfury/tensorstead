"""Peer-to-peer image distribution.

Building a spec independently on each node produces a *different* identifier
for the same recipe. That silently breaks the comparison and makes
`image_digest_mismatch` divergence meaningless — a failure that looks like
success, which this product has been bitten by more than once.

So an image is produced **once** and distributed, following the same shape
already established for model artifacts: the destination owns the
operation, pulls from the source, stages, verifies, and only then promotes.

No image byte passes through the coordinator. The coordinator
tells a destination where to fetch from; the transfer is agent to agent.
"""

from __future__ import annotations

import os
import shutil
import ssl
from pathlib import Path
from typing import Any

# A configured ceiling on one artifact transfer:
# the content endpoint below does not advertise an expected size up front, so
# this is the only bound available -- it stops a runaway or malicious peer
# from writing until the managed image store fills, since the request timeout
# bounds duration, not bytes.
_DEFAULT_MAX_TRANSFER_BYTES = 200 * 1024 * 1024 * 1024  # 200 GiB

# The preflight's own threshold, deliberately not the ceiling above -- see the
# identical constant and comment in replication.py, where review caught this
# comparison refusing an ordinary small
# transfer on a host with less free space than the (generous, worst-case)
# ceiling.
_MIN_FREE_BYTES_TO_ATTEMPT = 1 * 1024 * 1024 * 1024  # 1 GiB


def _max_transfer_bytes() -> int:
    raw = os.environ.get("TENSORSTEAD_MAX_ARTIFACT_BYTES")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_TRANSFER_BYTES
    return int(raw)


class ImageDistributionError(RuntimeError):
    """A distribution failed on this node, carrying the reference at fault."""

    def __init__(self, message: str, *, reference: str) -> None:
        super().__init__(message)
        self.message = message
        self.reference = reference


class ImageDistributionService:
    """Pull an image archive from a peer agent, verify it, then load it."""

    def __init__(self, engine: Any, store_path: str, ca_bundle: str | None = None) -> None:
        self._engine = engine
        self._store = Path(store_path)
        self._ca_bundle = ca_bundle

    def pull_from_peer(
        self,
        *,
        reference: str,
        source_endpoint: str,
        expected_image_id: str,
        token: str | None = None,
        fingerprint: str | None = None,  # noqa: ARG002 - see below
    ) -> str:
        """Fetch, stage, verify, then load. Never load an unverified archive.

        ``fingerprint`` is accepted and unused: certificate pinning is not
        this product's chosen trust mechanism (``coordinator/node_http.py``'s
        own ``_pin_fingerprint`` was already removed on the same grounds --
        "a control read as present while being absent"). CA verification via
        ``ca_bundle`` is the real, load-bearing check, immediately below. The
        parameter stays in the call shape so every peer client keeps the same
        signature. It is kept for compatibility rather than because enforcing
        it is planned.
        """
        import httpx

        if not source_endpoint.startswith("https://"):
            raise ImageDistributionError(
                f"refusing to fetch {reference!r} from non-https peer endpoint {source_endpoint!r}",
                reference=reference,
            )

        self._store.mkdir(parents=True, exist_ok=True)
        staged = self._store / f".staging-{expected_image_id.replace(':', '_')}.tar"
        url = f"{source_endpoint.rstrip('/')}/agent/v1/images/{reference}/content"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # Built as a context rather than passed as a path: httpx deprecated
        # ``verify=<str>`` and will remove it, and the path it would remove is
        # precisely the managed-CA path this hop depends on.
        # A deprecation on the only branch that establishes trust is not one to
        # leave until it becomes a failure.
        verify: bool | ssl.SSLContext = (
            ssl.create_default_context(cafile=self._ca_bundle) if self._ca_bundle else True
        )

        # Bounded against both a full disk and a runaway peer:
        # the content endpoint advertises no expected size, so the
        # preflight and the running total below are both against the
        # configured ceiling rather than a per-transfer figure.
        limit = _max_transfer_bytes()
        free = shutil.disk_usage(self._store).free
        if free < _MIN_FREE_BYTES_TO_ATTEMPT:
            raise ImageDistributionError(
                f"refusing to start pulling {reference!r}: only {free} bytes free, "
                f"below the {_MIN_FREE_BYTES_TO_ATTEMPT} byte minimum to attempt a transfer",
                reference=reference,
            )

        try:
            written = 0
            with (
                httpx.Client(verify=verify, timeout=1800.0) as client,
                client.stream("GET", url, headers=headers) as response,
                staged.open("wb") as handle,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > limit:
                        raise ImageDistributionError(
                            f"peer for {reference!r} sent more than the configured "
                            f"ceiling of {limit} bytes; aborting",
                            reference=reference,
                        )
                    handle.write(chunk)
        except Exception as exc:
            staged.unlink(missing_ok=True)
            if isinstance(exc, ImageDistributionError):
                raise
            raise ImageDistributionError(
                f"could not fetch {reference!r} from {source_endpoint}: {exc}",
                reference=reference,
            ) from exc

        return self._load_and_verify(
            staged, expected_image_id=expected_image_id, reference=reference
        )

    def _load_and_verify(self, staged: Path, *, expected_image_id: str, reference: str) -> str:
        """Load a staged archive and refuse it unless it matches ``expected_image_id``.

        Split out from ``pull_from_peer`` so the load/verify/cleanup sequence
        -- the part with security consequence -- is directly unit-testable
        without a real network fetch.
        """
        try:
            loaded = self._engine.import_image(archive_path=str(staged))
        except Exception as exc:
            raise ImageDistributionError(
                f"archive for {reference!r} could not be loaded: {exc}", reference=reference
            ) from exc
        finally:
            # The staged copy is removed whether or not the load succeeded: a
            # half-transferred archive left behind is a later import waiting to
            # pick up the wrong bytes.
            staged.unlink(missing_ok=True)

        if loaded != expected_image_id:
            # The mismatched image is already loaded in the daemon -- the
            # engine has no way to know its identity without loading it first
            # (base.py's import_image). Left alone, it is a poisoned tag the
            # local-first start path could pick up later despite this call
            # reporting failure. Removal
            # failing does not change the fact that mattered here: it is
            # appended to the message, not raised in place of it.
            cleanup_detail = ""
            try:
                self._engine.remove_image(image_id=loaded, force=True)
            except Exception as exc:
                cleanup_detail = f"; additionally, the rejected image could not be removed: {exc}"
            raise ImageDistributionError(
                f"{reference!r} arrived as {loaded!r}, expected {expected_image_id!r}; "
                f"every participating node must hold the same image{cleanup_detail}",
                reference=reference,
            )

        # Identity is not integrity. The check above says *which* image arrived;
        # this one says whether it arrived whole. An image whose config or layer
        # blobs never landed still loads under the expected id and still
        # resolves -- it fails only later, at container creation, as an HTTP 500
        # from the deployment route that names neither the image nor this hop
        # Distribution reported 200 OK twice over a TP=2
        # pair on exactly that path.
        try:
            self._engine.verify_image_materializable(image_id=loaded)
        except Exception as exc:
            # Removed on the same reasoning as a mismatch, and with more force
            # behind it: incomplete content in the daemon's store is sticky, so
            # a later load of this same id deduplicates against it, re-tags, and
            # reports success without repairing anything. Left in place it does
            # not merely poison this tag, it poisons every retry.
            cleanup_detail = ""
            try:
                self._engine.remove_image(image_id=loaded, force=True)
            except Exception as removal_exc:
                cleanup_detail = (
                    f"; additionally, the unusable image could not be removed: {removal_exc}"
                    f" -- remove it before retrying, or the retry will inherit this state"
                )
            raise ImageDistributionError(
                f"{reference!r} arrived as {loaded!r} but is not usable on this node: "
                f"{exc}{cleanup_detail}",
                reference=reference,
            ) from exc
        return str(loaded)
