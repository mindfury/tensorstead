"""Model-source port (seam).

A model source is authoritative for model identity, revision semantics, and
authorization. The v1 implementation is Hugging Face
(``adapters/sources/huggingface.py``); the port keeps the domain
independent of any particular upstream.

``supports_revision_pinning`` drives revision pinning: a source that cannot pin
must be reported as explicitly unpinned, never given a fabricated revision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AcquireProgress(Protocol):
    """A callable the source calls as retrieval makes progress."""

    def __call__(self, fraction: float, message: str) -> None: ...


class ModelSource(Protocol):
    """Retrieve a model revision into a local store on behalf of an agent."""

    supports_revision_pinning: bool
    requires_credential: bool

    def source_id(self) -> str:
        """Return the source identifier (e.g. ``"huggingface"``)."""

    def resolve(self, model_id: str, revision: str | None) -> str:
        """Resolve ``revision`` (or the source default) to a concrete revision.

        Returns the revision the source resolved. A source that cannot pin
        raises ``UnpinnableError`` so callers can record ``revision_pinned``
        explicitly rather than fabricate a value.
        """

    def acquire(
        self,
        model_id: str,
        revision: str,
        destination_dir: str,
        credential: str | None,
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> Any:
        """Download a resolved revision into ``destination_dir``.

        ``credential`` is the resolved secret value for this request; it must
        be passed in-memory and never written to disk or logs. Returns a
        result object exposing ``size_bytes`` and
        ``content_digest``.

        ``file_selector`` names which files of the revision to retrieve, as
        glob patterns; empty means all of them. It exists because a repository
        is not always one model -- GGUF publishers routinely ship twenty-odd
        quantizations of the same weights in one repo, and acquiring all of
        them to use one is not a realistic download. The patterns are recorded
        on the model because they are part of its identity (migration 0006);
        a source that cannot filter should retrieve everything rather than
        silently retrieve something else.
        """

    def verify(self, destination_dir: Path) -> None:
        """Structurally check a downloaded tree before it is promoted.

        Raises when the tree is not usable as the source expects a usable tree
        to look -- e.g. a missing ``config.json`` or a safetensors shard named
        by the index but absent or empty. The check is structural only; it is
        not a content digest (the source is authoritative for that on the
        upstream path; peer replication recomputes one). Source-format
        knowledge stays in the source adapter.
        """


class UnpinnableError(Exception):
    """The source cannot pin a revision.

    The caller must record ``revision_pinned = False`` and ``resolved_revision
    = None``, stating the lack of a guarantee rather than fabricating one.
    """
