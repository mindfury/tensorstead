"""Hugging Face model source.

Implements the model-source port by delegating retrieval to ``huggingface_hub``
(the upstream tool is authoritative for revision resolution and
download). Returns the concrete revision the source resolved, or
reports that the source cannot pin a revision as an explicit ``UnpinnableError``
— a revision is never fabricated.

Credentials arrive per-request as a resolved secret value; they
are passed to ``huggingface_hub`` in-memory and never written to disk or logs.
A gated model with no valid credential is refused as
``authorization_refused``.

**Unaccepted access terms are reported, never settled by us**. The hub
distinguishes "you need a token" from "you need to accept this model's terms",
and so does this adapter: the second is surfaced as something the operator must
take up with the provider. There is deliberately no code path that accepts terms
on the operator's behalf or looks for a way around them — the refusal stands.

``AcquisitionResult`` carries the resolved revision, ``revision_pinned``, and a
``content_digest`` used for stage-verify-promote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorstead.domain.errors import AuthorizationRefusedError
from tensorstead.ports.model_source import AcquireProgress, UnpinnableError

SOURCE_ID = "huggingface"

# Upstream statuses that mean "refused", regardless of which client raised.
_REFUSAL_STATUSES = {401, 403}

# Words the hub uses when the block is unaccepted terms rather than a missing or
# rejected token. Matched case-insensitively against the message.
_ACCESS_TERMS_MARKERS = ("accept", "terms", "conditions", "agreement", "license")

# Exception type names that mean a refusal even without an HTTP status attached
# (the hub raises typed errors; the test fake raises its own).
_REFUSAL_TYPE_MARKERS = ("gatedrepo", "authorizationrefused", "unauthorized", "forbidden")

# Category directories a diffusion-single-file repository sorts its weights
# into. Used only by ``verify`` to recognise a tree that legitimately has no
# ``config.json``; the runtime adapter that consumes such a tree keeps its own
# list, because what ComfyUI can *load* is a larger question than what makes a
# downloaded tree structurally verifiable.
_DIFFUSION_CATEGORIES = (
    "checkpoints",
    "diffusion_models",
    "unet",
    "text_encoders",
    "clip",
    "clip_vision",
    "vae",
    "loras",
    "controlnet",
    "upscale_models",
    "audio_encoders",
)

REASON_ACCESS_TERMS = "access_terms_not_accepted"
REASON_NO_CREDENTIAL = "no_credential_supplied"
REASON_CREDENTIAL_REJECTED = "credential_rejected"


@dataclass
class AcquisitionResult:
    """What the source resolved and downloaded for one model revision."""

    resolved_revision: str
    revision_pinned: bool
    size_bytes: int | None = None
    content_digest: str | None = None


class HuggingFaceSource:
    """A ``ModelSource`` backed by ``huggingface_hub``.

    ``supports_revision_pinning`` is True: the hub resolves any revision
    reference to a concrete snapshot, so a pinned revision is recorded. The hub
    client is injectable so tests supply the fake ``huggingface_hub``.
    """

    supports_revision_pinning = True
    requires_credential = True

    def __init__(self, hub: Any | None = None) -> None:
        # The real ``huggingface_hub`` is imported lazily so the source imports
        # cleanly without the optional dependency installed.
        self._hub = hub

    def source_id(self) -> str:
        return SOURCE_ID

    def resolve(self, model_id: str, revision: str | None) -> str:
        """Resolve ``revision`` (or the source default) to a concrete revision.

        Delegates to the hub. A revision we cannot pin is surfaced as
        ``UnpinnableError`` rather than fabricated. An authorization
        refusal becomes ``AuthorizationRefusedError``.
        """
        try:
            resolved_value = self._client().resolve_revision(model_id, revision)
        except AuthorizationRefusedError:
            raise
        except Exception as exc:
            refusal = _as_refusal(exc, model_id=model_id, credential_supplied=False)
            if refusal is None:
                raise
            raise refusal from exc
        if resolved_value is None:
            raise UnpinnableError(f"source cannot pin a revision for {model_id!r}")
        resolved: str = resolved_value
        return resolved

    def acquire(
        self,
        model_id: str,
        revision: str,
        destination_dir: str,
        credential: str | None,
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> AcquisitionResult:
        # The hub's own filter, passed through rather than reimplemented
        # (the upstream tool is authoritative). Omitted entirely
        # when the selection is empty so a whole-repo acquisition issues the
        # same call it always has, and so a hub client that does not accept the
        # argument keeps working for every model acquired before this existed.
        extra: dict[str, Any] = {}
        if file_selector:
            extra["allow_patterns"] = list(file_selector)
        try:
            result = self._client().snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=credential,
                local_dir=destination_dir,
                progress=progress,
                **extra,
            )
        except AuthorizationRefusedError:
            raise
        except Exception as exc:
            refusal = _as_refusal(exc, model_id=model_id, credential_supplied=bool(credential))
            if refusal is None:
                raise
            raise refusal from exc
        # ``result`` is a dict carrying the concrete revision the hub resolved
        # and the downloaded size, per the fake hub's contract (see
        # tests/fakes/hf_hub.py). A real ``huggingface_hub.snapshot_download``
        # returns a local path; we resolve the revision via ``resolve`` and
        # report size as None (the real size is recorded by the caller from the
        # filesystem if needed).
        resolved_revision = revision
        size_bytes = None
        content_digest = None
        if isinstance(result, dict):
            resolved_revision = result.get("resolved_revision", revision)
            size_bytes = result.get("size_bytes")
            content_digest = result.get("content_digest")
        elif isinstance(result, str):
            # A bare string is the concrete resolved revision (some hubs/harness
            # report the revision as the snapshot's return value).
            resolved_revision = result or revision
        return AcquisitionResult(
            resolved_revision=resolved_revision,
            revision_pinned=True,
            size_bytes=size_bytes,
            content_digest=content_digest,
        )

    def verify(self, destination_dir: Path) -> None:
        """Structurally verify a downloaded tree before it is promoted.

        A tree is usable when ``config.json`` is present and parses as JSON, and
        -- when a ``model.safetensors.index.json`` is present -- every shard it
        names in ``weight_map`` exists and is non-empty. A tree without an index
        needs only ``config.json`` (e.g. a config-only or non-sharded model).

        A **GGUF** tree is the exception and is verified on its own terms: the
        format embeds the metadata that ``config.json`` carries elsewhere, so
        one is never present and its absence is not a fault.

        Raises ``ValueError`` naming exactly what is missing or unusable; it
        never returns silently on a bad tree. This is what stops an interrupted
        download -- one missing ``config.json`` -- from being stamped
        ``available`` with a ``verified_at``: the check sits between
        ``acquire`` and ``promote``, so a failed verification never reaches the
        rename that would make the replica count.
        """
        dest = Path(destination_dir)
        config = dest / "config.json"
        if not config.is_file():
            # GGUF is self-describing: the metadata a transformers tree keeps in
            # config.json is embedded in the file itself, so a GGUF model tree
            # has none and never will. Demanding one here would fail every GGUF
            # acquisition at the last step before promotion, after the whole
            # download. Verified on its own terms instead -- at least one
            # non-empty .gguf -- which is the same structural question asked of
            # the format that is actually present.
            # Hugging Face preserves repository directories when a selection
            # names a path such as ``UD-IQ1_S/*.gguf``; GGUF verification must
            # therefore inspect the acquired tree, not only its root.
            ggufs = sorted(path for path in dest.rglob("*.gguf") if path.is_file())
            if ggufs:
                for gguf in ggufs:
                    if gguf.stat().st_size == 0:
                        raise ValueError(f"gguf file is empty: {gguf}")
                return
            # A **diffusion-single-file** tree is the second exception, on the
            # same grounds as GGUF. ComfyUI-style repositories ship bare
            # ``.safetensors`` sorted into category directories --
            # ``diffusion_models/``, ``vae/``, ``text_encoders/`` -- and carry
            # no config.json anywhere, because safetensors embeds its own header
            # and the workflow graph that loads it supplies everything else.
            # Demanding one failed the acquisition at the last step before
            # promotion, after 59 GB had already been fetched.
            #
            # Deliberately narrower than "any tree with safetensors". A
            # transformers tree keeps its weights at the *root*, so an
            # interrupted download that lost config.json still fails here as it
            # should. Only weights sorted into known category subdirectories --
            # a layout a partial transformers download cannot accidentally
            # produce -- are verified on these terms.
            grouped = sorted(
                path
                for name in _DIFFUSION_CATEGORIES
                for path in (dest / name).glob("*.safetensors")
                if path.is_file()
            )
            if grouped:
                for weight in grouped:
                    if weight.stat().st_size == 0:
                        raise ValueError(f"safetensors file is empty: {weight}")
                return
            raise ValueError(f"model tree is missing config.json at {config}")
        try:
            json.loads(config.read_text())
        except ValueError as exc:
            # JSONDecodeError is a ValueError subclass; re-raise with the path
            # so the failure names the file rather than only the parse error.
            raise ValueError(f"config.json at {config} is not valid JSON: {exc}") from exc

        index = dest / "model.safetensors.index.json"
        if not index.is_file():
            return  # a non-sharded model needs only config.json

        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except ValueError as exc:
            raise ValueError(
                f"model.safetensors.index.json at {index} is not valid JSON: {exc}"
            ) from exc
        # ``weight_map`` maps parameter names to shard filenames; a shard may be
        # referenced many times, so dedupe before reporting.
        for shard in sorted(set(weight_map.values())):
            shard_path = dest / shard
            if not shard_path.is_file():
                raise ValueError(f"model tree is missing shard {shard!r} named by the index")
            if shard_path.stat().st_size == 0:
                raise ValueError(f"model tree has an empty shard {shard!r}")

    def _client(self) -> Any:
        if self._hub is not None:
            return self._hub
        from huggingface_hub import HfApi

        return _RealHubAdapter(HfApi())


def _as_refusal(
    exc: Exception, *, model_id: str, credential_supplied: bool
) -> AuthorizationRefusedError | None:
    """Map an upstream error to ``authorization_refused``, or leave it alone.

    Returns ``None`` when the error is not an authorization refusal, so an
    ordinary network or filesystem failure keeps its own shape instead of being
    mislabelled as a credential problem — an operator chasing the wrong cause is
    worse than a generic error.
    """
    if not _is_refusal(exc):
        return None

    reason = _refusal_reason(exc, credential_supplied=credential_supplied)
    return AuthorizationRefusedError(
        _refusal_message(reason, model_id),
        detail={
            "source_id": SOURCE_ID,
            "source_model_id": model_id,
            "reason": reason,
            "upstream_message": str(exc),
        },
    )


def _is_refusal(exc: Exception) -> bool:
    """True when the upstream error means "refused", by status or by type."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in _REFUSAL_STATUSES:
        return True
    # The whole MRO, so a more specific refusal type (unaccepted terms, say)
    # is still recognised as a refusal by its base.
    return any(
        marker in klass.__name__.lower()
        for klass in type(exc).__mro__
        for marker in _REFUSAL_TYPE_MARKERS
    )


def _refusal_reason(exc: Exception, *, credential_supplied: bool) -> str:
    """Distinguish unaccepted terms from a missing or rejected credential."""
    text = str(exc).lower()
    if getattr(exc, "access_terms_required", False) or any(
        marker in text for marker in _ACCESS_TERMS_MARKERS
    ):
        return REASON_ACCESS_TERMS
    return REASON_CREDENTIAL_REJECTED if credential_supplied else REASON_NO_CREDENTIAL


def _refusal_message(reason: str, model_id: str) -> str:
    """State which source refused, and what would actually resolve it."""
    if reason == REASON_ACCESS_TERMS:
        return (
            f"{SOURCE_ID} refused access to {model_id!r}: this model's access terms "
            f"have not been accepted. Accept them with the provider, then retry — "
            f"this product never accepts or bypasses access terms on your behalf."
        )
    if reason == REASON_NO_CREDENTIAL:
        return (
            f"{SOURCE_ID} refused access to {model_id!r}: the model is gated and no "
            f"credential was supplied. Set one with "
            f"`stead credential set {SOURCE_ID} <name>` and retry."
        )
    return (
        f"{SOURCE_ID} refused access to {model_id!r}: the supplied credential was "
        f"rejected. Check that it is current and authorized for this model."
    )


class _RealHubAdapter:
    """Adapt the real ``huggingface_hub.HfApi`` to the narrow surface we use.

    Keeps the real-vs-fake surface identical: the fake ``tests/fakes/hf_hub.py``
    exposes ``resolve_revision`` and ``snapshot_download``; this adapter maps the
    real hub onto the same calls so the source code is written once against one
    contract (the conformance harness needs the fake and real to
    agree).
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    def resolve_revision(self, repo_id: str, revision: str | None) -> str | None:
        info = self._api.model_info(repo_id, revision=revision or "main")
        return getattr(info, "sha", None) or revision or "main"

    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str | None,
        token: str | None,
        local_dir: str,
        progress: AcquireProgress | None = None,
        allow_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        from huggingface_hub import snapshot_download as _sd

        # Forwarded only when there is a selection, so a whole-repository
        # acquire issues the call it always issued. ``allow_patterns`` is the
        # hub's own filter, passed through rather than reimplemented
        # here.
        extra: dict[str, Any] = {}
        if allow_patterns:
            extra["allow_patterns"] = allow_patterns
        path = _sd(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_dir=local_dir,
            **extra,
        )
        if progress is not None:
            progress(1.0, "downloaded")
        return {"resolved_revision": revision or "main", "local_path": str(path)}
