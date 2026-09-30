"""Agent image routes — ``POST /agent/v1/images:pull``.

Pulls an image via the container engine and returns the **platform-specific**
digest, not the multi-arch manifest-list digest.
On arm64 hosts the two differ, and only the former identifies the image that
actually ran. Management-token gated.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from tensorstead.agent.app import require_management, require_replication
from tensorstead.agent.container_engine.base import (
    ImageBuildError,
    ImageInUseError,
    ImageNotPresentError,
)

router = APIRouter(prefix="/agent/v1", tags=["images"])


class ImagePullRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str


class ImagePullResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    digest: str


@router.post(
    "/images:pull",
    response_model=ImagePullResponse,
    dependencies=[Depends(require_management)],
)
def pull_image(payload: ImagePullRequest, request: Request) -> ImagePullResponse:
    """Pull an image and return its platform-specific digest."""
    engine = request.app.state.container_engine
    digest = engine.pull_image(payload.reference)
    return ImagePullResponse(digest=digest)


class ImageRemoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str


class ImageRemoveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str
    # True when this call removed something; False when the daemon did not hold
    # the reference at all. Both are successes, and the coordinator reaps its
    # record either way -- the difference is only what it can honestly claim.
    removed: bool
    digest: str | None = None


@router.post(
    "/images:remove",
    response_model=ImageRemoveResponse,
    dependencies=[Depends(require_management)],
)
def remove_image(payload: ImageRemoveRequest, request: Request) -> ImageRemoveResponse:
    """Remove an image from this host.

    Unconditional at this layer, exactly as the model-replica delete is: the
    *referenced* check belongs to the coordinator, since only it knows the
    deployment graph. What the agent owns is the daemon's answer —
    whether the image was there, and whether a container still holds it.

    Idempotent. A reference the daemon does not hold is a 200 with
    ``removed: false``, not a failure: the coordinator is then reaping a record
    that outlived its object, which is a thing it should be able to finish.
    """
    engine = request.app.state.container_engine
    digest = engine.image_digest(payload.reference)
    if digest is None:
        return ImageRemoveResponse(reference=payload.reference, removed=False, digest=None)

    try:
        engine.remove_image(image_id=payload.reference)
    except ImageNotPresentError:
        # Raced with something else removing it. Same outcome as absent.
        return ImageRemoveResponse(reference=payload.reference, removed=False, digest=None)
    except ImageInUseError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "image_in_use",
                "message": str(exc),
                "detail": {"reference": payload.reference, "digest": digest},
            },
        ) from exc
    except ImageBuildError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "image_remove_failed",
                "message": str(exc),
                "detail": {"reference": payload.reference},
            },
        ) from exc

    return ImageRemoveResponse(reference=payload.reference, removed=True, digest=digest)


class ImageListEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: str
    digest: str


@router.get(
    "/images",
    response_model=list[ImageListEntry],
    dependencies=[Depends(require_management)],
)
def list_images(request: Request) -> list[ImageListEntry]:
    """What the daemon actually holds, so records can be compared to it.

    The coordinator's image records are written when it pulls, builds or
    imports and were never checked against the node again. Nothing in the
    product removed an image, so every out-of-band removal orphaned a record
    silently. This is the read that makes the comparison
    possible; the coordinator does the comparing.
    """
    engine = request.app.state.container_engine
    try:
        rows = engine.list_images()
    except ImageBuildError as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "image_list_failed", "message": str(exc), "detail": {}},
        ) from exc
    return [ImageListEntry(reference=r["reference"], digest=r["digest"]) for r in rows]


class ImageBuildRequest(BaseModel):
    """Body of ``POST /agent/v1/images:build``.

    Ordered steps, not a command line. There is no field carrying something to
    run on the host itself, and no interactive form — the narrowing of the
    rule covers recorded build specs and nothing else.

    ``entrypoint`` is the produced image's own, so an image that must prepare
    itself before serving can. It runs *in the container*,
    never on the host, which is the boundary this design draws.
    """

    model_config = ConfigDict(extra="forbid")

    reference: str
    base_image: str
    steps: list[str] = []
    # Absent for every build spec recorded before this existed, which means
    # "leave the base image's entrypoint alone" -- what they always did.
    entrypoint: list[str] = []


class ImageProducedResponse(BaseModel):
    """An image produced locally."""

    model_config = ConfigDict(extra="forbid")

    # Content-addressable identifier. Deliberately not named `digest`: a
    # locally produced image has no registry digest, and the rule forbids
    # presenting one as the other.
    image_id: str
    reference: str
    origin: str


class ImageImportRequest(BaseModel):
    """Body of ``POST /agent/v1/images:import``.

    ``archive_name`` is a file name within the node's managed image store, not
    a path. The design narrowed the rule for recorded build specs; it did not
    narrow the separate rule that no parameter accepts an arbitrary path, and
    the MCP guardrail caught the first draft doing exactly that.
    """

    model_config = ConfigDict(extra="forbid")

    archive_name: str
    reference: str
    # When given, the archive must produce this identifier or nothing becomes
    # available — the same rule applied to images.
    expected_image_id: str | None = None


@router.post(
    "/images:build",
    response_model=ImageProducedResponse,
    dependencies=[Depends(require_management)],
)
def build_image(payload: ImageBuildRequest, request: Request) -> ImageProducedResponse:
    """Build a recorded spec on this node."""
    engine = request.app.state.container_engine
    image_id = engine.build_image(
        reference=payload.reference,
        base_image=payload.base_image,
        steps=list(payload.steps),
        entrypoint=list(payload.entrypoint or []) or None,
    )
    return ImageProducedResponse(image_id=image_id, reference=payload.reference, origin="built")


def _managed_archive(request: Request, name: str) -> Path:
    """Resolve ``name`` inside the managed image store, refusing to escape it."""
    store = Path(getattr(request.app.state, "image_store_path", "/var/lib/tensorstead/images"))
    if "/" in name or name in ("", ".", ".."):
        raise ImageBuildError(
            f"archive must be a file name inside {store}, not a path", reference=name
        )
    candidate = (store / name).resolve()
    if candidate.parent != store.resolve():
        raise ImageBuildError(f"archive {name!r} escapes {store}", reference=name)
    if not candidate.is_file():
        raise ImageBuildError(f"archive {name!r} is not in {store}", reference=name)
    return candidate


@router.post(
    "/images:import",
    response_model=ImageProducedResponse,
    dependencies=[Depends(require_management)],
)
def import_image(payload: ImageImportRequest, request: Request) -> ImageProducedResponse:
    """Import a prebuilt archive, verifying it before it becomes available."""
    engine = request.app.state.container_engine
    archive = _managed_archive(request, payload.archive_name)
    image_id = engine.import_image(archive_path=str(archive))
    if payload.expected_image_id is not None and image_id != payload.expected_image_id:
        # Refused rather than reported: a mismatched archive must not become an
        # available image. The daemon already loaded it by this
        # point -- import_image has no way to know its identity without
        # loading it first -- so the rejection also removes it rather than
        # leaving a poisoned tag for a later start to pick up.
        cleanup_detail = ""
        try:
            engine.remove_image(image_id=image_id, force=True)
        except Exception as exc:
            cleanup_detail = f"; additionally, the rejected image could not be removed: {exc}"
        raise ImageBuildError(
            f"archive produced {image_id!r}, expected {payload.expected_image_id!r}"
            f"{cleanup_detail}",
            reference=payload.reference,
        )
    # Loading is not availability. An archive missing
    # config or layer content still loads under the right id and still resolves;
    # it fails only when a container is created from it. Checked here so the
    # refusal names the import, rather than surfacing much later as a bare 500
    # from the deployment route.
    try:
        engine.verify_image_materializable(image_id=image_id)
    except Exception as exc:
        cleanup_detail = ""
        try:
            engine.remove_image(image_id=image_id, force=True)
        except Exception as removal_exc:
            cleanup_detail = (
                f"; additionally, the unusable image could not be removed: {removal_exc}"
                f" -- remove it before retrying, or the retry will inherit this state"
            )
        raise ImageBuildError(
            f"archive loaded as {image_id!r} but no container can be created from it: "
            f"{exc}{cleanup_detail}",
            reference=payload.reference,
        ) from exc
    return ImageProducedResponse(image_id=image_id, reference=payload.reference, origin="imported")


class ImageDistributeRequest(BaseModel):
    """Body of ``POST /agent/v1/images:distribute``.

    The **destination** agent owns the operation, exactly as model replication
    does: it pulls from the source peer, writes to a staging
    path, verifies, and only then loads. A source that pushed would have to be
    trusted by the destination for more than reading bytes.
    """

    model_config = ConfigDict(extra="forbid")

    reference: str
    source_endpoint: str
    expected_image_id: str
    # No ``replication_token`` here, deliberately. The
    # destination uses *its own* configured replication credential to fetch from
    # the source, exactly as model replication does. A field here would mean the
    # credential travelling through the coordinator -- into its request bodies,
    # its logs, and its operation records -- to authenticate a hop between two
    # agents that both already hold it.
    #
    # It existed, was never sent, and the source endpoint requires the token, so
    # every multi-node image build would have failed its distribution step with
    # a 401 that named nothing.
    source_fingerprint: str | None = None


class ImageProbeRequest(BaseModel):
    """Body of ``POST /agent/v1/images:probe``.

    Asks an image what it accepts. Carries no configuration and starts no
    deployment: the container runs to completion with no model, no network, and
    no published port, and is removed either way.
    """

    model_config = ConfigDict(extra="forbid")

    reference: str
    runtime_type: str


class ImageProbeResponse(BaseModel):
    """What an image said it accepts, or that it could not be established."""

    model_config = ConfigDict(extra="forbid")

    reference: str
    image_id: str | None = None
    # ``known`` or ``unknown``. Never an empty option list standing in for a
    # failed probe -- "we did not find out" and "this runtime accepts nothing"
    # are different facts and the second one is never true.
    state: str
    options: list[dict[str, Any]] = []
    detail: str | None = None


@router.post(
    "/images:probe",
    response_model=ImageProbeResponse,
    dependencies=[Depends(require_management)],
)
def probe_image(payload: ImageProbeRequest, request: Request) -> ImageProbeResponse:
    """Ask an image what options it accepts.

    Runs here rather than on the coordinator because the answer is a property of
    *this build on this architecture*. The controller is x86 and these images are
    arm64; an emulated probe reported a platform failure that had nothing to do
    with the image. The node is the only place the question
    can be asked truthfully.
    """
    from tensorstead.agent.routes.deployments import _runtime_adapter

    engine = request.app.state.container_engine
    adapter = _runtime_adapter(request, payload.runtime_type)
    probe = adapter.schema_probe()
    image_id = engine.image_digest(payload.reference)

    if probe is None:
        return ImageProbeResponse(
            reference=payload.reference,
            image_id=image_id,
            state="unknown",
            detail=f"the {payload.runtime_type} adapter declares no schema probe",
        )

    # Fail closed on an unresolvable image rather than probing a reference and
    # attributing the answer to nothing. A tag is not an
    # identity; an answer that cannot be bound to one is not worth recording,
    # and recording it by digest afterwards cannot repair an attribution that
    # was already wrong.
    if image_id is None:
        return ImageProbeResponse(
            reference=payload.reference,
            image_id=None,
            state="unknown",
            detail=(
                f"{payload.reference!r} resolves to no image on this node, so there is "
                f"nothing whose surface could be established here"
            ),
        )

    try:
        output = engine.run_once(
            # The resolved identity, never the mutable reference. Running the
            # tag would let a retag between resolution and execution return
            # image B's schema labelled with image A's id -- the record and the
            # thing it describes diverging, silently, in the one operation whose
            # entire purpose is to describe something accurately.
            image=image_id,
            entrypoint=list(probe.entrypoint),
            command=list(probe.command),
            script=probe.script,
            with_accelerator=probe.needs_accelerator,
        )
        options = adapter.parse_schema(output)
    except Exception as exc:
        # Unknown, never empty. A probe that fails must not make an image look
        # like one that accepts nothing.
        return ImageProbeResponse(
            reference=payload.reference,
            image_id=image_id,
            state="unknown",
            detail=str(exc)[:1000],
        )

    return ImageProbeResponse(
        reference=payload.reference,
        image_id=image_id,
        state="known",
        options=[asdict(option) for option in options],
    )


@router.get(
    "/images/{reference:path}/content",
    dependencies=[Depends(require_replication)],
)
def serve_image_content(reference: str, request: Request) -> StreamingResponse:
    """Stream a locally held image archive to a requesting peer.

    Gated by the **replication** token, not the management one:
    an agent holding only this role can fetch artifacts and cannot issue
    management instructions anywhere. Read-only — it writes nothing on this
    host, and the export is produced into a temporary file that is removed
    once streamed.
    """
    engine = request.app.state.container_engine
    store = Path(getattr(request.app.state, "image_store_path", "/var/lib/tensorstead/images"))
    store.mkdir(parents=True, exist_ok=True)
    archive = store / f".serve-{reference.replace('/', '_').replace(':', '_')}.tar"
    try:
        engine.export_image(reference=reference, archive_path=str(archive))
    except Exception as exc:
        archive.unlink(missing_ok=True)
        raise HTTPException(
            status_code=404,
            detail={"code": "image_not_present", "message": str(exc), "detail": {}},
        ) from exc

    def _stream() -> Any:
        try:
            with archive.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    yield chunk
        finally:
            archive.unlink(missing_ok=True)

    return StreamingResponse(_stream(), media_type="application/x-tar")


@router.post(
    "/images:distribute",
    response_model=ImageProducedResponse,
    dependencies=[Depends(require_management)],
)
def distribute_image(payload: ImageDistributeRequest, request: Request) -> ImageProducedResponse:
    """Pull an image from a peer, verify it, then load it.

    Staged, verified, and only then loaded — the same order acquisition uses,
    so an interrupted transfer can never present itself as an available image
    (the same rule applied to images).

    The identifier is verified against the source's before loading. That is the
    entire point of distributing rather than rebuilding: every participating
    node must hold the *same* image, because different identifiers for one spec
    silently break the comparison.
    """
    service: Any = getattr(request.app.state, "image_distribution", None)
    if service is None:
        raise HTTPException(
            status_code=501,
            detail={
                "code": "distribution_unavailable",
                "message": "this agent has no image distribution service configured",
                "detail": {},
            },
        )
    image_id = service.pull_from_peer(
        reference=payload.reference,
        source_endpoint=payload.source_endpoint,
        expected_image_id=payload.expected_image_id,
        # This agent's own credential, not one handed to it.
        token=request.app.state.replication_token,
        fingerprint=payload.source_fingerprint,
    )
    return ImageProducedResponse(
        image_id=image_id, reference=payload.reference, origin="distributed"
    )
