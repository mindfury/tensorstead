"""Agent replication routes.

Two endpoints with deliberately different trust roles:

- ``POST /agent/v1/models:replicate`` — the coordinator instructing *this*
  agent to make a model locally available by pulling from a named peer. That is
  a management instruction, so it carries the **management** token.
- ``GET /agent/v1/models/{id}/content`` — a peer agent pulling bytes. That is
  the agent-to-agent hop, so it carries the **replication** token,
  and an agent compromised into holding only this token can copy
  artifacts and cannot manage anything. Its two sub-resources are the same
  content, finer grained: ``.../content/manifest`` lists every path and size,
  and ``.../content/file?path=`` serves one file, honouring a byte ``Range``,
  so the destination can pull the model over several connections at once.

The content endpoint is **read-only and refuses to serve a replica that is not
``available``** — a staging copy can never propagate onward and be promoted by
the receiver on the strength of having arrived.

No model byte transits the coordinator on either route.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from tensorstead.agent.app import require_management, require_replication
from tensorstead.agent.replication import ReplicationError, parse_byte_range

router = APIRouter(prefix="/agent/v1", tags=["replication"])

# Failure codes that are the caller's fault rather than this host's.
_STATUS_BY_CODE = {
    "not_found": 404,
    "replica_not_available": 409,
    "verification_failed": 422,
    "artifact_too_large": 413,
    "replication_unavailable": 503,
    "replication_failed": 502,
}


class ReplicateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_model_id: str
    resolved_revision: str | None = None
    content_digest: str | None = None


class ReplicateSource(BaseModel):
    """Where to pull from, and how to be sure it is who it claims to be."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    agent_endpoint: str
    # Supplied by the coordinator, which pinned it at registration, so the
    # destination can verify the source agent rather than trusting DNS/IP.
    agent_cert_fingerprint: str | None = None


class ReplicateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ReplicateModel
    source_replica: ReplicateSource


class ReplicateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    state: str
    content_digest: str | None = None
    size_bytes: int | None = None


def _model_id(model: ReplicateModel) -> str:
    return f"{model.source_id}:{model.source_model_id}"


@router.post(
    "/models:replicate",
    response_model=ReplicateResponse,
    dependencies=[Depends(require_management)],
)
def replicate(payload: ReplicateRequest, request: Request) -> ReplicateResponse:
    """Pull a model revision from a peer, verify it, then promote it.

    This agent owns the operation end to end; the coordinator named a source
    and is not otherwise involved.
    """
    service: Any = request.app.state.replication
    try:
        outcome = service.replicate(
            model_id=_model_id(payload.model),
            content_digest=payload.model.content_digest,
            resolved_revision=payload.model.resolved_revision,
            source_endpoint=payload.source_replica.agent_endpoint,
            source_node_id=payload.source_replica.node_id,
            source_fingerprint=payload.source_replica.agent_cert_fingerprint,
            replication_token=request.app.state.replication_token,
        )
    except ReplicationError as exc:
        raise HTTPException(
            status_code=_STATUS_BY_CODE.get(exc.code, 500),
            detail={"code": exc.code, "message": exc.message, "detail": exc.detail},
        ) from exc
    return ReplicateResponse(
        model_id=outcome.model_id,
        state=outcome.state,
        content_digest=outcome.content_digest,
        size_bytes=outcome.size_bytes,
    )


@router.get(
    "/models/{model_id:path}/content",
    dependencies=[Depends(require_replication)],
)
def serve_content(model_id: str, request: Request) -> StreamingResponse:
    """Stream a locally ``available`` replica to a requesting peer.

    Read-only: it mutates nothing on this host. Refuses anything not
    ``available``, so a staging copy cannot propagate.
    """
    service: Any = request.app.state.replication
    try:
        path, digest = service.serve(model_id)
    except ReplicationError as exc:
        raise HTTPException(
            status_code=_STATUS_BY_CODE.get(exc.code, 500),
            detail={"code": exc.code, "message": exc.message, "detail": exc.detail},
        ) from exc
    return StreamingResponse(
        service.stream_archive(path),
        media_type="application/x-tar",
        headers={"X-Content-Digest": digest},
    )


def _refusal(exc: ReplicationError) -> HTTPException:
    return HTTPException(
        status_code=_STATUS_BY_CODE.get(exc.code, 500),
        detail={"code": exc.code, "message": exc.message, "detail": exc.detail},
    )


@router.get(
    "/models/{model_id:path}/content/manifest",
    dependencies=[Depends(require_replication)],
)
def serve_manifest(model_id: str, request: Request) -> dict[str, Any]:
    """Every directory and file of a locally ``available`` replica, with sizes.

    Read-only, and refuses anything not ``available`` exactly as the archive
    route does. It describes the content; it is not trusted for it -- the
    destination still verifies the digest of what it pulled.
    """
    service: Any = request.app.state.replication
    try:
        entries = service.manifest(model_id)
    except ReplicationError as exc:
        raise _refusal(exc) from exc
    return {"entries": [entry.as_json() for entry in entries]}


@router.get(
    "/models/{model_id:path}/content/file",
    dependencies=[Depends(require_replication)],
)
def serve_file(model_id: str, path: str, request: Request) -> StreamingResponse:
    """One file of a locally ``available`` replica, or one byte range of it."""
    service: Any = request.app.state.replication
    try:
        target = service.file_for(model_id, path)
    except ReplicationError as exc:
        raise _refusal(exc) from exc
    size = target.stat().st_size
    try:
        requested = parse_byte_range(request.headers.get("range"), size)
    except ValueError as exc:
        raise HTTPException(
            status_code=416,
            detail={"code": "range_not_satisfiable", "message": str(exc), "detail": {}},
            headers={"Content-Range": f"bytes */{size}"},
        ) from exc
    start, end = requested or (0, size)
    headers = {"Content-Length": str(end - start), "Accept-Ranges": "bytes"}
    if requested is not None:
        headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"
    return StreamingResponse(
        service.stream_file(target, start, end),
        status_code=206 if requested is not None else 200,
        media_type="application/octet-stream",
        headers=headers,
    )
