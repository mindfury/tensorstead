"""Agent model routes — ``POST /agent/v1/models:acquire`` and
``GET /agent/v1/models``.

The agent-side model management surface. ``acquire`` delegates to the
acquisition service (stage → verify → promote) via the configured model
source, streaming progress. ``GET /agent/v1/models`` lists the replicas
present on this host from its filesystem marker store.

Management-token gated. These routes carry no inference payload.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from tensorstead.agent.acquisition import local_model_id
from tensorstead.agent.app import require_management
from tensorstead.domain.models import canonical_file_selector

router = APIRouter(prefix="/agent/v1", tags=["models"])


class AcquireRequest(BaseModel):
    """Body of ``POST /agent/v1/models:acquire``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_model_id: str
    revision: str | None = None
    credential: str | None = None
    # Which files of the repository to retrieve; empty means all of them.
    # Part of the model's identity upstream (migration 0006), so it also
    # distinguishes the on-node store directory below.
    file_selector: list[str] = Field(default_factory=list)


class AcquireResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved_revision: str
    revision_pinned: bool
    size_bytes: int | None = None
    content_digest: str | None = None


class ModelListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    state: str
    local_path: str | None = None
    verified_at: str | None = None


@router.post(
    "/models:acquire",
    response_model=AcquireResponse,
    dependencies=[Depends(require_management)],
)
def acquire(payload: AcquireRequest, request: Request) -> AcquireResponse:
    """Acquire a model revision from upstream, stage→verify→promote."""
    state = request.app.state
    acquisition = state.acquisition
    source = state.model_source

    progress = _ProgressReporter()

    selector = canonical_file_selector(payload.file_selector)
    replica = acquisition.acquire(
        source,
        model_id=local_model_id(payload.source_id, payload.source_model_id, selector),
        source_model_id=payload.source_model_id,
        revision=payload.revision,
        credential=payload.credential,
        progress=progress,
        file_selector=selector,
    )
    return AcquireResponse(
        resolved_revision=replica.resolved_revision or payload.revision or "main",
        revision_pinned=replica.resolved_revision is not None,
        size_bytes=replica.size_bytes,
        content_digest=replica.content_digest,
    )


@router.get(
    "/models",
    response_model=list[ModelListItem],
    dependencies=[Depends(require_management)],
)
def list_models(request: Request) -> list[ModelListItem]:
    """List the model replicas present on this host."""
    acquisition = request.app.state.acquisition
    replicas = acquisition.list_replicas()
    return [
        ModelListItem(
            model_id=r.model_id,
            state=r.state,
            local_path=r.local_path,
            verified_at=r.verified_at.isoformat() if r.verified_at else None,
        )
        for r in replicas
    ]


@router.delete(
    "/models/{model_id}",
    status_code=204,
    dependencies=[Depends(require_management)],
)
def delete_model(model_id: str, request: Request) -> None:
    """Delete a model replica from this host.

    Unconditional at this layer — the *referenced* check is the
    coordinator's, since only it knows the deployment graph.
    The agent removes the local replica from its filesystem marker store.
    """
    acquisition = request.app.state.acquisition
    acquisition.remove_replica(model_id)


class _ProgressReporter:
    """Collects acquisition progress into a bounded snapshot.

    The acquisition service drives a callback; the agent records the latest
    fraction and message. In v1 the acquire is synchronous and returns only on
    terminal outcome; progress is available for a future streaming surface but
    is not emitted here.
    """

    def __init__(self) -> None:
        self.fraction = 0.0
        self.message = ""

    def __call__(self, fraction: float, message: str) -> None:
        self.fraction = fraction
        self.message = message
