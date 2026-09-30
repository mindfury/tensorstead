"""Agent resources route — ``GET /agent/v1/resources``.

On-demand accelerator, memory, and managed-storage reading. The agent runs
no sampler and keeps no history, and reports the storage figures without
acting on them — nothing here refuses or defers work.

Management-token gated.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from tensorstead.agent.app import require_management
from tensorstead.agent.resources import (
    default_managed_paths,
    read_resources,
)
from tensorstead.contracts.agent import NodeResourceObservationResponse, NodeResourceStorage

router = APIRouter(prefix="/agent/v1", tags=["resources"])


@router.get(
    "/resources",
    response_model=NodeResourceObservationResponse,
    dependencies=[Depends(require_management)],
)
def get_resources(request: Request) -> NodeResourceObservationResponse:
    """On-demand resource reading; no sampler, no history."""
    state = request.app.state

    nvml_source = getattr(state, "nvml_source", None)
    fs_source = getattr(state, "fs_source", None)
    # Two different questions, and conflating them is what made this endpoint
    # lie. For the *reading*, an undeclared platform must be treated as not
    # unified: letting the host-memory figure stand in for accelerator memory on
    # a discrete-GPU node reports a number about the wrong pool. For the
    # *report*, an undeclared platform is unknown.
    #
    # This previously used `.get(..., False)` for both, so a node whose topology
    # the agent could not establish was published as having denied unified
    # memory -- observed on both DGX Sparks, where detection returned None and
    # this endpoint answered `false` about hardware that is unified.
    declared = getattr(state, "platform_facts", {}).get("memory_is_unified")

    models_path, images_path = _managed_paths(state)

    reading = read_resources(
        nvml_source=nvml_source,
        filesystem_source=fs_source,
        model_store_path=models_path,
        image_store_path=images_path,
        cache_path=_cache_path(state),
        memory_is_unified=bool(declared),
    )

    return NodeResourceObservationResponse(
        status=reading["status"],
        observed_at=reading["observed_at"],
        accelerator_utilization_pct=reading["accelerator_utilization_pct"],
        accelerator_memory_used=reading["accelerator_memory_used"],
        accelerator_memory_total=reading["accelerator_memory_total"],
        # The declared value, not the reading's coerced one: `None` here means
        # the agent could not establish the topology, which is a different fact
        # from a discrete-GPU node and must not render as one.
        memory_is_unified=declared,
        storage=[
            NodeResourceStorage(
                purpose=s["purpose"],
                path=s["path"],
                capacity_bytes=s["capacity_bytes"],
                available_bytes=s["available_bytes"],
            )
            for s in reading["storage"]
        ],
    )


def _managed_paths(state: object) -> tuple[str, str]:
    """Return the managed-storage paths from app.state or the environment."""
    models = getattr(state, "model_store_path", None)
    images = getattr(state, "image_store_path", None)
    if models and images:
        return str(models), str(images)
    return default_managed_paths()


def _cache_path(state: Any) -> str:
    """The managed compile-cache root for this agent."""
    from tensorstead.agent.resources import default_cache_path

    configured = getattr(state, "cache_path", None)
    return str(configured) if configured else default_cache_path()
