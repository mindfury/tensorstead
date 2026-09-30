"""Response-only domain state types.

Observed state is **never persisted as current** — that is a
domain-wide rule. ``ObservedState``, ``NodeResourceObservation``, and
``Divergence`` are response types only: they are obtained from the responsible
node at request time, always carry when they were read (``observed_at``),
and admit ``unknown`` / ``unreachable`` as ordinary values, so
a stale value is never presented as current.

``DesiredState`` is the sole per-deployment restart-on-boot control,
defined here as a shared enum.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The divergences the design detects. Kept as an enum so the
# divergence computation and the export/CLI surfaces share one spelling.
DivergenceKind = Literal[
    "declared_running_but_absent",
    "unexpected_instance",
    "revision_mismatch",
    "image_digest_mismatch",
    "endpoint_mismatch",
    # The container is up and the port accepts connections,
    # but the runtime does not answer its own API — the state the 2026-08-10
    # incident spent its entire window in while reporting no divergence at all.
    # It is deliberately not folded into ``endpoint_mismatch``: "nothing is
    # listening" and "something is listening but not serving" call for different
    # actions, and a single kind would erase the distinction the observed block
    # now goes to some trouble to establish.
    "inference_not_ready",
]


class DesiredState(enum.StrEnum):
    """Per-deployment desired lifecycle state."""

    STOPPED = "stopped"
    RUNNING = "running"


class ObservedState(BaseModel):
    """Declared-independent observation of one deployment, fetched on demand.

    Response type only. No table, no cache, no write path.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["running", "not_running", "unknown", "unreachable"]
    observed_at: datetime
    per_node: dict[str, PerNodeObservation] = Field(default_factory=dict)
    running_image_digest: str | None = None
    # Aggregated across participating nodes, worst-case-wins, exactly as
    # ``status`` already is. Surfaced at the top level because this is the block
    # an operator reads first, and the pair is the point: ``status: running``
    # with ``inference_ready: false`` is a deployment that exists and does not
    # serve — a sentence the product previously had no way to say.
    endpoint_reachable: bool | None = None
    inference_ready: bool | None = None
    detail: str | None = None


class PerNodeObservation(BaseModel):
    """Per-participating-node observation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["running", "not_running", "unknown", "unreachable"]
    running_image_digest: str | None = None
    running_revision: int | None = None
    # Transport reachability and inference readiness are separate facts, and a
    # client must be able to tell them apart. ``None`` on either
    # means the probe could not establish it — never "no".
    endpoint_reachable: bool | None = None
    inference_ready: bool | None = None
    detail: str | None = None


class StorageReading(BaseModel):
    """One managed-storage-location capacity/available reading."""

    model_config = ConfigDict(extra="forbid")

    purpose: str
    path: str
    capacity_bytes: int
    available_bytes: int


class NodeResourceObservation(BaseModel):
    """On-demand accelerator, memory, and managed-storage reading.

    Response type only. Never used by the product to
    gate an operation.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unknown", "unreachable"]
    observed_at: datetime
    accelerator_utilization_pct: float | None = None
    accelerator_memory_used: int | None = None
    accelerator_memory_total: int | None = None
    memory_is_unified: bool = False
    storage: list[StorageReading] = Field(default_factory=list)


class Divergence(BaseModel):
    """A declared-vs-observed disagreement.

    Produced as a by-product of an observed-state request, never by a scheduled
    scan. Detecting one mutates nothing; host state changes only on an
    explicit reconcile.
    """

    model_config = ConfigDict(extra="forbid")

    kind: DivergenceKind
    node_id: str
    declared: object | None = None
    observed: object | None = None


ObservedState.model_rebuild()
