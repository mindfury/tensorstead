"""Persisted domain entities — declared state only.

These are the *persisted* entities: the coordinator's authoritative record of
declared state. Observed state is never persisted as current
and therefore has no entity here.

The split that governs this module (its organizing principle):

- A **Deployment** is a stable identity plus a desired state.
- A **DeploymentRevision** is an immutable, numbered definition. Everything
  reproducible lives on the revision.

Entities are plain, platform-neutral dataclasses — the domain
package imports nothing OS- or vendor-specific. Persistence concerns (SQL,
migrations) are confined to the SQLite adapter.

Invariants asserted by tests and by this module's own factory helpers:
revisions are 1-based / monotonic / gapless, ``DeploymentRevision`` is
immutable, and ``Deployment.id`` is stable across revisions.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from .identity import assert_valid_ulid


class ReplicaState(enum.StrEnum):
    STAGING = "staging"
    AVAILABLE = "available"
    FAILED = "failed"


class OperationState(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OperationKind(enum.StrEnum):
    MODEL_ACQUIRE = "model_acquire"
    MODEL_REPLICATE = "model_replicate"
    IMAGE_ACQUIRE = "image_acquire"
    # A build produces an image; an acquire fetches one. Distinct kinds because
    # they are distinct verbs, and because an operator reading history needs to
    # know which one happened. (``IMAGE_ACQUIRE`` above is currently declared
    # and never used -- noted rather than repurposed, since renaming a verb to
    # avoid adding one is how records stop describing what occurred.)
    IMAGE_BUILD = "image_build"
    DEPLOYMENT_CREATE = "deployment_create"
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    REMOVE = "remove"
    RECONCILE = "reconcile"
    MODEL_DELETE = "model_delete"
    IMAGE_DELETE = "image_delete"


@dataclass(frozen=True)
class Node:
    """A registered, already-provisioned inference host."""

    id: str
    name: str
    agent_endpoint: str
    agent_contract_version: str
    agent_cert_fingerprint: str
    platform_facts: dict[str, Any]
    registered_at: datetime
    # Declared, never observed. An operator says "not this one" for a reason
    # nothing on the node itself would show up as -- a standalone service
    # outside Tensorstead's view, hardware set aside for something else -- and
    # every entry point that would put new work on this node (create, modify,
    # start) honours it before doing anything. Not the no-scheduling rule's
    # business: this never picks a node for the operator, it only
    # ever refuses one they picked.
    reserved: bool = False
    reserved_reason: str = ""
    # A per-node override for the coordinator-to-agent management credential:
    # a reference into the credential
    # provider, never a value -- the same rule applied to a new secret kind.
    # Empty means "no override": every agent still holds the fleet-wide
    # TENSORSTEAD_MGMT_TOKEN, so a compromised node discloses a credential that
    # controls every other node too. Set, it means this node was issued its
    # own credential and a compromise of it no longer implies the others.
    agent_management_token_ref: str = ""

    def __post_init__(self) -> None:
        assert_valid_ulid(self.id, what="node id")


@dataclass(frozen=True)
class ModelSource:
    """Authoritative for model identity, revision, and authorization."""

    id: str  # "huggingface" — the only v1 value
    supports_revision_pinning: bool
    requires_credential: bool


def canonical_file_selector(patterns: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a file selection into its canonical, comparable form.

    Selection is part of a model's identity, so two spellings of the same
    selection must compare equal or the uniqueness rule stops deduplicating.
    Sorted and deduplicated; blank entries dropped. ``()`` means the whole
    repository, which is what every model acquired before this existed is.

    Accepts a string (one pattern), an iterable of strings, or ``None``.
    """
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        candidates: list[str] = [patterns]
    else:
        candidates = [str(p) for p in patterns]
    return tuple(sorted({stripped for p in candidates if (stripped := p.strip())}))


def local_model_id(
    source_id: str, source_model_id: str, file_selector: tuple[str, ...] = ()
) -> str:
    """The node-local identity of one acquired model.

    The node stores each model in a directory derived from this id, so it has
    to distinguish everything the coordinator considers a distinct model. A
    file selection does (migration 0006): one GGUF repository yields twenty-odd
    models that differ only in which file was retrieved, and keying the store
    on the repository alone would put the second one in the first one's
    directory, where the reuse check -- which compares resolved revisions, and
    they match -- would hand back the wrong weights as though they were a hit.

    A whole-repository acquisition keeps the exact id it has always had, so no
    model acquired before this existed changes its path. A selection appends a
    short digest of the canonical patterns rather than the patterns themselves:
    bounded length, and safe regardless of what a glob contains.
    """
    base = f"{source_id}:{source_model_id}"
    selector = canonical_file_selector(file_selector)
    if not selector:
        return base
    digest = hashlib.sha256("\n".join(selector).encode()).hexdigest()[:12]
    return f"{base}#{digest}"


@dataclass(frozen=True)
class Model:
    """A logical model revision, independent of where copies live."""

    id: str
    source_id: str
    source_model_id: str
    resolved_revision: str | None = None
    revision_pinned: bool = True
    size_bytes: int | None = None
    content_digest: str | None = None
    # Which files of the upstream repository this model is. Empty means all of
    # them. Part of identity, not of the request that acquired it: a repo
    # carrying twenty-three quantizations yields a different model per
    # selection, and a record naming the repo while holding one file describes
    # something that does not exist (migration 0006).
    file_selector: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert_valid_ulid(self.id, what="model id")
        # Canonicalized here rather than trusted from the caller, so a model
        # constructed from a route payload and one read back from SQLite are
        # the same value.
        object.__setattr__(self, "file_selector", canonical_file_selector(self.file_selector))


@dataclass(frozen=True)
class InferenceCredential:
    """A credential authenticating inference clients to a runtime.

    Holds a *reference*, never the value — the value lives in the credential
    provider, exactly as model-source credentials do. The
    product owns this setting because it owns the rest of a deployment's
    runtime configuration; owning the record is not the same as being in the
    data path, which is a separate question.
    """

    name: str
    secret_ref: str
    set_at: datetime


@dataclass(frozen=True)
class ModelReplica:
    """Presence and verified state of one logical model revision on one node."""

    model_id: str
    node_id: str
    local_path: str
    state: ReplicaState
    verified_at: datetime | None = None


@dataclass(frozen=True)
class Credential:
    """A named credential reference for a source. The value is never a column."""

    source_id: str
    name: str
    secret_ref: str
    is_default: bool
    set_at: datetime


@dataclass(frozen=True)
class Deployment:
    """A stable identity plus a desired state; holds no definition of its own."""

    id: str
    name: str
    desired_state: str  # DesiredState value: "stopped" | "running"
    current_revision: int
    running_revision: int | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        assert_valid_ulid(self.id, what="deployment id")
        if self.current_revision < 1:
            raise ValueError("current_revision must be 1-based")
        if self.running_revision is not None and self.running_revision < 1:
            raise ValueError("running_revision must be 1-based")


@dataclass(frozen=True)
class DeploymentRevision:
    """An immutable, numbered definition of a deployment.

    No ``UPDATE`` is ever issued against this table: every accepted
    modification inserts revision *n+1*. Denormalized model identity keeps a
    revision sufficient to recreate itself even if the Model row is later
    deleted.
    """

    deployment_id: str
    revision: int
    model_id: str
    # Denormalized model identity, copied at creation.
    model_source_id: str
    source_model_id: str
    resolved_revision: str | None
    revision_pinned: bool
    runtime_type: str
    runtime_version: str
    image_reference: str
    image_digest: str
    runtime_config: dict[str, Any]
    participating_nodes: tuple[str, ...]
    endpoint: str
    origin_platform_facts: dict[str, Any]
    # Which reviewed approval authorized a code-loading option on this
    # revision, if any (``tensorstead.domain.approvals``). Provenance, never
    # permission: the decision is remade against the approvals table at create
    # and again at the agent, so a revision whose approval was later deleted
    # cannot start -- while its record still says what it was authorized by.
    code_approval_fingerprint: str = ""
    code_approval_id: str = ""
    # Whether this definition may be restored by the node's service manager
    # after a reboot.
    #
    # **Default off, and on the revision rather than the deployment.** A modify
    # produces revision n+1, which is an unproven definition; inheriting the
    # previous revision's boot persistence would let an untested change acquire
    # the right to run before anyone has seen it start. Declaring it here means
    # a change to it is a change to the deployment's definition, visible in the
    # record and in export like any other.
    #
    # It was previously derived from desired lifecycle state alone, so starting
    # a deployment once granted it persistence forever. A deployment that had
    # never completed a single successful start then deadlocked its node's GPU
    # driver on every subsequent boot.
    restore_on_boot: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        assert_valid_ulid(self.deployment_id, what="deployment id")
        if self.revision < 1:
            raise ValueError("revision must be 1-based")
        if not self.participating_nodes:
            raise ValueError("a deployment must name at least one node")
        # One physical agent cannot occupy two positions in a runtime group. A
        # repeated ID still counts toward `len(participating_nodes) > 1`, so the
        # revision is classified as distributed while naming one participant:
        # two rank-0 launches on the same host, and per-node outcomes keyed by
        # node ID that keep only the last. Defended here as
        # well as at the service boundary so import and any future surface
        # cannot construct the impossible revision by going around it.
        if len(set(self.participating_nodes)) != len(self.participating_nodes):
            raise ValueError("a deployment must name each participating node at most once")


@dataclass(frozen=True)
class Operation:
    """A unit of management work with progress and a terminal outcome."""

    id: str
    kind: OperationKind
    target_type: str
    target_id: str
    deployment_revision: int | None = None
    state: OperationState = OperationState.PENDING
    failure_reason: dict[str, Any] | None = None
    per_node_outcomes: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        assert_valid_ulid(self.id, what="operation id")
        assert_valid_ulid(self.target_id, what="operation target id")


class ImageOrigin(StrEnum):
    """How a managed image came to be on a node."""

    PULLED = "pulled"
    BUILT = "built"
    IMPORTED = "imported"


@dataclass(frozen=True)
class ImageBuildSpec:
    """A recorded, versioned recipe for a runtime image.

    Data, not a command. Recording one executes nothing; building it
    is a separate operation. That separation is what makes a build inspectable
    before it runs and reproducible after it has.
    """

    name: str
    base_image: str
    steps: tuple[str, ...]
    # The image's own ``ENTRYPOINT``, so a produced image can be *self-starting*
    # Some runtimes need work done between the container
    # starting and the server running -- installing a model-shipped encoder into
    # the runtime, for one -- and that work reads the mounted model directory,
    # which does not exist at build time.
    #
    # Recording it here keeps the mechanism declarative and provenance-tracked
    # while its *content* stays pinned to whatever model revision was acquired.
    # The alternative was a launch-time hook in the agent, which would make the
    # product know why a particular model needs a particular file moved -- and
    # that is the image's business, not the management plane's.
    entrypoint: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @property
    def base_is_pinned(self) -> bool:
        """Whether the base is pinned to an immutable digest.

        A moving base makes the build unreproducible. This is reported, never
        refused: pinning is the operator's judgement, on the same terms as
        the refusal to gate on the product's own observations.
        """
        return "@sha256:" in self.base_image


@dataclass(frozen=True)
class ImageRecord:
    """Tracks an image present on a node so deletion can be refused.

    ``digest`` is a registry digest for a pulled image and a content-addressable
    image identifier for one produced locally. Never present the
    second as the first: a locally built image has no registry digest, and
    conflating them makes an export look portable when it is not. ``origin``
    is what tells the two apart.
    """

    node_id: str
    reference: str
    digest: str
    pulled_at: datetime = field(default_factory=datetime.now)
    origin: ImageOrigin = ImageOrigin.PULLED
    # Which build spec or archive produced it; None when pulled.
    produced_by: str | None = None

    @property
    def is_registry_digest(self) -> bool:
        """False for locally produced images, which have no registry digest."""
        return self.origin is ImageOrigin.PULLED
