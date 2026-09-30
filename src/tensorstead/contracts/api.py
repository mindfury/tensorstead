"""Client-facing (coordinator API) models, including the structured failure shape.

These are the request/response types a *client* of the coordinator sees — the
authoritative management surface. CLI and MCP are
projections of this same service layer, so these models are what those surfaces
marshal and render.

The structured failure shape ``{code, message, node_id, detail}`` is defined
here so every error response identifies what failed and on which node in
fields, not only in a prose message.

No model here carries an inference request or response payload.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientContractModel(BaseModel):
    """Base for coordinator-API models: reject unrecognized fields."""

    model_config = ConfigDict(extra="forbid")


class FailureReason(ClientContractModel):
    """The structured failure shape every surface renders."""

    code: str
    message: str
    node_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class NodeRegisterRequest(ClientContractModel):
    name: str
    agent_endpoint: str


class NodeRegisterResponse(ClientContractModel):
    id: str
    name: str
    agent_endpoint: str
    agent_contract_version: str
    platform_facts: dict[str, Any]
    registered_at: datetime
    reserved: bool = False
    reserved_reason: str = ""
    # A boolean, never the credential -- the same no-read-path rule every
    # other secret this product tracks already follows.
    # Whether this node presents its own coordinator-to-agent
    # token instead of the fleet-wide one.
    has_management_token_override: bool = False


class NodeReserveRequest(ClientContractModel):
    note: str = ""


class NodeRotateManagementTokenRequest(ClientContractModel):
    """Tell the coordinator this node now has its own management credential.

    The value must already be installed in that agent's own environment
    (the same out-of-band step ``TENSORSTEAD_MGMT_TOKEN`` has always required)
    -- this call does not provision it, only records which value to present.
    """

    token: str = Field(min_length=1)


class NodeSummary(ClientContractModel):
    """One node in the inventory, as recorded when it registered.

    Deliberately *not* a live view. ``node.list`` is the cheap inventory call
    and fanning out a probe per node would make it as slow as the slowest agent
    and fail entirely when any node is unreachable.
    """

    id: str
    name: str
    agent_endpoint: str
    # Named for what it is. This was ``agent_contract_version``, which read as
    # the node's current version and is not: it is captured once at
    # registration and never rewritten. An operator watching
    # both nodes sit at "1.0" while every live probe said 1.6 raised it as a
    # defect, and was right to -- a caller gating on capability from this field
    # would refuse to use a capability the node has had for days, with nothing
    # in the response hinting the value was old.
    #
    # The value is unchanged and still correct. Only the claim it was making
    # about itself was wrong. Current state comes from ``node.reachability``.
    registered_contract_version: str
    registered_at: datetime
    # Declared, not observed -- belongs in this cheap, non-probing summary for
    # exactly the reason a live resource figure would not: an operator's own
    # statement, never rewritten except by them.
    reserved: bool = False
    reserved_reason: str = ""
    has_management_token_override: bool = False


class ReachabilityResponse(ClientContractModel):
    status: Literal["reachable", "unreachable"]
    observed_at: datetime
    # What the agent says it is *now*, as distinct from
    # ``Node.agent_contract_version``, which is what it said when it registered
    # and is never refreshed. The two disagree for the whole of a rolling
    # upgrade. `None` means an agent that did not report it.
    contract_version: str | None = None
    agent_version: str | None = None
    # What the node says about itself now, as distinct from
    # ``Node.platform_facts``, which is the registration snapshot and is never
    # rewritten. A fact the agent learned to report after a node registered --
    # accelerator compute capability, for one -- exists only here.
    platform_facts: dict[str, Any] = Field(default_factory=dict)
    # Which build the agent process is. Empty from an agent too old
    # to report it, which is not the same as a build it could not identify.
    build: dict[str, Any] = Field(default_factory=dict)


class RuntimeInfo(ClientContractModel):
    type: str
    versions: list[str]
    supports_distributed: bool
    # One or more image references an operator can try as a starting point,
    # each with a note saying what kind of suggestion it is. A
    # suggestion, never a selection: the operator chooses and the product
    # reports. Empty means the adapter could not name
    # one, reported as such rather than as a stale guess.
    suggested_images: list[dict[str, Any]] = Field(default_factory=list)


class ModelAcquireRequest(ClientContractModel):
    source_id: str
    source_model_id: str
    revision: str | None = None
    nodes: list[str]
    credential: str | None = None
    # Which files of the repository to acquire, as glob patterns; empty means
    # all of them. Part of the resulting model's identity, not a transient
    # option: one GGUF repository commonly ships twenty-odd quantizations, and
    # the selection is what says which of them this model is (migration 0006).
    file_selector: list[str] = Field(default_factory=list)


class OperationAcceptedResponse(ClientContractModel):
    """202 response for long-running work: an operation id to poll."""

    operation_id: str
    # Advisory notes that did not affect the outcome. Empty by
    # default, and never a reason a request was refused: this rule forbids
    # representing the pre-flight check as a guarantee, so it can only ever
    # inform.
    warnings: list[str] = Field(default_factory=list)


class ModelReplica(ClientContractModel):
    node_id: str
    state: Literal["staging", "available", "failed"]
    verified_at: datetime | None = None


class ModelResponse(ClientContractModel):
    id: str
    source_id: str
    source_model_id: str
    resolved_revision: str | None = None
    revision_pinned: bool
    size_bytes: int | None = None
    replicas: list[ModelReplica]
    # Which files of the repository this model is; empty means all of them.
    # Reported because it is part of identity: without it two rows naming one
    # repository at one revision are indistinguishable to anyone reading the
    # record, which is the situation file selection exists to avoid.
    file_selector: list[str] = Field(default_factory=list)


class CredentialRef(ClientContractModel):
    source_id: str
    name: str
    is_default: bool
    set_at: datetime


class CredentialSetRequest(ClientContractModel):
    """``PUT /v1/credentials/{source_id}/{name}`` — a value or a reference.

    Exactly one of ``secret``, ``from_env``, or ``from_file``.
    The **reference forms carry no secret**, which is what lets the MCP surface
    expose only those and still reach parity: an agent can set a credential
    without a secret ever entering its context.

    There is deliberately no response model carrying a value — no read path
    exists for one.
    """

    secret: str | None = None
    from_env: str | None = None
    from_file: str | None = None
    default: bool = False


class CredentialDeleteResponse(ClientContractModel):
    """What deleting a credential did to the source's default.

    Reported rather than applied silently: ``default_now`` is ``None`` when the
    source is left without a default, and a successor appears here only because
    the caller named one to promote.
    """

    source_id: str
    name: str
    was_default: bool
    default_now: str | None = None
    consequence: str


class DeploymentCreateRequest(ClientContractModel):
    # Chosen once and never changed. It is unique, it is the handle every
    # surface resolves a deployment by, and ``DeploymentModifyRequest`` omits
    # it deliberately rather than by oversight: a deployment
    # repointed to a different model keeps its name, the way a host keeps its
    # hostname when it is reimaged. Read the model from the revision, not from
    # this.
    name: str
    model_id: str
    runtime_type: str
    runtime_version: str
    image_reference: str
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    participating_nodes: list[str]
    endpoint: str
    # Whether the node's service manager may bring this back after a reboot.
    # **Off unless asked for.**
    #
    # It used to follow from desired lifecycle state, so starting a deployment
    # once granted it persistence forever -- including a deployment that had
    # never completed a single successful start. One such deployment deadlocked
    # its node's GPU driver and was restored into the same deadlock on every
    # subsequent boot, which is how a recoverable incident became a reimage.
    restore_on_boot: bool = False


class DeploymentModifyRequest(ClientContractModel):
    """PATCH payload; any subset of the definition.

    Fields omitted are left at their current value. The deployment id is
    unchanged.

    ``runtime_config`` **patches** the current config by default: keys present
    override, keys absent are left alone, so a one-setting modify can no longer
    drop its neighbours. Set ``replace_config`` to replace the whole map
    at once — the destructive reading, reachable only when named.
    """

    runtime_config: dict[str, Any] | None = None
    replace_config: bool = False
    image_reference: str | None = None
    runtime_version: str | None = None
    endpoint: str | None = None
    participating_nodes: list[str] | None = None
    model_id: str | None = None
    # Absent leaves the current setting alone, like every other field here.
    restore_on_boot: bool | None = None
    # The optimistic check, opt-in. Absent means "no opinion", so a
    # caller that never read a revision is unaffected; supplying one asks the
    # coordinator to refuse if the deployment moved since you looked.
    expected_revision: int | None = None


class DeploymentModifyResponse(ClientContractModel):
    """The outcome of a modification.

    ``revision`` is the new current revision. ``restart_required`` is
    **reported, never acted on** — a running deployment is not restarted as an
    implicit consequence. ``applied`` is always false because no
    host state changes here.

    ``runtime_config`` is the configuration actually recorded on the new
    revision, so a ``replace_config`` that dropped a key is visible at
    the moment it happens rather than only at the next ``deployment show`` —
    and a patch that kept one is too.
    """

    revision: int
    restart_required: bool
    applied: Literal[False] = False
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    # Advisory notes about the new configuration, never reasons it was refused.
    # Empty by default. This is the path that matters most
    # for them: a deployment gains speculative decoding by modification, not by
    # being created, so a note attached only to create would never be seen.
    warnings: list[str] = Field(default_factory=list)


class DeploymentObservedPerNode(ClientContractModel):
    status: Literal["running", "not_running", "unknown", "unreachable"]
    running_image_digest: str | None = None
    running_revision: int | None = None
    # Transport: something accepts a connection on the deployment's port.
    endpoint_reachable: bool | None = None
    # Inference: the runtime answers its own API as a serving model. A bound
    # port is not a serving runtime, and collapsing the two is what let a dead
    # deployment report itself healthy. None means the probe could
    # not establish the fact -- distinct from 'not ready'.
    inference_ready: bool | None = None
    # Whether this node's container serves at all. False for
    # a vLLM rank running headless: it participates in the group that serves
    # without serving itself, so the two fields above are ``None`` for it and
    # the deployment's verdict is taken from the ranks that do serve.
    #
    # Reported per node rather than hidden, because an operator looking at a
    # two-node deployment needs to see *why* one node shows nothing.
    serves_inference: bool = True
    # Whether the runtime was given an inference credential. A fact about
    # the endpoint, never the credential itself. None means the
    # agent did not say -- distinct from 'not authenticated'.
    endpoint_authenticated: bool | None = None
    detail: str | None = None


class DeploymentDeclared(ClientContractModel):
    """The declared block of a deployment response.

    Declared state is the coordinator's authoritative record — never merged
    with observed state. ``observed_at`` is absent here and present on
    ``DeploymentObserved`` so the two blocks are structurally distinguishable.
    """

    id: str
    name: str
    desired_state: Literal["stopped", "running"]
    current_revision: int
    running_revision: int | None = None
    revision: dict[str, Any]


class DeploymentObserved(ClientContractModel):
    """The observed block of a deployment response.

    Fetched from the responsible node at request time; carries ``observed_at``
    and admits ``unknown``/``unreachable`` as ordinary values.
    """

    status: Literal["running", "not_running", "unknown", "unreachable"]
    observed_at: datetime
    per_node: dict[str, DeploymentObservedPerNode] = Field(default_factory=dict)
    running_image_digest: str | None = None
    # Aggregated across participating nodes, worst case first. ``status:
    # running`` with ``inference_ready: false`` is the sentence this product
    # could not previously say about a deployment that exists but does not
    # serve.
    endpoint_reachable: bool | None = None
    inference_ready: bool | None = None
    detail: str | None = None


class DeploymentResponse(ClientContractModel):
    """Declared + observed, never merged.

    Two labelled blocks — ``declared`` and ``observed`` — so a reader can
    always tell what was recorded from what was seen right now. ``observed``
    carries ``observed_at``; ``declared`` does not, which is the structural
    separation the design requires.
    """

    declared: DeploymentDeclared
    observed: DeploymentObserved
    divergences: list[dict[str, Any]] = Field(default_factory=list)


class DeploymentStatusResponse(ClientContractModel):
    """Observed state only — ``deployment.status``.

    The observed portion of a deployment response, returned by the focused
    ``GET /v1/deployments/{id}/status`` endpoint. Declared state is not
    included; it is available from ``GET /v1/deployments/{id}``.
    """

    observed: DeploymentObserved
    divergences: list[dict[str, Any]] = Field(default_factory=list)


class DeploymentRuntimeResponse(ClientContractModel):
    """The runtime's own account of itself, per node.

    Argv, restart count, and recent output — the three things that explain a
    runtime failure that the design deliberately did not attempt to carry.
    Observed at request time and **stored nowhere**: no field here is written to
    a revision, an operation record, or an export. A runtime may log request
    content, and the management plane must not be where that comes to rest.
    """

    deployment_id: str
    observed_at: datetime
    # Keyed by node id. A node that could not be asked appears with a ``detail``
    # rather than being omitted — absence would read as "it had nothing to say".
    per_node: dict[str, dict[str, Any]] = Field(default_factory=dict)


class NodeResourcesResponse(ClientContractModel):
    """On-demand accelerator, memory, and managed-storage reading.

    Reported for operator judgment; never used by the product to gate an
    operation. ``status`` may be ``unknown`` or ``unreachable``. ``detail``
    carries the reason when status is not ``ok``.
    """

    status: Literal["ok", "unknown", "unreachable"]
    observed_at: datetime
    accelerator_utilization_pct: float | None = None
    accelerator_memory_used: int | None = None
    accelerator_memory_total: int | None = None
    # ``None`` means the agent could not establish the accelerator's memory
    # topology -- no driver answer and no recognised board. Distinct from
    # ``False``, which is a discrete-GPU node saying so.
    memory_is_unified: bool | None = None
    storage: list[dict[str, Any]] = Field(default_factory=list)
    detail: str | None = None


class DeploymentRevisionInfo(ClientContractModel):
    revision: int
    model: dict[str, Any] | None = None
    runtime_type: str
    runtime_version: str
    image_reference: str
    image_digest: str
    runtime_config: dict[str, Any]
    participating_nodes: list[str]
    endpoint: str
    created_at: datetime


class OperationResponse(ClientContractModel):
    id: str
    kind: str
    state: Literal["pending", "running", "succeeded", "failed"]
    deployment_revision: int | None = None
    per_node_outcomes: dict[str, Any] | None = None
    failure_reason: FailureReason | None = None
    progress: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
