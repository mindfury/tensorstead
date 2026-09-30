"""Coordinator to node-agent contract types.

These are the request/response models on the *agent hop only* — the internal
management surface between the coordinator and a node agent. They carry no
inference request or response payloads: the product records an endpoint; it
never stands on one.

Every model uses ``model_config = ConfigDict(extra="forbid")``: an
unrecognized field is rejected, never ignored. An older agent that silently
dropped a field from a newer coordinator would report success while leaving the
host in a state nobody asked for — fail-closed is the only acceptable
behavior.

These shared Pydantic types align authoring, validation, and OpenAPI generation,
but the published contract version (``contracts/version.py``) is the *only*
compatibility signal between two separately-upgraded sides.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentContractModel(BaseModel):
    """Base for agent-hop models: reject unrecognized fields."""

    model_config = ConfigDict(extra="forbid")


class ContainerEngineInfo(AgentContractModel):
    name: str
    version: str


class AgentInfoResponse(AgentContractModel):
    """Response for ``GET /agent/v1/info``.

    ``platform_facts`` is a flat open key/value record of what the node reported
    The domain knows no fixed enumeration of it.
    """

    contract_version: str
    agent_version: str
    # Which build this agent process is: build number, git revision, whether
    # the tree was dirty. Every value ``None`` where it cannot be established,
    # which is the case for any wheel not produced by ``scripts/build_package``
    # and for a source checkout. Unknown rather than guessed.
    build: dict[str, Any] = Field(default_factory=dict)
    platform_facts: dict[str, Any] = Field(default_factory=dict)
    service_manager: str
    container_engine: ContainerEngineInfo


class NodeResourceStorage(AgentContractModel):
    """One managed-storage-location reading."""

    purpose: str
    path: str
    capacity_bytes: int
    available_bytes: int


class NodeResourceObservationResponse(AgentContractModel):
    """Response for ``GET /agent/v1/resources``."""

    status: Literal["ok", "unknown", "unreachable"]
    observed_at: datetime
    accelerator_utilization_pct: float | None = None
    accelerator_memory_used: int | None = None
    accelerator_memory_total: int | None = None
    # ``None`` means the agent could not establish the accelerator's memory
    # topology -- no driver answer and no recognised board. Distinct from
    # ``False``, which is a discrete-GPU node saying so.
    memory_is_unified: bool | None = None
    storage: list[NodeResourceStorage] = Field(default_factory=list)


class ManagedContainer(AgentContractModel):
    """One container in the product's namespace on a node.

    Carries identity and liveness only. Deciding whether it *should* be there
    needs the coordinator's deployment records, which the agent does not hold.
    """

    name: str
    deployment_id: str | None = None
    running: bool = False


class ObservedStateResponse(AgentContractModel):
    """Response for ``GET /agent/v1/deployments/{id}/observed``."""

    status: Literal["running", "not_running", "unknown", "unreachable"]
    observed_at: datetime
    running_image_digest: str | None = None
    running_revision: int | None = None
    # Transport: something accepts a connection on the deployment's port.
    endpoint_reachable: bool | None = None
    # Inference: the runtime answers its own API as a serving model.
    # Separate from reachability because a bound port is not a serving model,
    # and that gap is where the 2026-08-10 incident lived. `None` means the
    # probe could not establish the fact -- including an agent predating
    # contract 1.2, which omits the field entirely.
    inference_ready: bool | None = None
    # Whether this container was ever meant to serve (contract 1.12).
    # A vLLM rank other than the head runs headless and
    # starts no API server; the two fields above are not unknown for it, there
    # is simply nothing to know.
    #
    # Defaults to True, which is what every agent below 1.12 means by omitting
    # it and what every single-node deployment is. An older agent in a
    # distributed group therefore still vetoes the verdict -- correct, because
    # such an agent genuinely did probe a headless rank and genuinely did get
    # nothing, and the fix is the upgrade rather than a coordinator that assumes
    # what the node did not say.
    serves_inference: bool = True
    # Whether the runtime was given an inference credential. A boolean, never
    # the value: the credential belongs to the data plane and is provisioned
    # outside this product. `None` means an agent too old to say.
    endpoint_authenticated: bool | None = None
    detail: str | None = None
    # The ``tensorstead-`` namespace as the node sees it, so the coordinator can
    # decide what is unexpected. An agent below contract 1.3 omits it
    # entirely, which means "this node did not enumerate" -- not the same fact
    # as an empty list, which means "it looked and the namespace was clean".
    managed_containers: list[ManagedContainer] = Field(default_factory=list)


class OperationOutcome(AgentContractModel):
    """Per-node outcome of an operation."""

    state: Literal["pending", "running", "succeeded", "failed"]
    code: str | None = None
    message: str | None = None


class FailureReason(AgentContractModel):
    """Structured failure shape. Named fields, not just text."""

    code: str
    message: str
    node_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
