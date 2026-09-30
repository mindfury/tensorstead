"""Node service — node management.

Node management is the entry point to a host. Registration **verifies and
records**; it provisions and installs nothing:

- ``register`` calls the node agent's ``GET /agent/v1/info`` to confirm
  reachability and contract compatibility, then captures the reported platform
  facts. The two failure modes are
  ``agent_unreachable`` and ``agent_version_incompatible`` —
  never a bootstrap attempt.
- ``deregister`` is **refused with ``still_referenced``** while any managed
  deployment names the node, listing the referrers.
  Participating-node sets are never rewritten.
- ``reachability`` is on-demand; ``resources`` is on-demand and
  never gates an operation.

The agent must be already installed and running; registration verifies that it
is, it does not install it.
"""

from __future__ import annotations

import builtins
from dataclasses import replace
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from tensorstead.contracts.version import (
    CODE_AGENT_VERSION_INCOMPATIBLE,
    OP_NODE_REGISTER,
    check_operation_supported,
)
from tensorstead.domain.errors import (
    AgentUnreachableError,
    AgentVersionIncompatibleError,
    AlreadyExistsError,
    InvalidAgentEndpointError,
    NodeReservedError,
    NotFoundError,
    OperationUnsupportedByAgentError,
    StillReferencedError,
)
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Node
from tensorstead.ports.node_client import AgentCallError
from tensorstead.ports.repository import Repository


def _require_valid_agent_endpoint(agent_endpoint: str) -> None:
    """Refuse an ``agent_endpoint`` before anything is ever sent to it.

    Registration probes this value immediately and attaches the fleet
    management bearer to that request (``NodeHTTPClient``), so the value is
    an authenticated request destination, not display text. A canonical
    origin only: ``https``, a host, no userinfo, no
    path beyond an empty one, no query, no fragment. Any of those could
    smuggle extra meaning into a URL an operator only skimmed -- userinfo
    that silently redirects credentials to a different host than the one
    that appears first, a fragment that never reaches the wire but changes
    what the reader thinks they approved.

    This does not vet *which* host -- an operator-approved inventory is a
    larger feature the design names as further hardening, not done here.
    Nor does it make plain ``verify=True`` (public CA trust, the fallback
    when ``TENSORSTEAD_AGENT_CA_BUNDLE`` is unset) mean anything by itself; an
    https requirement only closes the door it actually closes.
    """
    parsed = urlsplit(agent_endpoint)
    problems = []
    if parsed.scheme != "https":
        problems.append("scheme must be https")
    if not parsed.hostname:
        problems.append("a host is required")
    if parsed.username is not None or parsed.password is not None:
        problems.append("userinfo (user:pass@host) is not allowed")
    if parsed.path not in ("", "/"):
        problems.append("a path is not allowed")
    if parsed.query:
        problems.append("a query string is not allowed")
    if parsed.fragment:
        problems.append("a fragment is not allowed")
    if problems:
        raise InvalidAgentEndpointError(
            f"agent_endpoint {agent_endpoint!r} is invalid: {'; '.join(problems)}"
        )


def refuse_if_node_reserved(node: Node) -> None:
    """Raise ``node_reserved`` when an operator has declared this node off-limits.

    One function, imported by both ``DeploymentService`` (create, modify) and
    ``LifecycleService`` (start) rather than a check inlined in each. Those two
    already drifted once over a materially identical rule -- ``create``
    enforced node existence and ``modify`` did not --
    and marking a node reserved is exactly the kind of rule a second call site
    is easy to add without remembering the first one exists.
    """
    if node.reserved:
        raise NodeReservedError(
            f"node {node.name!r} is reserved: {node.reserved_reason}",
            node_id=node.id,
            detail={"reserved_reason": node.reserved_reason},
        )


class NodeService:
    """Coordinator-side node management over the repository and node client."""

    def __init__(
        self, repository: Repository, node_client: Any, credential_provider: Any = None
    ) -> None:
        self._repo = repository
        self._client = node_client
        # Optional, not because the token is optional -- the field it backs
        # defaults to "no override" for exactly this reason -- but because
        # callers that never touch rotate/clear (most of the existing test
        # suite) should not have to construct one.
        self._credential_provider = credential_provider

    # ------------------------------------------------------------------ nodes
    def register(
        self,
        *,
        name: str,
        agent_endpoint: str,
        agent_cert_fingerprint: str = "",
    ) -> Node:
        """Register a host whose agent is already running.

        Verifies reachability and contract compatibility, then captures the
        agent's reported platform facts. Installs nothing.
        """
        _require_valid_agent_endpoint(agent_endpoint)
        if self._repo.get_node_by_name(name) is not None:
            raise AlreadyExistsError(f"a node named {name!r} is already registered")

        # Build a provisional node to drive the info call.
        provisional = Node(
            id=new_ulid(),
            name=name,
            agent_endpoint=agent_endpoint,
            agent_contract_version="",
            agent_cert_fingerprint=agent_cert_fingerprint,
            platform_facts={},
            registered_at=datetime.now().astimezone(),
        )
        try:
            info = self._client.get_info(provisional)
        except AgentCallError as exc:
            raise AgentUnreachableError(
                f"node {name!r} agent unreachable: {exc.message}", node_id=provisional.id
            ) from exc

        contract_version = info.get("contract_version", "")
        # Version negotiation on register: the node must speak our
        # major contract version for the register operation itself.
        try:
            check_operation_supported(OP_NODE_REGISTER, contract_version)
        except Exception as exc:
            code = getattr(exc, "code", "")
            if code == CODE_AGENT_VERSION_INCOMPATIBLE:
                raise AgentVersionIncompatibleError(
                    f"node {name!r} agent contract version {contract_version!r} "
                    "is incompatible with the coordinator's"
                ) from exc
            raise OperationUnsupportedByAgentError(
                f"node {name!r} agent does not support register: {exc}"
            ) from exc

        node = Node(
            id=provisional.id,
            name=name,
            agent_endpoint=agent_endpoint,
            agent_contract_version=contract_version,
            agent_cert_fingerprint=agent_cert_fingerprint,
            platform_facts=info.get("platform_facts", {}),
            registered_at=datetime.now().astimezone(),
        )
        self._repo.save_node(node)
        return node

    def list(self) -> list[Node]:
        return self._repo.list_nodes()

    def get(self, node_id: str) -> Node:
        node = self._repo.get_node(node_id)
        if node is None:
            raise NotFoundError(f"no node with id {node_id!r}")
        return node

    # --------------------------------------------------------- node marking
    def reserve(self, node_id: str, note: str) -> Node:
        """Declare a node off-limits to new work (create, modify, start).

        A statement, not an observation: nothing about the node itself
        changes, and nothing already running on it is touched. The only
        effect is that ``DeploymentService`` and ``LifecycleService`` refuse
        to name this node going forward, with ``node_reserved`` and this
        note, until it is un-reserved.
        """
        node = self.get(node_id)
        updated = replace(node, reserved=True, reserved_reason=note)
        self._repo.save_node(updated)
        return updated

    def unreserve(self, node_id: str) -> Node:
        node = self.get(node_id)
        updated = replace(node, reserved=False, reserved_reason="")
        self._repo.save_node(updated)
        return updated

    # ---------------------------------------------- per-node management token
    def rotate_agent_management_token(self, node_id: str, token: str) -> Node:
        """Give this node its own coordinator-to-agent credential.

        The operator is expected to have already installed ``token`` in that
        agent's own environment (the same out-of-band step
        ``TENSORSTEAD_MGMT_TOKEN`` has always required) before calling this --
        registration verifies an agent is already running rather than
        provisioning it, and the same is true here. This call
        only tells the coordinator which value to present to this node from
        now on.

        Stored via the credential provider, never as a column value --
        the same rule already applied to every
        other secret this product tracks. A prior rotation's stored value is
        deleted, not merely overwritten as a dangling reference.
        """
        if not token:
            raise ValueError("token must not be empty; use clear_agent_management_token instead")
        node = self.get(node_id)
        ref = self._credential_provider.store("node-management-token", node.id, token)
        old_ref = node.agent_management_token_ref
        updated = replace(node, agent_management_token_ref=ref)
        self._repo.save_node(updated)
        if old_ref:
            self._credential_provider.delete(old_ref)
        return updated

    def clear_agent_management_token(self, node_id: str) -> Node:
        """Revert this node to the fleet-wide ``TENSORSTEAD_MGMT_TOKEN``."""
        node = self.get(node_id)
        old_ref = node.agent_management_token_ref
        updated = replace(node, agent_management_token_ref="")
        self._repo.save_node(updated)
        if old_ref:
            self._credential_provider.delete(old_ref)
        return updated

    # ------------------------------------------------------------- deregister
    def deregister(self, node_id: str) -> None:
        """Remove a node, refused while any deployment names it.

        Participating-node sets are never rewritten; a deployment that names a
        node is a referrer that must be modified or removed first.
        """
        node = self.get(node_id)
        referrers = self._referring_deployments(node.id)
        if referrers:
            raise StillReferencedError(
                f"node {node.name!r} is referenced by deployment(s): "
                f"{', '.join(referrers)}; modify or remove them first",
                detail={"referrers": referrers},
            )
        self._repo.delete_node(node.id)

    def _referring_deployments(self, node_id: str) -> builtins.list[str]:
        referrers: builtins.list[str] = []
        for deployment in self._repo.list_deployments():
            revision = self._repo.get_revision(deployment.id, deployment.current_revision)
            if revision is not None and node_id in revision.participating_nodes:
                referrers.append(deployment.name)
        return referrers

    # ------------------------------------------------------------ reachability
    def reachability(self, node_id: str) -> dict[str, Any]:
        """On-demand reachability check, and what the agent says it is.

        The agent's *current* contract and release versions are reported here
        because this call already has them: it asks ``/agent/v1/info`` and,
        until then, discarded the reply. The only version the product otherwise
        holds is ``Node.agent_contract_version``, captured once at registration
        and never written again — so a node upgraded afterwards kept being
        described by a snapshot from the day it was registered. During a rolling
        upgrade, which is the moment the question is actually asked, that answer
        is wrong in the direction of "this node cannot do the new thing".

        Observed, therefore, and never written back to the record: the stored
        value is what the node said when it registered, and overwriting it would
        destroy that fact rather than complement it.
        """
        node = self.get(node_id)
        try:
            info = self._client.get_info(node)
        except AgentCallError as exc:
            raise AgentUnreachableError(
                f"node {node.name!r} unreachable: {exc.message}", node_id=node.id
            ) from exc
        return {
            "status": "reachable",
            "observed_at": datetime.now().astimezone().isoformat(),
            # `None` where an agent predates reporting it, which is not the same
            # as an agent that reports an empty version.
            "contract_version": info.get("contract_version") or None,
            "agent_version": info.get("agent_version") or None,
            # The same reasoning, one field wider. `Node.platform_facts` is also
            # captured once at registration, so a fact the agent only learned to
            # report in a later release -- the accelerator's compute capability,
            # for instance -- can never appear for a node registered before it.
            # The quantization advisory hit exactly that: the agents
            # reported the capability and the coordinator was reading a
            # three-day-old snapshot that predated the field.
            #
            # Observed, never written back. The record still says what the node
            # said when it registered.
            "platform_facts": dict(info.get("platform_facts") or {}),
            # What is *actually deployed* on this node, which no other field
            # could answer.
            "build": dict(info.get("build") or {}),
        }

    # --------------------------------------------------------------- resources
    def resources(self, node_id: str) -> dict[str, Any]:
        """On-demand accelerator, memory, and managed-storage reading.

        Degrades to ``unreachable`` on transport failure. Reported
        for operator judgment only; never used to gate an operation.
        No sampler, no history.
        """
        node = self.get(node_id)
        try:
            raw: dict[str, Any] = self._client.get_resources(node)
            return raw
        except AgentCallError as exc:
            return {
                "status": "unreachable",
                "observed_at": datetime.now().astimezone().isoformat(),
                "accelerator_utilization_pct": None,
                "accelerator_memory_used": None,
                "accelerator_memory_total": None,
                "memory_is_unified": False,
                "storage": [],
                "detail": exc.message,
            }
