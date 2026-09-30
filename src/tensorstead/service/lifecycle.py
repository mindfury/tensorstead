"""Lifecycle service.

Drives a deployment's desired-state transitions through the node client.
This module owns the *desired-state* transitions and the coordinator's
record of what it asked the agent to do. The agent owns the host-side
execution. A running deployment survives a coordinator outage because
the boot-restoration unit the agent installed is self-sufficient
and does not depend on the coordinator.

Lifecycle operations:

- **start** — drive the agent to materialize the current revision,
  record ``running_revision``, set desired state ``running``, and enable
  boot restoration.
- **stop** — stop the container, disable and remove the unit so
  a reboot does not revive it, set desired state ``stopped``.
- **restart** — stop then start; equivalent to stop+start in
  sequence, producing a new operation.
- **remove** — remove the runtime instance and its boot arrangement;
  model artifacts and images are retained. Fails rather than
  orphaning when a node is unreachable.

Idempotent state handling: a request for a state
already held is a satisfied request, not an error — the CLI exits 0.

Per-deployment operation serialization: a per-deployment
lock plus an optimistic revision check, so the loser of a concurrent
conflict gets a reported outcome rather than an interleaved partial
effect.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from tensorstead.domain.errors import (
    AgentUnreachableError,
    AlreadyInStateError,
    NotFoundError,
    PartialFailureError,
)
from tensorstead.domain.models import Deployment, ImageRecord, Node, ReplicaState
from tensorstead.ports.node_client import AgentCallError
from tensorstead.ports.repository import Repository
from tensorstead.ports.runtime_adapter import NodePosition
from tensorstead.service.nodes import refuse_if_node_reserved

# What a start operation has and has not established.
#
# `_drive_nodes(..., "start", ...)` returns when every node's container has been
# created and started. That is not the same as the deployment serving: a 27B
# model spends minutes loading weights and compiling afterwards, during which
# the container is up, the port is bound, and no inference is possible.
#
# The operation record was reporting `succeeded` for the first fact while an
# operator read it as the second. Saying so costs one line and closes the gap
# between "the command worked" and "the thing you wanted is true".
_STARTED_NOT_SERVING = (
    "containers started; the runtime is not serving until it has loaded. "
    "A large model takes minutes, during which observed status is running "
    "with inference_ready false. Check `deployment show <name>`, and "
    "`deployment runtime <name>` if it does not settle"
)


def _fabric_address(node: Node) -> str:
    """The host a peer reaches ``node`` at, taken from its agent endpoint.

    The agent endpoint is the one address the product already holds for every
    node and has already proven reachable. Deriving from it avoids a second,
    separately maintained address that could disagree with it -- and a fabric
    address that disagrees with the management address is a record that stopped
    matching reality somewhere nobody would think to look.
    """
    return str(urlsplit(node.agent_endpoint).hostname or node.agent_endpoint)


class LifecycleService:
    """Drive a deployment's desired-state transitions through the node client."""

    def __init__(
        self,
        repository: Repository,
        node_client: Any,
        runtime_adapters: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repository
        self._client = node_client
        # Only to ask whether a runtime's ranks must be started in order.
        # Optional so every existing construction keeps working and
        # behaves exactly as it did: no adapters means no staging, which is the
        # concurrent dispatch this service has always done.
        self._adapters = runtime_adapters or {}
        # Per-deployment lock for operation serialization.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, deployment_id: str) -> threading.Lock:
        """Return the per-deployment lock, creating it on first use."""
        with self._locks_guard:
            if deployment_id not in self._locks:
                self._locks[deployment_id] = threading.Lock()
            return self._locks[deployment_id]

    # ----------------------------------------------------------------- start
    def start(self, deployment_id: str) -> Deployment:
        """Start a deployment.

        Already running → ``already_in_state`` with no second instance.
        On start we drive the agent to materialize the
        current revision, record ``running_revision``, and set desired state to
        ``running`` so the boot-restoration unit stays enabled.
        """
        with self._lock_for(deployment_id):
            deployment = self._get_deployment(deployment_id)
            if deployment.desired_state == "running":
                raise AlreadyInStateError(
                    f"deployment {deployment.name!r} is already running",
                    detail={"deployment_id": deployment.id},
                )

            revision = self._repo.get_revision(deployment.id, deployment.current_revision)
            if revision is None:
                raise NotFoundError(
                    f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
                )

            # Drive the agent on every participating node, collecting per-node
            # outcomes. Raises before the save below if any node
            # failed, so desired state is never recorded as changed on a
            # half-applied operation.
            self._drive_nodes(revision, "start", self._create_on)

            # Records the revision the nodes were last successfully *asked* to
            # run. Declared state, not observed: it is written because the
            # start call returned, and a container that starts and then dies
            # during load leaves this value describing an intent rather than a
            # reality. The measured counterpart is the per-node
            # `running_revision` in observed state, read back from the
            # container's own label, and the two are deliberately
            # separate.
            updated = replace(
                deployment,
                desired_state="running",
                running_revision=deployment.current_revision,
            )
            self._repo.save_deployment(updated)
            return updated

    def start_advisories(self) -> list[str]:
        """What a completed start or restart has actually established.

        Returned to the caller so the operation's success can be read for what
        it is. See ``_STARTED_NOT_SERVING``.
        """
        return [_STARTED_NOT_SERVING]

    # ------------------------------------------------------------------ stop
    def stop(self, deployment_id: str) -> Deployment:
        """Stop a deployment.

        Already stopped → ``already_in_state``. Stops the
        container, disables and removes the unit so a reboot does not revive
        it, and records desired state ``stopped``.
        """
        with self._lock_for(deployment_id):
            deployment = self._get_deployment(deployment_id)
            if deployment.desired_state == "stopped":
                raise AlreadyInStateError(
                    f"deployment {deployment.name!r} is already stopped",
                    detail={"deployment_id": deployment.id},
                )

            revision = self._repo.get_revision(deployment.id, deployment.current_revision)
            if revision is None:
                raise NotFoundError(
                    f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
                )

            self._drive_nodes(revision, "stop", self._stop_on)

            updated = replace(deployment, desired_state="stopped")
            self._repo.save_deployment(updated)
            return updated

    # --------------------------------------------------------------- restart
    def restart(self, deployment_id: str) -> Deployment:
        """Restart a deployment — stop then start.

        Produces a fresh start on the current revision. A deployment that is
        stopped will simply start. A running deployment is stopped and then
        started again.
        """
        with self._lock_for(deployment_id):
            deployment = self._get_deployment(deployment_id)
            # If running, stop first. If already stopped, just start.
            if deployment.desired_state == "running":
                self._stop_on_nodes(deployment)
            # Start on the current revision.
            self._start_on_nodes(deployment)
            updated = replace(
                deployment,
                desired_state="running",
                running_revision=deployment.current_revision,
            )
            self._repo.save_deployment(updated)
            return updated

    # ---------------------------------------------------------------- remove
    def remove(self, deployment_id: str) -> dict[str, Any]:
        """Remove a deployment — instance + unit go, model and image stay.

        Removal of a running deployment stops the instance and removes its
        boot arrangement. Fails rather than orphaning when a node is
        unreachable. Returns what was removed and what was retained.
        """
        with self._lock_for(deployment_id):
            deployment = self._get_deployment(deployment_id)
            revision = self._repo.get_revision(deployment.id, deployment.current_revision)
            if revision is None:
                raise NotFoundError(
                    f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
                )

            per_node: dict[str, dict[str, Any]] = {}
            for node_id in revision.participating_nodes:
                node = self._repo.get_node(node_id)
                if node is None:
                    raise NotFoundError(f"no node with id {node_id!r}")
                try:
                    self._client.remove_deployment(node, deployment.id)
                    per_node[node_id] = {"state": "succeeded"}
                except AgentCallError as exc:
                    per_node[node_id] = {"state": "failed", "code": exc.code}
                    raise AgentUnreachableError(
                        f"remove failed on node {node.name!r}: {exc.message}",
                        node_id=node.id,
                        detail={"code": exc.code},
                    ) from exc

            # Delete the deployment and its revisions from the store.
            self._repo.delete_deployment(deployment.id)

            return {
                "removed": ["container", "systemd_unit"],
                "retained": ["model_artifacts", "image"],
                "per_node": per_node,
            }

    # The optimistic revision check used to be implemented *twice*: here, reachable
    # from nothing, and as ``DeploymentService._check_expected_revision``, which
    # is the one ``modify`` calls. No lifecycle surface accepts an expected
    # revision -- ``expected_revision`` exists only on ``DeploymentModifyRequest``
    # -- so this copy could never run. It was deleted rather than wired: two
    # implementations of one rule is how they drift, and the live one is the
    # deployments-side check.

    # --------------------------------------------------- per-node execution
    def _drive_nodes(
        self,
        revision: Any,
        action: str,
        call: Callable[..., None],
    ) -> dict[str, dict[str, Any]]:
        """Run ``call`` on every participating node and judge the whole.

        Every node is attempted, rather than stopping at the first failure, so
        the reported outcome describes the state the cluster is actually in. The
        verdict then depends on the mix:

        - **all succeeded** → return the outcomes; the caller records the new
          desired state.
        - **some succeeded, some failed** → ``partial_failure``. The nodes that
          changed are left as they are and surface as divergence; rolling them
          back would be a mutation nobody requested.
        - **all failed** → the single-node shape (``agent_unreachable``), so a
          one-node deployment reports exactly what it did before.
        """
        per_node: dict[str, dict[str, Any]] = {}
        failures: list[tuple[Node, AgentCallError]] = []

        nodes: list[Node] = []
        for node_id in revision.participating_nodes:
            node = self._repo.get_node(node_id)
            if node is None:
                raise NotFoundError(f"no node with id {node_id!r}")
            # Refused only for "start" -- a node marked reserved means "put no
            # new work here", not "nothing already there may be stopped". The
            # single entry point ``start`` and a stop-then-start cycle both
            # reach (create/modify's own check lives in DeploymentService,
            # sharing the same function so the two cannot drift the way
            # create/modify itself once did).
            if action == "start":
                refuse_if_node_reserved(node)
            nodes.append(node)

        # Start the calls before awaiting any result.  A node can spend several
        # minutes pulling an image or warming a runtime; serial dispatch would
        # otherwise prevent every later nominated node from being attempted at
        # all, and a partial cluster would stay hidden behind one
        # stalled request.
        # ...unless the runtime's own group cannot form that way, in which case
        # the ranks are ordered.
        #
        # **Workers first, head last.** That is the order of the reference
        # recipe this estate's DeepSeek deployment was derived from, and the
        # order that demonstrably works: every non-head rank runs with
        # `--headless`, waits for the head's rendezvous, and the head is started
        # last as the API process that completes the group.
        #
        # It was the inverse until now -- head dispatched alone and awaited, then
        # workers -- and a recorded finding named that as the last of three
        # differences from the reference shape. The other two are closed: the
        # adapter no longer emits `--distributed-executor-backend`, and
        # `--headless` is rendered per rank. This is the remainder.
        #
        # Worth being honest about what is and is not known. Head-first was not
        # *proven* wrong; six TP=2 attempts failed for reasons that were each
        # identified and separately fixed, and none was traced to ordering. The
        # argument for changing it is narrower: we have exactly one known-good
        # sequence, no evidence that deviating from it buys anything, and a first
        # hardware attempt is a bad place to be testing two hypotheses at once.
        #
        # Every nominated node is still attempted, and the workers
        # among them are still dispatched
        # concurrently with each other. Only the head is ordered, now last.
        # ``nodes`` stays the whole declared group throughout: every rank derives
        # its position from it, so it must not be narrowed to whoever is being
        # dispatched right now. ``dispatch`` is who still needs calling.
        dispatch = list(nodes)
        staged = action == "start" and len(nodes) > 1 and self._requires_staged_start(revision)
        if staged:
            lead, workers = nodes[0], nodes[1:]
            with ThreadPoolExecutor(max_workers=max(1, len(workers))) as executor:
                worker_futures = {
                    node.id: executor.submit(call, node, revision, nodes) for node in workers
                }
                for node in workers:
                    try:
                        worker_futures[node.id].result()
                        per_node[node.id] = {"state": "succeeded"}
                    except AgentCallError as exc:
                        per_node[node.id] = {
                            "state": "failed",
                            "code": exc.code,
                            "message": exc.message,
                        }
                        failures.append((node, exc))

            if failures:
                # The head is not started. It would come up expecting a full
                # group, wait for a rank that is not coming, and hold accelerator
                # memory while doing it. The workers that did start are unwound
                # by the caller.
                per_node[lead.id] = {
                    "state": "failed",
                    "code": "group_workers_unavailable",
                    "message": (
                        f"not started: {len(failures)} of {len(workers)} worker rank(s) "
                        f"did not become available, so the group cannot form"
                    ),
                }
                dispatch = []
            else:
                dispatch = [lead]

        with ThreadPoolExecutor(max_workers=max(1, len(dispatch))) as executor:
            # ``nodes`` is passed through rather than re-resolved per thread.
            # An earlier draft looked each peer up from inside the worker, which
            # put concurrent reads on one sqlite connection and broke two
            # multi-node tests -- the resolved objects were already here.
            futures = {node.id: executor.submit(call, node, revision, nodes) for node in dispatch}
            for node in dispatch:
                future = futures[node.id]
                try:
                    future.result()
                    per_node[node.id] = {"state": "succeeded"}
                except AgentCallError as exc:
                    per_node[node.id] = {
                        "state": "failed",
                        "code": exc.code,
                        "message": exc.message,
                    }
                    failures.append((node, exc))

        if not failures:
            return per_node

        # Read the verdict *before* unwinding. The cleanup rewrites a started
        # rank's state to ``stopped``, and computing the verdict afterwards saw
        # no successes at all -- so a group that half-formed and was tidied up
        # reported ``agent_unreachable``, when nothing was unreachable. The
        # operation partially succeeded and was then unwound, and the record has
        # to say both.
        succeeded = [nid for nid, outcome in per_node.items() if outcome["state"] == "succeeded"]
        if staged:
            self._unwind_partial_group(revision, nodes, per_node)
        failed_node, failure = failures[0]
        if succeeded:
            failed_names = ", ".join(
                repr(self._node_name(nid))
                for nid, outcome in per_node.items()
                if outcome["state"] == "failed"
            )
            raise PartialFailureError(
                f"{action} succeeded on {len(succeeded)} of "
                f"{len(per_node)} nodes; failed on {failed_names}",
                node_id=failed_node.id,
                detail={
                    "action": action,
                    "per_node": per_node,
                    "desired_state_changed": False,
                },
            )
        raise AgentUnreachableError(
            f"{action} failed on node {failed_node.name!r}: {failure.message}",
            node_id=failed_node.id,
            detail={"code": failure.code, "per_node": per_node},
        )

    def _unwind_partial_group(
        self, revision: Any, nodes: list[Node], per_node: dict[str, dict[str, Any]]
    ) -> None:
        """Stop the ranks that started when the group failed to form.

        **Why this reverses work when the rest of the product does not.** The
        standing rule is that a partial multi-node operation leaves what
        succeeded alone: rolling it back would be a mutation nobody requested,
        and a distributed image build deliberately keeps the image it managed to
        produce. A half-formed *runtime group* is the case where that reasoning
        does not hold. A rank whose peers never joined serves nothing and cannot
        be made to -- it holds accelerator memory for a group that does not
        exist. The first live TP=2 retry left exactly that: the head exited, the
        worker stayed up logging broken pipes, and an operator had to stop it by
        hand.

        The narrowness is what keeps it honest: this unwinds only ranks *this
        operation started*, only for a runtime that declared its ranks form one
        group, and only when that group failed. It never touches state it did
        not create.

        **Evidence is captured before stopping**, because stopping destroys it.
        A cleanup that erased why the group failed would trade one unusable
        state for another.
        """
        started = [node for node in nodes if per_node.get(node.id, {}).get("state") == "succeeded"]
        for node in started:
            outcome = per_node[node.id]
            try:
                outcome["runtime_evidence"] = self._client.get_runtime(
                    node, revision.deployment_id, tail=200
                )
            except Exception as exc:  # evidence is best-effort; cleanup is not
                outcome["runtime_evidence_error"] = str(exc)
            try:
                self._client.stop_deployment(node, revision.deployment_id)
                outcome["state"] = "stopped"
                outcome["code"] = "group_did_not_form"
                outcome["message"] = (
                    "this rank started but the group did not form; it was stopped "
                    "rather than left holding accelerator memory for a group that "
                    "does not exist"
                )
            except Exception as exc:
                # Reported, never swallowed: a rank that could not be stopped is
                # exactly the state an operator has to know about.
                outcome["cleanup_failed"] = str(exc)

    def _requires_staged_start(self, revision: Any) -> bool:
        """Whether this runtime's ranks must be started in order.

        Read from the adapter's declared container requirements rather than
        from a name or a list kept here: the fact that vLLM's group needs a head
        up first is a fact about vLLM, and the *same* declaration
        tells the agent which port to wait on. One source, so the two halves of
        the mechanism cannot disagree about whether a runtime needs staging.

        Silent about anything it cannot determine -- no adapter, no
        requirements, an adapter that predates this -- and silence means the
        concurrent dispatch every deployment has always had.
        """
        adapter = self._adapters.get(getattr(revision, "runtime_type", ""))
        if adapter is None:
            return False
        position = NodePosition(
            node_index=0,
            node_count=len(revision.participating_nodes),
            self_address="",
            peer_addresses=[],
        )
        try:
            requirements = adapter.container_requirements(dict(revision.runtime_config), position)
        except Exception:
            # An adapter that cannot describe its requirements must not be able
            # to block a start. Staging is a refinement; its absence is the
            # behaviour that shipped.
            return False
        return getattr(requirements, "rendezvous_port", None) is not None

    def _node_name(self, node_id: str) -> str:
        node = self._repo.get_node(node_id)
        return node.name if node is not None else node_id

    def _create_on(
        self,
        node: Node,
        revision: Any,
        peers: list[Node] | None = None,
        *,
        credential: str | None = None,
    ) -> None:
        replica = self._repo.get_replica(revision.model_id, node.id)
        if replica is None or replica.state != ReplicaState.AVAILABLE:
            raise AgentCallError(
                "model_not_available",
                f"model {revision.source_model_id!r} is not available on {node.name!r}",
                node_id=node.id,
            )
        payload: dict[str, Any] = {
            "deployment_id": revision.deployment_id,
            "revision": revision.revision,
            "runtime_type": revision.runtime_type,
            "image_reference": revision.image_reference,
            "runtime_config": revision.runtime_config,
            "model_path": replica.local_path,
            "endpoint": revision.endpoint,
            # Declared on the revision, default off. Sent on
            # every create so the node never has to infer it: an agent below
            # contract 1.13 refuses the unknown key outright rather than
            # silently ignoring it, which is the correct failure -- an agent
            # that dropped it would arrange boot restoration the old way and
            # nobody would know.
            "restore_on_boot": bool(getattr(revision, "restore_on_boot", False)),
        }
        # Where this node already sits among the declared ones.
        # Derived from the revision's ordered participating_nodes, never
        # configured -- the product states a position it already knows and
        # leaves the adapter to turn it into whatever its runtime needs.
        # Omitted entirely for a single-node deployment, so nothing about an
        # existing deployment changes.
        if peers and len(peers) > 1:
            addresses = [_fabric_address(peer) for peer in peers]
            index = [peer.id for peer in peers].index(node.id)
            payload["node_position"] = {
                "node_index": index,
                "node_count": len(peers),
                "self_address": addresses[index],
                "peer_addresses": addresses,
            }
        # Resolved once by the caller (_start_on_nodes), before any node was
        # dispatched, and held nowhere beyond this one request body over TLS
        # -- the agent writes none of it.
        # Omitted entirely when there is none, so the request has no
        # credential key at all rather than a null one.
        if credential is not None:
            payload["inference_credential"] = credential

        # The grant travels with the deployment that needs it, and only then.
        #
        # It is not a permission slip the agent takes on trust: it carries the
        # whole tuple, and the agent re-derives the model revision from its own
        # replica marker and the image digest from the image it actually
        # resolved, then recomputes the fingerprint. A grant that does not
        # describe what the node is about to run is refused there. Looked up
        # fresh from the approvals table on every start rather than read off the
        # revision, so a revoked approval stops working immediately instead of
        # at the next modify.
        grant = self._code_execution_grant(revision)
        if grant is not None:
            payload["code_execution_grant"] = grant

        response = self._client.create_deployment(node, payload)
        self._record_image(node, revision, response)

    def _code_execution_grant(self, revision: Any) -> dict[str, Any] | None:
        """The approval authorizing this revision's code-loading option, if any.

        Read from the approvals table by the fingerprint the revision recorded,
        not from the revision itself. The distinction is the whole point of
        revocation: the revision remembers what authorized it, and this asks
        whether that authorization still stands. A deleted approval yields
        ``None``, the agent receives no grant, and its own validation refuses
        the option -- so ``code_approval_delete`` stops the next start without
        anything having to walk the deployment table.
        """
        recorded = getattr(revision, "code_approval_fingerprint", "")
        if not recorded:
            return None
        approval = self._repo.find_code_approval(recorded)
        if approval is None:
            return None
        return approval.as_grant()

    def _resolve_inference_credential(self, deployment_id: str) -> str | None:
        """Resolve the deployment's bound inference credential, if any.

        Returns None when nothing is bound, which means "whatever this node was
        provisioned with applies" -- the pre-existing behaviour, and the
        migration path for deployments created before the product owned this.

        Raises ``CredentialResolutionError`` (uncaught, deliberately) when a
        binding exists but could not be resolved. This used to be caught here
        and turned into the same ``None`` "nothing bound" returns, on a
        comment claiming the start would then fail on the agent side with the
        runtime's own error -- it would not: the agent reads an absent
        credential as permission to fall back to its own node-wide key, or to
        none at all, so a broken binding silently became an unauthenticated
        or wrongly-keyed deployment instead of a failed start.
        ``resolve_for_deployment`` now raises
        this exactly when that distinction matters, so it is not caught here.
        """
        service = getattr(self, "_inference_credentials", None)
        if service is None:
            return None
        return service.resolve_for_deployment(deployment_id)  # type: ignore[no-any-return]

    def _record_image(self, node: Node, revision: Any, response: Any) -> None:
        """Record the image the agent actually pulled.

        The agent resolves and returns the platform-specific digest, and this
        response was previously discarded. Nothing called ``save_image``, so
        ``image_records`` was never written: `image list` was permanently empty
        and `image delete` could never find anything to remove or to refuse.
        An inventory that is always empty is indistinguishable from a node with
        no images, which is exactly the state a first-time user is in.
        """
        digest = (response or {}).get("image_digest") if isinstance(response, dict) else None
        if not digest:
            return
        self._repo.save_image(
            ImageRecord(
                node_id=node.id,
                reference=revision.image_reference,
                digest=str(digest),
                pulled_at=datetime.now().astimezone(),
            )
        )

    def _stop_on(
        self,
        node: Node,
        revision: Any,
        peers: list[Node] | None = None,  # noqa: ARG002 -- part of the _drive_nodes call shape
    ) -> None:
        self._client.stop_deployment(node, revision.deployment_id)

    # --------------------------------------------------------------- helpers
    def _start_on_nodes(self, deployment: Deployment) -> None:
        """Drive the agent to start the deployment on each participating node."""
        revision = self._repo.get_revision(deployment.id, deployment.current_revision)
        if revision is None:
            raise NotFoundError(
                f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
            )
        # Resolved once, here, rather than once per node inside _create_on
        # (which used to re-resolve it concurrently, redundantly, in every
        # dispatched thread). The binding is per-deployment, not per-node, so
        # a broken credential is the identical failure on every node and
        # should fail the whole start immediately -- before any node is
        # touched -- rather than surface as several identical per-node
        # failures. Uncaught deliberately: see
        # _resolve_inference_credential.
        credential = self._resolve_inference_credential(revision.deployment_id)
        self._drive_nodes(
            revision, "start", functools.partial(self._create_on, credential=credential)
        )

    def _stop_on_nodes(self, deployment: Deployment) -> None:
        """Drive the agent to stop the deployment on each participating node."""
        revision = self._repo.get_revision(deployment.id, deployment.current_revision)
        if revision is None:
            raise NotFoundError(
                f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
            )
        self._drive_nodes(revision, "stop", self._stop_on)

    def _get_deployment(self, deployment_id: str) -> Deployment:
        deployment = self._repo.get_deployment(deployment_id)
        if deployment is None:
            raise NotFoundError(f"no deployment with id {deployment_id!r}")
        return deployment
