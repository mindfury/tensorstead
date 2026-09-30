"""Observed-state retrieval and divergence computation.

The two separable questions:

- **Declared state** — the coordinator's authoritative record.
- **Observed state** — what the responsible node reports right now,
  fetched at request time, carrying when it was read (``observed_at``),
  and degrading to ``unknown`` or ``unreachable`` rather than to a
  cached value.

The two are returned as **separate labelled blocks, never merged**.
Declared state is still returned in full alongside an unobtainable
observation. No previously observed value is presented as
current.

**Divergence** is computed as a by-product of observation over the
bounded set the design defines: ``declared_running_but_absent``,
``unexpected_instance``, ``revision_mismatch``, ``image_digest_mismatch``,
``endpoint_mismatch``. Detecting a divergence mutates nothing; host state
changes only when a client explicitly invokes reconcile.
No scheduled scan looks for them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tensorstead.domain.errors import NotFoundError
from tensorstead.domain.models import Deployment, DeploymentRevision, Node
from tensorstead.domain.state import Divergence
from tensorstead.ports.node_client import AgentCallError
from tensorstead.ports.repository import Repository


class ObservationService:
    """Coordinator-side observed-state retrieval and divergence."""

    def __init__(self, repository: Repository, node_client: Any) -> None:
        self._repo = repository
        self._client = node_client

    def observe(
        self, deployment_id: str
    ) -> tuple[Deployment, DeploymentRevision, dict[str, Any], list[Divergence]]:
        """Fetch observed state from the responsible node at request time.

        Returns ``(deployment, revision, observed_dict, divergences)``. The
        observed dict carries ``status``, ``observed_at``, ``per_node``, and
        ``running_image_digest``. On an unreachable node, status degrades to
        ``unreachable`` and ``observed_at`` is set to the attempt time —
        declared state is still returned in full by the caller.
        """
        deployment = self._repo.get_deployment(deployment_id)
        if deployment is None:
            raise NotFoundError(f"no deployment with id {deployment_id!r}")

        revision = self._repo.get_revision(deployment.id, deployment.current_revision)
        if revision is None:
            raise NotFoundError(
                f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
            )

        per_node: dict[str, dict[str, Any]] = {}
        overall_status = "not_running"
        latest_image_digest: str | None = None
        now = datetime.now().astimezone()

        for node_id in revision.participating_nodes:
            node = self._repo.get_node(node_id)
            if node is None:
                per_node[node_id] = {
                    "status": "unknown",
                    "running_image_digest": None,
                    "running_revision": None,
                    "endpoint_reachable": None,
                    "inference_ready": None,
                    "endpoint_authenticated": None,
                    "detail": "node not found in inventory",
                }
                continue

            node_obs = self._observe_node(node, deployment.id)
            per_node[node_id] = node_obs

            if node_obs["status"] == "unreachable":
                overall_status = "unreachable"
            elif node_obs["status"] == "running" and overall_status != "unreachable":
                overall_status = "running"
            elif node_obs["status"] == "unknown" and overall_status == "not_running":
                overall_status = "unknown"

            if node_obs.get("running_image_digest"):
                latest_image_digest = node_obs["running_image_digest"]

        # If any node is unreachable, the overall status reflects the worst case.
        if any(pn["status"] == "unreachable" for pn in per_node.values()):
            overall_status = "unreachable"

        observed = {
            "status": overall_status,
            "observed_at": now.isoformat(),
            "per_node": per_node,
            "running_image_digest": latest_image_digest,
            "endpoint_reachable": self._worst_case(per_node, "endpoint_reachable"),
            "inference_ready": self._worst_case(per_node, "inference_ready"),
            "detail": None,
        }

        divergences = self._compute_divergences(deployment, revision, per_node)
        return deployment, revision, observed, divergences

    @staticmethod
    def _worst_case(per_node: dict[str, dict[str, Any]], field: str) -> bool | None:
        """Aggregate a per-node boolean, worst case first.

        One node not serving means the deployment is not serving, so ``False``
        wins over everything. ``True`` is claimed only when every node said so:
        a single unknown collapses the answer to unknown, because "we could not
        tell about one of them" must not be reported as a clean bill of health.

        **Nodes that do not serve are not counted**. A vLLM
        rank other than the head runs headless and starts no API server, so it
        has no endpoint fact to contribute -- and counting its silence would
        make every correctly-configured two-node deployment report
        ``inference_ready: false`` for as long as it ran. Worst-case aggregation
        over a set that includes a member which can never be well is not
        caution, it is a permanent false negative.

        A group where *no* node serves still yields ``None`` rather than
        ``True``: the empty-values guard below catches it, so "nobody serves"
        cannot come out as a clean answer either.
        """
        values = [obs.get(field) for obs in per_node.values() if obs.get("serves_inference", True)]
        if not values:
            return None
        if any(value is False for value in values):
            return False
        if all(value is True for value in values):
            return True
        return None

    def _observe_node(self, node: Node, deployment_id: str) -> dict[str, Any]:
        """Fetch observed state from one node; degrade to unreachable on failure."""
        try:
            raw = self._client.get_observed(node, deployment_id)
        except AgentCallError as exc:
            return {
                "status": "unreachable",
                "running_image_digest": None,
                "running_revision": None,
                "endpoint_reachable": None,
                "inference_ready": None,
                "detail": exc.message,
            }

        return {
            "status": raw.get("status", "unknown"),
            "running_image_digest": raw.get("running_image_digest"),
            "running_revision": raw.get("running_revision"),
            "endpoint_reachable": raw.get("endpoint_reachable"),
            "inference_ready": raw.get("inference_ready"),
            # An agent below contract 1.12 omits this, and ``True`` is what such
            # an agent meant: it probed, so it has an endpoint fact to
            # contribute.
            "serves_inference": raw.get("serves_inference", True),
            "endpoint_authenticated": raw.get("endpoint_authenticated"),
            "detail": raw.get("detail"),
            "unexpected_instance": raw.get("unexpected_instance", False),
            # Absent from an agent older than this field. ``None`` records
            # "this agent does not enumerate" and is not the same fact as an
            # empty list, which would mean "the namespace is clean".
            "managed_containers": raw.get("managed_containers"),
        }

    def _compute_divergences(
        self,
        deployment: Deployment,
        revision: DeploymentRevision,
        per_node: dict[str, dict[str, Any]],
    ) -> list[Divergence]:
        """Compute the bounded divergence set.

        Detecting a divergence mutates nothing. The five
        kinds are:

        - ``declared_running_but_absent`` — desired running but not running.
        - ``unexpected_instance`` — a managed container the coordinator did
          not record. Reported, never killed.
        - ``revision_mismatch`` — running a different revision than declared.
        - ``image_digest_mismatch`` — running a different image digest.
        - ``endpoint_mismatch`` — the endpoint is not reachable.
        """
        divergences: list[Divergence] = []
        recorded_ids = {record.id for record in self._repo.list_deployments()}

        for node_id, obs in per_node.items():
            status = obs.get("status", "unknown")

            # declared_running_but_absent
            if (
                deployment.desired_state == "running"
                and status != "running"
                and status != "unreachable"
            ):
                divergences.append(
                    Divergence(
                        kind="declared_running_but_absent",
                        node_id=node_id,
                        declared="running",
                        observed=status,
                    )
                )

            # unexpected_instance — reported, never killed.
            #
            # Judged here because this is the only side that knows what was
            # recorded. The agent used to answer this alone and could not: it
            # sees its node's containers and nothing else, so "not recorded by
            # the coordinator" was a question it had no way to evaluate.
            divergences.extend(self._unexpected_instances(node_id, obs, recorded_ids))

            if obs.get("unexpected_instance"):
                divergences.append(
                    Divergence(
                        kind="unexpected_instance",
                        node_id=node_id,
                        declared=None,
                        observed="managed container whose label identity contradicts its name",
                    )
                )

            # revision_mismatch
            running_revision = obs.get("running_revision")
            if (
                status == "running"
                and running_revision is not None
                and running_revision != deployment.running_revision
            ):
                divergences.append(
                    Divergence(
                        kind="revision_mismatch",
                        node_id=node_id,
                        declared=deployment.running_revision,
                        observed=running_revision,
                    )
                )

            # image_digest_mismatch
            running_digest = obs.get("running_image_digest")
            if (
                status == "running"
                and running_digest is not None
                and revision.image_digest
                and running_digest != revision.image_digest
            ):
                divergences.append(
                    Divergence(
                        kind="image_digest_mismatch",
                        node_id=node_id,
                        declared=revision.image_digest,
                        observed=running_digest,
                    )
                )

            # endpoint_mismatch
            endpoint_reachable = obs.get("endpoint_reachable")
            if status == "running" and endpoint_reachable is False:
                divergences.append(
                    Divergence(
                        kind="endpoint_mismatch",
                        node_id=node_id,
                        declared=revision.endpoint,
                        observed="unreachable",
                    )
                )

            # inference_not_ready — the container runs and the port
            # answers, but the runtime does not serve. Only raised when the
            # probe actually established it: ``None`` means we could not tell,
            # and reporting a divergence we did not observe would trade a false
            # negative for a false positive rather than fixing anything.
            if (
                status == "running"
                and endpoint_reachable is not False
                and obs.get("inference_ready") is False
            ):
                divergences.append(
                    Divergence(
                        kind="inference_not_ready",
                        node_id=node_id,
                        declared=revision.endpoint,
                        observed=obs.get("detail") or "runtime is not serving",
                    )
                )

        return divergences

    @staticmethod
    def _unexpected_instances(
        node_id: str, obs: dict[str, Any], recorded_ids: set[str]
    ) -> list[Divergence]:
        """Managed containers on a node that no deployment record accounts for.

        The expected case: something is running in the product's namespace that
        the product did not put there — an operator's manual ``docker run``, or
        a container that outlived the deployment record it belonged to.
        **Reported, never killed.**

        ``managed_containers`` absent means an agent too old to enumerate, and
        no claim is made. An empty list is the opposite: an agent that looked
        and found nothing.

        A container is identified by its ``tensorstead.deployment_id`` label,
        falling back to its name. A container carrying neither an identifiable
        id nor the managed prefix is left alone rather than guessed at.
        """
        managed = obs.get("managed_containers")
        if not isinstance(managed, list):
            return []

        divergences: list[Divergence] = []
        for entry in managed:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "")
            identity = entry.get("deployment_id")
            if not identity and name.startswith("tensorstead-"):
                identity = name[len("tensorstead-") :]
            if not identity or identity in recorded_ids:
                continue
            divergences.append(
                Divergence(
                    kind="unexpected_instance",
                    node_id=node_id,
                    declared=None,
                    observed=f"{name} is running in the managed namespace but is not recorded"
                    if entry.get("running")
                    else f"{name} exists in the managed namespace but is not recorded",
                )
            )
        return divergences

    def runtime(self, deployment_id: str, *, tail: int = 500) -> dict[str, Any]:
        """Fetch each participating node's runtime report.

        The runtime's own account of itself: the argv it was launched with, how
        often it has been restarted, and its recent output. Observation could say
        a deployment stopped serving; this is what says why.

        Per node, because a distributed deployment fails on one node at a time
        and the failing one is the whole point of asking.

        Read through and discarded. No part of this is written to the
        repository, attached to a revision, or included in an export — a
        runtime may log request content, and the management plane must not be
        where that comes to rest.
        """
        deployment = self._repo.get_deployment(deployment_id)
        if deployment is None:
            raise NotFoundError(f"no deployment with id {deployment_id!r}")
        revision = self._repo.get_revision(deployment.id, deployment.current_revision)
        if revision is None:
            raise NotFoundError(
                f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
            )

        per_node: dict[str, dict[str, Any]] = {}
        for node_id in revision.participating_nodes:
            node = self._repo.get_node(node_id)
            if node is None:
                per_node[node_id] = {"detail": "node not found in inventory"}
                continue
            # Checked rather than caught. Wrapping the call in `except
            # AttributeError` would also swallow an AttributeError raised
            # anywhere *inside* a working client and report it as an old agent
            # -- a real bug rendered as a version difference, which is the
            # hardest kind to find.
            ask = getattr(self._client, "get_runtime", None)
            if ask is None:
                per_node[node_id] = {
                    "detail": "this coordinator's node client cannot ask for runtime state"
                }
                continue
            try:
                per_node[node_id] = ask(node, deployment.id, tail=tail)
            except AgentCallError as exc:
                # An unreachable node is reported as such rather than omitted.
                # A missing key in this map would read as "that node had
                # nothing to say", which is a different and more reassuring
                # fact than "we could not ask it".
                per_node[node_id] = {"detail": f"unreachable: {exc.message}"}

        return {
            "deployment_id": deployment.id,
            "observed_at": datetime.now().astimezone().isoformat(),
            "per_node": per_node,
        }

    def reconcile(self, deployment_id: str) -> dict[str, Any]:
        """Converge a deployment toward its declared state.

        Full convergence reports success; partial convergence reports failure
        naming each remaining divergence. Applied changes are never reverted,
        and the deployment is left in a state a later reconcile can act on.
        This is the only code path that mutates a host in response to
        divergence, and only because the coordinator explicitly called it
        after a client explicitly asked.
        """
        deployment = self._repo.get_deployment(deployment_id)
        if deployment is None:
            raise NotFoundError(f"no deployment with id {deployment_id!r}")

        revision = self._repo.get_revision(deployment.id, deployment.current_revision)
        if revision is None:
            raise NotFoundError(
                f"deployment {deployment.name!r} has no revision {deployment.current_revision}"
            )

        per_node: dict[str, dict[str, Any]] = {}
        any_remaining = False

        for node_id in revision.participating_nodes:
            node = self._repo.get_node(node_id)
            if node is None:
                per_node[node_id] = {
                    "state": "failed",
                    "code": "node_not_found",
                    "changed": [],
                    "remaining": [{"kind": "node_not_found"}],
                }
                any_remaining = True
                continue

            try:
                result = self._client.reconcile_deployment(node, deployment.id)
                per_node[node_id] = {
                    "state": "succeeded" if not result.get("remaining") else "partial",
                    "changed": result.get("changed", []),
                    "remaining": result.get("remaining", []),
                }
                if result.get("remaining"):
                    any_remaining = True
            except AgentCallError as exc:
                per_node[node_id] = {
                    "state": "failed",
                    "code": exc.code,
                    "changed": [],
                    "remaining": [{"kind": "agent_unreachable", "detail": exc.message}],
                }
                any_remaining = True

        return {
            "status": "succeeded" if not any_remaining else "partial",
            "deployment_id": deployment_id,
            "per_node": per_node,
            # One entry per remaining divergence, tagged with the node it is on.
            #
            # ``info["remaining"]`` is a *list*, and ``**`` over a list raises
            # ``TypeError: 'list' object is not a mapping``. The comprehension
            # only evaluated for nodes that had something remaining, so
            # reconcile raised 500 exactly when it had something to report and
            # returned cleanly whenever there was nothing to say -- which is
            # why it looked like it worked.
            "remaining": [
                {"node_id": nid, **item}
                for nid, info in per_node.items()
                for item in (info.get("remaining") or [])
            ],
        }
