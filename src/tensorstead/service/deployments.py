"""Deployment service.

``create`` validates the runtime config against the selected runtime adapter
before the deployment is valid, then writes revision 1 with
``desired_state: stopped`` and **alters no host state**.

Endpoint-collision refusal: a known managed deployment on the same
node+endpoint is refused with ``endpoint_conflict``, naming it — a refusal on
**identity** grounds, never capacity. ``runtime_not_distributed`` rejects a
multi-node request against a runtime that cannot distribute.

Deployment is a stable identity + desired state; every definition lives on an
immutable numbered revision. ``current_revision`` on the
Deployment row points at the latest revision.

``modify`` extends this module: it inserts revision *n+1* with no
``UPDATE`` against the revision table, reports ``restart_required``
without acting on it, lists and retrieves retained revisions, and
``create_from_export`` maps an exported revision back onto a
create request.
"""

from __future__ import annotations

import builtins
import contextlib
from dataclasses import replace as deploy_replace
from datetime import datetime
from typing import Any

from tensorstead.domain.errors import (
    EndpointConflictError,
    InvalidDeploymentError,
    NotFoundError,
    RuntimeNotDistributedError,
)
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import (
    Deployment,
    DeploymentRevision,
    canonical_file_selector,
)
from tensorstead.ports.repository import Repository
from tensorstead.service.approvals import authorize_code_options
from tensorstead.service.nodes import refuse_if_node_reserved


class DeploymentService:
    """Coordinator-side deployment definition and inspection."""

    def __init__(
        self,
        repository: Repository,
        runtime_adapters: dict[str, Any],
        node_client: Any | None = None,
    ) -> None:
        self._repo = repository
        self._adapters = runtime_adapters
        # Optional: without it the advisory pre-flight check is simply not
        # performed. It is advisory, so its absence must never change
        # whether a deployment can be created.
        self._client = node_client

    # ------------------------------------------------------------------ create
    def create(
        self,
        *,
        name: str,
        model_id: str,
        runtime_type: str,
        runtime_version: str,
        image_reference: str,
        runtime_config: dict[str, Any],
        participating_nodes: list[str],
        endpoint: str,
        restore_on_boot: bool = False,
    ) -> Deployment:
        """Create a deployment at revision 1; alters no host state.

        Validates runtime config against the selected adapter and
        enforces the endpoint-collision and not-distributed refusals
        before the deployment is treated as valid.
        """
        adapter = self._adapter(runtime_type)

        # The model is resolved *before* validation, not after, because a
        # config declaring a code-loading option can only be validated against
        # the approval that authorizes it, and an approval is keyed by this
        # model's source and immutable revision. Previously this lookup happened
        # after validate_config; moving it up changes no outcome for any
        # deployment that loads no code.
        model = self._repo.get_model(model_id)
        if model is None:
            raise NotFoundError(f"no model with id {model_id!r}")

        approved_options, approval = authorize_code_options(
            self._repo,
            runtime_type=runtime_type,
            runtime_config=runtime_config,
            model=model,
            image_reference=image_reference,
            participating_nodes=participating_nodes,
        )

        # Validate runtime config against the runtime's own schema.
        try:
            validated_config = adapter.validate_config(
                runtime_config, approved_options=approved_options
            )
        except ValueError as exc:
            raise InvalidDeploymentError(f"invalid {runtime_type} config: {exc}") from exc

        # Multi-node requires a distributed-capable runtime.
        if len(participating_nodes) > 1 and not adapter.supports_distributed:
            raise RuntimeNotDistributedError(
                f"runtime {runtime_type!r} cannot distribute; a deployment naming "
                f"more than one node is refused",
                detail={"runtime": runtime_type, "nodes": len(participating_nodes)},
            )

        self._require_distinct_nodes(participating_nodes)
        self._require_registered_nodes(participating_nodes)
        self._validate_distribution(adapter, runtime_type, validated_config, participating_nodes)

        # Endpoint-collision refusal on identity grounds.
        self._check_endpoint_conflict(participating_nodes, endpoint, exclude_deployment=None)

        deployment_id = new_ulid()
        now = datetime.now().astimezone()
        origin_facts = self._origin_platform_facts(participating_nodes)
        deployment = Deployment(
            id=deployment_id,
            name=name,
            desired_state="stopped",
            current_revision=1,
            running_revision=None,
            created_at=now,
        )
        self._repo.save_deployment(deployment)

        revision = DeploymentRevision(
            deployment_id=deployment_id,
            revision=1,
            model_id=model.id,
            model_source_id=model.source_id,
            source_model_id=model.source_model_id,
            resolved_revision=model.resolved_revision,
            revision_pinned=model.revision_pinned,
            runtime_type=runtime_type,
            runtime_version=runtime_version,
            image_reference=image_reference,
            # Deliberately empty and derived later, not "resolved at start"
            # as this comment used to claim while nothing resolved it. The
            # digest is recorded when the agent pulls, and read back from the
            # image records when reporting or exporting.
            image_digest="",
            runtime_config=validated_config,
            participating_nodes=tuple(participating_nodes),
            endpoint=endpoint,
            restore_on_boot=restore_on_boot,
            origin_platform_facts=origin_facts,
            created_at=now,
            code_approval_fingerprint=approval.fingerprint if approval else "",
            code_approval_id=approval.id if approval else "",
        )
        self._repo.insert_revision(revision)
        return deployment

    def _require_distinct_nodes(self, node_ids: builtins.list[str]) -> None:
        """Refuse a participant named more than once.

        Refused rather than deduplicated, deliberately. Declared order carries
        rank in a distributed group, so silently dropping a repeat would both
        hide the operator's mistake and change which node becomes rank 0.
        """
        duplicates = sorted({nid for nid in node_ids if node_ids.count(nid) > 1})
        if duplicates:
            raise InvalidDeploymentError(
                f"participating nodes must be distinct; named more than once: "
                f"{', '.join(duplicates)}",
                detail={"duplicates": duplicates},
            )

    def _validate_distribution(
        self,
        adapter: Any,
        runtime_type: str,
        config: dict[str, Any],
        node_ids: builtins.list[str],
    ) -> None:
        """Refuse a config that cannot describe the declared group.

        The adapter owns the rule, because what a parallelism setting means is a
        fact about the runtime and not about deployments. The
        service owns *when* it is asked: before the revision is stored, so the
        contradiction is caught while it is still only a record.
        """
        try:
            adapter.validate_distribution(config, node_count=len(node_ids))
        except ValueError as exc:
            raise InvalidDeploymentError(f"invalid {runtime_type} config: {exc}") from exc

    def _require_registered_nodes(self, node_ids: builtins.list[str]) -> None:
        """Every named participant must be a registered, unreserved node.

        ``create`` enforced registration and ``modify`` did not, so a PATCH
        could store a revision naming a node that does not exist. Nothing
        refused it: the definition became the deployment's current revision
        and failed only later at start, by which point the record and any
        possible reality had already diverged.

        Both callers share this so the two paths cannot drift again -- they
        drifted originally because one grew a rule the other never learned.
        """
        for node_id in node_ids:
            node = self._repo.get_node(node_id)
            if node is None:
                raise NotFoundError(f"no node with id {node_id!r}")
            refuse_if_node_reserved(node)

    def _check_expected_revision(self, deployment_id: str, expected: int) -> None:
        """Refuse when the deployment moved since the caller last read it."""
        from tensorstead.domain.errors import ConcurrentModificationError

        deployment = self._repo.get_deployment(deployment_id)
        if deployment is None:
            raise NotFoundError(f"deployment {deployment_id!r} was not found")
        if deployment.current_revision != expected:
            raise ConcurrentModificationError(
                f"deployment {deployment.name!r} was modified concurrently: "
                f"expected revision {expected}, actual {deployment.current_revision}",
                detail={"expected": expected, "actual": deployment.current_revision},
            )

    def config_advisories(
        self,
        runtime_type: str,
        config: dict[str, Any],
        node_names: builtins.list[str] | None = None,
    ) -> builtins.list[str]:
        """Adapter notes about a valid configuration.

        Advisory and never gating: these configurations run. The runtime simply
        does something the operator probably did not intend and mentions it in a
        startup line nobody reads until something is already wrong.

        An adapter that does not implement this contributes nothing, so a
        runtime nobody has written notes for behaves exactly as before.
        """
        adapter = self._adapter(runtime_type)
        declare = getattr(adapter, "config_advisories", None)
        if declare is None:
            return []

        # One node's facts at a time, because hardware need not be uniform: a
        # deployment spanning two nodes can be accelerated on one and not the
        # other, and collapsing that to a single answer would hide the case
        # worth knowing about. Deduplicated in order, since the common case is
        # identical hardware saying the same thing twice.
        notes: builtins.list[str] = []
        for facts in self._participating_platform_facts(node_names):
            try:
                produced = list(declare(config, facts))
            except Exception:
                # A broken advisory must never cost an operator their
                # deployment. It is a note; failing to produce one is not a
                # reason to refuse.
                produced = []
            for note in produced:
                if note not in notes:
                    notes.append(note)
        return notes

    def _participating_platform_facts(
        self, node_names: builtins.list[str] | None
    ) -> builtins.list[dict[str, Any]]:
        """Platform facts per named node; one empty mapping when none are named.

        The empty mapping matters: an adapter must still get the chance to say
        something that does not depend on hardware, and returning no mappings at
        all would silence every advisory whenever the caller omitted nodes.
        """
        if not node_names:
            return [{}]
        facts: builtins.list[dict[str, Any]] = []
        for name in node_names:
            node = self._repo.get_node_by_name(name) or self._repo.get_node(name)
            if node is None:
                facts.append({})
                continue
            facts.append(self._live_platform_facts(node))
        return facts or [{}]

    def _live_platform_facts(self, node: Any) -> dict[str, Any]:
        """Ask the node what it is now, falling back to its registration record.

        The record is a snapshot taken when the node registered and never
        rewritten, which is correct as a record and wrong as an input to a
        judgement about the hardware *today*. A fact the agent only learned to
        report in a later release cannot appear in it at all -- which is exactly
        what happened to the quantization advisory, whose measured
        capability could never have been rendered on an estate registered days
        before the agent began reporting it.

        Never raises. This feeds a note, and a note must not be able to cost an
        operator their deployment.
        """
        # Merged, not replaced, and the difference is a real trap. The two sets
        # are not the same shape: the appliance reports
        # `accelerator_compute_capability` today and no longer reports
        # `memory_is_unified`, which its registration record still holds. A live
        # read that wins wholesale would silently drop every fact the agent has
        # stopped volunteering -- a *non-empty* answer looking authoritative
        # while being partial, which is harder to notice than an empty one.
        #
        # So: the record is the floor, and anything the node says now wins over
        # it key by key.
        facts = dict(getattr(node, "platform_facts", None) or {})
        ask = getattr(self._client, "get_info", None)
        if ask is not None:
            try:
                live = dict((ask(node) or {}).get("platform_facts") or {})
            except Exception:
                live = {}
            facts.update(live)
        return facts

    def endpoint_preflight(self, participating_nodes: list[str], endpoint: str) -> list[str]:
        """Advisory, read-only check of whether the endpoint is already in use.

        The design permits this and is emphatic about its limits: the
        runtime's own bind result is what actually enforces availability, and
        this must never be represented as a guarantee. A port free when checked
        can be taken before the runtime starts.

        So it warns and never refuses. Every failure mode -- an unreachable
        node, an agent too old to answer, a transport error -- yields no
        warning rather than an error, because a check that could block a
        creation would be enforcement, which is forbidden on anything but
        identity grounds.

        The agent implements this endpoint and nothing called it until now.
        """
        if self._client is None:
            return []
        try:
            port = int(endpoint.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return []

        warnings: list[str] = []
        for node_id in participating_nodes:
            node = self._repo.get_node(node_id)
            if node is None:
                continue
            with contextlib.suppress(Exception):
                # Advisory: a check we could not perform is not a finding, and
                # must not become one. Deliberately silent rather than logged —
                # an unreachable node during create is ordinary, and a log line
                # per node would train operators to ignore the category.
                result = self._client.check_endpoint(node, port)
                if isinstance(result, dict) and result.get("bound"):
                    warnings.append(
                        f"port {port} already appears to be in use on {node.name!r}. "
                        "This is advisory only: the check is a moment-in-time "
                        "observation and the runtime's own bind result is what "
                        "decides. The deployment was created regardless."
                    )
        return warnings

    def _check_endpoint_conflict(
        self,
        participating_nodes: list[str],
        endpoint: str,
        *,
        exclude_deployment: str | None,
    ) -> None:
        """Refuse when a known managed deployment holds the same node+endpoint.

        A refusal on identity grounds, never capacity.
        ``exclude_deployment`` lets modification skip the deployment
        being modified.
        """
        for deployment in self._repo.list_deployments():
            if exclude_deployment is not None and deployment.id == exclude_deployment:
                continue
            revision = self._repo.get_revision(deployment.id, deployment.current_revision)
            if revision is None:
                continue
            # Only a node+endpoint match is a conflict.
            if revision.endpoint == endpoint and any(
                n in revision.participating_nodes for n in participating_nodes
            ):
                raise EndpointConflictError(
                    f"endpoint {endpoint!r} is already held by managed deployment "
                    f"{deployment.name!r} on the same node",
                    detail={"conflicting_deployment": deployment.name, "endpoint": endpoint},
                )

    # ------------------------------------------------------------------ read
    def list(self) -> list[Deployment]:
        return self._repo.list_deployments()

    def get(self, deployment_id: str) -> Deployment:
        deployment = self._repo.get_deployment(deployment_id)
        if deployment is None:
            raise NotFoundError(f"no deployment with id {deployment_id!r}")
        return deployment

    def get_revision(self, deployment_id: str, revision: int | None = None) -> DeploymentRevision:
        deployment = self.get(deployment_id)
        revision_number = revision if revision is not None else deployment.current_revision
        rev = self._repo.get_revision(deployment.id, revision_number)
        if rev is None:
            raise NotFoundError(f"deployment {deployment.name!r} has no revision {revision_number}")
        return rev

    # ------------------------------------------------- modification & export
    def modify(
        self,
        deployment_id: str,
        *,
        runtime_config: dict[str, Any] | None = None,
        replace_config: bool = False,
        image_reference: str | None = None,
        runtime_version: str | None = None,
        endpoint: str | None = None,
        participating_nodes: builtins.list[str] | None = None,
        model_id: str | None = None,
        restore_on_boot: bool | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Modify a deployment definition — insert revision *n+1*.

        ``Deployment.id`` stays stable. No ``UPDATE`` is ever issued
        against the revision table: an accepted modification always inserts a
        new numbered revision. The resulting definition is validated
        against the runtime adapter and the endpoint-collision refusal
        is enforced, excluding the deployment being modified.

        ``restart_required`` is **reported, never acted on**: a
        running deployment is never restarted as an implicit consequence, and
        ``applied`` is always false because no host state changes here.

        ``runtime_config`` **merges** at the key level by default: passing one
        setting changes that key and leaves every other at its current value,
        so ``modify --config tool_call_parser=...`` can no longer silently drop
        ``gpu_memory_utilization`` and its neighbours. Whole-map
        replacement is the destructive reading and is reachable only with
        ``replace_config=True`` — an explicit, named-for-destruction switch.
        """
        # The optimistic half of concurrency control. The per-deployment lock
        # serialises operations, but a caller that read revision N, lost a
        # race, and then acted would still succeed against a definition that
        # changed underneath it -- no conflict reported, which is the outcome
        # the lock's optimistic half requires. Optional so callers with no
        # opinion are unaffected.
        if expected_revision is not None:
            self._check_expected_revision(deployment_id, expected_revision)

        deployment = self.get(deployment_id)
        current = self._repo.get_revision(deployment.id, deployment.current_revision)
        if current is None:
            raise NotFoundError(f"deployment {deployment.name!r} has no current revision")

        # Apply the partial update over the current definition.
        new_nodes = (
            list(participating_nodes)
            if participating_nodes is not None
            else list(current.participating_nodes)
        )
        # ``runtime_config`` merges by default so a one-key patch can
        # never drop its neighbours. Whole-map replacement requires the explicit
        # ``replace_config`` switch — the destructive reading must be named, not
        # reached by accident. The merge is done here, inside the per-deployment
        # lock the caller already holds, rather than by a caller that reads the
        # current config, merges, and PATCHes: that would race a concurrent
        # modify and overwrite its change, which is exactly what the lock
        # exists to prevent.
        if runtime_config is None:
            new_config = dict(current.runtime_config)
        elif replace_config:
            new_config = dict(runtime_config)
        else:
            new_config = {**dict(current.runtime_config), **runtime_config}
        # The merge is at the *key* level, and a nested map is one key. So a
        # modify naming `host_config` to change `ipc_mode` replaces the whole
        # map, silently dropping whatever else was in it. That is the documented
        # contract, and it is still a trap: it cost a revision on the appliance
        # by dropping a retained collective-library debug setting while an
        # operator believed they were changing shared memory alone
        # (a recorded finding from the appliance).
        #
        # Reported rather than deep-merged. Deep-merging would leave no way to
        # *remove* a nested key without inventing a sentinel, and changing merge
        # semantics under existing records is a larger claim than this defect
        # justifies. Naming what was dropped makes the surprise visible at the
        # moment it happens, which is all the operator needed.
        dropped = _nested_keys_dropped(dict(current.runtime_config), new_config)
        new_model_id = model_id if model_id is not None else current.model_id
        new_image = image_reference if image_reference is not None else current.image_reference
        new_version = runtime_version if runtime_version is not None else current.runtime_version
        new_endpoint = endpoint if endpoint is not None else current.endpoint

        # Validate the resulting config against the runtime's own schema.
        adapter = self._adapter(current.runtime_type)

        # Re-decided from scratch, never inherited from ``current``. A modify
        # can change the model, the image, or the config, and each of those
        # changes the tuple an approval is keyed by -- so carrying the previous
        # revision's authorization forward is precisely the bug this guards
        # against. The model is fetched here rather than below for the same
        # reason as in ``create``.
        new_model = self._repo.get_model(new_model_id)
        if new_model is None:
            raise NotFoundError(f"no model with id {new_model_id!r}")
        approved_options, approval = authorize_code_options(
            self._repo,
            runtime_type=current.runtime_type,
            runtime_config=new_config,
            model=new_model,
            image_reference=new_image,
            participating_nodes=new_nodes,
        )
        try:
            validated_config = adapter.validate_config(
                new_config, approved_options=approved_options
            )
        except ValueError as exc:
            raise InvalidDeploymentError(f"invalid {current.runtime_type} config: {exc}") from exc

        if len(new_nodes) > 1 and not adapter.supports_distributed:
            raise RuntimeNotDistributedError(
                f"runtime {current.runtime_type!r} cannot distribute; a deployment "
                f"naming more than one node is refused",
                detail={"runtime": current.runtime_type, "nodes": len(new_nodes)},
            )

        # Same definition-validity rule as create, and for the same reason: a
        # revision naming an unregistered node is not a deployment anyone can
        # start. Before the insert, never after.
        self._require_distinct_nodes(new_nodes)
        self._require_registered_nodes(new_nodes)
        self._validate_distribution(adapter, current.runtime_type, validated_config, new_nodes)

        # Endpoint-collision refusal on identity grounds,
        # excluding the deployment being modified.
        self._check_endpoint_conflict(new_nodes, new_endpoint, exclude_deployment=deployment.id)

        model = new_model
        next_revision = deployment.current_revision + 1
        new_rev = DeploymentRevision(
            deployment_id=deployment.id,
            revision=next_revision,
            model_id=model.id,
            model_source_id=model.source_id,
            source_model_id=model.source_model_id,
            resolved_revision=model.resolved_revision,
            revision_pinned=model.revision_pinned,
            runtime_type=current.runtime_type,
            runtime_version=new_version,
            image_reference=new_image,
            # As above: derived from the image record, never predicted here.
            image_digest="",
            runtime_config=validated_config,
            participating_nodes=tuple(new_nodes),
            endpoint=new_endpoint,
            # Carried forward when the caller says nothing, like every other
            # field here. A revision that *changes* it says so explicitly, so
            # the record shows who asked for boot persistence and when.
            restore_on_boot=(
                current.restore_on_boot if restore_on_boot is None else restore_on_boot
            ),
            origin_platform_facts=self._origin_platform_facts(new_nodes),
            created_at=datetime.now().astimezone(),
            code_approval_fingerprint=approval.fingerprint if approval else "",
            code_approval_id=approval.id if approval else "",
        )
        self._repo.insert_revision(new_rev)

        # Advance the deployment's current_revision; identity and desired state
        # are untouched.
        self._repo.save_deployment(
            deploy_replace(
                deployment,
                current_revision=next_revision,
            )
        )

        restart_required = deployment.running_revision is not None
        return {
            "revision": next_revision,
            "restart_required": restart_required,
            "applied": False,  # host untouched
            # The configuration actually recorded on the new revision, so the
            # surface can show it: a ``--replace-config`` that dropped a
            # key is visible at the moment it happens rather than only at the
            # next ``deployment show`` — and a merge that kept one is too.
            "runtime_config": dict(validated_config),
            # Notes about the configuration just recorded. Attached here rather
            # than only to create, because a deployment gains speculative
            # decoding by *modification* -- which is exactly how revision 12
            # got MTP -- and a warning only ever shown at create would never be
            # seen by the person who needed it.
            "warnings": dropped
            + self.config_advisories(
                current.runtime_type, validated_config, list(new_rev.participating_nodes)
            ),
        }

    def list_revisions(self, deployment_id: str) -> builtins.list[DeploymentRevision]:
        """List every retained revision, oldest first."""
        deployment = self.get(deployment_id)
        return self._repo.list_revisions(deployment.id)

    # ----------------------------------------------------------------- re-import
    def create_from_export(
        self,
        export: dict[str, Any],
        *,
        target_endpoint: str | None = None,
    ) -> Deployment:
        """Recreate a deployment from an exported artifact.

        Maps an ``deployment.export`` document onto a create request, resolving
        participating node **names** against the current inventory and erroring
        on a missing name rather than substituting silently. The recreation is
        a **new deployment with a new id** at revision 1 — not a continuation of
        the original.

        Comparability warnings are surfaced in ``app.state``-free form via the
        returned ``warnings`` list: a loud warning on
        cpu_arch/os_family mismatch, an advisory note on accelerator or memory,
        and **nothing refused** on either count.
        """
        placement = export.get("placement", {}) or {}
        node_names = placement.get("nodes", []) or []
        node_ids: builtins.list[str] = []
        for name in node_names:
            node = self._repo.get_node_by_name(name)
            if node is None:
                raise NotFoundError(
                    f"export references node {name!r} which is not in the inventory; "
                    f"register it first",
                    detail={"missing_node": name},
                )
            node_ids.append(node.id)

        model_block = export.get("model", {}) or {}
        source_id = str(model_block.get("source") or "")
        source_model_id = str(model_block.get("id") or "")
        resolved_revision = model_block.get("revision")
        # The file selection is part of the model's identity (migration 0006),
        # so it is part of the lookup. Without it an export naming one
        # quantization of a GGUF repository cannot find the model it names --
        # and, worse, could find a *different* selection of the same repository
        # at the same revision and recreate the deployment against the wrong
        # weights. Absent means the whole repository, which is what every
        # export written before file selection existed describes.
        file_selector = canonical_file_selector(model_block.get("file_selector"))
        model = self._repo.find_model(
            source_id,
            source_model_id,
            str(resolved_revision) if resolved_revision is not None else None,
            file_selector,
        )
        if model is None:
            selection = ", ".join(file_selector) if file_selector else "the whole repository"
            raise NotFoundError(
                f"export references model {source_id}:{source_model_id} "
                f"@{resolved_revision} ({selection}) which is not in the inventory; "
                f"acquire it first",
                detail={
                    "source_id": source_id,
                    "source_model_id": source_model_id,
                    "file_selector": list(file_selector),
                },
            )

        runtime = export.get("runtime", {}) or {}
        image = export.get("image", {}) or {}
        deployment = self.create(
            name=export.get("deployment", {}).get("name") or "recreated-deployment",
            model_id=model.id,
            runtime_type=runtime.get("type", "vllm"),
            runtime_version=runtime.get("version", "latest"),
            image_reference=image.get("reference") or "",
            runtime_config=export.get("runtime_config", {}) or {},
            participating_nodes=node_ids,
            endpoint=target_endpoint or placement.get("endpoint", ""),
        )
        return deployment

    def comparability_warnings(
        self, export: dict[str, Any], node_names: builtins.list[str]
    ) -> builtins.list[dict[str, Any]]:
        """Warn on platform comparability between the export's origin and targets.

        One hard dimension and one advisory one, both non-refusing:

        - ``cpu_arch`` / ``os_family`` mismatch → **loud** warning (the
          platform-specific digest cannot resolve on the target).
        - accelerator model / memory difference → **advisory** note (affects
          whether it runs, not whether declared values match).

        Nothing is refused on comparability grounds.
        """
        origin = export.get("origin_platform", {}) or {}
        warnings: builtins.list[dict[str, Any]] = []
        for name in node_names:
            node = self._repo.get_node_by_name(name)
            if node is None:
                continue
            facts = node.platform_facts or {}
            target_cpu = facts.get("cpu_arch")
            target_os = facts.get("os_family")
            severity: str | None = None
            reason = ""
            if origin.get("cpu_arch") and target_cpu and origin.get("cpu_arch") != target_cpu:
                severity = "warning"
                reason = f"cpu_arch {origin.get('cpu_arch')!r} != {target_cpu!r}"
            elif origin.get("os_family") and target_os and origin.get("os_family") != target_os:
                severity = "warning"
                reason = f"os_family {origin.get('os_family')!r} != {target_os!r}"
            elif origin.get("accelerator_model") and facts.get("accelerator_model"):
                if origin.get("accelerator_model") != facts.get("accelerator_model"):
                    severity = "note"
                    reason = "accelerator_model differs"
            if severity is not None:
                warnings.append(
                    {
                        "node": name,
                        "severity": severity,
                        "reason": reason,
                        "refused": False,
                    }
                )
        return warnings

    # ------------------------------------------------------------------ helpers
    def _origin_platform_facts(self, participating_nodes: builtins.list[str]) -> dict[str, Any]:
        """Snapshot the originating node's platform facts.

        Uses the first participating node's registered facts. Records, never
        constrains: the snapshot exists so a later export can warn on
        comparability, and nothing is decided on it.
        """
        for node_id in participating_nodes:
            node = self._repo.get_node(node_id)
            if node is not None and node.platform_facts:
                return dict(node.platform_facts)
        return {}

    def _adapter(self, runtime_type: str) -> Any:
        adapter = self._adapters.get(runtime_type)
        if adapter is None:
            raise InvalidDeploymentError(f"unknown runtime type {runtime_type!r}")
        return adapter


def _nested_keys_dropped(before: dict[str, Any], after: dict[str, Any]) -> builtins.list[str]:
    """Warn about entries a key-level merge removed from inside a nested map.

    Only nested maps, and only removals. Changing a value is what a modify is
    for; losing one the caller never mentioned is the surprise worth naming.
    """
    warnings: builtins.list[str] = []
    for key, previous in before.items():
        if not isinstance(previous, dict):
            continue
        replacement = after.get(key)
        if not isinstance(replacement, dict):
            continue
        lost = sorted(set(previous) - set(replacement))
        if lost:
            warnings.append(
                f"{key} was replaced whole, dropping {', '.join(lost)}. A modify "
                f"merges at the key level, so a nested map must be supplied "
                f"complete; resupply the dropped entries if they were wanted"
            )
    return warnings
