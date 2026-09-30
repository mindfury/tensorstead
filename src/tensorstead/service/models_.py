"""Model acquisition service.

Acquires a logical model revision onto one or more participating nodes, once
from upstream and then reusing verified replicas. The coordinator
orchestrates and holds authoritative placement metadata; **no model byte
transits the coordinator process**.

- ``acquire`` drives the model source (via the agent) for the first node and
  reuses an existing verified replica for a node that already holds it at the
  same revision — transferring nothing.
- A staging or failed replica is never presented as available.
- ``list`` / ``get`` report the logical model and its per-node replica states.

Credentials are resolved **here and passed per-request**:
the request names a credential (or names none and gets the source's default),
the coordinator resolves that reference to a value, and the value travels to the
agent for that one call. Neither the coordinator's database nor the agent
persists it. An upstream refusal comes back as ``authorization_refused``.
"""

from __future__ import annotations

import builtins
import threading
from datetime import datetime
from typing import Any

from tensorstead.contracts.version import CompatibilityRefusal
from tensorstead.domain.errors import (
    AuthorizationRefusedError,
    NodeOperationFailedError,
    NotFoundError,
    StillReferencedError,
)
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import (
    ImageBuildSpec,
    ImageOrigin,
    ImageRecord,
    Model,
    ModelReplica,
    ModelSource,
    ReplicaState,
    canonical_file_selector,
    local_model_id,
)
from tensorstead.ports.model_source import UnpinnableError
from tensorstead.ports.node_client import AgentCallError
from tensorstead.ports.repository import Repository


def _store_path(source_id: str, source_model_id: str, file_selector: tuple[str, ...] = ()) -> str:
    """Where the node keeps this model, derived the way the node derives it.

    The coordinator records a replica's ``local_path`` and the agent creates
    the directory; two derivations of one fact is how they drift, so both call
    ``local_model_id``. A whole-repository acquisition yields the exact string
    this recorded before file selection existed.
    """
    return (
        f"/var/lib/tensorstead/models/{local_model_id(source_id, source_model_id, file_selector)}"
    )


class ModelService:
    """Coordinator-side model acquisition and inspection."""

    def __init__(
        self,
        repository: Repository,
        node_client: Any,
        credentials: Any | None = None,
    ) -> None:
        self._repo = repository
        self._client = node_client
        # The credential service resolves a named reference to a value for the
        # duration of one acquisition. Optional so a coordinator wired
        # without credentials still acquires ungated models.
        self._credentials = credentials
        # Per-key lock for acquire serialization. Two concurrent acquires of
        # the same (source, model, requested revision) must not both download:
        # the first installs a verified replica, the second reuses it. The lock
        # is held across the per-node loop so the reuse check sees the first's
        # committed replica rather than racing past an empty store. Mirrors the
        # per-deployment lock in lifecycle.py.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(
        self,
        source_id: str,
        source_model_id: str,
        requested_revision: str,
        file_selector: tuple[str, ...] = (),
    ) -> threading.Lock:
        """Return the per-key acquire lock, creating it on first use.

        The selection is part of the key because it is part of identity: two
        quantizations of one repository are different models and must not
        serialize behind each other, and -- more importantly -- must not be
        treated as the same acquisition by the reuse checks this lock guards.
        """
        key = local_model_id(source_id, source_model_id, file_selector)
        key = f"{key}@{requested_revision}"
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    # ------------------------------------------------------------------ model
    def acquire(
        self,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        nodes: list[str],
        credential: str | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> Model:
        """Acquire ``source_model_id`` onto ``nodes``.

        Resolves the concrete revision (the source decides when the operator
        gave none), reuses a verified replica where present at that revision,
        and records the logical model plus its per-node replicas. A
        staging/failed replica is never treated as available.

        ``credential`` is a **name**, not a value. Naming none applies
        the source's default, which is how the automatic application works.
        """
        requested_revision = self._requested_revision(revision)
        selector = canonical_file_selector(file_selector)
        with self._lock_for(source_id, source_model_id, requested_revision, selector):
            return self._acquire_under_lock(
                source_id=source_id,
                source_model_id=source_model_id,
                revision=revision,
                requested_revision=requested_revision,
                nodes=nodes,
                credential=credential,
                file_selector=selector,
            )

    def _acquire_under_lock(
        self,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        requested_revision: str,
        nodes: list[str],
        credential: str | None,
        file_selector: tuple[str, ...] = (),
    ) -> Model:
        """The acquire body, run under the per-key lock.

        Serialized per ``(source, model, requested revision)`` so a concurrent
        second acquire reuses the first's verified replica instead of racing
        it to the same on-node target directory. ``requested_revision`` is the
        upstream reference (``revision or "main"``); the resolved immutable
        revision comes back from the agent and is what gets recorded.
        """
        source = self._source(source_id)

        # The agent owns the upstream client and therefore resolves the source
        # reference.  Its successful response returns the immutable revision
        # that was actually downloaded; never retain a moving label such as
        # ``main`` as though it were reproducible.
        resolved: str | None = None

        # Resolve the credential reference to a value once, for this
        # acquisition only. Held in memory across the per-node calls below and
        # never written anywhere.
        secret = self._resolve_credential(source_id, credential)

        # Acquire once from upstream, then replicate to the rest.
        # ``holder`` is the first node known to hold an available replica; every
        # later node prefers pulling from it over a second upstream download.
        replicas: list[ModelReplica] = []
        holder: str | None = None
        for node_id in nodes:
            existing = (
                self._verified_replica_at(
                    source_id, source_model_id, requested_revision, node_id, file_selector
                )
                if revision is not None
                else None
            )
            if existing is not None:
                # Reuse the verified replica; transfer nothing.
                replicas.append(existing)
                holder = holder or node_id
                resolved = requested_revision
                continue

            if (
                holder is not None
                and resolved is not None
                and self._try_replicate(
                    source_id,
                    source_model_id,
                    resolved,
                    source_node_id=holder,
                    dest_node_id=node_id,
                    file_selector=file_selector,
                )
            ):
                replicas.append(
                    self._record_available(
                        source_id, source_model_id, resolved, node_id, file_selector
                    )
                )
                continue

            result = self._acquire_on_node(
                source, source_model_id, requested_revision, node_id, secret, file_selector
            )
            actual_revision = self._resolved_revision_from_agent(result, source_model_id)
            if resolved is not None and actual_revision != resolved:
                raise UnpinnableError(
                    f"source returned inconsistent revisions for {source_model_id!r}: "
                    f"{resolved!r} then {actual_revision!r}"
                )
            resolved = actual_revision
            # Record the replica first: it creates the logical model row that
            # the digest is then written onto.
            replicas.append(
                self._record_available(source_id, source_model_id, resolved, node_id, file_selector)
            )
            self._record_content_digest(source_id, source_model_id, resolved, result, file_selector)
            holder = holder or node_id

        if resolved is None:
            raise UnpinnableError(
                f"source did not resolve an immutable revision for {source_model_id!r}"
            )
        return self._find_or_create_model(
            source_id, source_model_id, resolved, replicas, file_selector
        )

    def _try_replicate(
        self,
        source_id: str,
        source_model_id: str,
        revision: str,
        *,
        source_node_id: str,
        dest_node_id: str,
        file_selector: tuple[str, ...] = (),
    ) -> bool:
        """Ask ``dest`` to pull the model from ``source``; report whether it did.

        Returns ``False`` rather than raising when replication is not possible —
        an unreachable peer, no usable route, an agent too old to support the
        operation. **Local replication is an optimization, not a precondition**,
        so the caller falls back to acquiring
        from upstream instead of failing the whole request.
        """
        source_node = self._repo.get_node(source_node_id)
        dest_node = self._repo.get_node(dest_node_id)
        if source_node is None or dest_node is None:
            return False

        model = self._repo.find_model(source_id, source_model_id, revision, file_selector)
        if model is None or model.content_digest is None:
            # Nothing to verify the transfer against; the destination would
            # refuse to promote it anyway. Go upstream instead.
            return False

        try:
            self._client.replicate_model(
                dest_node,
                model={
                    "source_id": source_id,
                    "source_model_id": source_model_id,
                    "resolved_revision": revision,
                    "content_digest": model.content_digest,
                },
                source_node=source_node,
            )
        except (AgentCallError, CompatibilityRefusal, AttributeError, NotImplementedError):
            return False
        return True

    def _record_content_digest(
        self,
        source_id: str,
        source_model_id: str,
        revision: str,
        result: dict[str, Any] | None,
        file_selector: tuple[str, ...] = (),
    ) -> None:
        """Persist the digest the agent reported, so peers can verify against it.

        Without this the second node has nothing to check a peer transfer
        against, and replication correctly refuses to promote — so recording it
        is what makes acquire-once-then-replicate work at all.
        """
        if not isinstance(result, dict):
            return
        digest = result.get("content_digest")
        size = result.get("size_bytes")
        if digest is None and size is None:
            return
        model = self._repo.find_model(source_id, source_model_id, revision, file_selector)
        if model is None:
            return
        self._repo.update_model_content(model.id, content_digest=digest, size_bytes=size)

    def _resolve_credential(self, source_id: str, credential: str | None) -> str | None:
        """Resolve a credential *name* to its value for one acquisition.

        ``None`` when the source has no credential to apply — not an error here.
        Whether one is required is upstream's judgement, and it arrives as
        ``authorization_refused`` when it is.
        """
        if self._credentials is None:
            return None
        resolved: str | None = self._credentials.resolve_for_acquisition(source_id, credential)
        return resolved

    @staticmethod
    def _requested_revision(revision: str | None) -> str:
        """Return the upstream request reference; it is not recorded as identity."""
        return revision or "main"

    @staticmethod
    def _resolved_revision_from_agent(result: dict[str, Any] | None, model_id: str) -> str:
        """Require the immutable revision that the agent actually acquired."""
        if not isinstance(result, dict) or result.get("revision_pinned") is not True:
            raise UnpinnableError(f"source cannot pin a revision for {model_id!r}")
        resolved = result.get("resolved_revision")
        if not isinstance(resolved, str) or not resolved or resolved == "main":
            raise UnpinnableError(f"source did not return an immutable revision for {model_id!r}")
        return resolved

    def _acquire_on_node(
        self,
        source: ModelSource,
        source_model_id: str,
        revision: str,
        node_id: str,
        secret: str | None,
        file_selector: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        """Drive the agent's acquire for one node.

        ``secret`` is the resolved credential value, passed for this one call
        and held nowhere. Returns the agent's response, which
        carries the ``content_digest`` later nodes verify their peer transfer
        against.
        """
        node = self._repo.get_node(node_id)
        if node is None:
            raise NotFoundError(f"no node with id {node_id!r}")
        try:
            response: dict[str, Any] | None = self._client.acquire_model(
                node,
                source_id=source.id,
                source_model_id=source_model_id,
                revision=revision,
                credential=secret,
                file_selector=file_selector,
            )
            return response
        except AgentCallError as exc:
            # Record a failed replica so it is never presented as available.
            self._save_replica(
                source_id=source.id,
                source_model_id=source_model_id,
                revision=revision,
                node_id=node_id,
                state=ReplicaState.FAILED,
                file_selector=file_selector,
            )
            # The reason is not stored on the replica: ModelReplica has no field
            # for it, and it was previously passed here and discarded. It is
            # recorded on the Operation instead, which is the requirement.
            refusal = self._as_authorization_refusal(exc, source, source_model_id, node_id)
            if refusal is None:
                raise
            raise refusal from exc

    def _as_authorization_refusal(
        self,
        exc: AgentCallError,
        source: ModelSource,
        source_model_id: str,
        node_id: str,
    ) -> AuthorizationRefusedError | None:
        """Promote an upstream authorization refusal to a typed domain failure.

        An agent-transport error is an internal shape; ``authorization_refused``
        is the one an operator must be able to act on, so it becomes a typed
        domain error the API renders as ``403 authorization_refused`` and the
        CLI exits ``4`` on. Every other agent failure keeps its
        existing shape, so this is a promotion and not a catch-all.

        The message is made to **name the source** as well as the model, because
        a refusal an operator cannot attribute to an upstream is not actionable.
        """
        if exc.code != "authorization_refused":
            return None
        detail = dict(exc.detail)
        detail.setdefault("source_id", source.id)
        detail.setdefault("source_model_id", source_model_id)
        message = exc.message
        if source.id not in message:
            message = f"{source.id}: {message}"
        return AuthorizationRefusedError(message, node_id=node_id, detail=detail)

    def _record_available(
        self,
        source_id: str,
        source_model_id: str,
        revision: str,
        node_id: str,
        file_selector: tuple[str, ...] = (),
    ) -> ModelReplica:
        """Record an available replica for a node at a resolved revision."""
        replica = ModelReplica(
            model_id=self._find_or_create_model(
                source_id, source_model_id, revision, [], file_selector
            ).id,
            node_id=node_id,
            local_path=_store_path(source_id, source_model_id, file_selector),
            state=ReplicaState.AVAILABLE,
            verified_at=datetime.now().astimezone(),
        )
        self._repo.save_replica(replica)
        return replica

    def _save_replica(
        self,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        node_id: str,
        state: ReplicaState,
        file_selector: tuple[str, ...] = (),
    ) -> None:
        """Persist a model row and its replica on one node.

        This is the **failure** path: its one caller records a FAILED replica
        after an acquire raised. ``revision`` is therefore the reference the
        operator *asked for*, not one any source resolved -- routinely the
        moving label ``main``.

        So the model is recorded explicitly unpinned. It previously inherited
        ``revision_pinned = revision is not None``, which read ``main`` as a
        concrete revision and wrote a row claiming a moving label had been
        pinned. Nothing had noticed, because the
        row is only written once an acquire has already failed.
        """
        model = self._find_or_create_model(
            source_id, source_model_id, revision, [], file_selector, revision_pinned=False
        )
        replica = ModelReplica(
            model_id=model.id,
            node_id=node_id,
            local_path=_store_path(source_id, source_model_id, file_selector),
            state=state,
            verified_at=datetime.now().astimezone() if state == ReplicaState.AVAILABLE else None,
        )
        self._repo.save_replica(replica)

    def _verified_replica_at(
        self,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        node_id: str,
        file_selector: tuple[str, ...] = (),
    ) -> ModelReplica | None:
        """Return the existing available replica for this node at this revision.

        ``None`` when the node does not already hold a verified replica at the
        same revision, so the coordinator transfers nothing it does not need to.
        """
        model = self._repo.find_model(source_id, source_model_id, revision, file_selector)
        if model is None:
            return None
        replica = self._repo.get_replica(model.id, node_id)
        if replica is None or replica.state != ReplicaState.AVAILABLE:
            return None
        return replica

    def _find_or_create_model(
        self,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        replicas: list[ModelReplica],
        file_selector: tuple[str, ...] = (),
        *,
        revision_pinned: bool | None = None,
    ) -> Model:
        """Return the logical model for this revision, creating it if absent.

        ``revision_pinned`` defaults to "a revision was supplied", which is
        right on the success path: that revision came back from the agent and
        is concrete. A caller holding a reference the source never resolved
        must say so, rather than let this infer pinning from mere presence.
        """
        model = self._repo.find_model(source_id, source_model_id, revision, file_selector)
        if model is None:
            pinned = (revision is not None) if revision_pinned is None else revision_pinned
            model = Model(
                id=new_ulid(),
                source_id=source_id,
                source_model_id=source_model_id,
                resolved_revision=revision,
                revision_pinned=pinned,
                file_selector=file_selector,
            )
            self._repo.save_model(model)
        # Persist any replicas carried by the caller.
        for replica in replicas:
            self._repo.save_replica(replica)
        return model

    # ----------------------------------------------------------------- source
    def _source(self, source_id: str) -> ModelSource:
        source = self._repo.get_model_source(source_id)
        if source is None:
            # Register the ratified source (huggingface) on first use.
            source = ModelSource(
                id=source_id,
                supports_revision_pinning=True,
                requires_credential=True,
            )
            self._repo.save_model_source(source)
        return source

    # --------------------------------------------------------------- read side
    def list(self) -> list[Model]:
        return self._repo.list_models()

    def get(self, model_id: str) -> Model:
        model = self._repo.get_model(model_id)
        if model is None:
            raise NotFoundError(f"no model with id {model_id!r}")
        return model

    # ------------------------------------------------------------- delete side
    def delete(self, model_id: str) -> None:
        """Delete a logical model and its replicas.

        Refused with ``still_referenced`` while any retained deployment
        revision references the logical model, naming the referring
        deployments. Only the coordinator knows the deployment
        graph, so this check is here, not on the agent.
        """
        model = self.get(model_id)

        # Check that no deployment revision references this model.
        referrers = self._model_referrers(model_id)
        if referrers:
            names = ", ".join(referrers)
            raise StillReferencedError(
                f"model {model.source_model_id!r} is referenced by deployments: {names}",
                detail={"referrers": referrers, "model_id": model_id},
            )

        # Remove replicas from each node that holds one. Best-effort: an agent
        # that cannot be reached does not prevent the coordinator from dropping
        # its declared record, and the HTTP side-effect cannot be rolled back.
        import contextlib

        for replica in self._repo.list_replicas(model_id):
            node = self._repo.get_node(replica.node_id)
            if node is not None:
                with contextlib.suppress(AgentCallError):
                    self._client.delete_model(node, model_id)

        # Remove the model row and its replicas from the store in one
        # transaction, so the delete is atomic: a model referenced by a
        # retained deployment revision is caught by the check above (a typed
        # ``still_referenced``), and any other failure rolls the replicas back
        # rather than leaving ``replicas: []`` under a model row that remains.
        self._repo.delete_model_and_replicas(model_id)

    def _model_referrers(self, model_id: str) -> builtins.list[str]:
        """Return deployment names whose *any retained revision* references this model.

        A deployment retains every revision for rollback, and each
        retained revision's ``model_id`` is a live reference: deleting the model
        would break a rollback to that revision. The check therefore inspects
        every retained revision, not only ``current_revision`` -- which is what
        the ``delete`` docstring already claimed, and what the enforced
        foreign key ``deployment_revisions.model_id -> models.id`` makes
        structural. Checking only the current revision let the guard pass and
        the FK fire after the replicas had already committed, surfacing as
        ``internal_error`` on a call that had partially succeeded.
        """
        referrers: list[str] = []
        for deployment in self._repo.list_deployments():
            for revision in self._repo.list_revisions(deployment.id):
                if revision.model_id == model_id:
                    referrers.append(deployment.name)
                    break  # one match per deployment is enough to refuse
        return referrers

    # ----------------------------------------------------------- image side
    def list_images(self) -> builtins.list[Any]:
        """List images present on nodes."""
        return self._repo.list_images()

    def delete_image(self, node_id: str, digest: str) -> dict[str, Any]:
        """Delete an image from a node — the bytes, then the record.

        Refused with ``still_referenced`` while **any retained** deployment
        revision references the image digest, naming the referring deployments.

        This used to delete the record and nothing else, while reporting
        "Image removed from node". It freed no bytes, and because the record is
        the only handle the product has on the object, deleting it made the
        image unreachable rather than gone: 48 deletes, 0 bytes, 430 GB that
        no record named any more.

        So the node goes first and the record follows only if the node agreed:

        * removed, or already absent → delete the record. "Already absent" is
          the reaping case, and it is how a record that outlived its object
          gets cleaned up rather than becoming permanent.
        * a container still holds it → ``AgentCallError`` propagates and **the
          record stays**, because the bytes stay. An operator who cannot see the
          image in ``image list`` cannot be asked to deal with it.
        * node unreachable → same. A delete that cannot reach the node has not
          deleted anything, and saying otherwise is the whole defect.
        """
        referrers = self._image_referrers(node_id, digest)
        if referrers:
            from tensorstead.domain.errors import StillReferencedError

            names = ", ".join(referrers)
            raise StillReferencedError(
                f"image {digest!r} on node {node_id!r} is referenced by deployments: {names}",
                detail={"referrers": referrers, "node_id": node_id, "digest": digest},
            )

        record = self._image_record(node_id, digest)
        if record is None:
            raise NotFoundError(f"no image {digest!r} recorded on node {node_id!r}")
        node = self._repo.get_node(node_id)
        if node is None:
            raise NotFoundError(f"no node with id {node_id!r}")

        # Deliberately **not** gated on ``node.agent_contract_version``. That
        # field is the registration snapshot and is never rewritten, so gating
        # on it would refuse a capability an upgraded node has had for days --
        # the defect already recorded against it in contracts/api.py. An agent
        # too old to serve the route answers 404, which arrives here as
        # ``AgentCallError`` and leaves the record standing, which is the
        # property that actually matters.

        # By reference, not by digest: an object carrying several tags has one
        # record per tag, and removing a tag is what a per-reference record
        # means. The object goes when its last tag does.
        result = self._client.remove_image(node, reference=record.reference)
        removed = bool(result.get("removed")) if isinstance(result, dict) else False

        self._repo.delete_image(node_id, digest)
        return {
            "node_id": node_id,
            "digest": digest,
            "reference": record.reference,
            "removed_from_node": removed,
            # Both outcomes delete the record; only one of them freed anything.
            "outcome": "removed" if removed else "record_reaped",
        }

    def _image_record(self, node_id: str, digest: str) -> Any | None:
        """The recorded image for this node and digest, or None."""
        for image in self._repo.list_images():
            if image.node_id == node_id and image.digest == digest:
                return image
        return None

    def reconcile_images(self, node_id: str | None = None) -> dict[str, Any]:
        """Compare image records against what the nodes hold, and reap the orphans.

        Exists because the records drifted badly and silently: 50 records
        against 25 real objects, one object recorded as two, and two images
        present on the nodes with no record at all. Nothing
        compared them, and until image deletion reached the node nothing could have
        kept them aligned.

        Read-only against the nodes. It deletes **records**, never images: a
        record whose object the node does not hold is the one thing here that
        is certainly wrong, and it is what makes the bytes unreachable. An
        image present with no record is reported, not invented into one —
        adopting it would guess an origin and a ``produced_by`` the product
        does not know, and the distinction is load-bearing.

        A node that cannot be reached is reported and its records left alone.
        Absence of evidence is not evidence of absence, which is exactly the
        inference that produced the drift.
        """
        nodes = [n for n in self._repo.list_nodes() if node_id is None or n.id == node_id]
        if node_id is not None and not nodes:
            raise NotFoundError(f"no node with id {node_id!r}")

        records = self._repo.list_images()
        reaped: list[dict[str, str]] = []
        unrecorded: list[dict[str, str]] = []
        unreachable: list[dict[str, str]] = []

        for node in nodes:
            try:
                present = self._client.list_images(node)
            except CompatibilityRefusal:
                # An agent too old to be asked has told us nothing, which is
                # the same standing as an unreachable one: report it, touch
                # none of its records. Not pre-checked against the registration
                # snapshot, for the reason delete_image gives.
                unreachable.append({"node_id": node.id, "reason": "operation_unsupported_by_agent"})
                continue
            except AgentCallError as exc:
                unreachable.append({"node_id": node.id, "reason": exc.code})
                continue

            present_digests = {row.get("digest") for row in present}
            present_pairs = {(row.get("reference"), row.get("digest")) for row in present}

            for record in [r for r in records if r.node_id == node.id]:
                if (record.reference, record.digest) in present_pairs:
                    continue
                # A digest the node still holds under a different reference is
                # a retagging, not an orphan. Reaping it would delete the only
                # record of bytes that are really there.
                if record.digest in present_digests:
                    continue
                self._repo.delete_image(record.node_id, record.digest)
                reaped.append(
                    {
                        "node_id": record.node_id,
                        "reference": record.reference,
                        "digest": record.digest,
                    }
                )

            recorded_pairs = {(r.reference, r.digest) for r in records if r.node_id == node.id}
            for row in present:
                if (row.get("reference"), row.get("digest")) not in recorded_pairs:
                    unrecorded.append(
                        {
                            "node_id": node.id,
                            "reference": str(row.get("reference")),
                            "digest": str(row.get("digest")),
                        }
                    )

        return {
            "reaped": reaped,
            "unrecorded": unrecorded,
            "unreachable": unreachable,
        }

    def _image_referrers(self, node_id: str, digest: str) -> builtins.list[str]:
        """Return deployment names whose *any retained revision* uses this image here.

        A deployment retains every revision for rollback, so a
        retained revision's ``image_digest`` is a live reference: deleting the
        image would break a rollback to it. This inspected only
        ``current_revision`` while its own docstring promised every retained
        one — the identical defect ``_model_referrers`` had, fixed for models by
        a recorded finding and left standing here for another six weeks.
        """
        referrers: list[str] = []
        for deployment in self._repo.list_deployments():
            for revision in self._repo.list_revisions(deployment.id):
                if revision.image_digest == digest and node_id in revision.participating_nodes:
                    referrers.append(deployment.name)
                    break  # one match per deployment is enough to refuse
        return referrers


class ImageBuildService:
    """Managed runtime images: recorded build specs, builds, and imports.

    Exists because the product could not repair a broken upstream image. When
    `nvcr.io/nvidia/vllm:26.07-py3` shipped an xgrammar too old for its own
    tool-calling path, the only route available to an operator or an agent was
    SSH — which is the workflow this design exists to replace. Declining to
    support image production did not prevent the manipulation, it relocated it
    outside the system where nothing recorded it.

    The design named "image acquisition **or build**" from the beginning; only the
    implementation was missing.
    """

    def __init__(self, repository: Any, node_client: Any) -> None:
        self._repo = repository
        self._client = node_client

    # ------------------------------------------------------------ build specs
    def record_spec(
        self,
        name: str,
        *,
        base_image: str,
        steps: list[str],
        entrypoint: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a build spec. Executes nothing."""
        spec = ImageBuildSpec(
            name=name,
            base_image=base_image,
            steps=tuple(steps),
            entrypoint=tuple(entrypoint or ()),
        )
        self._repo.save_build_spec(spec)
        result: dict[str, Any] = self._spec_summary(spec)
        if not spec.base_is_pinned:
            # Reported, never refused: pinning is the operator's judgement, on
            # the same terms as the refusal to gate on our observations.
            result["warning"] = (
                f"base image {spec.base_image!r} is not pinned to a digest, so this "
                "build is not reproducible; pin it with @sha256:... to make it so"
            )
        return result

    def list_specs(self) -> list[dict[str, Any]]:
        return [self._spec_summary(s) for s in self._repo.list_build_specs()]

    @staticmethod
    def _spec_summary(spec: ImageBuildSpec) -> dict[str, Any]:
        """One shape for a build spec, used by every path that returns one.

        There were two. ``record_spec`` returned ``entrypoint`` unconditionally
        and ``list_specs`` never returned it, so the same object described
        itself differently depending on which call you made -- noticed reading
        `buildspec list` against the live store right after deploying the
        column. Small, and the same shape as every larger instance of it.

        An empty entrypoint is omitted rather than rendered as ``[]``: a spec
        that does not set one is using the image's own, which is what every
        spec recorded before the column was doing. Showing an empty list on
        each of them would announce a feature nobody used, which is the trap
        ``extra_args`` set when it defaulted to an empty dict.
        """
        summary: dict[str, Any] = {
            "name": spec.name,
            "base_image": spec.base_image,
            "steps": list(spec.steps),
            "base_is_pinned": spec.base_is_pinned,
            "created_at": spec.created_at.isoformat(),
        }
        if spec.entrypoint:
            summary["entrypoint"] = list(spec.entrypoint)
        return summary

    def delete_spec(self, name: str) -> dict[str, Any]:
        """Delete a spec, refused while an image it produced is referenced."""
        if self._repo.get_build_spec(name) is None:
            raise NotFoundError(f"build spec {name!r} is not recorded")
        referrers = self._referring_deployments(produced_by=name)
        if referrers:
            raise StillReferencedError(
                f"build spec {name!r} produced an image still used by: {', '.join(referrers)}",
                detail={"referrers": referrers},
            )
        self._repo.delete_build_spec(name)
        return {"name": name, "status": "deleted"}

    # ------------------------------------------------------- build and import
    def require_spec(self, name: str) -> ImageBuildSpec:
        """Resolve a recorded build spec, refusing an unknown name.

        Split out so the route can refuse an unrecorded spec *before* accepting
        the operation. A build now returns 202 and runs in the background,
        and a name that was never recorded is knowable
        without doing any work -- so it stays an immediate 404 rather than
        becoming a failed operation the caller has to poll to discover.
        """
        spec: ImageBuildSpec | None = self._repo.get_build_spec(name)
        if spec is None:
            raise NotFoundError(f"build spec {name!r} is not recorded")
        return spec

    def build(self, name: str, *, node_id: str, reference: str) -> dict[str, Any]:
        """Build a recorded spec on one node and record its provenance.

        Produced **once** on a nominated node. Building separately on each node
        yields different identifiers for the same spec, which breaks the export
        comparison silently — the distribution step exists for that
        reason and follows the acquire-once-then-replicate pattern.
        """
        spec = self.require_spec(name)
        node = self._require_node(node_id)

        try:
            response = self._client.build_image(
                node,
                {
                    "reference": reference,
                    "base_image": spec.base_image,
                    "steps": list(spec.steps),
                    "entrypoint": list(spec.entrypoint),
                },
            )
        except AgentCallError as exc:
            # The agent's own detail is merged in, not replaced. It carries the
            # build log tail and the failing step; keeping only the coordinator's
            # three identifiers is how a failed build used to reach the operation
            # record naming the node and the exit code and nothing else -- which
            # sent an operator to reproduce the build over SSH, off this control
            # plane and outside its audit trail. The coordinator's keys are
            # written last so they win a collision: which node ran the build is
            # something only this side knows.
            raise NodeOperationFailedError(
                f"build of {reference!r} failed on node {node.name!r}: {exc.message}",
                detail={
                    **exc.detail,
                    "reference": reference,
                    "spec": name,
                    "node_id": node_id,
                },
            ) from exc
        image_id = str((response or {}).get("image_id", ""))
        self._record_produced(node_id, reference, image_id, ImageOrigin.BUILT, name)
        return {
            "reference": reference,
            "image_id": image_id,
            "origin": ImageOrigin.BUILT.value,
            "produced_by": name,
            "node_id": node_id,
            # Named image_id, never digest: a locally produced image has no
            # registry digest and the design forbids presenting one as the other.
            "is_registry_digest": False,
        }

    def build_and_distribute(
        self, name: str, *, nodes: list[str], reference: str
    ) -> dict[str, Any]:
        """Build once on the first node, then distribute to the rest.

        Building on each node independently would produce a different
        identifier for the same spec, which breaks the export comparison and
        makes divergence detection meaningless. Producing once is the whole
        point of this operation existing.

        A node that cannot receive the image is a failure naming that node,
        and the overall outcome is failure. The
        nodes that did receive it are left alone rather than unwound: the
        product never reverses work it has done on a host without being asked
        (this product's rule).
        """
        if not nodes:
            raise NotFoundError("at least one node is required to build on")

        built = self.build(name, node_id=nodes[0], reference=reference)
        image_id = str(built["image_id"])
        per_node: dict[str, Any] = {nodes[0]: {"status": "built", "image_id": image_id}}
        failures: list[str] = []

        source = self._require_node(nodes[0])
        for node_id in nodes[1:]:
            node = self._repo.get_node(node_id)
            if node is None:
                failures.append(node_id)
                per_node[node_id] = {"status": "failed", "detail": "node not registered"}
                continue
            try:
                response = self._client.distribute_image(
                    node,
                    {
                        "reference": reference,
                        "source_endpoint": source.agent_endpoint,
                        "expected_image_id": image_id,
                    },
                )
            except Exception as exc:
                failures.append(node.name)
                # The agent's structured ``code`` is kept alongside its message
                # rather than flattened into one string. An
                # operator reading a per-node outcome needs to know *which*
                # failure it was -- a refused credential and an identifier
                # mismatch are different problems with different next steps --
                # and ``str(exc)`` on an ``AgentCallError`` drops the code.
                per_node[node_id] = {
                    "status": "failed",
                    "code": getattr(exc, "code", "image_distribution_failed"),
                    "detail": str(exc),
                }
                continue
            arrived = str((response or {}).get("image_id", ""))
            if arrived != image_id:
                failures.append(node.name)
                per_node[node_id] = {"status": "failed", "detail": "identifier mismatch"}
                continue
            self._record_produced(node_id, reference, arrived, ImageOrigin.BUILT, name)
            per_node[node_id] = {"status": "distributed", "image_id": arrived}

        return {
            "reference": reference,
            "image_id": image_id,
            "produced_by": name,
            "per_node": per_node,
            # Partial success is an overall failure, and the nodes that failed
            # are named.
            "status": "failed" if failures else "succeeded",
            "failed_nodes": sorted(failures),
        }

    def import_archive(
        self, *, node_id: str, reference: str, archive_name: str, expected_image_id: str | None
    ) -> dict[str, Any]:
        """Import a prebuilt archive, verified before it becomes available."""
        node = self._require_node(node_id)
        try:
            response = self._client.import_image(
                node,
                {
                    "archive_name": archive_name,
                    "reference": reference,
                    "expected_image_id": expected_image_id,
                },
            )
        except AgentCallError as exc:
            raise NodeOperationFailedError(
                f"import of {archive_name!r} failed on node {node.name!r}: {exc.message}",
                detail={
                    **exc.detail,
                    "reference": reference,
                    "archive": archive_name,
                    "node_id": node_id,
                },
            ) from exc
        image_id = str((response or {}).get("image_id", ""))
        self._record_produced(node_id, reference, image_id, ImageOrigin.IMPORTED, archive_name)
        return {
            "reference": reference,
            "image_id": image_id,
            "origin": ImageOrigin.IMPORTED.value,
            "produced_by": archive_name,
            "node_id": node_id,
            "is_registry_digest": False,
        }

    # ----------------------------------------------------------------- helpers
    def _record_produced(
        self, node_id: str, reference: str, image_id: str, origin: ImageOrigin, produced_by: str
    ) -> None:
        if not image_id:
            return
        self._repo.save_image(
            ImageRecord(
                node_id=node_id,
                reference=reference,
                digest=image_id,
                pulled_at=datetime.now().astimezone(),
                origin=origin,
                produced_by=produced_by,
            )
        )

    def _require_node(self, node_id: str) -> Any:
        node = self._repo.get_node(node_id)
        if node is None:
            raise NotFoundError(f"node {node_id!r} is not registered")
        return node

    def _referring_deployments(self, *, produced_by: str) -> list[str]:
        references = {
            image.reference
            for image in self._repo.list_images()
            if getattr(image, "produced_by", None) == produced_by
        }
        referrers: list[str] = []
        for deployment in self._repo.list_deployments():
            revision = self._repo.get_revision(deployment.id, deployment.current_revision)
            if revision is not None and revision.image_reference in references:
                referrers.append(deployment.name)
        return sorted(referrers)
