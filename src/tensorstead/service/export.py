"""Deployment export service.

A pure projection of one revision into the reproducible artifact:
YAML a person can read months later and recreate from, and that a parser can
map back onto a create request.

- **Secrets are structurally absent**: the projection
  draws only from a ``DeploymentRevision``, which holds no field that ever
  carries a secret value. There is nothing for a redaction step to miss.
- **Unpinned is stated, never implied**: when
  ``revision_pinned`` is false the artifact carries an explicit prose note,
  so a reader cannot believe a guarantee the source never made.
- **``origin_platform`` records, never constrains**: the
  originating node's facts are rendered so a re-import can warn on
  comparability, but nothing is refused on it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import yaml

from tensorstead.domain.models import Deployment, DeploymentRevision, Operation

_APIVER = "tensorstead/v1"
_KIND = "DeploymentExport"

# Unpinned prose note: must be present whenever revision_pinned is false.
_UNPINNED_NOTE = (
    "This source does not expose a stable revision identifier. Recreating this "
    "deployment may not reproduce identical model weights."
)

# Significant lifecycle history to include in the operation block.
_OPERATION_KINDS = {
    "deployment_create": "deployment_create",
    "start": "start",
    "stop": "stop",
    "restart": "restart",
    "remove": "remove",
    "reconcile": "reconcile",
}


class ExportService:
    """Render one revision of a deployment as the reproducible YAML artifact."""

    def __init__(self, repository: Any) -> None:
        self._repo = repository

    # ---------------------------------------------------------------- export
    def export(
        self,
        deployment: Deployment,
        revision: DeploymentRevision,
        *,
        operations: list[Operation] | None = None,
    ) -> str:
        """Render ``revision`` as the YAML document.

        ``exported_from_revision`` identifies which revision this represents;
        ``origin_platform`` is rendered from the revision's recorded
        facts; the unpinned note is present exactly when the
        source never pinned a revision.
        """
        data = self._project(deployment, revision, operations=operations or [])
        body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        if isinstance(body, bytes):
            body = body.decode()
        return str(body).rstrip() + "\n"

    def as_dict(
        self,
        deployment: Deployment,
        revision: DeploymentRevision,
        *,
        operations: list[Operation] | None = None,
    ) -> dict[str, Any]:
        """Return the projection as a plain dict (used by tests and the API)."""
        return self._project(deployment, revision, operations=operations or [])

    def _project(
        self,
        deployment: Deployment,
        revision: DeploymentRevision,
        *,
        operations: list[Operation],
    ) -> dict[str, Any]:
        model_block: dict[str, Any] = {
            "source": revision.model_source_id,
            "id": revision.source_model_id,
            "revision": revision.resolved_revision,
            "revision_pinned": revision.revision_pinned,
        }
        if not revision.revision_pinned:
            # The prose note must travel with the explicit unpinned flag.
            model_block["note"] = _UNPINNED_NOTE

        # A model may be part of its repository, and which part is identity
        # (migration 0006). An export that omitted it would name a repository at
        # a revision and recreate against whichever selection happened to match
        # -- a different set of weights under the same three fields. Read from
        # the model rather than the revision because the revision records the
        # model's id, and identity belongs to the model.
        #
        # Emitted only when there is a selection, so every export of a
        # whole-repository model is byte-identical to what it was before this.
        recorded = self._repo.get_model(revision.model_id)
        selector = list(getattr(recorded, "file_selector", ()) or ()) if recorded else []
        if selector:
            model_block["file_selector"] = selector

        image = {
            "reference": revision.image_reference,
            # Derived, not declared. The digest of an image cannot be known at
            # create time without pulling it, and pulling would break "the
            # host is untouched until start". It is recorded when the
            # agent pulls, and read back here, so the export states the digest
            # that actually ran rather than one predicted before it did.
            "digest": revision.image_digest or self._recorded_digest(revision.image_reference),
        }
        platform = self._platform(revision)
        if platform:
            image["platform"] = platform
        image.update(self._image_origin(revision.image_reference))

        return {
            "apiVersion": _APIVER,
            "kind": _KIND,
            "exported_at": datetime.now().astimezone().isoformat(),
            "exported_from_revision": revision.revision,
            "deployment": {
                "name": deployment.name,
                "id": deployment.id,
                "desired_state": deployment.desired_state,
            },
            "model": model_block,
            "runtime": {
                "type": revision.runtime_type,
                "version": revision.runtime_version,
            },
            "image": image,
            "runtime_config": revision.runtime_config,
            "placement": {
                "nodes": self._node_names(revision.participating_nodes),
                "endpoint": revision.endpoint,
            },
            "origin_platform": self._origin_platform(revision),
            "operations": self._operations_block(operations),
            **self._code_approval_block(revision),
        }

    def _code_approval_block(self, revision: Any) -> dict[str, Any]:
        """What authorized this revision to load code, and whether it still does.

        Omitted entirely when the revision loads no code, so no existing export
        grows a key announcing a feature it does not use.

        Reports both what the revision recorded and what the approvals table
        says now, because they answer different questions. The recorded
        fingerprint is history and cannot change. ``status`` is the live
        reading: ``revoked`` means the approval that authorized this revision no
        longer exists, so the definition in this export will be refused if
        anyone tries to start it -- which an export exists to tell you *before*
        you carry it to another estate, not after.
        """
        recorded = getattr(revision, "code_approval_fingerprint", "")
        if not recorded:
            return {}
        approval = self._repo.find_code_approval(recorded)
        block: dict[str, Any] = {
            "fingerprint": recorded,
            "approval_id": getattr(revision, "code_approval_id", ""),
            "status": "active" if approval is not None else "revoked",
        }
        if approval is not None:
            block.update(
                {
                    "option": approval.option,
                    "reason": approval.reason,
                    "approved_by": approval.approved_by,
                    "policy_version": approval.policy_version,
                    "approved_at": approval.created_at.isoformat(),
                    "model_revision": approval.model_revision,
                    "image_digest": approval.image_digest,
                }
            )
        return {"code_execution_approval": block}

    def _recorded_digest(self, reference: str) -> str | None:
        """The digest observed when this image was pulled or produced.

        `DeploymentRevision.image_digest` is empty on every deployment: it was
        persisted as "" behind a comment claiming resolution happened at start,
        and nothing resolved it. Rather than make create pull an image, the
        recorded observation is the answer -- and it is a better one, because
        it is what was seen rather than what was expected.
        """
        for record in self._repo.list_images():
            if record.reference == reference and record.digest:
                return str(record.digest)
        return None

    def _image_origin(self, reference: str) -> dict[str, Any]:
        """How a non-pulled image came to exist, so it can be reproduced.

        An export is meant to be sufficient to recreate a deployment elsewhere.
        For an image built or imported on this estate, the reference
        alone is not: another estate cannot pull `local/vllm:patched`. Naming
        the build spec makes the image reproducible; naming the archive at
        least makes its provenance findable.

        Contents are never included, only provenance. The build spec
        itself is exported so the recipe travels, which is the whole point --
        it is ordered steps over a base image, and holds no secret.
        """
        records = [
            record
            for record in self._repo.list_images()
            if record.reference == reference and getattr(record, "produced_by", None)
        ]
        if not records:
            return {}
        record = records[0]
        origin = getattr(record, "origin", None)
        origin_value = getattr(origin, "value", origin)
        block: dict[str, Any] = {"origin": origin_value, "produced_by": record.produced_by}

        if origin_value == "built":
            spec = self._repo.get_build_spec(record.produced_by)
            if spec is not None:
                block["build_spec"] = {
                    "base_image": spec.base_image,
                    "steps": list(spec.steps),
                    "base_is_pinned": spec.base_is_pinned,
                }
                if not spec.base_is_pinned:
                    block["note"] = (
                        "the base image is not pinned to a digest, so rebuilding "
                        "this spec elsewhere may not produce the same image"
                    )
        else:
            block["note"] = (
                "this image was imported from an archive; recreating this "
                "deployment elsewhere requires that archive"
            )
        return block

    @staticmethod
    def _platform(revision: DeploymentRevision) -> str | None:
        """Render the platform portion of the image block when known."""
        facts = revision.origin_platform_facts or {}
        cpu = facts.get("cpu_arch")
        os_family = facts.get("os_family")
        if cpu and os_family:
            return f"linux/{cpu}"
        return None

    @staticmethod
    def _origin_platform(revision: DeploymentRevision) -> dict[str, Any]:
        """Render the recorded origin platform facts."""
        facts = revision.origin_platform_facts or {}
        return {
            "cpu_arch": facts.get("cpu_arch"),
            "os_family": facts.get("os_family"),
            "accelerator_model": facts.get("accelerator_model"),
            "accelerator_memory_total": facts.get("accelerator_memory_total"),
            "memory_is_unified": bool(facts.get("memory_is_unified", False)),
        }

    def _node_names(self, node_ids: tuple[str, ...] | list[str]) -> list[str]:
        """Resolve participating node ids to readable names for the export.

        The revision stores node **ids**; the export renders **names**, because a
        person must be able to read and recreate from them, and re-import
        resolves names against the current inventory. When a
        node is no longer in the inventory the id is rendered as a fallback so the
        artifact stays self-consistent.
        """
        names: list[str] = []
        for node_id in node_ids:
            node = self._repo.get_node(node_id)
            names.append(node.name if node is not None else node_id)
        return names

    @staticmethod
    def _operations_block(operations: list[Operation]) -> list[dict[str, Any]]:
        """Significant lifecycle history as a bounded, human-readable list."""
        block: list[dict[str, Any]] = []
        for op in operations:
            kind = _OPERATION_KINDS.get(op.kind.value)
            if kind is None:
                continue
            block.append(
                {
                    "kind": kind,
                    "revision": op.deployment_revision,
                    "state": op.state.value,
                    "finished_at": op.finished_at.isoformat() if op.finished_at else None,
                }
            )
        return block
